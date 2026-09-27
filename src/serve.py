# -*- coding: utf-8 -*-
r"""阶段四 本地问答页面：标准库 http.server，**不装任何新包**（requests/openai 本机都没有）。

设计上有三处是**踩过的坑逼出来的**，不是风格偏好：

1. **白名单路由，绝不用 SimpleHTTPRequestHandler(directory=ROOT)**。
   那个类会把整棵目录树静态发出去——`src/`（含本文件与 key 的读取逻辑）和
   `out/`（含 chunks.jsonl、索引）都能被任人下载。这里每个路由都手写，其余一律 404。

2. **只绑 127.0.0.1，且不发 `Access-Control-Allow-Origin`**。
   key 活在服务进程里，浏览器永远拿不到它。但服务一旦绑 `0.0.0.0`，
   同一个局域网里任何人都能拿这个页面当免费 LLM 代理刷 key。这个风险不值得冒。

3. **服务端渲染，只有一个渲染器**。`GET /?q=…` 直接吐出完整 HTML，
   页面里的提交按钮走的也是这个 GET（不是 fetch + 前端拼 DOM）。
   这样「截图看到的」与「用户看到的」是同一段代码——阶段四的交付物就是截图，
   两套渲染器一旦漂移，截图就不再是证据了。
   `POST /api/ask` 仍然保留，供阶段五/六取**结构化**的逐题记录（JSON）。

key 只从环境变量读，**不进 JS、不进响应**。返回前有一句 assert 把这件事变成代码断言，
而不是口头承诺——「绝不发给前端」这话，不断言就等于没说。

用法：
    PY=/c/Users/yexinyu/AppData/Local/Python/bin/python.exe   # 裸 python 是 Store 存根
    $PY -X utf8 -u src/serve.py                 # 起服务（必须 -X utf8）
    $PY -X utf8 -u src/serve.py --port 8010
"""
import argparse
import csv
import html
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import search as S            # noqa: E402
import ask as A               # noqa: E402

PDF_DIR = os.path.join(ROOT, "data", "pdf")
CSV = os.path.join(ROOT, "data", "companies.csv")

log = logging.getLogger("serve")
IX = None                     # 进程内只建一次，见 main()
LOCK = threading.Lock()       # 检索本身只读，但嵌入模型不是线程安全的，排队最省心
PDF_OK = set()                # 允许发的 PDF 文件名**白名单**（来自 companies.csv）


# ------------------------------------------------------------------ 元数据
def load_companies():
    """读 companies.csv（utf-8-sig，**带 BOM**，用 -sig 免得第一列键名变成 '\\ufeffcode'）。
    返回 {公司名: 行}。PDF 白名单也从这里来，不另维护一份。"""
    out = {}
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["name"].strip()] = row
    return out


def page_note(meta, company):
    """页眉/页脚偏移的口径说明，直接取阶段一实测的字段——不在这里重新算。

    **`page_offset` 不一定是数字**：国货航那一格是 `twoup`（双页拼版，99 个 PDF 页
    拆成 196 个页单元，左页 = 2p−6）。第一版只管了数字和 0/空，遇到 twoup 会印出
    「印页 = 物理页 − twoup（…）」这种**看着有内容其实是废话**的字。跑起来才发现，
    所以这里按实测值分档写明。"""
    m = meta.get(company) or {}
    off = (m.get("page_offset") or "").strip()
    note = (m.get("note") or "").strip()
    if off in ("", "0"):
        return "印页 = 物理页"
    if off == "twoup":
        return ("**双页拼版**：印页与物理页不是简单偏移（实测 左页 = 2p−6）；"
                "跳过去会是跨页版式，属已知偏差")
    return "印页 = 物理页 − %s（%s）" % (off, note[:80] if note else "阶段一实测")


def _em(escaped):
    """`**强调**` → `<b>强调</b>`。**没有 markdown 渲染器**，不转的话页面上就是
    一排光秃秃的星号（第一版就是这样）。入参必须**已经 HTML 转义过**。"""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


# ------------------------------------------------------------------ 渲染
def _link_cites(escaped_text, n_ev):
    """把 [n] 变成页内锚点。**先转义再链接**，顺序反了就是 XSS。"""
    def f(m):
        n = int(m.group(1))
        if not (1 <= n <= n_ev):
            # 越界角标：**不装作有出处**，划成删除线让人一眼看见模型编了个号
            return '<del class="badcite">[%d]</del>' % n
        return '<a class="cite" href="#ev%d">[%d]</a>' % (n, n)
    return A.CITE_RE.sub(f, escaped_text)


def _ev_html(e):
    """一块证据。出处抬头 + 两个页码 + 跳 PDF 的链接。"""
    comp = e.get("company") or ""
    printed = e.get("printed") or e.get("printed_note") or "无印页"
    phys = e.get("phys")
    q = urllib.parse.quote(comp + ".pdf")
    if phys:
        lnk = ('<a class="pdf" target="_blank" href="/pdf/%s#page=%s">'
               'PDF 第 %s 物理页 ↗</a>' % (q, phys, phys))
    else:
        lnk = '<span class="muted">无物理页</span>'
    return (
        '<div class="ev" id="ev%d">'
        '<div class="evh"><b>[%d]</b> <span class="co">%s</span>'
        ' <span class="sec">%s</span> <span class="pg">印 %s</span>'
        ' <span class="cid">%s</span> <span class="muted">RRF %.5f</span> %s</div>'
        '<pre class="evt">%s</pre></div>'
        % (e["n"], e["n"], html.escape(comp), html.escape(e.get("section") or "（未归节）"),
           html.escape(str(printed)), html.escape(e.get("chunk_id") or ""),
           e.get("score") or 0.0, lnk, html.escape(e.get("text") or "")))


def render_result(out, meta, view="full"):
    """把 answer_question 的返回渲染成 HTML。**全站唯一的渲染器**（见文件头第 3 条）。

    `view="ev"` 只渲染证据区（跳过答案正文）。加这个不是为了好看——**没有 key 时
    答案=原文片段，同一批文本在页面上出现两遍，整页高到 8000px 都截不全**，
    而截图要证明的恰恰是证据区那几列（公司/章节/印页/物理页/块号/分数）。
    所以给一个「只看证据」的入口。**两个视图渲染出来的证据区是同一段代码**，
    不存在截图和页面对不上的可能。"""
    ev = out["evidence"]
    parts = []
    # 顶部：问题 + 用了什么模型（**只写模型名，绝不写 key**）
    llm = ("大模型：%s" % out["model"]) if out["used_llm"] else "**未调用大模型**"
    parts.append('<div class="qbox"><div class="q">%s</div>'
                 '<div class="meta">%s ｜ 用时 %ss ｜ k=%d ｜ %s</div></div>'
                 % (html.escape(out["question"]), html.escape(str(llm)),
                    out["elapsed"], out["k"], html.escape(out["note"] or "正常")))
    q = urllib.parse.quote(out["question"])
    if view == "ev":
        parts.append('<div class="nav">视图：<b>只看证据</b> ｜ '
                     '<a href="/?q=%s">看完整页（含答案正文）</a></div>' % q)
    else:
        parts.append('<div class="nav">视图：<b>完整页</b> ｜ '
                     '<a href="/?q=%s&amp;view=ev">只看证据</a></div>' % q)
    # 子查询：对比题为什么能答出来，全在这行
    subs = " ； ".join("%s→%s" % (s["company"] or "·", s["query"]) for s in out["sub_queries"])
    parts.append('<div class="subs">实际发出的子查询：%s</div>' % html.escape(subs))
    # 告警：红框，位置在答案**上面**——先看风险再看结论
    if out["warnings"]:
        parts.append('<div class="warn"><b>⚠ 这版可能有问题（%d 条）</b><ul>%s</ul></div>'
                     % (len(out["warnings"]),
                        "".join("<li>%s</li>" % _em(html.escape(w)) for w in out["warnings"])))
    if out["no_citation"]:
        parts.append('<div class="nocite">该答案<b>一个可核对的出处都没有</b>——'
                     '不要采信，请直接看下面的原文</div>')
    if view != "ev":
        parts.append('<h2>答案</h2><pre class="ans">%s</pre>'
                     % _link_cites(_em(html.escape(out["answer"] or "")), len(ev)))
    # 角标总览
    if out["citations"]:
        parts.append('<div class="cits">用到的出处：%s</div>'
                     % " ".join('<a class="cite" href="#ev%d">[%d]</a>' % (n, n)
                                for n in out["citations"]))
    # 证据：**永远全展开**，不只展开被引用的（阶段三「召回了但没写对」只能靠这个看出来）
    parts.append('<h2>检索到的证据（共 %d 块，全部摊开）</h2>' % len(ev))
    parts.append("".join(_ev_html(e) for e in ev))
    # 页码口径 + 已知偏差，写在最下面当脚注
    cos = sorted({e.get("company") for e in ev if e.get("company")})
    notes = "".join("<li>%s：%s</li>" % (html.escape(c), html.escape(page_note(meta, c)))
                    for c in cos)
    parts.append(
        '<div class="foot"><b>页码口径</b><ul>%s</ul>'
        '<b>已知偏差（阶段一二已披露，不是链接坏了）</b><ul>'
        '<li>国货航是**双页拼版**：99 个 PDF 页被拆成 196 个页单元，跳过去是跨页版式。</li>'
        '<li>申通快递有 8 张纯图隔页被归给下一节，这类块的印页会比正文小 1。</li>'
        '<li>上表的「物理页」才是 PDF 阅读器认的那个页码；「印页」是报告自己印的页码。'
        '两者在多数公司差 0，少数差 1～3（见上）。**跳转一律用物理页。**</li>'
        '</ul></div>' % notes)
    return "".join(parts)


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>财报知识库问答 · 13 家物流公司 2025 年报</title>
<style>
 body{{font:15px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
      margin:0;background:#f6f7f9;color:#222}}
 .wrap{{max-width:960px;margin:0 auto;padding:20px}}
 h1{{font-size:19px;margin:0 0 4px}} .top{{color:#666;font-size:13px;margin-bottom:14px}}
 form{{display:flex;gap:8px;margin:0 0 18px}}
 input[type=text]{{flex:1;padding:10px 12px;font-size:15px;border:1px solid #ccd;border-radius:6px}}
 button{{padding:10px 20px;font-size:15px;border:0;border-radius:6px;background:#2f6fdb;color:#fff;cursor:pointer}}
 .qbox{{background:#fff;border-left:4px solid #2f6fdb;padding:10px 14px;border-radius:4px;margin-bottom:10px}}
 .q{{font-size:17px;font-weight:600}} .meta{{color:#666;font-size:12.5px;margin-top:3px}}
 .subs{{color:#666;font-size:12.5px;margin:0 0 12px;word-break:break-all}}
 .nav{{font-size:12.5px;color:#666;margin:0 0 10px}}
 h2{{font-size:15px;margin:20px 0 8px;color:#333;border-bottom:1px solid #e0e3e8;padding-bottom:5px}}
 pre{{white-space:pre-wrap;word-break:break-word;font-family:inherit;margin:0}}
 .ans{{background:#fff;border:1px solid #e0e3e8;border-radius:6px;padding:14px}}
 .warn{{background:#fff6e5;border:1px solid #f0c060;border-radius:6px;padding:10px 14px;margin:10px 0}}
 .warn ul{{margin:6px 0 0 18px;padding:0}} .warn li{{margin:3px 0}}
 .nocite{{background:#ffe9e9;border:1px solid #e88;border-radius:6px;padding:10px 14px;margin:10px 0;color:#a11}}
 .cits{{margin:8px 0;font-size:13.5px}}
 a.cite{{display:inline-block;background:#e8f0ff;color:#1c4fa8;border-radius:3px;
        padding:0 4px;margin:0 1px;text-decoration:none;font-size:12.5px}}
 a.cite:hover{{background:#2f6fdb;color:#fff}}
 del.badcite{{color:#bbb;font-size:12.5px;text-decoration:line-through}}
 .ev{{background:#fff;border:1px solid #e0e3e8;border-radius:6px;padding:10px 12px;margin:0 0 8px}}
 .ev:target{{border-color:#2f6fdb;box-shadow:0 0 0 2px #cfe0ff}}
 .evh{{font-size:12.5px;color:#444;margin-bottom:6px;line-height:1.9}}
 .co{{font-weight:600}} .sec{{color:#555}} .pg{{color:#1c4fa8}}
 .cid{{color:#888;font-family:Consolas,monospace}} .muted{{color:#999}}
 .evt{{font-size:13px;color:#333;max-height:230px;overflow:auto;
      background:#fbfbfc;border-radius:4px;padding:8px}}
 a.pdf{{color:#1c4fa8;text-decoration:none;border-bottom:1px dotted #1c4fa8}}
 .foot{{margin-top:22px;color:#666;font-size:12.5px;background:#fff;
       border:1px solid #e0e3e8;border-radius:6px;padding:12px 14px}}
 .foot ul{{margin:4px 0 10px 18px;padding:0}}
</style></head><body><div class="wrap">
<h1>财报知识库问答</h1>
<div class="top">语料：13 家物流运输公司 2025 年年度报告全文 ｜ {corpus} ｜
 检索：向量 + BM25 混合（RRF k={rrf}，无 reranker） ｜ {llm}</div>
<form method="get" action="/">
 <input type="text" name="q" value="{q}" placeholder="例：申通快递和圆通速递的业务量对比" autofocus>
 <button type="submit">提问</button>
</form>
{body}
</div></body></html>
"""


def render_page(q, body, corpus, llm):
    return PAGE.format(corpus=html.escape(corpus), rrf=S.RRF_K, llm=_em(html.escape(llm)),
                       q=html.escape(q or ""), body=body)


# ------------------------------------------------------------------ HTTP
class H(BaseHTTPRequestHandler):
    server_version = "finrag-ask/1.0"
    protocol_version = "HTTP/1.1"          # 有 Content-Length 才敢用 1.1；Range 也需要它

    def log_message(self, fmt, *args):
        # 默认实现往 stderr 写；这里走 logging，免得和 search.py 的 print 混在一起
        log.info("%s - %s", self.address_string(), fmt % args)

    # ---- 工具
    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        s = json.dumps(obj, ensure_ascii=False, indent=1)
        # **把「key 绝不发给前端」变成断言**，不是口头承诺。key 为空时这条恒真，
        # 但只要有 key 在，任何一次泄漏都会在这里炸出来，而不是悄悄流到浏览器。
        key = A.env_conf()[0]
        assert not key or key not in s, "key 出现在要发给浏览器的响应里——拒绝发送"
        self._send(code, s, "application/json; charset=utf-8")

    def _notfound(self, why="没有这个路由"):
        self._send(404, "404 %s\n" % why)

    # ---- GET
    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(u.path)
        if path == "/":
            g = urllib.parse.parse_qs(u.query)
            self._index(g.get("q", [""])[0].strip(),
                        (g.get("view", ["full"])[0] or "full").strip())
        elif path == "/api/health":
            self._health()
        elif path == "/api/ask":
            self._api_ask(urllib.parse.parse_qs(u.query).get("q", [""])[0].strip())
        elif path.startswith("/pdf/"):
            self._pdf(path[len("/pdf/"):])
        elif path == "/favicon.ico":
            self._send(204)
        else:
            self._notfound()

    do_HEAD = do_GET

    # ---- POST /api/ask
    def do_POST(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path != "/api/ask":
            return self._notfound()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            q = (data.get("q") or "").strip()
        except Exception as e:
            return self._json({"error": "请求体解析失败：%s" % e}, 400)
        self._api_ask(q, post=True)

    # ---- 各路由
    def _health(self):
        man = IX.man
        key, base, model = A.env_conf()
        self._json({
            "ok": True, "n_chunks": man["n_chunks"], "n_vectors": man["n_vectors"],
            "embed_model": man["model"], "dim": man["dim"],
            "rrf_k": S.RRF_K, "route_top": S.ROUTE_TOP, "k_default": A.K_DEFAULT,
            "plan_default": "V2+u", "reranker": False,
            # 只报「配没配」「模型名」，**不报 key 本身，连长度前缀也不报**（探路模式才报前缀）
            "llm_configured": bool(key), "llm_base": base if key else None,
            "llm_model": model or ("自动从 /models 挑" if key else None),
        })

    def _api_ask(self, q, post=False):
        if not q:
            return self._json({"error": "缺少 q"}, 400)
        t0 = time.time()
        with LOCK:
            out = A.answer_question(IX, q)
        out["server_elapsed"] = round(time.time() - t0, 2)
        # 结构化记录里**不含 key**：answer_question 只回显模型名
        self._json(out)

    def _index(self, q, view="full"):
        meta = META
        if not q:
            body = ('<div class="foot">在上面的框里输入问题。'
                    '答案里的 <a class="cite" href="#ev1">[1]</a> 角标点开就是原文，'
                    '再点「PDF 第 N 物理页」跳到报告原页。<br>'
                    '跨公司对比题会自动**拆成每家一条子查询**再融合——'
                    '这是阶段三实测到的：一句话问两家，两家的答案块都召不回来。</div>')
        else:
            with LOCK:
                out = A.answer_question(IX, q)
            body = render_result(out, meta, view)
        llm = ("LLM：%s" % (A.env_conf()[2] or "已配 key，自动挑模型")) if A.env_conf()[0] \
            else "**未配 key → 答案走原文片段回退（不假装是大模型答的）**"
        page = render_page(q, body, "%d 块 / %d 家" % (IX.man["n_chunks"],
                                                      len(IX.man.get("companies", [])) or 13), llm)
        self._send(200, page, "text/html; charset=utf-8")

    def _pdf(self, name):
        """**手写 Range/206**。Python 3.14 的 SimpleHTTPRequestHandler 源码里没有 Range，
        它只会整文件 200 —— 顺丰那份 23.7 MB，浏览器跳页时会先下完整份再跳，
        再加上 `#page=` 的锚点行为不可靠。所以这里按 RFC 7233 自己切。"""
        if name not in PDF_OK:
            return self._notfound("不在 PDF 白名单里：%s" % name)
        path = os.path.join(PDF_DIR, name)
        if not os.path.isfile(path):
            return self._notfound("白名单里有但这个文件不在磁盘上：%s" % name)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end, code = 0, size - 1, 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng.strip())
            if not m:
                return self._send(416, "Range 头看不懂：%s\n" % rng,
                                  extra={"Content-Range": "bytes */%d" % size})
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            elif m.group(2):                       # bytes=-N：最后 N 字节
                start = max(0, size - int(m.group(2)))
            if start >= size or start > end:
                return self._send(416, "Range 越界\n",
                                  extra={"Content-Range": "bytes */%d" % size})
            end = min(end, size - 1)
            code = 206
        n = end - start + 1
        extra = {"Accept-Ranges": "bytes"}
        if code == 206:
            extra["Content-Range"] = "bytes %d-%d/%d" % (start, end, size)
        self.send_response(code)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Content-Length", str(n))
        self.send_header("Cache-Control", "no-store")
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = n
            while left > 0:
                chunk = f.read(min(262144, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return                          # 浏览器提前关掉连接，正常，不刷栈
                left -= len(chunk)


class Server(ThreadingHTTPServer):
    """**Windows 上必须关掉 `allow_reuse_address`。**

    `HTTPServer` 默认开 `SO_REUSEADDR`。在 Linux 上这跟「端口被占」不冲突，
    但在 **Windows 上它允许两个 socket 绑同一个地址**——我重启服务时就没报错，
    结果 8000 端口上同时挂着新旧两个进程（PID 27820 与 3848），请求按内核心情
    分给谁就是谁：新代码的改动时有时无，**而且完全不报错**。这比崩掉还坏，
    因为测量出来的数都变成不可信的。

    关掉之后第二次绑会直接 OSError，启动失败看得见。"""
    allow_reuse_address = False
    daemon_threads = True


# ------------------------------------------------------------------ main
def main():
    global IX, PDF_OK, META
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-warm", action="store_true", help="跳过预热（只用于快速改前端）")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    # httpx/huggingface_hub 每个模型文件都打一条 INFO，几十行把起服务那几行有用的日志
    # 埋掉了（第一次跑就是这样）。这里压到 WARNING，只留自己的日志。
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub",
                  "sentence_transformers", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    meta = load_companies()
    for name in meta:
        PDF_OK.add(name + ".pdf")
    META = meta

    t0 = time.time()
    IX = S.Index(a.model, a.device)
    log.info("索引载入 %.2fs：%d 块 / %d 向量 / %s", time.time() - t0,
             IX.man["n_chunks"], IX.man["n_vectors"], IX.man["model"])

    if a.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("**绑在 %s 上**：key 只在服务进程里，局域网内谁都能拿这个页面当 LLM 代理。"
                    "本作业只需本机访问，建议改回 127.0.0.1。", a.host)
    # **先绑端口再预热**。反过来的话，端口被占时要先白等 30 多秒预热才报错
    # ——实测就是这样：第二次起的那次，31.8 s 的功全白做，人还以为是启动慢。
    try:
        srv = Server((a.host, a.port), H)
    except OSError as e:
        # 端口被占时**说清楚是这个原因**并给出退路，不要让它变成一个看不懂的栈
        log.error("绑不上 %s:%d（%s）。端口被占了——先关掉在跑的那个，"
                  "或者换 --port 起一个。", a.host, a.port, e)
        return 1

    if not a.no_warm:
        # **必须预热**。实测冷启动后第一次 hybrid() 要 27.2 s（bge-small 权重加载 + 首次编码），
        # 之后同一进程内每次 0.0x s。不预热的话，用户点第一次「提问」会以为页面卡死了。
        t0 = time.time()
        with LOCK:
            A.answer_question(IX, "顺丰控股2025年营业收入是多少", use_llm=False)
        log.info("预热完成 %.2fs（第一次提问不再吃这份延迟）", time.time() - t0)

    log.info("已起：http://%s:%d/  （PDF 白名单 %d 份；Ctrl-C 停）", a.host, a.port, len(PDF_OK))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("收到 Ctrl-C，停。")
    return 0


META = {}

if __name__ == "__main__":
    sys.exit(main())
