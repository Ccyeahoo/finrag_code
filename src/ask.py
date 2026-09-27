# -*- coding: utf-8 -*-
r"""阶段四 问答：检索 → 组证据 → 大模型成文 → 回填出处角标。

**这个设计不是拍脑袋，是阶段三实测的 5/9 逼出来的**（依据见 out/stage3_report.md §3）：
  1. **对比题必须拆子查询**。一句话里问两家公司，两家的答案块都召不回来；
     拆成每家一条子查询再融合，两家都能排到 hybrid 第 1。不做这步，对比题直接废。
  2. **取 k≥10**。中国外运那道题的答案块排在 hybrid 第 9；只看 top-3 会答错。
  3. **证据全摊给页面看**，不只摊被引用的。阶段三栽过一次「自检报绿而实际没召回」，
     把证据摊开才能一眼分清是「没召回」还是「召回了但没写对」。

key 只从环境变量读：FINRAG_API_KEY / FINRAG_BASE_URL / FINRAG_MODEL。
绝不写进文件、不进 git、**绝不发给前端**——LLM 调用只发生在服务进程里，
serve.py 返回浏览器的 JSON 里连 key 的影子都不该有（那边有一句 assert 把关）。
另外：**绝不把 key 打印出来**，本文件的打印语句一律只打长度或前缀。

用法：
    PY=/c/Users/yexinyu/AppData/Local/Python/bin/python.exe   # 裸 python 是 Store 存根
    $PY src/ask.py --probe                   # 探通路（要 key），只打状态码与模型名
    $PY src/ask.py --plan-check              # **不要 key**：比对子查询拆分规则
    $PY src/ask.py --selftest                # 端到端自检（有 key 就用，没有就走模板回退）
    $PY src/ask.py --q "顺丰控股2025年营业收入是多少"
    $PY src/ask.py --q "..." --no-llm        # 只检索、不调大模型
"""
import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search as S

BASE_DEFAULT = "https://api.deepseek.com/v1"
K_DEFAULT = 10                       # 阶段三实测：答案块可能排到第 9，k 必须 ≥10
ROUTE_TOP = S.ROUTE_TOP

# 简称 → 语料里的 canonical 公司名。**这张表是手工列的**（语料里的 company 字段本身
# 就是这 13 个短名，见 chunk_audit.md），列在这里只是为了让「顺丰」「外运」这类
# 口语问法也能认出来。认不出来就当全局问题处理，不猜。
ALIAS = {
    "顺丰": "顺丰控股", "顺丰速运": "顺丰控股",
    "外运": "中国外运", "中外运": "中国外运",
    "圆通": "圆通速递", "申通": "申通快递", "韵达": "韵达股份",
    "中储": "中储股份", "中谷": "中谷物流", "华贸": "华贸物流",
    "嘉友": "嘉友国际", "辽港": "辽港股份", "铁龙": "铁龙物流",
    "东航": "东航物流", "国货": "国货航",
}

# 问题里的连接词/疑问词，拆子查询时去掉（它们只增加噪声 token）。
STOPWORDS = ("和", "与", "、", "及", "跟", "对比", "比较", "相比", "分别", "各自",
             "哪个", "哪家", "谁", "更高", "更低", "怎么样", "如何", "情况",
             "三家", "两家", "几家", "这家", "那家")

# 同义词扩展：**针对阶段三实测到的用词不对齐**。圆通原文写「快递业务完成量」，
# 而人问「业务量」——BM25 凭空少一个大 token，那题就是因此漏召回的。
SYN = {
    "业务量": ["业务完成量", "完成量", "件量", "包裹量"],
    "营业收入": ["营收", "营业总收入"],
    "净利润": ["归母净利润", "归属于上市公司股东的净利润"],
    "同比": ["同比增减", "增长率"],
    "毛利率": ["毛利率"],
    "研发": ["研发投入", "研发费用"],
}


# ------------------------------------------------------------------ 公司识别
def companies(ix):
    """语料里实际有哪些公司——**从 chunks.jsonl 现读，不手工维护列表**。"""
    return sorted({r["company"] for r in ix.rows if r.get("company")})


def find_companies(q, names):
    """题面里点到了哪几家公司，按**在问句里出现的先后**返回。

    长名优先、简称兜底，同一个 canonical 只算一次（「中国外运」和「外运」都命中
    也只算一家）。认不出来就返回空表 = 当成全局问题，不硬猜。"""
    hit = {}
    for key, canon in list(ALIAS.items()) + [(n, n) for n in names]:
        i = q.find(key)
        if i >= 0:
            hit[canon] = min(hit.get(canon, i), i)
    return [c for c, _ in sorted(hit.items(), key=lambda kv: kv[1])]


# ------------------------------------------------------------------ 查询规划
def plan_queries(q, comps, names, mode="V2"):
    """把一句问题拆成若干条子查询，返回 [(公司或 None, 子查询), …]。

    - 没点到公司 / 只点到一家 → 原问题原样发一条（不做无谓改写）。
    - 点到 ≥2 家 → **每家一条**（阶段三实测：这是对比题能不能做出来的分水岭）。

    mode 控制「拆」的力度，取值见 --plan-check：
      V1 = 公司名 + 原问题去掉所有公司名与连接词
      V2 = V1 + 同义词扩展（默认）
      V3 = 公司名 + 原问题原样（不删其它公司名，作对照）
    """
    if len(comps) < 2:
        return [(comps[0] if comps else None, q)]
    out = []
    for c in comps:
        if mode == "V3":
            out.append((c, "%s %s" % (c, q)))
            continue
        rest = q
        # **必须长名优先**。先替换简称会把「顺丰控股」削成「控股」、「圆通速递」削成
        # 「速递」，子查询里就混进「控股 速递 股份」这种垃圾 token——我第一版就是这么错的，
        # 对照表里那串 '顺丰控股 控股 速递 股份三家2025年营业收入' 就是证据。
        for k in sorted(set(list(ALIAS) + list(names)), key=len, reverse=True):
            rest = rest.replace(k, " ")
        for w in STOPWORDS:
            rest = rest.replace(w, " ")
        rest = re.sub(r"\s+", " ", rest).strip()
        sq = ("%s %s" % (c, rest)).strip()
        if mode == "V2":
            extra = []
            for w, syns in SYN.items():
                if w in q:
                    extra += [w] + syns
            if extra:
                sq += " " + " ".join(extra)
        out.append((c, sq))
    return out


# ------------------------------------------------------------------ 检索
def parse_plan(s):
    """'V2+u' → ('V2', 'union')；后缀 f=只过滤、u=过滤与不过滤都跑、无=不过滤。"""
    mode, _, suf = s.partition("+")
    return mode, {"f": True, "u": "union"}.get(suf, False)


def retrieve(ix, q, k=K_DEFAULT, mode="V2", filt="union"):
    """按子查询分别召回，再用 rr_fuse_many 统一融合（公式与阶段三同一份）。

    返回 (plan, comps, evidence)。evidence 是 [(chunk_id, 融合分, 块记录), …]。

    filt 三档，**都实测过**（`--plan-check` 的对照表就是这份数据）：
      False = 不过滤；True = 只留该公司的块；"union" = **两种都跑一遍再一起融合**。
    单用过滤在对比题上不一定更好（实测圆通 @2 → @5），两路都算等于
    「哪一路觉得它靠前都算数」，代价只是多跑一次纯本地检索（0.03 s 量级）。"""
    names = companies(ix)
    comps = find_companies(q, names)
    plan = plan_queries(q, comps, names, mode)
    routes = []
    for comp, sq in plan:
        if filt and comp:
            v, b = ix.routes(sq, top=ROUTE_TOP, comp=comp)
            routes += [v, b]
            if filt == "union":
                v2, b2 = ix.routes(sq, top=ROUTE_TOP)
                routes += [v2, b2]
        else:
            v, b = ix.routes(sq, top=ROUTE_TOP)
            routes += [v, b]
    ev = []
    for cid, sc in S.rr_fuse_many(routes, top=k):
        r = ix.by_id.get(cid)
        if r:
            ev.append((cid, sc, r))
    return plan, comps, ev


# ------------------------------------------------------------------ 组 prompt
SYS_PROMPT = (
    "你是财报问答助手。规则：\n"
    "1. **只依据【资料】回答**，不得使用资料以外的知识，不得自行推算资料里没有的数字。\n"
    "2. 每一条事实后面**必须紧跟来源编号**，格式如 [1]；一句话用了多块就连着写 [1][3]。\n"
    "3. 资料里没有的，直接回答「资料里没有」，**不要猜、不要用常识补**。\n"
    "4. 引用数字时**保持原文写法**（千分位、单位、百分号、精度一律照抄），不要改写、不要四舍五入。\n"
    "5. 做对比时逐家列出，各带各自的编号，不要合并成一个数。"
)


def build_messages(q, ev):
    """证据块编号 [1..n]，每块抬头写「公司｜章节｜印页（块号 xxx）」——

    编号 [n] 是我们要求的角标写法；**块号也放进抬头**，因为实测有些模型不写 [n]、
    直接把抬头里的块号抄进答案。两种写法都认（见 map_citations），所以得让它看得见。"""
    parts = []
    for n, (cid, sc, r) in enumerate(ev, 1):
        parts.append("[%d] %s（块号 %s）\n%s" % (n, S.where(r), cid, r["text"]))
    user = "【资料】\n%s\n\n【问题】%s" % ("\n\n".join(parts), q)
    return [{"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user}]


# ------------------------------------------------------------------ 角标回填
CITE_RE = re.compile(r"\[(\d{1,2})\]")
CHUNKID_RE = re.compile(r"\b(\d{6}-\d{4})\b")     # 块号写法：002352-0070


def map_citations(answer, ev):
    """把答案里的出处回填成证据编号，**两条路**：

    1. `[n]` 角标 —— SYS_PROMPT 里要求的写法；
    2. `002352-0070` 这种**块号原样**出现 —— 有些模型不写 [n]，直接抄抬头里的块号。
       这同样是个可核对的出处，认它。

    越界的编号/块号**丢掉但记账**（返回 dropped）。不记账就等于静默吞掉——
    阶段三那次「自检报绿而实际没召回」就是静默吞出来的，这里不再犯。

    一个合法出处都没有 → no_citation=True，页面红字提示。
    **宁可提示无出处，也不替它编一个。**"""
    used, dropped = [], []
    cid2n = {cid: n for n, (cid, _, _) in enumerate(ev, 1)}
    for m in CITE_RE.finditer(answer or ""):
        n = int(m.group(1))
        if 1 <= n <= len(ev):
            if n not in used:
                used.append(n)
        elif n not in dropped:
            dropped.append(n)
    for m in CHUNKID_RE.finditer(answer or ""):
        cid = m.group(1)
        n = cid2n.get(cid)
        if n is None:
            if cid not in dropped:
                dropped.append(cid)
        elif n not in used:
            used.append(n)
    return sorted(used), dropped


# ------------------------------------------------------------------ 反向核查
NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?\s*[%％]?")
# 「…（中略 268 字）…」——_snip 自己写的省略标记，里面那个数是**字数**不是数据，
# 反向核查前必须抠掉（_nums 里抠），否则每条长块都会贡献一个查不到的假数字。
_ELIDE_RE = re.compile(r"…（中略 \d+ 字）…")


def _norm_num(s):
    return (s.replace(",", "").replace("，", "").replace("％", "%")
             .replace(" ", "").replace(" ", "").strip())


def _nums(text):
    """从答案里取出「可核对的数字串」。取之前先抠掉**我们自己生成的东西**：

      · `[n]` 角标、`002352-0070` 块号；`印57–58` 出处抬头里的印页；
      · `…（中略 268 字）…` 里的字数——**这是我写的省略计数，不是财报数字**。
        （第一版没抠，它被当成「资料里找不到的数字 268」报了 5 条假警报。
         假警报会训练人忽略告警，比不报还糟，所以这里必须抠干净。）

    抠完剩下的才算「答案里出现的数字」，拿去反向核查。"""
    t = text or ""
    t = CHUNKID_RE.sub(" ", t)
    t = CITE_RE.sub(" ", t)
    t = _ELIDE_RE.sub(" ", t)
    t = re.sub(r"印[\d]+(?:[–—\-][\d]+)?", " ", t)
    out = []
    for m in NUM_RE.finditer(t):
        s = _norm_num(m.group(0))
        if len(re.sub(r"\D", "", s)) < 2:
            continue                      # 单个数字多半是序号/章节号，不查
        out.append(s)
    return list(dict.fromkeys(out))


def check_numbers(answer, ev, cited):
    """**反向核查**：答案里的数字，能不能在**被引用的原文块**里找到原样写法？

    这是防「模型自己算了一个数/记错了/把两块拼成一个数」的。只查正向（块里有没有
    答案串）挡不住编造，必须掉头查一遍答案。

    返回找不到出处的数字。`in_any` 有而 `in_cited` 没有 → 数字确实在资料里但
    **角标指错了块**；两处都没有 → **资料里根本没这个数**，这档最该看。"""
    if not answer:
        return []
    norm = [_norm_num(re.sub(r"\s+", "", r["text"])) for _, _, r in ev]
    bad = []
    for s in _nums(answer):
        in_cited = [n for n in cited if s in norm[n - 1]]
        if in_cited:
            continue
        in_any = [n for n in range(1, len(ev) + 1) if s in norm[n - 1]]
        bad.append({"num": s, "in_cited": [], "in_any": in_any})
    return bad


def audit(q, comps, ev, cited, dropped, bad_nums):
    """把「这版可能哪里不对」写成给人看的清单，**页面和报告都要摊出来**。

    宁可多报一条让人去核，也不要闷着——用户的原话是「不可靠时请明确指出，
    不要假装成功」。"""
    w = []
    if not comps:
        # 问句里一个语料内的公司名都没认出来 → 只能按整句去检索，没有按公司过滤/拆分。
        # 这是「问了一家不在语料里的公司」最可能的入口，必须说出来而不是照答。
        w.append("题面里**没认出语料中的任何公司名**，无法按公司拆分检索"
                 "——若问的公司不在这 13 家里，答案必然不对口")
    if dropped:
        w.append("模型写了不存在的出处 %s，已丢弃未回填（可能是编的）"
                 "、".join(str(d) for d in dropped))
    for c in comps:
        if not any(r.get("company") == c for _, _, r in ev):
            w.append("题面点到「%s」，但 %d 条证据里**没有一块属于它**——这家必答不出来"
                     % (c, len(ev)))
    for d in bad_nums:
        if d["in_any"]:
            w.append("数字 %s 在资料里（证据 %s），但**不在被引用的块里**——角标可能指错"
                     % (d["num"], d["in_any"]))
        else:
            w.append("数字 %s **在全部证据里都找不到**——可能是模型自己算的，须人工核"
                     % d["num"])
    return w


# ------------------------------------------------------------------ 大模型
def env_conf():
    return (os.environ.get("FINRAG_API_KEY", "").strip(),
            os.environ.get("FINRAG_BASE_URL", BASE_DEFAULT).rstrip("/"),
            os.environ.get("FINRAG_MODEL", "").strip())


def _get(url, key, timeout=30):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _post(url, key, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def pick_model(base, key):
    """从 /models 列表里挑一个聊天模型。优先 deepseek-chat，其次名字带 chat 的，
    再其次带 pro 的，最后取第一个。**只打印模型名，不打印 key。**"""
    d = _get(base + "/models", key)
    ids = [m["id"] for m in d.get("data", [])]
    for want in ("deepseek-chat",):
        if want in ids:
            return want, ids
    for kw in ("chat", "pro"):
        for i in ids:
            if kw in i.lower():
                return i, ids
    return (ids[0] if ids else None), ids


def call_llm(messages, base, key, model, timeout=180):
    """**标准库 urllib，不装 requests/openai。** temperature=0 求可复现。"""
    d = _post(base + "/chat/completions", key,
              {"model": model, "messages": messages, "temperature": 0, "stream": False},
              timeout=timeout)
    ch = d.get("choices") or []
    if not ch:
        raise RuntimeError("响应里没有 choices：%s" % json.dumps(d, ensure_ascii=False)[:300])
    return ch[0]["message"]["content"], d.get("usage") or {}


# ------------------------------------------------------------------ 模板回退
_PAGER_RE = re.compile(r"(19|20)\d{2}\s*年?\s*(半)?年度报告(全文)?")


def _is_pager(s):
    """页眉/页脚行：公司全名 + 年份 + 「年度报告」，没有别的信息。

    判据取「含年报字样 + 短 + 无标点」三条同时成立才丢。真句子几乎都带标点，
    或者超 32 字（如「九、是否存在半数以上董事无法保证…年度报告的真实性」，37 字），
    不会被误伤——**丢错内容比留一行页脚更糟**，所以判据往保守里写。
    实测这条命中同一行 447 次（`中储发展股份有限公司2025 年年度报告`），确实只是页脚。
    """
    if len(s) > 32 or not _PAGER_RE.search(s):
        return False
    return not re.search(r"[。，；：%、？]", s)


def _clean_text(text, prefix):
    t = (text or "").replace(prefix or "", "")
    keep = [ln for ln in t.split("\n") if not _is_pager(ln.strip())]
    return re.sub(r"[ \t]+", " ", "\n".join(keep)).strip()


def _snip(body, head=400, tail=500):
    """长块**不许掐尾巴**。阶段三的教训写得很清楚：表格题的答案常在后半段，
    只给前 220 字会把答案切掉**而且看不出来**（截断是静默的，错了也不报）。
    改成「前 400 + …（中略 N 字）… + 后 500」，**中略多少字明写出来**。"""
    if len(body) <= head + tail:
        return body
    return "%s\n…（中略 %d 字）…\n%s" % (body[:head], len(body) - head - tail, body[-tail:])


def template_answer(ev):
    """**没有 key / 调不通时的回退**：不假装是大模型答的，直接把原文片段和出处摊出来。"""
    lines = ["（没有可用的 LLM，下面是检索到的原文片段，未经组织——**这不是生成的答案**）", ""]
    for n, (cid, sc, r) in enumerate(ev, 1):
        body = _clean_text(r["text"], r.get("prefix"))
        lines.append("[%d] %s（块号 %s）" % (n, S.where(r), cid))
        lines.append("    " + _snip(body).replace("\n", "\n    "))
    return "\n".join(lines)


def answer_question(ix, q, k=K_DEFAULT, mode="V2", filt="union", use_llm=True):
    """一次问答的完整流程，返回一个统一结构的 dict（serve.py 直接 json 出去）。"""
    t0 = time.time()
    plan, comps, ev = retrieve(ix, q, k=k, mode=mode, filt=filt)
    key, base, model = env_conf()
    used_llm, note, usage = False, "", {}
    if not use_llm:
        # 不写「已按 --no-llm」——没设 key 时也是走到这里，写死会看不出真实原因。
        ans, note = template_answer(ev), "未调用大模型（--no-llm 或未设 key），下面是原文片段"
    elif not key:
        ans, note = template_answer(ev), "未设 FINRAG_API_KEY，走模板回退"
    else:
        if not model:
            try:
                model, ids = pick_model(base, key)
                print("[llm] /models 列表：%s → 选 %s" % (ids, model))
            except Exception as e:
                return _pack(q, ev, plan, comps, template_answer(ev),
                             "选模型失败：%s；走模板回退" % type(e).__name__, False, k, {})
        try:
            ans, usage = call_llm(build_messages(q, ev), base, key, model)
            used_llm = True
        except urllib.error.HTTPError as e:
            ans = template_answer(ev)
            note = "LLM HTTP %s：%s；走模板回退" % (e.code, e.read()[:200])
        except Exception as e:
            ans = template_answer(ev)
            note = "LLM 调用失败（%s: %s）；走模板回退" % (type(e).__name__, e)
    out = _pack(q, ev, plan, comps, ans, note, used_llm, k, usage)
    out["model"] = model if used_llm else None
    out["elapsed"] = round(time.time() - t0, 2)
    return out


def _pack(q, ev, plan, comps, ans, note, used_llm, k, usage):
    cits, dropped = map_citations(ans, ev)
    bad_nums = check_numbers(ans, ev, cits)
    evidence = []
    for n, (cid, sc, r) in enumerate(ev, 1):
        evidence.append({
            "n": n, "chunk_id": cid, "score": round(sc, 6),
            "company": r.get("company"), "section": r.get("section"),
            "printed": (None if r.get("printed_lo") is None
                        else "%s%s" % (r["printed_lo"],
                                       "" if r.get("printed_hi") in (None, r["printed_lo"])
                                       else "–%s" % r["printed_hi"])),
            "printed_note": None if r.get("printed_lo") is not None else r.get("page_note"),
            "phys": r.get("phys_lo"),          # 跳 PDF 用**物理页**，见 serve.py
            "text": r["text"],
        })
    return {"question": q, "answer": ans, "note": note,
            "used_llm": used_llm, "no_citation": (used_llm and not cits),
            "citations": cits, "dropped_citations": dropped,
            "num_check": bad_nums,
            "warnings": audit(q, comps, ev, cits, dropped, bad_nums),
            "evidence": evidence, "k": k,
            "sub_queries": [{"company": c, "query": sq} for c, sq in plan],
            "companies": comps, "usage": usage}


# ------------------------------------------------------------------ CLI
def cmd_probe(a):
    key, base, model = env_conf()
    print("base=%s" % base)
    if not key:
        print("**没设 FINRAG_API_KEY**——阶段四要它才能调模型。")
        return
    print("key：长度 %d、前缀 '%s'（**只打前缀，不打全文**）" % (len(key), key[:3]))
    try:
        got, ids = pick_model(base, key)
        print("HTTP 200｜模型列表：%s" % ids)
        print("将使用：%s" % got)
    except urllib.error.HTTPError as e:
        print("HTTP %s：%s" % (e.code, e.read()[:200]))
        print("→ key 不被接受或接口地址不对，**停下**，不重试。")
    except Exception as e:
        print("连不上（%s: %s）→ **停下**，不瞎试。" % (type(e).__name__, e))


PLAN_CHECK = [
    ("申通快递和圆通速递的业务量对比", None),
    ("顺丰控股2025年营业收入是多少", None),
    ("中国外运2025年营业收入同比变化", None),
    ("顺丰控股、圆通速递、韵达股份三家2025年营业收入对比", None),
]
PLAN_NEEDLES = {
    "申通快递和圆通速递的业务量对比": [("申通快递", ["261.39"]), ("圆通速递", ["311.44"])],
    "顺丰控股2025年营业收入是多少": [("顺丰控股", ["3,082.3亿", "3,082亿"])],
    "中国外运2025年营业收入同比变化": [("中国外运", ["8.34%"])],
    # 圆通/韵达这两条我原来写的是 ['56','营业收入'] / ['营业收入']——**那是假绿**：
    # 「营业收入」四个字满篇都是，随便哪个块都能满足。数字是现从语料里查出来的原文写法
    # （圆通 600233-0018 印17、韵达 002120-0014 印11），不是估的。
    "顺丰控股、圆通速递、韵达股份三家2025年营业收入对比": [
        ("顺丰控股", ["3,082.3亿", "3,082亿"]), ("圆通速递", ["752.77"]),
        ("韵达股份", ["514.75"])],
}


def _digits(s):
    """只留数字：'3,082.3亿' → '30823'。比数字串时用它，千分位/小数点的写法差异就抹平了。"""
    return re.sub(r"\D", "", _norm_num(s))


# 同一笔钱的**另一种原文写法**：叙述段写「514.75 亿元」，合并报表写「51,474,506,545.37」元。
# 两者都是年报自己印的，**不是我在放水**——但口径不一样，所以**分开报、不合并**：
# 严格口径（叙述段的亿元写法）没过就是没过，这张表只是说明「没过的原因是哪种没过」。
PLAN_NEEDLES_ALT = {
    "顺丰控股、圆通速递、韵达股份三家2025年营业收入对比": [
        ("顺丰控股", ["308,226,647"]),          # 千元口径（002352-0010/-0071 都印这个）
        ("圆通速递", ["7,527,684.74", "75,276,847,420.82"]),
        ("韵达股份", ["51,474,506,545.37"])],
}


def cmd_selftest(a):
    """端到端自检：**判据是「答案块进没进喂给模型的那批 ctxs」**，不是「检索榜上有没有」。

    阶段三的 selftest 查的是检索榜，它证明不了模型看没看到答案——中间还隔着子查询拆分
    与融合。这一步直接跑完整流水线，查三件事：

      1. **召回**：喂进去的证据里，每家公司有没有一块含答案数字串（k 取够大）；
      2. **成文**：答案正文里有没有照写那个数字。**召回到了却没写对是另一种失败**，
         与「没召回」的修法完全不同，所以要分开报，不能混成一个「没过」。
      3. **出处**：角标有没有、越界的有没有被丢、答案里的数字在原文里找不找得到。

    走模板回退时答案就是原文片段，且长块有「…（中略 N 字）…」，中略里的数字会被算成
    「没写」——**这是口径问题不是失败**，所以上面那行会先把口径打出来。"""
    ix = S.Index(a.model, a.device)
    key, _, model = env_conf()
    use_llm = bool(key) and not a.no_llm
    mode, filt = parse_plan(a.plan)
    print("索引 %d 块｜检索模型 %s｜拆分 %s" % (ix.man["n_chunks"], ix.man["model"], a.plan))
    print("LLM：%s" % ("用（%s）" % (model or "自动从 /models 挑") if use_llm
                     else "**不用**（%s）→ 走模板回退，答案=原文片段，按片段口径判"
                          % ("未设 FINRAG_API_KEY" if not key else "--no-llm")))
    print("判据：每家公司的答案数字串要出现在**喂给模型的证据**里（k=%d）；"
          "再看答案正文有没有照写、出处有没有落空。\n" % a.top)
    n_grp = n_rec = n_wrote = 0
    for q, _ in PLAN_CHECK:
        groups = PLAN_NEEDLES[q]
        out = answer_question(ix, q, k=a.top, mode=mode, filt=filt, use_llm=use_llm)
        print("=" * 78)
        print("【题】%s" % q)
        print("  子查询：%s" % " ; ".join("%s→%r" % (s["company"] or "·", s["query"])
                                          for s in out["sub_queries"]))
        print("  模型 %s｜%ss%s" % (out["model"], out["elapsed"],
                                   "｜" + out["note"] if out["note"] else ""))
        ans_nums = _nums(out["answer"])
        alt = dict(PLAN_NEEDLES_ALT.get(q) or [])
        for comp, alts in groups:
            n_grp += 1
            rk = None
            for e in out["evidence"]:
                if e["company"] == comp and S.hit_ctx(e["text"], alts):
                    rk = e["n"]
                    break
            wrote = any(_digits(x) == _digits(nd) for nd in alts for x in ans_nums)
            n_rec += 1 if rk else 0
            n_wrote += 1 if wrote else 0
            tag = ("召回#%d" % rk if rk else "**未召回**")
            if rk and not wrote:
                tag += "｜**召回到了但答案没写**"
            extra = ""
            if not rk and alt.get(comp):
                # 没召回时补一句：同一笔钱的**另一种原文写法**（元口径表格）在不在？
                # 不合并口径，只是让人看清「是这家完全没戏，还是只是叙述段没进来」。
                r2 = None
                for e in out["evidence"]:
                    if e["company"] == comp and S.hit_ctx(e["text"], alt[comp]):
                        r2 = e["n"]
                        break
                extra = "（另一口径%s）" % ("#%d" % r2 if r2 else "也没进来")
            print("   %-8s %-12s 答案%s%s"
                  % (comp, tag, "照写了" if wrote else "没有这个数", extra))
        if out["warnings"]:
            print("   ⚠ 告警：")
            for w in out["warnings"]:
                print("     · %s" % w)
        if out["used_llm"]:
            print("   角标 %s｜越界丢弃 %s｜无出处 %s"
                  % (out["citations"] or "无", out["dropped_citations"] or "无",
                     "**是**" if out["no_citation"] else "否"))
    print("\n" + "=" * 78)
    print("汇总：**召回 %d/%d**｜答案照写 %d/%d" % (n_rec, n_grp, n_wrote, n_grp))
    print("（召回=检索的锅，照写=模型的锅。两个数分开看，别合成一个。）")


def cmd_plan_check(a):
    """步骤 1 的 go/no-go：规则版拆分能不能复现我手写子查询的那次成功。
    **本模式不需要 key。**"""
    ix = S.Index(a.model, a.device)
    names = companies(ix)
    print("语料里 %d 家公司：%s" % (len(names), "、".join(names)))
    print("判据与阶段三一致：块里必须出现**答案数字串**；对比题要求**每家都进 top-%d**。\n" % a.top)
    combos = ["V1", "V1+f", "V2", "V2+f", "V2+u", "V3", "V3+f"]
    print("口径说明：**严格口径**要求块里出现该数字的**叙述段写法**（如「514.75 亿元」）；"
          "后面括号里的**另一口径**是同一笔钱的**合并报表写法**（如「51,474,506,545.37」元）。\n"
          "两者都是年报自己印的，但**绝不合并成一个数**——严格口径没过就是没过。\n")
    for q, _ in PLAN_CHECK:
        print("=" * 78)
        print("【题】%s" % q)
        groups = PLAN_NEEDLES[q]
        alt_groups = PLAN_NEEDLES_ALT.get(q)
        rows = []
        for tag in combos:
            mode, filt = parse_plan(tag)
            plan, comps, ev = retrieve(ix, q, k=a.top, mode=mode, filt=filt)
            per, per_alt = [], []
            for comp, alts in groups:
                rank = None
                for n, (cid, sc, r) in enumerate(ev, 1):
                    if r["company"] != comp:
                        continue
                    if S.hit_ctx(r["text"], alts):
                        rank = n
                        break
                per.append(rank)
            for comp, alts in (alt_groups or []):
                rank = None
                for n, (cid, sc, r) in enumerate(ev, 1):
                    if r["company"] != comp:
                        continue
                    if S.hit_ctx(r["text"], alts):
                        rank = n
                        break
                per_alt.append(rank)
            ok = all(x is not None for x in per)
            rows.append((tag, ok, per, per_alt, plan))
            fmt = lambda xs: "/".join(str(x) if x else "未召回" for x in xs)
            print(" %-16s %s 排 %-14s%s | %s" % (
                tag, "过  " if ok else "**没过**", fmt(per),
                ("（另一口径 %s）" % fmt(per_alt)) if per_alt else "",
                " ; ".join("%s→%r" % (c or "·", sq) for c, sq in plan)))
        print("\n | 变体 | 结果 |")
        print(" |---|---|")
        for tag, ok, per, per_alt, plan in rows:
            fmt = lambda xs: "/".join(str(x) if x else "未召回" for x in xs)
            print(" | %s | %s %s%s |" % (
                tag, "过" if ok else "没过", fmt(per),
                ("（另一口径 %s）" % fmt(per_alt)) if per_alt else ""))
        print()


def cmd_query(a):
    ix = S.Index(a.model, a.device)
    mode, filt = parse_plan(a.plan)
    out = answer_question(ix, a.q, k=a.top, mode=mode, filt=filt, use_llm=not a.no_llm)
    print("问题：%s" % a.q)
    print("子查询：%s" % " ; ".join("%s→%r" % (c or "·", sq) for c, sq in
                                   [(s["company"], s["query"]) for s in out["sub_queries"]]))
    print("模型：%s｜用了大模型：%s｜耗时 %ss%s"
          % (out["model"], out["used_llm"], out["elapsed"],
             ("｜" + out["note"]) if out["note"] else ""))
    print("\n── 答案 ──\n%s" % out["answer"])
    if out["no_citation"]:
        print("\n**该答案没有可核对的出处（一个合法角标都没有）**")
    else:
        print("\n角标：[%s]" % "] [".join(str(c) for c in out["citations"]))
    if out["dropped_citations"]:
        print("越界丢弃的出处：%s" % out["dropped_citations"])
    if out["warnings"]:
        print("\n── 告警（**没写对的地方都在这**）──")
        for w in out["warnings"]:
            print(" ⚠ %s" % w)
    print("\n── 检索到的证据（k=%d，全部摊开）──" % out["k"])
    for e in out["evidence"]:
        mark = "★" if e["n"] in out["citations"] else " "
        print(" %s[%2d] %-14s %-28s 印%-8s 物理页%-4s %.5f"
              % (mark, e["n"], e["company"], (e["section"] or "（未归节）")[:28],
                 e["printed"] or (e["printed_note"] or "无"), e["phys"], e["score"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="探通路（要 key）")
    ap.add_argument("--plan-check", action="store_true", help="比对子查询拆分规则（不要 key）")
    ap.add_argument("--selftest", action="store_true", help="端到端自检（无 key 则走模板回退）")
    ap.add_argument("--q")
    ap.add_argument("--top", type=int, default=K_DEFAULT)
    ap.add_argument("--no-llm", action="store_true", help="只检索，不调大模型")
    ap.add_argument("--plan", default="V2+u",
                    help="V1/V2/V3，后缀 +f 只过滤、+u 过滤与不过滤都跑（默认）")
    ap.add_argument("--model", default=S.DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.probe:
        cmd_probe(a)
    elif a.plan_check:
        cmd_plan_check(a)
    elif a.selftest:
        cmd_selftest(a)
    elif a.q:
        cmd_query(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
