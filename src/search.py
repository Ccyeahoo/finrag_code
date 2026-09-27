# -*- coding: utf-8 -*-
r"""阶段三 检索：向量 + BM25 混合召回，RRF(k=60) 融合，**不加 reranker**。

为什么不加 reranker：课件实测加了更差（半年报表格题 0.30 vs 0.90）。理由在
`out/stage1_report.md` 的同源结论里，本脚本不改这个决定。

用法：
    PY=/c/Users/yexinyu/AppData/Local/Python/bin/python.exe   # 裸 python 是 Store 存根
    $PY src/search.py --build                     # 建索引 → out/index/
    $PY src/search.py --q "顺丰控股2025年营业收入" --mode hybrid --top 10
    $PY src/search.py --q "..." --mode vector     # 或 --mode bm25
    $PY src/search.py --selftest                  # 3 个已知答案的题，贴对照

产物（out/index/）：
    emb.npy           N×D float32，已 L2 归一化
    emb_meta.json     行号 → chunk_id
    bm25_tokens.jsonl 每块的分词
    manifest.json     模型 id/revision/维度/块数/分词器/chunks.jsonl 的 sha256

**key 全程不需要**：嵌入与 BM25 都在本地跑。阶段四/五要用 LLM 组织答案时才要 key，
按环境变量读、绝不落盘、绝不进 git、绝不发给前端。

依赖（阶段三安装，见计划体积表）：numpy / rank_bm25 / jieba / sentence-transformers
    - rank_bm25 必须走 wheel（sdist 的 setup.py 引用不存在的 version.py）
    - jieba 装不上就自动退到「CJK 二元 + 字母数字串」分词，manifest 里如实记哪个生效
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time

if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract as E

OUT = E.OUTDIR
CHUNKS = os.path.join(OUT, "chunks.jsonl")
IDX = os.path.join(OUT, "index")

# 模型选择（**本机实测后改过一次，理由见 out/stage3_report.md 体积/耗时表**）：
#   计划选的是 Qwen/Qwen3-Embedding-0.6B，但本机是 i7-1165G7（4 核、无 AVX512-BF16、
#   无 CUDA），实测 fp32 **8.1 s/块 → 4503 块要 10.1 小时**，不可行；且该仓权重是
#   bf16，本机 CPU 不支持会**静默降级成模拟**，慢到 21 s/块。
#   改用它：bge-small-zh-v1.5（24M、512 维、512 token）实测 **0.15 s/块 → 约 11 分钟**。
#   代价如实记：中文检索质量不如 0.6B 那档。想换回来：
#       $PY src/search.py --build --model BAAI/bge-base-zh-v1.5    # 0.83 s/块 ≈ 1.0 h
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："   # bge-zh-v1.5 官方查询前缀
RRF_K = 60
ROUTE_TOP = 20                             # 每路召回条数，再融合

# 块正文最长 ~1168 字 ≈ 1200 token，超过模型 512 的上限。**不截断**：
# 按 token 切成重叠窗口，每窗一个向量，检索时同一个 chunk_id 取最高分。
WINDOW = 480
STRIDE = 384


# ------------------------------------------------------------------ 基础
def ensure_hf_endpoint():
    """本机实测 `huggingface.co` 连不上（15s 超时）、`hf-mirror.com` 通。
    在**导入 huggingface_hub 之前**定 `HF_ENDPOINT`，否则模型下载会挂住。
    只在未设 + 官方站不通时才动，并且**打印出来**，不做静默切换。"""
    if os.environ.get("HF_ENDPOINT"):
        print("[hf] 沿用已设的 HF_ENDPOINT=%s" % os.environ["HF_ENDPOINT"])
        return
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=4).close()
        return                                    # 官方站通，什么都不做
    except Exception as e:
        print("[hf] huggingface.co 不通（%s）→ 改用镜像 https://hf-mirror.com" % e)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


ensure_hf_endpoint()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_chunks():
    return [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]


def brief(r, n=110):
    t = re.sub(r"\s+", " ", r["text"].replace(r["prefix"], "")).strip()
    return t[:n] + ("…" if len(t) > n else "")


def where(r):
    """出处一行：公司 / 章节 / 印页（没有印页就写说明，不编造）"""
    pg = r.get("printed_lo")
    if pg is None:
        pg = r.get("page_note") or "无印页"
    elif r.get("printed_hi") not in (None, pg):
        pg = "%s–%s" % (pg, r["printed_hi"])
    sec = r.get("section") or "（未归节）"
    return "%s｜%s｜印%s" % (r["company"], sec, pg)


# ------------------------------------------------------------------ 分词
def make_tokenizer():
    """返回 (name, func)。jieba 优先；装不上退到 CJK 二元，不静默换口径。"""
    try:
        import jieba
        jieba.setLogLevel(60)

        def f(s):
            return [w for w in jieba.lcut(s) if w.strip()]
        return "jieba.lcut", f
    except Exception as e:
        sys.stderr.write("[warn] jieba 不可用（%s）→ 退到 CJK 二元分词\n" % e)

    def g(s):
        out, buf = [], []
        for ch in s:
            if "一" <= ch <= "鿿":
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(ch)
            elif ch.isalnum():
                buf.append(ch)
            elif ch.isspace():
                if buf:
                    out.append("".join(buf))
                    buf = []
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(ch)
        if buf:
            out.append("".join(buf))
        # CJK 单字 + 相邻二元，模拟"词"
        cjk = [w for w in out if len(w) == 1 and "一" <= w <= "鿿"]
        bg = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return out + bg
    return "cjk-uni+bigram（jieba 缺失的退路）", g


# ------------------------------------------------------------------ 向量
def load_embedder(model, device="cpu"):
    from sentence_transformers import SentenceTransformer
    # 一律 fp32：本机 CPU 无 bf16，若沿用仓里的 bf16 权重，torch 会**静默**降级成
    # 模拟实现，实测慢 2.6 倍（21 s/块 vs 8.1 s/块），而且不报错。
    m = SentenceTransformer(model, device=device,
                            model_kwargs={"dtype": __import__("torch").float32})
    # bge-zh-v1.5 的查询前缀。ST 读出来的 prompts 可能是空串（实测本机就是空），
    # 空着会让查询与文档**同分布**，检索质量下降但**不会报错**——所以显式设死。
    try:
        if not (m.prompts or {}).get("query"):
            m.prompts["query"] = QUERY_PROMPT
    except Exception:
        pass
    return m


def window_texts(tok, text, win=WINDOW, stride=STRIDE):
    """长文本切成 token 窗口再解码回字符串。**不丢字**：
    覆盖不足 win 的尾段也保留。窗口重叠 stride<win，避免切断的关键词两边都查不到。"""
    lim, tok.model_max_length = tok.model_max_length, 10 ** 9   # 免得每次超长都刷警告
    try:
        ids = tok(text, add_special_tokens=False)["input_ids"]
    finally:
        tok.model_max_length = lim
    if len(ids) <= win:
        return [text]
    out, s = [], 0
    while True:
        out.append(tok.decode(ids[s:s + win]))
        if s + win >= len(ids):
            break
        s += stride
    return out


def embed_docs(m, texts, batch=16):
    """文档侧编码。Qwen3-Embedding 的 base 仓自带 prompts 配置，
    encode_document 会按它决定要不要加前缀；没有就退回 encode。"""
    fn = getattr(m, "encode_document", None) or m.encode
    return fn(list(texts), batch_size=batch, normalize_embeddings=True,
              show_progress_bar=True, convert_to_numpy=True)


def embed_query(m, q):
    fn = getattr(m, "encode_query", None) or m.encode
    return fn([q], normalize_embeddings=True, convert_to_numpy=True)[0]


# ------------------------------------------------------------------ 建索引
def cmd_build(a):
    import numpy as np
    rows = load_chunks()
    if a.min_chars:
        rows = [r for r in rows if r["n_chars"] >= a.min_chars]
    os.makedirs(IDX, exist_ok=True)
    t0 = time.time()
    print("块数 %d（chunks.jsonl 的 sha256 %s…）"
          % (len(rows), sha256_file(CHUNKS)[:12]))

    # BM25 分词
    tname, tok = make_tokenizer()
    toks = [tok(r["text"]) for r in rows]
    with open(os.path.join(IDX, "bm25_tokens.jsonl"), "w", encoding="utf-8") as f:
        for r, t in zip(rows, toks):
            f.write(json.dumps({"chunk_id": r["chunk_id"], "tokens": t},
                               ensure_ascii=False) + "\n")
    print("分词器 %s，平均 %d 词/块" % (tname, sum(len(t) for t in toks) // max(len(toks), 1)))

    # 向量
    print("加载模型 %s（首次会下载到 HF 缓存；离线请用 --model 指本地目录）" % a.model)
    m = load_embedder(a.model, a.device)
    try:
        dim = m.get_embedding_dimension()
    except AttributeError:
        dim = m.get_sentence_embedding_dimension()

    # 长块切窗（不截断）。flat 与 owner 一一对应：第 i 个向量属于 rows[owner[i]]。
    tokf = m.tokenizer
    flat, owner, n_trunc_free = [], [], 0
    for i, r in enumerate(rows):
        ws = window_texts(tokf, r["text"])
        if len(ws) > 1:
            n_trunc_free += 1
        for w in ws:
            flat.append(w)
            owner.append(i)
    longest = max(len(tokf(t, add_special_tokens=False)["input_ids"]) for t in flat)
    print("窗口：%d 块 → %d 个向量（%d 块超 %d token 被切窗）；最长窗 %d token（上限 %d）"
          % (len(rows), len(flat), n_trunc_free, WINDOW, longest, m.max_seq_length))
    assert longest <= m.max_seq_length, "有窗口仍超模型上限，会静默截断"

    t1 = time.time()
    emb = embed_docs(m, flat, a.batch)
    emb = np.asarray(emb, dtype="float32")
    np.save(os.path.join(IDX, "emb.npy"), emb)
    json.dump([rows[i]["chunk_id"] for i in owner],
              open(os.path.join(IDX, "emb_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    import chunk as C
    man = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": a.model, "dim": int(dim), "n_chunks": len(rows),
        "n_vectors": len(flat), "window": {"win": WINDOW, "stride": STRIDE},
        "chunks_over_window": n_trunc_free, "longest_window_tokens": int(longest),
        "embed_seconds": round(time.time() - t1, 1),
        "query_prompt": (m.prompts or {}).get("query") or "",
        "chunks_sha256": sha256_file(CHUNKS), "min_chars": a.min_chars,
        "tokenizer": tname, "rrf_k": RRF_K, "route_top": ROUTE_TOP,
        "no_reranker": True,
        "chunk_params": {"SIZE": C.SIZE, "OVERLAP": C.OVERLAP, "COV_MIN": C.COV_MIN},
        "numpy": np.__version__,
    }
    try:
        import sentence_transformers as st
        man["sentence_transformers"] = st.__version__
        man["max_seq_length"] = int(m.max_seq_length)
    except Exception:
        pass
    json.dump(man, open(os.path.join(IDX, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("向量 %s  维度 %d  用时 %.1fs" % (emb.shape, dim, time.time() - t0))
    print("已写 %s/" % IDX)


# ------------------------------------------------------------------ 检索
class Index:
    def __init__(self, model=None, device="cpu"):
        import numpy as np
        from rank_bm25 import BM25Okapi
        self.rows = load_chunks()
        man_p = os.path.join(IDX, "manifest.json")
        self.man = json.load(open(man_p, encoding="utf-8"))
        # 索引与语料必须对得上——对不上就报错，不静默用旧索引
        cur = sha256_file(CHUNKS)
        if cur != self.man.get("chunks_sha256"):
            raise SystemExit("索引与 chunks.jsonl 不匹配（索引建后语料变过）：\n"
                             "  索引 %s…  现在 %s…\n请重跑 --build"
                             % (self.man["chunks_sha256"][:12], cur[:12]))
        self.emb = np.load(os.path.join(IDX, "emb.npy"))
        ids = json.load(open(os.path.join(IDX, "emb_meta.json"), encoding="utf-8"))
        self.by_id = {r["chunk_id"]: r for r in self.rows}
        self.meta = ids                       # **窗口级**（一个长块占多行），只给向量路用
        # BM25 是**块级**的。它的 chunk_id 必须从 bm25_tokens.jsonl 自己读，
        # **不能借用 self.meta**：两者长度不同（8631 vs 4503），借了就是下标错位，
        # 返回一堆看着正常、其实对不上的块，而且不报错。这个 bug 我踩过一次，留字为证。
        bt = [json.loads(l)
              for l in open(os.path.join(IDX, "bm25_tokens.jsonl"), encoding="utf-8")]
        self.bm25_ids = [b["chunk_id"] for b in bt]
        self.toks = [b["tokens"] for b in bt]
        assert len(self.emb) == len(self.meta), "emb.npy 与 emb_meta.json 行数不符"
        assert len(self.bm25_ids) == len(self.toks), "BM25 词表与 id 数不符"
        assert set(self.bm25_ids) <= set(self.by_id), "BM25 的 id 不在 chunks.jsonl 里"
        self.bm25 = BM25Okapi(self.toks)
        self._tok = make_tokenizer()[1]
        self._model = None
        self.model_name = model or self.man["model"]
        self.device = device

    def ensure_model(self):
        if self._model is None:
            self._model = load_embedder(self.model_name, self.device)
        return self._model

    def _keep(self, cid, comp):
        """comp=None 全要；否则只留该公司的块。company 字段来自 chunks.jsonl。"""
        if comp is None:
            return True
        r = self.by_id.get(cid)
        return bool(r) and r["company"] == comp

    def vector(self, q, top=ROUTE_TOP, comp=None):
        """一个块可能有多个窗口向量（长块切窗），**按 chunk_id 取最高分**再排序。
        comp 给公司名就只在那一家里排（阶段四的对比题要用）。"""
        import numpy as np
        qv = embed_query(self.ensure_model(), q)
        s = self.emb @ np.asarray(qv, dtype="float32")
        best = {}
        for i, sc in enumerate(s):
            cid = self.meta[i]
            if not self._keep(cid, comp):
                continue
            f = float(sc)
            if cid not in best or f > best[cid]:
                best[cid] = f
        return sorted(best.items(), key=lambda kv: -kv[1])[:top]

    def bm25v(self, q, top=ROUTE_TOP, comp=None):
        import numpy as np
        s = self.bm25.get_scores(self._tok(q))
        out = []
        # 降序扫，跳过不属于 comp 的；**遇到第一个非正分就停**（后面全是 0 或负）
        for i in np.argsort(-s):
            if s[i] <= 0:
                break
            cid = self.bm25_ids[i]
            if not self._keep(cid, comp):
                continue
            out.append((cid, float(s[i])))
            if len(out) >= top:
                break
        return out

    def routes(self, q, top=ROUTE_TOP, comp=None):
        """两路各跑一次，返回 (向量, BM25)。查询只嵌入一次，别为了好看重复算。"""
        return self.vector(q, top, comp), self.bm25v(q, top, comp)

    def hybrid(self, q, top=10, k=RRF_K, comp=None):
        """RRF：score = Σ 1/(k + rank)，两路各取 ROUTE_TOP 条。"""
        return rr_fuse(*self.routes(q, comp=comp), top=top, k=k)


def rr_fuse_many(routes, top=10, k=RRF_K):
    """RRF 的通式：把**任意多条**召回路线（向量、BM25、以及阶段四各条子查询的
    两路）一视同仁地按 1/(k+rank) 累加。

    阶段三的 rr_fuse 就是「两条路线」的特例——所以 rr_fuse 直接调它，
    公式只留这一份，避免两处实现漂移。"""
    acc = {}
    for route in routes:
        for rank, (cid, _) in enumerate(route, 1):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(acc.items(), key=lambda kv: -kv[1])[:top]


def rr_fuse(v, b, top=10, k=RRF_K):
    """两条路线（向量、BM25）的融合。等价于 rr_fuse_many([v, b], ...)。"""
    return rr_fuse_many([v, b], top=top, k=k)


def print_hits(items, rows_by_id, head):
    print(head)
    for n, (cid, sc) in enumerate(items, 1):
        r = rows_by_id.get(cid)
        if r is None:
            print("  %2d  %s  （不在 chunks.jsonl 里）" % (n, cid))
            continue
        print("  %2d  %-14s %.4f  %s" % (n, cid, sc, where(r)))
        print("       %s" % brief(r))


def cmd_query(a):
    t0 = time.time()
    ix = Index(a.model, a.device)
    print("索引：%d 块｜模型 %s｜维度 %d｜分词 %s｜无 reranker｜RRF k=%d"
          % (ix.man["n_chunks"], ix.man["model"], ix.man["dim"],
             ix.man["tokenizer"], ix.man["rrf_k"]))
    print("问题：%s   模式：%s\n" % (a.q, a.mode))
    v, b = ix.routes(a.q)
    if a.mode == "vector":
        items = v[:a.top]
    elif a.mode == "bm25":
        items = b[:a.top]
    else:
        items = rr_fuse(v, b, top=a.top)
        # 混合模式额外贴两路各自的独有项，证明融合真的多召回
        vs, bs = {c for c, _ in v}, {c for c, _ in b}
        print("两路独有：向量独有 %d 条、BM25 独有 %d 条、交集 %d 条\n"
              % (len(vs - bs), len(bs - vs), len(vs & bs)))
    print_hits(items, ix.by_id, "── 结果（%.1fs）──" % (time.time() - t0))


# ------------------------------------------------------------------ 自检
# 判据说明（**这里改过一次，记下为什么**）：
#   上一版判据是「top-N 里有没有块同时含公司名 + 关键词（如「营业收入」）」。那条太松——
#   它报出 9/9 命中，但实际上「业务量对比」那题，圆通**含数字的答案块三路都没召回**；
#   命中的只是含「业务量」三个字的叙述段。**自检报绿而实际没做出来，比没有自检更糟。**
#   现在改成严判据：块里必须出现**答案数字串**本身。
#   - 「数字串」是我**人工读 PDF 抄下来的**，等于把答案写死做成回归测试；
#   - 对比题要求**每一家**都进 top-N，只进一家不算过（否则「对比」根本做不了）。
SELFTEST = [
    ("顺丰控股2025年营业收入是多少",
     [("顺丰控股", ["3,082.3亿", "3,082亿"])]),
    ("中国外运2025年营业收入同比变化",
     [("中国外运", ["8.34%"])]),          # 要的是「同比变化」，只出现 968.09亿 不算
    ("申通快递和圆通速递的业务量对比",
     [("申通快递", ["261.39"]), ("圆通速递", ["311.44"])]),
]


def hit_ctx(text, needles, w=55):
    """把「答案串」前后各 w 字打出来，好人工核对答案是不是真在里面。
    返回 None 表示这个块里根本没有答案串（不算命中）。"""
    for x in needles:
        i = text.find(x)
        if i >= 0:
            a = max(0, i - w)
            return "%s【%s】%s" % (re.sub(r"\s+", " ", text[a:i]), x,
                                  re.sub(r"\s+", " ", text[i + len(x):i + len(x) + w]))
    return None


def cmd_selftest(a):
    ix = Index(a.model, a.device)
    print("索引：%d 块｜模型 %s｜维度 %d｜判据：答案数字串须进 top-%d"
          % (ix.man["n_chunks"], ix.man["model"], ix.man["dim"], a.top))
    print("（对比题「申通↔圆通」要求两家都进；只进一家 = 这题没过）\n")
    verdict = []
    for q, groups in SELFTEST:
        print("=" * 78)
        print("【题】%s" % q)
        v, b = ix.routes(q)
        for mode, items in (("vector", v[:a.top]), ("bm25", b[:a.top]),
                            ("hybrid", rr_fuse(v, b, top=a.top))):
            per = []
            for comp, alts in groups:
                rank, ctx = None, None
                # 同一家可能有多个答案块，**只记第一个排名**（per 记的是「第几组过了」，不是命中次数）
                for n, (cid, _) in enumerate(items, 1):
                    r = ix.by_id.get(cid)
                    if not r or r["company"] != comp:
                        continue
                    c = hit_ctx(r["text"], alts)
                    if c:
                        rank, ctx = n, "【%s】%s" % (cid, c)
                        break
                per.append(rank)              # 未召回就是 None，留在表里看得见
            ok = all(x is not None for x in per) and len(per) == len(groups)
            verdict.append((q, mode, ok, per))
            print(" %-7s %s 排 %-4s | top5：%s"
                  % (mode, "过" if ok else "**没过**",
                     "/".join(str(x) if x else "未召回" for x in per),
                     " ".join("%s(印%s)" % (c, (ix.by_id[c].get("printed_lo")))
                              for c, _ in items[:5] if c in ix.by_id)))
            if ctx:
                print("         └ 答案块原文：%s" % ctx[:190])
        print_hits(rr_fuse(v, b, top=a.top)[:5], ix.by_id, " hybrid top5 明细：")
    print("\n" + "=" * 78)
    print("| 题 | vector | bm25 | hybrid |")
    print("|---|---|---|---|")
    for q, groups in SELFTEST:
        row = {m: (p, ok) for qq, m, ok, p in verdict if qq == q}
        cells = []
        for m in ("vector", "bm25", "hybrid"):
            p, ok = row.get(m, ([], False))
            cells.append("%s%s" % ("过 " if ok else "没过 ",
                                   "/".join(str(x) if x else "未召回" for x in p)))
        print("| %s | %s |" % (q, " | ".join(cells)))
    n_ok = sum(1 for _, _, ok, _ in verdict if ok)
    print("\n三种检索 × %d 题 = %d 格，**过 %d 格**" % (len(SELFTEST), len(verdict), n_ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF 仓库名或本地模型目录（离线用本地目录）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--min-chars", type=int, default=0,
                    help="建索引时丢掉短于该字数的块（默认 0=全要）")
    a = ap.parse_args()
    if a.build:
        cmd_build(a)
    elif a.selftest:
        cmd_selftest(a)
    elif a.q:
        cmd_query(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
