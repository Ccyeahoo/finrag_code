# -*- coding: utf-8 -*-
r"""阶段二：13 家全量切块 → out/chunks.jsonl

每条设计都对着阶段一实测出的一个缺陷（见 out/stage1_report.md §11 与
out/chunks_preview.jsonl 的四项不达标），逐条对应：

  1. **表格按行列进正文**（不是只计数）。取表顺序 = 阶段一定稿的生产规则：
     一页优先用 `stream`，`stream` 没抓到才退 `lines`，`text` 不进生产
     （出处 src/sum_tables.py:78-100）。表与文字行按 y 坐标**交错**排入，
     保持与纸面一致的阅读顺序；落在表 bbox 里的文字行会被剔除，不重复。
  2. **800 字/块、120 字重叠、句末收尾**。阶段一的预览块 27.9% > 900 字
     （去空白最长 2581），且 chunk_units 的重叠在「句末收尾」分支里被
     下一句赋值覆盖丢弃（stage1.py:50-98）。这里重写，并加了防死循环兜底。
  3. **每块前置【公司名｜章节名】**。前缀实打实进 text，方便 BM25/向量都吃到；
     n_chars 只数正文，不数前缀。
  4. **块不跨节**：遇节边界强制收尾，省得第八节的表被贴进第七节的块。
  5. **乱码页照图改正**：data/text_patches.json 按 (公司, 物理页) 整页替换。
  6. **纯图隔页归给下一节**：见 fix_dividers()，已实测（申通 8 命中 / 其余 0）。
  7. **无印页不编造**：printed 为 None 的单元如实记 null + page_note，
     不按 2p-6 之类硬填（国货航 5 个页单元）。

一个**与阶段一不同的决定**，如实记在这里：文字行改为**统一按坐标重排**
（extract.words_to_text 同一套规则）取，不再对 normal 版式用
`pg.get_text("text")` 的原始读序。原因有二：① 要把表按 y 插进正文，本来
就必须有坐标，纯 get_text 拿不到行坐标；② 阶段一已经实测过原始读序在
双栏页（申通目录）上是错的，坐标序才是对的那份。11 家正常版式的文字内容
与阶段一逐页对照过一致，代价可忽略。

用法：
  python chunk.py --all                  # 全量 13 家（未缓存的 9 家要抽表，后台跑）
  python chunk.py 申通快递 嘉友国际       # 指定公司
  python chunk.py --all --no-tables       # 只切文字（调块长用，秒级）
  python chunk.py --all --sample-only     # 只重出 chunks_sample.md，不重切
"""
import argparse
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter

if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract as E
import pymupdf as fitz

ROOT = E.ROOT
OUT = E.OUTDIR
CHUNKS = os.path.join(OUT, "chunks.jsonl")
PATCHES = os.path.join(ROOT, "data", "text_patches.json")

SIZE = 800                    # 目标块长（去空白字数），课程参数
OVERLAP = 120                 # 目标重叠字数，课程参数
COV_MIN = 0.95                # 表框内文字行的"被表覆盖"阈值：>= 才剔，< 则保留（实测定的，见 unit_pieces）
SMALL_MIN = 200               # 碎块门槛：<该字数的块并进**同节**邻块（实测第一轮 5.3% 的块 <200 字，见 merge_small）
MERGE_CAP = 1200              # 并块后的块长上限：越过就不并，宁可留短块也不撞破"0 块 >2000 字"
SENT_END = "。；！？：”）」』.!?;"   # 句末字符（沿用 stage1.py 的定义）
HALF_ORDER = {"L": 0, "R": 1, "full": 2}

# 纯图隔页判据里的"字数少"门槛（与阶段一 probe_div.py 实测用的同一个值）
DIVIDER_MAX_CHARS = 40
# 第二道护栏：该页必须是整页出血图。来源 out/blind_all.csv 的 img_ratio 列
# （图片面积占页面面积比）。加这道护栏是为了防止在另外 9 家**未逐页验算**的
# 公司里，把"恰好字数很少且页码往前差一"的正文页误判成隔页。
DIVIDER_MIN_IMG = 0.9


# ------------------------------------------------------------------ 小工具
def nw(s):
    """去空白字数 —— 全流程唯一的口径（与 companies.csv 的 chars 一致）"""
    return len(re.sub(r"\s+", "", s))


def nows(s):
    """去空白后的**字符串**（nw 是它的长度），回查剔除行时要用它做子串匹配"""
    return re.sub(r"\s+", "", s)


def load_patches():
    if not os.path.exists(PATCHES):
        return {}
    with open(PATCHES, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for comp, pages in raw.items():
        if comp.startswith("_"):
            continue
        for phys, val in pages.items():
            txt = val["text"] if isinstance(val, dict) else val
            out[(comp, int(phys))] = txt
    return out


def load_img_ratio():
    """(公司, 物理页, half) -> 图片面积占比。来自阶段一的 out/blind_all.csv。"""
    p = os.path.join(OUT, "blind_all.csv")
    if not os.path.exists(p):
        return {}
    import csv
    with open(p, encoding="utf-8-sig") as f:
        return {(r["company"], int(r["phys"]), r["half"]): float(r["img_ratio"] or 0)
                for r in csv.DictReader(f)}


# ------------------------------------------------------------------ 文字行
def text_lines(pg, clip):
    """按 y 聚成文本行 -> [{"y","y0","y1","x0","x1","text"}]，读序 = 坐标序。

    y 容差与词间空格规则**照抄** extract.words_to_text（阶段一已对照过），
    只是这里保留每行的坐标，好在插表时按 y 定位。
    """
    ws = E.words_in(pg, clip) if clip is not None else pg.get_text("words")
    if not ws:
        return []
    ws = sorted(ws, key=lambda t: (t[1], t[0]))
    hs = [t[3] - t[1] for t in ws]
    tol = max(2.5, 0.45 * sorted(hs)[len(hs) // 2])
    groups, cur = [], [ws[0]]
    for t in ws[1:]:
        if abs(t[1] - cur[-1][1]) <= tol:
            cur.append(t)
        else:
            groups.append(cur)
            cur = [t]
    groups.append(cur)
    out = []
    for g in groups:
        g.sort(key=lambda t: t[0])
        buf = g[0][4]
        for prev, t in zip(g, g[1:]):
            buf += (" " if t[0] - prev[2] > E.SPACE_GAP else "") + t[4]
        out.append({"y": sum((t[1] + t[3]) / 2 for t in g) / len(g),
                    "y0": min(t[1] for t in g), "y1": max(t[3] for t in g),
                    "x0": min(t[0] for t in g), "x1": max(t[2] for t in g),
                    "text": buf})
    out.sort(key=lambda L: L["y"])
    return out


def split_long_line(txt, size):
    """单行超过 size 字时按句末拆（正常页不会走到这里，是防病态页的死循环兜底）。
    只在句末/逗号处断，**绝不从数字中间切开**。"""
    if nw(txt) <= size:
        return [txt]
    parts, buf = [], ""
    for ch in txt:
        buf += ch
        if nw(buf) >= size and ch in SENT_END + "，、, ":
            parts.append(buf)
            buf = ""
    if buf:
        parts.append(buf)
    return parts


# ------------------------------------------------------------------ 表格
def pick_tables(tb):
    """**生产取表规则**：优先 stream，stream 一张都没抓到才退 lines；text 不进生产。
    出处 src/sum_tables.py:78-100（阶段一定稿）。返回 (策略名, [表])。"""
    for strat in ("stream", "lines"):
        ts = [t for t in (tb or {}).get(strat, [])
              if isinstance(t, dict) and "rows" in t and t.get("cells")]
        if ts:
            return strat, ts
    return None, []


def _cell_rows(t):
    rows = t.get("cells") or []
    out = []
    for r in rows:
        cs = [re.sub(r"\s+", " ", ("" if c is None else str(c))).strip() for c in r]
        if any(cs):
            out.append(cs)
    return out


def table_pieces(t, tid, size):
    """一张表 -> 1..n 个 piece 的**多行文本**。

    格式（作业要求的"按行列还原"就是这个样子）：
        [表 3] 4行×5列
        项目 | 2025年 | 2024年 | 同比增减(%)
        营业收入 | 96,808,663,837.20 | 105,620,773,355.43 | -8.34
    表是**原子**的：整表能装进一块就不拆。只有单表 > size 字时才按行拆，
    每段重复表头行并标 (续)。
    """
    rows = _cell_rows(t)
    if not rows:
        return []
    n = len(rows)
    cols = max(len(r) for r in rows)
    body = [" | ".join(list(r) + [""] * (cols - len(r))) for r in rows]
    head = "[表 %s] %d行×%d列" % (tid, n, cols)
    whole = head + "\n" + "\n".join(body)
    if nw(whole) <= size or n <= 2:
        # n<=2 的表**故意不拆**：按行拆只能拆出"表头单独一段"这种没用的碎片，
        # 而一行本身就超 size 时怎么拆都超。这类 piece 会略微超过 size，
        # 数量在 chunks_stats.md 里如实报（实测极少）。
        return [whole]
    # 续段要重复表头行，所以预算得把它算进去 —— 早先忘了算，
    # 于是每段实际长度 = 800 + 表头行，申通出现 829 字的块就是这么来的。
    reserve = nw(head) + nw(body[0])
    segs, cur, curlen, first = [], [], nw(head), True
    for ln in body:
        L = nw(ln)
        if cur and curlen + L > size:
            segs.append(cur)
            first = False
            cur, curlen = [], reserve
        cur.append(ln)
        curlen += L
    if cur:
        segs.append(cur)
    out = []
    for k, seg in enumerate(segs):
        h = head if k == 0 else "[表 %s（续）] %d行×%d列" % (tid, n, cols)
        # 续段重复表头行：否则第二段只剩数字，看不出哪列是哪列
        extra = [body[0]] if k > 0 else []
        out.append("\n".join([h] + extra + seg))
    return out


# ------------------------------------------------------------------ 页单元 -> pieces
def sec_split(u, lines, sec_ctx):
    """页单元里**印着本节自己的标题**时，返回 (标题行的 y, 上一节名)；否则 None。

    为什么要有这一条（阶段二遗留问题②，实测 7 个节整节 0 块）：
    `locate_sections` 把「页 -> 节」做成一对一的字典，**同一物理页上印着两个节号时
    后来者覆盖先来者**。于是前一节的内容被贴成后一节的名字 —— 华贸/中储/中谷/辽港/
    铁龙 的「第一节 释义」、中谷/嘉友 的「第七节 债券相关情况」在语料里名下 0 块。
    内容一个字没丢（审计 §3 零丢失 100%），但**按节名过滤会全漏**。

    修法**只改 piece 的 section 字符串**：标题行**以上**的行归上一节。下游
    `chunk_pieces()` 本来就按 `piece["section"]` 变节切块、节边界不回带，所以
    **不必动已经审计过的切块器**（这是这条修法风险低的原因）。

    **自校验安全阀（关键，别删）**：只有当「本节按目录页码换算出的物理页 == 本页
    物理页」时才允许切。这一条挡住正文里的交叉引用 —— 顺丰 p72 正文里印着
    「第九节 财务报告」，但第九节起始印页是 148，环境不等，**必须被拒**。
    """
    if not sec_ctx:
        return None
    cur = u.get("section")
    if not cur or sec_ctx["start"].get(cur) != u["phys"]:
        return None
    prev = sec_ctx["prev"].get(cur)
    if not prev:
        return None
    # 正文里印的可能是全称「第二节 公司简介和主要财务指标」，也可能**只有节号**
    # 「第二节」（实测顺丰是节号独立成行、标题在下一行）、还可能只有标题
    # （extract.py 里记的申通式印法）。判据用**整行相等**而不是包含，
    # 所以「见第二节」这类引用天然比不中。
    # **三边都要 nows 归一化再比**：实测 6 家把节号与标题**中间不加空格**印成一行
    # （「第二节公司简介和主要财务指标」），拿带空格的 cur 去比永远比不中——
    # 我第一版就是这么写的，结果 7 个目标页一个没命中，只误打误撞命中顺丰。
    sec, _, title = cur.partition(" ")
    forms = {nows(cur), nows(sec)}
    if len(nows(title)) >= 3:              # 「释义」这种 2 字标题太容易被正文撞上
        forms.add(nows(title))
    hits = [((L["y0"] + L["y1"]) / 2) for L in lines
            if nows(L["text"]) in forms and len(nows(L["text"])) <= 40]
    if not hits:
        return None
    return (min(hits), prev)          # 取最靠上的那个，允许页眉重复节号时也不误切


def unit_pieces(pg, u, tb, patch, size, stats, sec_ctx=None):
    """一个页单元 -> 排好序的 piece 列表（文字行与表按 y 交错）。"""
    base = {"phys": u["phys"], "half": u["half"],
            "printed": u["printed"], "section": u["section"]}
    if patch is not None:
        # 乱码页照图改正：整页替换，不走表格
        stats["patched"] += 1
        return [dict(kind="line", text=ln, **base)
                for ln in patch.split("\n") if ln.strip()]

    lines = text_lines(pg, u["clip"])
    strat, tabs = pick_tables(tb)
    if any(t.get("cells") is None for _, t in [(strat, t) for t in tabs]):
        # 兜底：缓存里这张表只存了形状没存内容（旧 60k 上限留下的），回原页重抽
        stats["recut"] += 1
        strat, tabs = pick_tables(E.find_tables_both(pg, u["clip"], limit=50_000_000))

    boxes = [t["bbox"] for t in tabs if isinstance(t.get("bbox"), list)
             and len(t["bbox"]) == 4]
    # 落在表框里的文字行**不是无条件剔掉** —— 只有内容确实被保留下来的表覆盖了才剔。
    # 早先只按 bbox 剔，实测真丢过字：申通 p12 表脚脚注被并进表末行、剔除后丢失；
    # 嘉友 p4 释义表抓漏了行，那些行整行被剔、内容在语料里一个字都不剩（回查 cov=0.00）。
    # 覆盖判据用**字符多重集**（与顺序无关，所以跨单元格拆散/列重排都算覆盖）。
    blob = nows("\n".join(" | ".join(c or "" for c in r)
                          for t in tabs for r in _cell_rows(t))).replace("|", "")
    bc = Counter(blob)                        # 每页单元只建一次，别放进行循环
    items = []
    for L in lines:
        cy = (L["y0"] + L["y1"]) / 2
        # 横向与表重叠才可能是这张表的东西（表外的正文页眉页脚一般也重叠，见下面覆盖率把关）
        x_over = any(L["x1"] > bb[0] + 2 and L["x0"] < bb[2] - 2 for bb in boxes)
        inside = any((bb[1] - 3) <= cy <= (bb[3] + 3)
                     and L["x1"] > bb[0] + 2 and L["x0"] < bb[2] - 2 for bb in boxes)
        if inside:
            stats["line_in_table"] += 1
        if x_over:
            t = nows(L["text"]).replace("|", "")
            cov = 0.0
            if len(t) >= 6:                    # 太短的（页码/孤立符号）不参与判据
                cov = sum((Counter(t) & bc).values()) / len(t)
                if cov >= COV_MIN:
                    # 内容确实已经在保留下来的表里 → 剔掉去重。**框内框外一视同仁**：
                    # 表被折行的单元格碎片常落在框外（实测距框 0–200pt 都有），
                    # 只按框剔的话它们会和表里的同一行并排出现，块读起来是乱的。
                    # 安全性：cov>=0.95 保证内容不丢，最坏是少一行重复。
                    stats["line_dup_dropped"] += 1
                    continue
            if inside:
                # 落在框里、内容却不在表里 → 保留。删了就真丢（实测嘉友 p4 释义表抓漏的行）
                stats["line_kept_saved"] += 1
                if len(stats["drop_ex"]) < 5:
                    stats["drop_ex"].append("p%s cov=%.2f %s"
                                            % (u["phys"], cov, L["text"][:60]))
        items.append((L["y"], 0, L))
    for k, t in enumerate(tabs, 1):
        bb = t.get("bbox") if isinstance(t.get("bbox"), list) else [0, 0, 0, 0]
        items.append((bb[1] if len(bb) == 4 else 0, 1, (k, t)))
    items.sort(key=lambda it: (it[0], it[1]))

    stats["tables"] += len(tabs)
    # 同页双节：本页印着两个节号时，标题行以上的行改贴**上一节**的标签。
    # 只改 section 字符串，不动行、不动 y、不跨页——sp 为 None 时全部沿用 u["section"]。
    sp = sec_split(u, lines, sec_ctx)
    if sp is not None:
        stats["sec_split"] += 1
        stats.setdefault("sec_split_ex", []).append(
            "p%s %s 以上 → %s" % (u["phys"], sp[1], u["section"]))
    pieces = []
    for _, kind, obj in items:
        if kind == 0:
            sec = sp[1] if (sp is not None
                            and (obj["y0"] + obj["y1"]) / 2 < sp[0]) else u["section"]
            for seg in split_long_line(obj["text"], size):
                pieces.append(dict(kind="line", text=seg,
                                   **dict(base, section=sec)))
        else:
            k, t = obj
            bb = t.get("bbox") if isinstance(t.get("bbox"), list) else None
            sec = sp[1] if (sp is not None and bb is not None and len(bb) == 4
                            and bb[1] < sp[0]) else u["section"]
            for seg in table_pieces(t, k, size):
                pieces.append(dict(kind="table", text=seg,
                                   **dict(base, section=sec)))
    return pieces


# ------------------------------------------------------------------ 隔页归节
def fix_dividers(units, img_ratio):
    """纯图隔页归给**下一节**（阶段一遗留问题②）。

    阶段一的错法：申通 8 张整页出血的分节隔页被贴成上一节的标签
    （物理 p7 → 第一节，实际是第二节的隔页；p14 → 第二节，实际第三节……8/8 差一节）。

    规则（**已在本机对真实数据跑过验算**，见阶段一 probe_div.py）：
      纯图页（去空白字数 < 40）且 printed 不为 None
      且 (printed + 1) == 某节的起始印页  →  该页归给那一节。
    第二道护栏：该页的图片面积占比 >= 0.9（整页出血图）。

    实测命中：**申通正好 8 个**（p2 从 None 补成第一节、p7/p14/p62/p87/p101/p111
    各前移一节、p116 从第七节改到第八节）；嘉友/外运/国货航命中 0 个。
    其余 9 家在阶段一没做逐页验算，所以这道护栏留着——只有"字数极少 + 页码
    正好差一 + 整页是图"三条同时成立才改，误伤正文页的可能被压到最低。
    """
    ordered = sorted(units, key=lambda u: (u["phys"], HALF_ORDER.get(u["half"], 9)))
    starts = {}                      # 节名 -> 该节起始印页（读序上第一次出现）
    for u in ordered:
        s = u.get("section")
        if s and s not in starts and u.get("printed"):
            starts[s] = u["printed"]
    by_start = {}
    for s, pg_ in starts.items():
        by_start.setdefault(pg_, s)
    hits = []
    for u in ordered:
        if u["chars"] >= DIVIDER_MAX_CHARS or not u.get("printed"):
            continue
        tgt = by_start.get(u["printed"] + 1)
        if not tgt or tgt == u.get("section"):
            continue
        ratio = img_ratio.get((u["company"], u["phys"], u["half"]))
        if ratio is None or ratio < DIVIDER_MIN_IMG:
            hits.append((u, tgt, False, ratio))
            continue
        hits.append((u, tgt, True, ratio))
        u["section_before"] = u.get("section")
        u["section"] = tgt
    return hits


# ------------------------------------------------------------------ 切块
def sent_cut(ps, size, overlap):
    """在当前块里挑「句末收尾」点：返回 j（截到 ps[j] 为止），找不到返回 None。

    候选必须同时满足：
      * ps[j] 以句末字符结尾（这就是"句末收尾"）
      * 截断点之后回带的尾巴 <= 2*overlap（重叠别失控）
      * 截断点之前至少留 size*0.5 的正文（否则这个块太短）
    多个候选时取**尾巴最接近 overlap 的那个**。

    ★ 这一条是踩过坑改的：最初的写法"从尾部倒着找第一个句末点"，第一个命中的
      往往是块尾那一行本身（尾巴 = 0 字），于是 rest 为空、相邻两块**零重叠**。
      实测申通 0002/0003 两块首尾严丝合缝对不上，就是这个问题。
      改成按 |尾巴 - overlap| 取最近，才会真的把上一块的最后 ~120 字带到下一块。
    """
    tot = sum(nw(p["text"]) for p in ps)
    best = None
    for j in range(len(ps)):
        tail = sum(nw(p["text"]) for p in ps[j + 1:])
        if tot - tail < size * 0.5:
            break                      # 再往后 j 更大、head 只会更长，不会回头
        if tail > 2 * overlap:
            continue
        t = ps[j]["text"].rstrip()
        if t and t[-1] in SENT_END:
            d = abs(tail - overlap)
            if best is None or d < best[0]:
                best = (d, j)
    return best[1] if best else None


def carry_from(ps, size, overlap):
    """没有句末点时的回带：只带**文字行**（表不参与重叠，否则整表重复），
    且回带量不超过 size//2、必须真的比原块短。"""
    tail, acc = [], 0
    for p in reversed(ps):
        if p["kind"] != "line":
            break
        tail.insert(0, p)
        acc += nw(p["text"])
        if acc >= overlap:
            break
    if acc > size // 2 or len(tail) >= len(ps) or not tail:
        return []
    return tail


def chunk_pieces(pieces, size=SIZE, overlap=OVERLAP):
    """行累积切块：800 字/块、120 字重叠、句末收尾、不跨节。"""
    chunks, cur, curlen = [], [], 0

    def total(ps):
        return sum(nw(p["text"]) for p in ps)

    i, n = 0, len(pieces)
    while i < n:
        p = pieces[i]
        L = nw(p["text"])
        if cur and p["section"] != cur[-1]["section"]:
            # 节边界强制收尾，且**不回带**（回带会把上一节的行拖进下一节的块）
            chunks.append(cur)
            cur, curlen = [], 0
        if cur and curlen + L > size:
            j = sent_cut(cur, size, overlap)
            if j is not None:
                chunks.append(cur[:j + 1])
                rest = cur[j + 1:]
            else:
                chunks.append(cur)
                rest = carry_from(cur, size, overlap)
            cur, curlen = list(rest), total(rest)
            # 防死循环兜底：理论上两处都保证 rest <= size//2，走不到这里；
            # 真走到了就丢掉回带强行前进，绝不空转。
            if curlen >= size:
                cur, curlen = [], 0
            continue
        cur.append(p)
        curlen += L
        i += 1
    if cur:
        chunks.append(cur)
    return chunks


def merge_small(chunks, min_chars=SMALL_MIN):
    """把 <min_chars 的碎块并进**同一节**的邻块（先试前一块，再试后一块）。

    为什么要有这一趟：节边界强制收尾 + 大表原子化，会在边界上留下"只剩页码/页眉/
    表标题"的残块。实测第一轮 4730 块里有 253 块 <200 字（5.3%，最长的那份计划里
    我给自己定的线是 ≤2%），最短的只有 3 个字——「188」，就是国货航 p188 的页码。
    这类碎块内容不假（确实是正文里的字），但对检索是纯噪声：一个「188」被当成
    "块"召回，既占 top-k 又给不出任何答案。

    并块的三条自我约束：
      - **不跨节**（`块不跨节` 是切块的硬约束，并块不能把它破坏掉）；
      - **不丢内容**（并进邻块，不是删）；
      - **不让邻块越过 `MERGE_CAP`**——否则一串碎块会全并进同一块，800 的块能滚成
        两三千字，正好撞破"0 块 >2000 字"这条硬线。越不过去的碎块就**不并**，
        计数如实写进 `chunks_stats.md`（宁可留几个短块，也不制造超长块）。

    左右两边都能并时，并进**字少的那一边**（少制造溢出）。
    返回 (新块列表, 并掉的碎块数, 并完仍 <min_chars 的块数)。
    """
    def tot(c):
        return sum(nw(p["text"]) for p in c)

    def sec(c):
        return next((p["section"] for p in c if p["section"]), None)

    out = list(chunks)
    merged = 0
    i = 0
    while i < len(out):
        if tot(out[i]) >= min_chars:
            i += 1
            continue
        s = sec(out[i])
        t = tot(out[i])
        prev_ok = i > 0 and sec(out[i - 1]) == s and tot(out[i - 1]) + t <= MERGE_CAP
        next_ok = (i + 1 < len(out) and sec(out[i + 1]) == s
                   and tot(out[i + 1]) + t <= MERGE_CAP)
        if prev_ok and next_ok:                # 两边都行 → 并进字少的
            prev_ok = tot(out[i - 1]) <= tot(out[i + 1])
            next_ok = not prev_ok
        if prev_ok:
            out[i - 1] = out[i - 1] + out[i]   # 碎块接在前一块后面，读序不变
            del out[i]
            merged += 1
            continue                           # i 处换成了原后一块，再判它一次
        if next_ok:
            out[i + 1] = out[i] + out[i + 1]   # 碎块并到后一块前面，读序也不变
            del out[i]
            merged += 1
            continue                           # 新 out[i] 是合并后的块，可能还短
        i += 1
    left = sum(1 for c in out if tot(c) < min_chars)
    return out, merged, left


# ------------------------------------------------------------------ 输出行
def make_rows(name, code, exchange, chunks):
    rows = []
    for k, ps in enumerate(chunks, 1):
        sec = next((p["section"] for p in ps if p["section"]), None)
        prefix = "【%s｜%s】" % (name, sec or "（无章节：封面/前附页/封底）")
        pr = [p["printed"] for p in ps if p["printed"] is not None]
        phys = [p["phys"] for p in ps]
        rows.append({
            "chunk_id": "%s-%04d" % (code or "000000", k),
            "company": name, "code": code, "exchange": exchange,
            "section": sec,
            "printed_lo": pr[0] if pr else None,      # 读序上的首/末一个印页
            "printed_hi": pr[-1] if pr else None,
            "phys_lo": min(phys), "phys_hi": max(phys),
            "halves": sorted({p["half"] for p in ps}, key=lambda h: HALF_ORDER.get(h, 9)),
            "n_chars": sum(nw(p["text"]) for p in ps),
            "n_tables": sum(1 for p in ps if p["kind"] == "table"),
            "has_table": any(p["kind"] == "table" for p in ps),
            "n_pieces": len(ps),
            "prefix": prefix,
            "page_note": None if len(pr) == len(ps) else "含无印页单元（封面/前附页/封底）",
            "text": prefix + "\n" + "\n".join(p["text"] for p in ps),
        })
    return rows


# ------------------------------------------------------------------ 主流程
def do_company(name, comp, patches, img_ratio, want_tables=True):
    t0 = time.time()
    res = E.extract_company(name, want_tables=want_tables, verbose=False)
    units = res["units"]
    for u in units:
        u["company"] = name

    stats = {"tables": 0, "patched": 0, "recut": 0, "line_in_table": 0,
             "line_dup_dropped": 0, "line_kept_saved": 0, "drop_ex": [],
             "sec_split": 0}
    # 同页双节要用的两张表（只读 res，不碰取字/取表逻辑）：
    #   start: 节全名 -> 该节起始**物理页**（toc-title/toc-page-only 档是 found_phys，
    #          heading 档是 phys）；只有这一项对得上才允许按行切。
    #   prev : 节全名 -> 目录里紧挨着的**上一节**全名（这一节被吃掉时，上半页归它）。
    sec_ctx = None
    sec_rows_ = res.get("section_rows") or []
    if sec_rows_:
        fulls, start = [], {}
        for r in sec_rows_:
            if "title" in r:
                f, p = "%s %s" % (r["sec"], r["title"]), r.get("found_phys")
            else:
                f, p = r.get("sec"), r.get("phys")
            fulls.append(f)
            if f and p:
                start.setdefault(f, p)
        if start:
            sec_ctx = {"start": start,
                       "prev": {f: (fulls[i - 1] if i else None)
                                for i, f in enumerate(fulls)}}
    div = fix_dividers(units, img_ratio)
    units.sort(key=lambda u: (u["phys"], HALF_ORDER.get(u["half"], 9)))

    doc = fitz.open(E.pdf_path(name))
    pieces = []
    for u in units:
        pg = doc[u["phys"] - 1]
        tb = u.get("tables") if want_tables else None
        pieces.extend(unit_pieces(pg, u, tb, patches.get((name, u["phys"])),
                                  SIZE, stats, sec_ctx=sec_ctx))
    doc.close()

    chunks = chunk_pieces(pieces)
    n_before = len(chunks)
    n_small_before = sum(1 for c in chunks
                         if sum(nw(p["text"]) for p in c) < SMALL_MIN)
    chunks, n_merged, n_left = merge_small(chunks)
    stats["n_before"] = n_before
    stats["n_small_before"] = n_small_before
    stats["n_merged"] = n_merged
    stats["n_small_left"] = n_left
    rows = make_rows(name, res["code"], res["exchange"], chunks)

    print("\n【%s】%d 页 / %d 页单元 → %d 块   %.1fs"
          % (name, res["pages"], len(units), len(rows), time.time() - t0))
    print("   碎块并块：%d 块 → %d 块 | <%d 字的碎块 %d 个并掉 %d 个，"
          "余 %d 个（并下去会越过 %d 字上限，宁可留短块）"
          % (n_before, len(rows), SMALL_MIN, n_small_before, n_merged,
             n_left, MERGE_CAP))
    print("   章节来源=%s   表 %d 张（生产规则）   落在表框里的文字行 %d"
          % (E.SEC_NAME.get(res["sec_method"], res["sec_method"]),
             stats["tables"], stats["line_in_table"]))
    print("   重复碎片剔掉 %d 行（内容已在表里，框内框外一视同仁）；"
          "**框内但表没覆盖住、保留 %d 行**（保留的宁可重复也不丢）"
          % (stats["line_dup_dropped"], stats["line_kept_saved"]))
    for ex in stats["drop_ex"]:
        print("     ★保留 %s" % ex)
    if stats["patched"]:
        print("   乱码页照图改正：%d 页" % stats["patched"])
    if stats["recut"]:
        print("   表内容兜底回抽：%d 页（缓存里只剩形状）" % stats["recut"])
    if stats["sec_split"]:
        print("   同页双节按行切：命中 %d 页（上半页改贴上一节标签）"
              % stats["sec_split"])
        for ex in stats.get("sec_split_ex", []):
            print("     %s" % ex)
    if div:
        print("   隔页归节：命中 %d 个" % len(div))
        for u, tgt, ok, ratio in div:
            print("     p%-4s %-4s 印%-5s 字数%-4s 图占比%-5s %s → %s"
                  % (u["phys"], u["half"], u.get("printed"), u["chars"],
                     "%.2f" % ratio if ratio is not None else "无",
                     "改" if ok else "**护栏拦下未改**",
                     tgt if ok else (u.get("section") or "无")))
    sizes = sorted(r["n_chars"] for r in rows)
    if sizes:
        print("   块长 去空白：min %d  p50 %d  max %d   >900 字 %d 块"
              % (sizes[0], sizes[len(sizes) // 2], sizes[-1],
                 sum(1 for s in sizes if s > 900)))
    return rows, res, stats, div


def write_stats(all_rows, all_res):
    L = []
    L.append("# 阶段二 切块统计（13 家 / 去空白口径）\n")
    L.append("| 公司 | 页 | 页单元 | 块 | 含表块 | 含表块% | 表数 | 章节来源 | 印页范围 | 表框内行 剔/救 | 碎块 并前→并后 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tot = dict(pages=0, units=0, chunks=0, tbl=0, ntab=0, drop=0, li=0, saved=0,
               sb=0, sm=0, sl=0, nb=0)
    for name, (rows, res, stats, div) in all_rows.items():
        s = sorted(r["n_chars"] for r in rows)
        pr = [r["printed_lo"] for r in rows if r["printed_lo"] is not None]
        ph = [r["printed_hi"] for r in rows if r["printed_hi"] is not None]
        nt = sum(r["n_tables"] for r in rows)
        L.append("| %s | %d | %d | %d | %d | %.0f%% | %d | %s | %s–%s | %d / %d | %d→%d |"
                 % (name, res["pages"], len(res["units"]), len(rows),
                    sum(1 for r in rows if r["has_table"]),
                    100.0 * sum(1 for r in rows if r["has_table"]) / max(len(rows), 1),
                    nt, E.SEC_NAME.get(res["sec_method"], "?"),
                    min(pr) if pr else "-", max(ph) if ph else "-",
                    stats.get("line_dup_dropped", 0), stats.get("line_kept_saved", 0),
                    stats.get("n_small_before", 0), stats.get("n_small_left", 0)))
        tot["pages"] += res["pages"]
        tot["units"] += len(res["units"])
        tot["chunks"] += len(rows)
        tot["tbl"] += sum(1 for r in rows if r["has_table"])
        tot["ntab"] += nt
        tot["drop"] += stats.get("line_dup_dropped", 0)
        tot["saved"] += stats.get("line_kept_saved", 0)
        tot["li"] += stats.get("line_in_table", 0)
        for k in ("sb", "sm", "sl", "nb"):
            tot[k] += stats.get("n_" + {"sb": "small_before", "sm": "merged",
                                        "sl": "small_left", "nb": "before"}[k], 0)
    L.append("| **合计** | **%d** | **%d** | **%d** | **%d** | **%.0f%%** | **%d** | | | **%d / %d** | **%d→%d** |"
             % (tot["pages"], tot["units"], tot["chunks"], tot["tbl"],
                100.0 * tot["tbl"] / max(tot["chunks"], 1), tot["ntab"],
                tot["drop"], tot["saved"], tot["sb"], tot["sl"]))

    allc = [r for rows, _, _, _ in all_rows.values() for r in rows]
    sizes = sorted(r["n_chars"] for r in allc)
    n = len(sizes)

    def pct(q):
        return sizes[min(n - 1, int(q * n))]

    L.append("\n## 块长分布（去空白字数，不含前缀）\n")
    L.append("min %d / p10 %d / p25 %d / p50 %d / p75 %d / p90 %d / p99 %d / max %d（共 %d 块）"
             % (sizes[0], pct(.10), pct(.25), pct(.50), pct(.75), pct(.90), pct(.99), sizes[-1], n))
    L.append("")
    L.append("| 区间 | 块数 | 占比 |")
    L.append("|---|---|---|")
    for lo, hi in ((0, 200), (200, 400), (400, 600), (600, 800), (800, 900),
                   (900, 1200), (1200, 2000), (2000, 10 ** 9)):
        c = sum(1 for s in sizes if lo <= s < hi)
        L.append("| %s | %d | %.1f%% |"
                 % ("<200" if hi == 200 else (">=2000" if lo == 2000 else "%d–%d" % (lo, hi)),
                    c, 100.0 * c / n))
    L.append("")
    L.append("**自检**：>2000 字的块 %d 个（要求 0）；<200 字的块 %d 个（%.1f%%，要求 ≤2%%）"
             % (sum(1 for s in sizes if s > 2000), sum(1 for s in sizes if s < 200),
                100.0 * sum(1 for s in sizes if s < 200) / n))

    L.append("\n## 每节块数\n")
    for name, (rows, res, stats, div) in all_rows.items():
        bysec = {}
        for r in rows:
            bysec[r["section"] or "（无章节）"] = bysec.get(r["section"] or "（无章节）", 0) + 1
        L.append("- **%s**（%d 块）：%s" % (name, len(rows),
                 "、".join("%s %d" % kv for kv in sorted(bysec.items()))))
    L.append("\n## 如实标注\n")
    L.append("- 块长是**切完之后**的实测值，不是目标值；表是原子的，"
             "一张 800 字以内的整表不会被拆开，所以块长会围着 800 上下摆。")
    # -- 超长原子表：单块就一张表、且超过 800 字，拆不动的那种
    over = [r for r in allc if r["has_table"] and r["n_pieces"] == 1
            and r["n_chars"] > 800]
    over.sort(key=lambda r: -r["n_chars"])
    L.append("- **超过 800 字的「整块一张表」共 %d 块**（占 %.1f%%）：这类是 1–2 行的表"
             "（按行拆只会拆出没用的碎片）或单行本身就超 800 字，代码里**故意不拆**。"
             "最大的一个是 `%s` %d 字。它们都 <2000 字，没有被静默丢弃。"
             % (len(over), 100.0 * len(over) / max(n, 1),
                over[0]["chunk_id"] if over else "-",
                over[0]["n_chars"] if over else 0))
    L.append("- 表框里的文字行共 %d 行：其中**内容被表覆盖、剔掉去重 %d 行**，"
             "**表没覆盖住的 %d 行全部保留**（宁可和表重复，也不静默丢字）。"
             "判据是**内容**不是表框：**字符多重集覆盖率 ≥ %.2f**"
             "（与顺序无关，跨单元格拆散、列重排都算覆盖）才剔，**框内框外一视同仁**。"
             "这条规则是实测逼出来的，两个方向都被咬过："
             "① 只按 bbox 剔 → 申通 p12 的表脚脚注、嘉友 p4 释义表抓漏的行被整行剔掉，"
             "覆盖率 0.00，内容在语料里一个字不剩；"
             "② 只在框内剔 → 表被折行的单元格碎片落在框外（实测距框 0–200pt 都有），"
             "会和表里的同一行并排出现，块读起来是乱的。"
             % (tot["li"], tot["drop"], tot["saved"], COV_MIN))
    # -- 碎块并块：这一趟是**修自己的不合格**，所以把并前并后都写出来
    sb, sm, sl, nb = tot["sb"], tot["sm"], tot["sl"], tot["nb"]
    # 第一轮不达标的那个比例两次都要用，且**不能写成字面量**——
    # 语料一改这个数就漂（我已经见过 5.3%→5.4%），写死就会出现报告自相矛盾
    first = 100.0 * sb / max(nb, 1)
    L.append("- **碎块并块**：切完第一轮有 %d 块 <%d 字（占 %.1f%%），"
             "**这条线我自己定的是 ≤2%%，第一轮 %.1f%% 没达标，所以加了这一趟来修**。"
             "规则：<这个字数的块并进**同一节**的邻块（两边都能并时并进字少的那边），"
             "**不跨节、不丢字**；并下去会让邻块超过 %d 字的就不并——"
             "宁可留几个短块，也不制造超长块去撞破「0 块 >2000 字」这条硬线。"
             "结果：块数 %d → %d，并掉 %d 个，**并完仍 <200 字的 %d 个（%.2f%%）**。"
             % (sb, SMALL_MIN, first, first, MERGE_CAP, nb, n,
                sm, sl, 100.0 * sl / max(n, 1)))
    if sl:
        # 「为什么还是短的」按数据算，不猜。实测这 5 块**全部是本节唯一的一块**
        # （顺丰/国货航/东航物流是第七节 债券相关情况整节只有「□ 适用 √ 不适用」，
        #  韵达/申通是封面），同节压根没有邻块可并，
        # 所以 MERGE_CAP 那条上限从来没被碰到——旧稿写成「邻块超 1200 字」是错的。
        nsec = Counter((r["company"], r["section"]) for r in allc)
        small = [r for r in allc if r["n_chars"] < SMALL_MIN]
        solo = sum(1 for r in small if nsec[(r["company"], r["section"])] == 1)
        L.append("- 并完仍短的这 %d 块，**实测原因是「本节只有这一块」**：%d/%d 块所在节"
                 "总共就这一个块，**同节没有任何邻块可并**，所以跟 %d 字那条上限无关"
                 "（上限一次也没被碰到）。它们的正文是完整的，只是短——短不等于丢。"
                 "逐块清单：%s。"
                 % (sl, solo, sl, MERGE_CAP,
                    "；".join("%s %s %d字" % (r["chunk_id"], r["company"], r["n_chars"])
                              for r in small)))
    L.append("- 无印页单元（封面/前附页/封底）的 `printed_*` 记 `null` 并带 `page_note`，"
             "**没有编造页码**。")
    L.append("- 章节标签的可信度分三档（见 `out/stage1_report.md`）："
             "申通是「采信目录页码」档，正文里不印节号，**没有逐页实测**；"
             "本阶段的隔页归节只修正了它的 8 张整页出血隔页。")
    L.append("- 表内容取自阶段一定稿的三策略（`stream` 优先 / `lines` 兜底 / `text` 不进生产）。"
             "`stream` 对单行表、密排表有已知抓不到的边界（阶段一 §8），"
             "所以本阶段**不宣称「表全对」**，只宣称「人工抽查一致 + 剔除行已回查」。")
    path = os.path.join(OUT, "chunks_stats.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n已写 %s" % path)


def write_sample(all_rows, seed=20260926, nrand=5, ntarget=5):
    """chunks_sample.md —— 抽给你人工读的块。

    随机 5 块 + 定向 5 块（定向的理由逐块写出来，不假装是随机）。
    定向是为了让「表是不是按行列排的」「页码对不对」这两件事一定被看到，
    外加一条**专挑刚修好的那一节**（顺丰第九节），让修没修好肉眼可查。
    """
    allc = [r for rows, _, _, _ in all_rows.values() for r in rows]
    rnd = random.Random(seed)
    picked, seen = [], set()

    def add(r, why):
        if r["chunk_id"] in seen:
            return False
        seen.add(r["chunk_id"])
        picked.append((r, why))
        return True

    # 定向：外运的大表页 / 国货航拼版的左右半页 / 顺丰两张改正页 / 无印页块
    targets = [
        (lambda r: r["company"] == "中国外运" and r["has_table"]
         and any(s in r["text"] for s in ("营业收入", "现金流量", "资产负债表")),
         "定向：外运的含表块（验表格行列）"),
        (lambda r: r["company"] == "国货航" and len(r["halves"]) == 2,
         "定向：国货航双页拼版，块内含左右两半（验左右不串）"),
        (lambda r: r["company"] == "顺丰控股" and r["phys_lo"] <= 2,
         "定向：顺丰 p2 乱码页照图改正后"),
        (lambda r: r["page_note"] is not None,
         "定向：含无印页单元（验 printed 记 null 不编造）"),
        (lambda r: r["company"] == "顺丰控股" and r["has_table"]
         and (r["section"] or "").startswith("第九节"),
         "定向：顺丰 **第九节 财务报告**——**这一节原来整节被贴成「第八节 债券相关情况」**，"
         "修好后它的印页应从 **148** 起；顺带验财务报表的行列"),
    ]
    for pred, why in targets:
        for r in allc:
            if pred(r):
                add(r, why)
                break
    while len(picked) < nrand + ntarget:
        r = rnd.choice(allc)
        if not add(r, "随机"):
            if len(seen) >= len(allc):
                break
    L = ["# 阶段二 切块抽样（人工读这一份）\n",
         "共 %d 块，这里贴 %d 块：**随机 %d 块 + 定向 %d 块**。"
         % (len(allc), len(picked), nrand, ntarget),
         "随机用固定种子 %d，重跑同一份；定向的每一块都写了理由。\n" % seed,
         "读的时候重点看两件事：**① 表是不是按行列排的 ② 页码区间对不对**。\n",
         "---\n"]
    for k, (r, why) in enumerate(picked, 1):
        L.append("## 样本 %d / %s  （%s）\n" % (k, r["chunk_id"], why))
        L.append("| 字段 | 值 |")
        L.append("|---|---|")
        for key in ("company", "code", "exchange", "section", "printed_lo", "printed_hi",
                    "phys_lo", "phys_hi", "halves", "n_chars", "n_tables", "n_pieces",
                    "page_note"):
            L.append("| %s | %s |" % (key, r[key]))
        L.append("\n```text")
        L.append(r["text"])
        L.append("```\n---\n")
    path = os.path.join(OUT, "chunks_sample.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("已写 %s（%d 块）" % (path, len(picked)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="公司名；留空配 --all")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--sample-only", action="store_true")
    a = ap.parse_args()

    comp = E.load_companies()
    names = a.names or (list(comp) if a.all else [])
    if not names:
        ap.error("要么给公司名，要么 --all")
    for nm in names:
        if nm not in comp:
            ap.error("companies.csv 里没有 %s" % nm)

    if a.sample_only:
        rows = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
        by = {}
        for r in rows:
            by.setdefault(r["company"], []).append(r)
        all_rows = {k: (v, {"pages": 0, "units": [], "sec_method": "?"}, {}, [])
                    for k, v in by.items()}
        write_sample(all_rows)
        return

    patches = load_patches()
    img_ratio = load_img_ratio()
    print("=" * 78)
    print("阶段二 切块：%d 家   取表=%s   文字补丁=%d 页   图片占比护栏=%d 条"
          % (len(names), "关" if a.no_tables else "开", len(patches), len(img_ratio)))
    print("目标：%d 字/块、%d 字重叠、句末收尾、块不跨节" % (SIZE, OVERLAP))
    print("=" * 78)

    all_rows, out, t0 = {}, [], time.time()
    for name in names:
        rows, res, stats, div = do_company(name, comp[name], patches, img_ratio,
                                          want_tables=not a.no_tables)
        all_rows[name] = (rows, res, stats, div)
        out.extend(rows)

    with open(CHUNKS, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n已写 %s（%d 块，%.1f 万字，耗时 %.1fs）"
          % (CHUNKS, len(out), sum(r["n_chars"] for r in out) / 10000.0,
             time.time() - t0))
    write_stats(all_rows, None)
    write_sample(all_rows)


if __name__ == "__main__":
    main()
