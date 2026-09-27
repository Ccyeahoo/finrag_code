# -*- coding: utf-8 -*-
"""财报提取核心：文字（按词+坐标）/ 表格（两种策略）/ 双页拼版切分 / 页码对齐 / 目录

设计要点（每条都对应阶段一里实测出来的坑）：
  * 文字用 get_text("words") 拿坐标，而不是 get_text("blocks")——
    国货航双页版面上实测有 93 个 block 横跨中线（PyMuPDF 把左右两页同高度的
    文字粘成了一个 block），按 block 切必错，按 word 的 x 中心切才对。
  * 表格同时跑 strategy="lines" 和 strategy="text" 并对比——
    实测嘉友 p42 有框线的表：lines 给 10行×12列（干净），text 给 68行×14列
    （把标题也吞进去了）。所以 lines 优先，text 只在 lines 空的时候补。
  * 双页判据是「页宽 > 1000 且页眉有两个相差 1 的整数」，不是只看宽度——
    申通 p60 页宽同为 1191，但页眉只有一个 059，是 A3 横排宽表页，不是拼版。
  * 页脚页码要用词的下边缘 y1 判断，不是上边缘 y0——实测中储 p45 的页码
    y0=762、距底 67pt，用 y0 会被 72pt 的门槛漏掉（我就漏过一次）。

用法：
  python extract.py --file 嘉友国际 --pages 1-60
  python extract.py --file 国货航 --pages 20-20 --no-tables
  python extract.py --list
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time

if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pymupdf as fitz

# 项目根从本文件位置推（src/ 的上一级），不写死绝对路径——
# 写死的话别人把仓库 clone 到别的目录就跑不起来。chunk.py 直接继承这个 ROOT。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDFDIR = os.path.join(ROOT, "data", "pdf")
OUTDIR = os.path.join(ROOT, "out")
CACHEDIR = os.path.join(OUTDIR, "cache")

HEADER_BAND = 100.0     # 页眉取多高（pt）
FOOTER_BAND = 100.0     # 页脚取多高（pt）
WIDE_PT = 1000.0        # 判定"宽页"的宽度门槛
SPACE_GAP = 1.5         # 词间水平间隙超过这个值就补一个空格
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百]+节$")
SEC_NAME = {"toc-title": "目录标题回正文找（有逐节实测）",
            "heading": "扫正文「第X节」（有实测，无目录页时用）",
            "toc-page-only": "★采信目录页码（正文不印标题，未逐页实测）",
            "none": "没有定位到章节"}


# ----------------------------------------------------------------- 公司清单
def load_companies():
    with open(os.path.join(ROOT, "data", "companies.csv"), encoding="utf-8-sig") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def pdf_path(name):
    return os.path.join(PDFDIR, name + ".pdf")


# ----------------------------------------------------------------- 文字重组
def words_to_text(words):
    """把词按坐标重排成文本。换行看 y，词间空格看水平间隙。"""
    if not words:
        return ""
    ws = sorted(words, key=lambda t: (t[1], t[0]))
    hs = [t[3] - t[1] for t in ws]
    tol = max(2.5, 0.45 * (sorted(hs)[len(hs) // 2]))
    lines, cur = [], [ws[0]]
    for t in ws[1:]:
        if abs(t[1] - cur[-1][1]) <= tol:
            cur.append(t)
        else:
            lines.append(cur)
            cur = [t]
    lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda t: t[0])
        buf = ln[0][4]
        for prev, t in zip(ln, ln[1:]):
            buf += (" " if t[0] - prev[2] > SPACE_GAP else "") + t[4]
        out.append(buf)
    return "\n".join(out)


def words_in(pg, clip):
    """取 clip 内（按词中心点判定）的词"""
    ws = pg.get_text("words")
    x0, y0, x1, y1 = clip
    return [t for t in ws if x0 <= (t[0] + t[2]) / 2 < x1 and y0 <= (t[1] + t[3]) / 2 < y1]


# ----------------------------------------------------------------- 页眉页脚
def header_numbers(pg):
    """页眉里的纯数字词 -> [(x0, 值)]"""
    ws = pg.get_text("words")
    return [(t[0], int(t[4])) for t in ws
            if t[3] <= HEADER_BAND and re.fullmatch(r"\d{1,4}", t[4].strip())]


def footer_numbers(pg):
    """页脚里的纯数字词。**用下边缘 y1 判断**，不然会漏（踩过这个坑）。"""
    h = pg.rect.height
    ws = pg.get_text("words")
    return [(t[0], int(t[4])) for t in ws
            if t[1] >= h - FOOTER_BAND and re.fullmatch(r"\d{1,4}", t[4].strip())]


def twoup_pair(pg):
    """**判据**：这一页是不是双页拼版？是则返回 (左页码, 右页码)，否则 None。
    要求页宽 > 1000 且页眉有两个相差 1 的整数分居中线两侧。
    只用来判定整份文档的版式（doc_layout），不用来逐页切——
    因为有的拼版页只有一侧印了页码（国货航 p3 左半是印页 0），这里会漏判。"""
    if pg.rect.width <= WIDE_PT:
        return None
    mid = pg.rect.width / 2
    nums = header_numbers(pg)
    # 排除明显是年份的（2025 之类），只留 1-3 位
    cand = [(x, v) for x, v in nums if v < 1000]
    left = [v for x, v in cand if x < mid]
    right = [v for x, v in cand if x >= mid]
    for a in left:
        for b in right:
            if b - a == 1:
                return (a, b)
    return None


def twoup_halves(pg):
    """**切法**：把一张拼版宽页算成左右两半的印页码。
    与 twoup_pair 的区别：这里只要一侧有页码就能推出另一侧（相邻页差 1），
    所以国货航 p3（右半=1，左半=0 未印）也能切。
    注意必须先由 doc_layout 确认整份文档是 twoup 才可调用——
    申通那种 A3 横排宽表页也是 1191 宽，但页码居中，不能切。"""
    mid = pg.rect.width / 2
    cand = [(x, v) for x, v in header_numbers(pg) if v < 1000]
    ln = next((v for x, v in cand if x < mid), None)
    rn = next((v for x, v in cand if x >= mid), None)
    if ln is None and rn is not None:
        ln = rn - 1
    if rn is None and ln is not None:
        rn = ln + 1
    return (ln, rn)


def doc_layout(doc):
    """整份文档判一次：抽样宽页，八成以上是拼版才算 twoup"""
    wide = [i for i in range(doc.page_count) if doc[i].rect.width > WIDE_PT]
    if len(wide) < 5:
        return "normal", {"wide": len(wide), "twoup_hits": 0}
    sample = wide[:60]
    hits = sum(1 for i in sample if twoup_pair(doc[i]))
    layout = "twoup" if hits >= 0.8 * len(sample) else "normal"
    return layout, {"wide": len(wide), "sampled": len(sample), "twoup_hits": hits}


# ----------------------------------------------------------------- 表格
# 单张表的 cells 序列化后超过这个字符数就只留形状、不留内容（缓存保护）。
# 阶段一原值是 60000，阶段二抬高到 200 万：**作业硬要求就是"表格按行列还原"，
# 60k 的上限会让「所有者权益变动表」这类 A4 横排大表静默丢掉全部单元格文字**，
# 而丢掉时不报错、只在 blob 里留一个 "表太大，未存内容"。
# 实测：13 家全量跑完后 out/cache 里 cells=None 的条目数为 0（见 out/chunks_stats.md），
# 也就是说这个上限被抬高后**一张表的真实内容都没被截断**。
# 保留上限本身是为了防病态页（几千行的排版垃圾）把缓存撑爆，不是留作内容裁剪用。
CELLS_MAX_JSON = 2_000_000


def cap_cells(tbl, limit=None):
    """把一张表抽成 dict（行主序，None→空串）。
    limit=None 表示用 CELLS_MAX_JSON；传 None 之外的值可覆盖（切块阶段回抽兜底用）。"""
    lim = CELLS_MAX_JSON if limit is None else limit
    try:
        rows = tbl.extract()
    except Exception as e:
        return {"rows": tbl.row_count, "cols": tbl.col_count,
                "bbox": [round(v) for v in tbl.bbox], "error": repr(e), "cells": None}
    cells = [[("" if c is None else c) for c in r] for r in rows]
    blob = {"rows": tbl.row_count, "cols": tbl.col_count,
            "bbox": [round(v) for v in tbl.bbox]}
    if lim is not None and len(json.dumps(cells, ensure_ascii=False)) > lim:
        blob["cells"] = None
        blob["note"] = "表太大，未存内容（上限 %d 字符）" % lim
    else:
        blob["cells"] = cells
    return blob


def find_tables_both(pg, clip=None, limit=None):
    """同一页/同一半页跑三种策略，返回 {lines:[...], text:[...], stream:[...]}"""
    out = {}
    for strat in ("lines", "text"):
        try:
            tf = pg.find_tables(clip=clip, strategy=strat)
            out[strat] = [cap_cells(t, limit) for t in tf.tables]
        except Exception as e:
            out[strat] = [{"error": repr(e)}]
    try:
        out["stream"] = table_stream(pg, clip)
    except Exception as e:
        out["stream"] = [{"error": repr(e)}]
    return out


# ------------------------------------------------ 只画横线的表格（外运/国货航）
def h_rules(pg, clip=None, min_len=25.0):
    """取页面上足够长的**横线段** y 坐标（去重）。

    必须逐 items 看，不能看 d["rect"]：一个绘图组里常有好几条线，
    组的 rect 又高又宽，按组筛会把整组线一起丢掉
    （外运 p30 有 11 条横线，按组筛只剩 3 条——这个坑踩过一次）。
    """
    ys = []
    for d in pg.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p1, p2 = it[1], it[2]
                if abs(p2.y - p1.y) <= 1 and abs(p2.x - p1.x) >= min_len:
                    x0, x1 = min(p1.x, p2.x), max(p1.x, p2.x)
                    y = (p1.y + p2.y) / 2
                else:
                    continue
            elif it[0] == "re":
                r = it[1]
                if not (r.height <= 1.5 and r.width >= min_len):
                    continue
                x0, x1, y = r.x0, r.x1, (r.y0 + r.y1) / 2
            else:
                continue
            if clip is not None and (x0 < clip.x0 - 2 or x1 > clip.x1 + 2
                                     or y < clip.y0 - 2 or y > clip.y1 + 2):
                continue
            ys.append(round(y, 1))
    ys.sort()
    merged = []
    for y in ys:
        if not merged or y - merged[-1] > 3:
            merged.append(y)
    return merged


_h_rules = h_rules


def _col_bounds(rows_words, min_gap=7.0):
    """按词的水平空档切列：对所有行的词按 x 区间投票，
    某个 x 处如果**每一行**都是空档，那就是列边界。"""
    if not rows_words:
        return []
    xs = sorted({(round(w[0]), round(w[2])) for r in rows_words for w in r})
    bounds = []
    for i in range(len(xs) - 1):
        gap = xs[i + 1][0] - xs[i][1]
        if gap >= min_gap:
            hit = all(any(w[2] > xs[i][1] and w[0] < xs[i + 1][0] for w in r)
                      for r in rows_words if r)
            if hit:
                bounds.append((xs[i][1] + xs[i + 1][0]) / 2)
    return bounds


def table_stream(pg, clip=None, min_gap=20.0):
    """**第三种策略：纯空白切分（stream 模式）**，专治「没有框线」的财报表格。

    实测中国外运 p30：横线只有整张表的上下两条（242.0 / 386.3），
    行与行之间**一条线都没有**，行标签（营业收入）与数字还在不同的基线上
    （261.3 vs 259.8）。所以既不能靠竖线定列，也不能靠横线定行——
    只能靠空白：表里的列间距 ≥20pt，正文里词间距只有 3~8pt，
    用这个差别先把「表格行」挑出来，再在表格行都空着的位置定列边界。
    """
    words = words_in(pg, clip) if clip is not None else pg.get_text("words")
    if not words:
        return []
    # 1) 按 y 聚成文本行（容差取半个字高，营业收入与它右边的数字能并进同一行）
    lines = []
    for w in sorted(words, key=lambda t: (t[1], t[0])):
        h = w[3] - w[1]
        y = (w[1] + w[3]) / 2
        for L in lines:
            if abs(L["y"] - y) <= max(2.2, 0.5 * h):
                L["ws"].append(w)
                L["y"] = sum((q[1] + q[3]) / 2 for q in L["ws"]) / len(L["ws"])
                break
        else:
            lines.append({"y": y, "ws": [w], "h": h})
    lines.sort(key=lambda L: L["y"])

    # 2) 行内出现 ≥min_gap 的横向空档 → 这是表格行，不是正文
    for L in lines:
        L["ws"].sort(key=lambda t: t[0])
        L["gaps"] = [L["ws"][i + 1][0] - L["ws"][i][2] for i in range(len(L["ws"]) - 1)]
        L["is_row"] = bool(L["gaps"]) and max(L["gaps"]) >= min_gap
    rows_idx = [i for i, L in enumerate(lines) if L["is_row"]]
    if len(rows_idx) < 2:
        return []

    # 3) 列边界 = 绝大多数表格行都没盖到的 x 区段（取区段中点）
    #    阈值不能取「一行都没盖到」：国货航 p20 有单元格内换行的行，
    #    它横跨了本该是列边界的位置，卡死 100% 会让整张表只剩 2 列（实测踩过）。
    rowlines = [lines[i] for i in rows_idx]
    lo = min(w[0] for L in rowlines for w in L["ws"])
    hi = max(w[2] for L in rowlines for w in L["ws"])
    grid = 1.0
    n = int((hi - lo) / grid) + 1
    cover = [0] * n
    for L in rowlines:
        for w in L["ws"]:
            for k in range(max(0, int((w[0] - lo) / grid)),
                           min(n, int((w[2] - lo) / grid) + 1)):
                cover[k] += 1
    nr = len(rowlines)
    bounds, run = [], None
    for k in range(n):
        if cover[k] <= 0.3 * nr:
            run = k if run is None else run
        else:
            if run is not None and (k - run) * grid >= 8:
                bounds.append(lo + (run + k) / 2 * grid)
            run = None
    if run is not None and (n - run) * grid >= 8:
        bounds.append(lo + (run + n) / 2 * grid)
    edges = [lo] + bounds + [hi]
    if len(edges) < 3:
        return []

    # 4) 表格行按列边界切格；连续的行并成一张表（断 2 行以上算新表）
    lh = sorted(L["h"] for L in rowlines)[len(rowlines) // 2] or 10.0
    tables, cur = [], []
    for i in rows_idx:
        L = lines[i]
        cells = [""] * (len(edges) - 1)
        for w in L["ws"]:
            cx = (w[0] + w[2]) / 2
            for j in range(len(edges) - 1):
                if edges[j] <= cx < edges[j + 1]:
                    cells[j] = (cells[j] + " " + w[4]).strip()
                    break
        if len([c for c in cells if c]) < 2:
            continue
        if cur and L["y"] - cur[-1]["y"] > 2.2 * lh:
            tables.append(cur)
            cur = []
        cur.append({"y": L["y"], "cells": cells})
    if cur:
        tables.append(cur)

    out = []
    for tb in tables:
        if len(tb) < 2:
            continue
        out.append({"rows": len(tb), "cols": len(edges) - 1,
                    "bbox": [round(lo), round(tb[0]["y"]), round(hi), round(tb[-1]["y"])],
                    "cells": [t["cells"] for t in tb], "strategy": "stream"})
    return out


# 兼容旧名（早期写的是按横线切，实测行不通，已换成 stream）
table_from_hlines = table_stream


# ----------------------------------------------------------------- 页面单元
def page_units(doc, i, layout):
    """把物理页 i（0-based）拆成 1 或 2 个单元。
    单元 = {phys, printed, half, text, clip, twoup_pair}"""
    pg = doc[i]
    phys = i + 1
    if layout == "twoup" and pg.rect.width > WIDE_PT:
        pair = twoup_halves(pg)
        mid = pg.rect.width / 2
        h = pg.rect.height
        L = fitz.Rect(0, 0, mid, h)
        R = fitz.Rect(mid, 0, pg.rect.width, h)
        # 印页 0 是不存在的（封面内页没印页码），如实留 None，不编一个数
        lp = pair[0] if pair and pair[0] and pair[0] >= 1 else None
        rp = pair[1] if pair and pair[1] and pair[1] >= 1 else None
        return [
            {"phys": phys, "printed": lp, "half": "L", "pair": pair,
             "text": words_to_text(words_in(pg, L)), "clip": L},
            {"phys": phys, "printed": rp, "half": "R", "pair": pair,
             "text": words_to_text(words_in(pg, R)), "clip": R},
        ]
    if layout == "twoup":
        # 拼版文档里的窄页（封面、封底）
        return [{"phys": phys, "printed": None, "half": "full",
                 "text": pg.get_text("text"), "clip": None, "pair": None}]
    return [{"phys": phys, "printed": None, "half": "full",
             "text": pg.get_text("text"), "clip": None, "pair": None}]


def apply_offset(units, offset):
    """给普通版式的单元贴上印页码 = 物理页 - offset"""
    for u in units:
        if u["printed"] is None and u["half"] == "full" and isinstance(offset, int):
            u["printed"] = u["phys"] - offset
    return units


# ----------------------------------------------------------------- 目录
def parse_toc(doc, layout, max_scan=14):
    """在前若干页找目录页，抽出 [(节号, 节名, 印页码)]。

    同一页两种读序各解一遍，取**页码单调递增**的那一份：
      * get_text("text") 的读序在申通那种双栏卡片目录上，会把 8 个页码
        全排到标题前面去（实测），配不上号；
      * 按坐标重排（words_to_text）反而是对的。
    单调递增是硬校验——章节本来就得按顺序排，不递增就是配错了。
    """
    best, best_score = {"toc_phys_page": None, "toc_half": None, "items": []}, -1
    for i in range(min(max_scan, doc.page_count)):
        # 走 page_units 而不是整页文本：拼版页的目录只印在右半页，
        # 用整页合并文本会把左右两半串行（国货航 p3 就是这么漏掉的）。
        for u in page_units(doc, i, layout):
          # 两种读序都给一遍：normal 页 clip 是 None，也要按整页坐标重排一次，
          # 否则申通那种双栏目录配不上页码（get_text 把 8 个页码全排到前面）。
          cands = [u["text"],
                   words_to_text(words_in(doc[i], u["clip"]) if u["clip"]
                                 else doc[i].get_text("words"))]
          for chunk in cands:
            flat = re.sub(r"[.\u2026\s]+", " ", chunk)
            hits = list(re.finditer(r"第([一二三四五六七八九十]+)节", flat))
            if len(hits) < 4:
                continue
            items = []
            for k, m in enumerate(hits):
                seg = flat[m.end(): hits[k + 1].start() if k + 1 < len(hits) else len(flat)]
                seg = seg.strip()
                if not seg:
                    continue
                # 取**第一个**独立数字当页码：兼容 "标题 4" 和 "标题\n4"，
                # 也避免把后面「备查文件目录」那一段整块吞进标题里。
                mm = re.search(r"(?<!\d)(\d{1,4})(?!\d)", seg)
                # 标题 = 按数字切开后第一段非空文字。这样三种版式通吃：
                #   嘉友 "释义 4"              -> 释义
                #   外运 "释义\n4"             -> 释义
                #   申通 "002 重要提示、目录和释义" -> 重要提示、目录和释义（三行一组，页码在标题前）
                parts = [p for p in re.split(r"(?<!\d)\d{1,4}(?!\d)", seg) if p.strip()]
                title = re.sub(r"[\s:：]+", "", parts[0])[:40] if parts else ""
                items.append((m.group(0), title, int(mm.group(1)) if mm else None))
                if len(items) < 4:
                    continue
                score, mono = toc_score(items)
                if score > best_score:
                    best_score = score
                    best = {"toc_phys_page": i + 1, "toc_half": u["half"],
                            "items": items, "monotonic": mono}
    # 选中的这一份，再用**同行几何**把页码复核一遍（见 toc_pages_by_line）。
    # 只认严格更好的结果：几何配出的节数 <= 文本启发式时原样返回，
    # 所以只有「文本启发式整体错位」那种文档会被改到（实测就顺丰一家）。
    if best["toc_phys_page"] and best["items"]:
        pg = doc[best["toc_phys_page"] - 1]
        for u in page_units(doc, best["toc_phys_page"] - 1, layout):
            if u["half"] != best["toc_half"]:
                continue
            wl = (words_in(pg, u["clip"]) if u["clip"] else pg.get_text("words"))
            fix = toc_pages_by_line(wl)
            if fix:
                # 页码与**标题**都取几何的：标题直接从标题词里读出来，
                # 不像文本启发式那样会被目录页尾部文字污染
                # （顺丰第九节实测被解成「财务报告目录」，正文印的是「财务报告」，
                #  于是 sec+title 永远配不上，整节被丢掉）。
                fixed = []
                for s, t, p in best["items"]:
                    gp, gt = fix.get(s, (None, None))
                    fixed.append((s, gt or t, gp if gp is not None else p))
                sc, mo = toc_score(fixed)
                if sc > best_score:
                    nfix = sum(1 for (_, _, a), (_, _, b) in zip(best["items"], fixed)
                               if a != b)
                    best["items"] = fixed
                    best["monotonic"] = mo
                    best["page_fix"] = ("同行几何复核：%d 节页码与文本启发式不同，"
                                        "得分 %d→%d" % (nfix, best_score, sc))
            break
    return best


def toc_score(items):
    """目录候选打分：解出页码的节数 + 单调递增加 10 分。
    页码一个都没解出来的读序不加单调分——empty 列表会假通过。"""
    pages = [t[2] for t in items if t[2]]
    mono = len(pages) >= 4 and all(a < b for a, b in zip(pages, pages[1:]))
    return len(pages) + (10 if mono else 0), mono


def toc_pages_by_line(words):
    """按**同一行**把「第X节」配到它的页码与标题，返回 {节号: (页码, 标题或 None)}。

    标题只在标题词里确实带着标题时给出（顺丰「第一节　重要提示、目录和释义」
    这种一行印完的），只有光秃秃「第一节」的就给 None，让调用方保留文本启发式的标题。

    为什么需要它（实测）：顺丰把页码印成独立一列在标题**左边**
    （004 在 x=346、标题在 x=374，同一行、y 区间重叠 11.8pt、中心差 0.0），
    而上面那份文本启发式取「标题之后第一个数字」——顺丰的目录行里标题之后
    就是**下一节**的标题，于是每一节都偷走了下一节的页码，9 节整体错位一节
    （第一节@6、第二节@17…第九节被挤掉，实测）。正文实测：第一节在印4、
    第二节印6、第三节印17…… 与目录原印的 004/006/017 完全吻合，
    且落在**左/右两侧共 4 家**的坐标配对上无歧义。

    只认同行（y 区间有重叠）且紧邻（在标题左/右侧、不与标题重叠）的纯数字，
    配不上就不返回这一节——**宁可漏配回落到文本启发式，也不乱配**。
    圆通/韵达/外运/申通/东航/中储/中谷/铁龙/嘉友 的页码本来就在标题右边
    或下一行（ov>0 或完全不相干），文本启发式已经配对正确，
    这里配出的节数不比它多，调用方按「严格更好才采纳」处理，等于不动。
    """
    sec_re = re.compile(r"^(第[一二三四五六七八九十]+节)[\s　]*(.*)$")
    num_re = re.compile(r"^\d{1,4}$")
    out = {}
    for h in words:
        m = sec_re.match(h[4].strip())
        if not m:
            continue
        left, right = [], []
        for w in words:
            if not num_re.match(w[4].strip()):
                continue
            if min(h[3], w[3]) - max(h[1], w[1]) <= 0:
                continue                      # y 区间不重叠 = 不在同一行
            if w[2] <= h[0] + 1:
                left.append(w)
            elif w[0] >= h[2] - 1:
                right.append(w)
        pick = None
        if left:
            pick = max(left, key=lambda w: w[2])       # 离标题最近的那个
        elif right:
            pick = min(right, key=lambda w: w[0])
        if pick is not None:
            ttl = re.sub(r"[\s:：]+", "", m.group(2))[:40] or None
            out.setdefault(m.group(1), (int(pick[4].strip()), ttl))
    return out


def find_toc_units(doc, layout, max_scan=16, min_hits=4):
    """找出「目录页」单元：同一页上出现 >=4 个 "第X节" 就认定是目录。
    目录页必须从正文扫描里排掉（申通 p4 实测：目录把 8 个节全吸到自己头上）。
    可能不止一页（申通目录就跨 p4/p5）。"""
    out = set()
    for i in range(min(max_scan, doc.page_count)):
        for u in page_units(doc, i, layout):
            n = len(re.findall(r"第[一二三四五六七八九十]+节", u["text"]))
            if n >= min_hits:
                out.add((u["phys"], u["half"]))
    return out


def locate_sections(doc, layout, toc, offset, look_ahead=12, skip=()):
    """**主用办法**：拿目录里的 (节号, 标题)，回到正文里按标题定位起始页。

    为什么不能靠扫 "第X节"：申通正文里根本不印节号，只有标题
    （p3 正文首行就是「重要提示、目录和释义」，没有"第一节"四个字）；
    国货航连目录页都没有。所以标题定位是唯一通用的办法。
    定位窗口卡在「目录给的印页 ± look_ahead」，找不到就如实记 not found，不乱猜。

    skip：目录页自身的单元。**必须跳过**——申通的目录卡片就是「节号 + 标题 + 页码」，
    把标题读成独立一行，跟正文标题一模一样；不跳的话第二节会定到目录页上。

    返回 (page2sec, rows)；rows 每项含目录页码与实测页码，供人工核对。
    """
    items = [it for it in toc["items"] if it[2] and it[1]]
    if len(items) < 4:
        return {}, []
    order = []
    for i in range(doc.page_count):
        for u in page_units(doc, i, layout):
            if offset is not None:
                u = apply_offset([u], offset)[0]
            order.append(u)
    ok = [(k, u) for k, u in enumerate(order) if (u["phys"], u["half"]) not in set(skip)]

    def norm_lines(u):
        return [re.sub(r"\s+", "", ln) for ln in u["text"].split("\n")]

    # 游标精确到「单元 + 该单元内的行号」：嘉友第七节与第八节**同页起始**
    # （都印 85），只按页推进的话第八节会被跳过。
    rows = []
    cur_n, cur_line = 0, -1          # 游标按 ok 的下标走，避免 skip 造成下标错位
    for sec, title, tpage in items:
        exp = (tpage + offset) if isinstance(offset, int) else None
        lo, hi = cur_n, len(ok) - 1
        if exp is not None:
            lo = max(lo, next((n for n, (_, u) in enumerate(ok)
                               if u["phys"] >= exp - look_ahead and u["half"] in ("L", "full")), 0))
            hi = min(hi, next((n for n, (_, u) in enumerate(ok) if u["phys"] > exp + look_ahead),
                              len(ok) - 1))
        hit = None
        for n in range(lo, hi + 1):
            _, u = ok[n]
            lines = norm_lines(u)
            start = cur_line + 1 if n == cur_n else 0
            for j in range(start, len(lines)):
                # 正文标题有两种印法：单独一行只印标题（申通），
                # 或者「第一节　标题」整行（顺丰，实测标题与节号同行、
                # 中间一个全角空格）。归一化后正好等于 sec + title。
                if lines[j] == title or lines[j] == sec + title:
                    hit = (n, j)
                    break
            if hit:
                break
        if hit is None:
            rows.append({"sec": sec, "title": title, "toc_page": tpage,
                         "found_phys": None, "found_half": None,
                         "found_printed": None, "delta": None})
            continue
        cur_n, cur_line = hit
        k = ok[cur_n][0]
        u = order[k]
        rows.append({"sec": sec, "title": title, "toc_page": tpage,
                     "found_phys": u["phys"], "found_half": u["half"],
                     "found_printed": u["printed"],
                     "delta": (u["printed"] - tpage) if u["printed"] is not None else None})

    # 逐单元贴章节：从各节起始位置往后铺
    page2sec = {}
    cur = None
    # 按 (物理页, 半页) 建索引——双页拼版时左右半页共用同一个物理页号，
    # 只用 phys 做键会互相覆盖。
    starts = {(r["found_phys"], r["found_half"]): r for r in rows if r["found_phys"]}
    for u in order:
        r = starts.get((u["phys"], u["half"]))
        if r and (u["half"] == r["found_half"] or r["found_half"] == "full"):
            cur = "%s %s" % (r["sec"], r["title"])
        page2sec[(u["phys"], u["half"])] = cur
    return page2sec, rows


def sections_from_toc(doc, layout, toc, offset):
    """**第三档（兜底）**：正文里连标题都不印的文档（申通实测），
    直接按目录给的印页往下铺，偏移由目录与正文实测吻合过（申通=1）。

    这一步**没有逐页实测**，是「采信目录」——报告里必须写明，
    不能与「实测吻合」混为一谈。
    返回 (page2sec, rows)，rows 带 src="toc" 标记。
    """
    order = []
    for i in range(doc.page_count):
        for u in page_units(doc, i, layout):
            if offset is not None:
                u = apply_offset([u], offset)[0]
            order.append(u)
    rows = []
    for sec, title, tpage in toc["items"]:
        if not tpage:
            continue
        phys = tpage + offset if isinstance(offset, int) else None
        rows.append({"sec": sec, "title": title, "toc_page": tpage,
                     "found_phys": phys, "found_half": "full",
                     "found_printed": tpage, "delta": 0, "src": "toc"})
    page2sec = {}
    cur = None
    starts = {r["found_phys"]: r for r in rows if r["found_phys"]}
    for u in order:
        r = starts.get(u["phys"])
        if r and u["half"] in ("L", "full"):
            cur = "%s %s" % (r["sec"], r["title"])
        page2sec[(u["phys"], u["half"])] = cur
    return page2sec, rows


def scan_headings(doc, layout, offset=None, skip=()):
    """**交叉校验**：扫正文里单独成行的 "第X节"，得到另一份 页->节 映射。
    只在正文真的印了节号的文档（嘉友、国货航）上有效；申通用它是空的。

    skip：必须把目录页排掉。目录页上列着全部 "第X节" 字样，
    不排掉的话 8 个节会被一起贴到目录页上（申通 p4 实测就是这个错）。
    """
    cur = None
    page2sec = {}
    starts = []
    for i in range(doc.page_count):
        units = page_units(doc, i, layout)
        if offset is not None:
            units = apply_offset(units, offset)
        for u in units:
            if (u["phys"], u["half"]) in skip:
                page2sec[(u["phys"], u["half"])] = cur
                continue
            lines = [re.sub(r"\s+", "", ln) for ln in u["text"].split("\n")]
            lines = [ln for ln in lines if ln]
            for j, ln in enumerate(lines):
                if SECTION_RE.match(ln):
                    # 节名 = 紧随其后的非空行
                    nm = lines[j + 1] if j + 1 < len(lines) else ""
                    if 2 <= len(nm) <= 30 and not SECTION_RE.match(nm):
                        cur = "%s %s" % (ln, nm)
                        starts.append({"phys": u["phys"], "half": u["half"],
                                       "printed": u["printed"], "sec": cur})
            page2sec[(u["phys"], u["half"])] = cur
    return page2sec, starts


# ----------------------------------------------------------------- 表格缓存
def _cache_path(name):
    os.makedirs(CACHEDIR, exist_ok=True)
    return os.path.join(CACHEDIR, name + ".tables.json")


def load_tcache(name):
    p = _cache_path(name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_tcache(name, cache):
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ----------------------------------------------------------------- 主流程
def extract_company(name, pages=None, want_tables=True, verbose=True):
    comp = load_companies()[name]
    doc = fitz.open(pdf_path(name))
    npages = doc.page_count
    layout, linfo = doc_layout(doc)
    offset = comp["page_offset"]
    offset = int(offset) if str(offset).isdigit() else offset

    if verbose:
        print("【%s】%d 页  版式=%s %s  偏移=%s" % (name, npages, layout, linfo, offset))

    oi = offset if isinstance(offset, int) else None
    toc = parse_toc(doc, layout)
    toc_units = find_toc_units(doc, layout)
    page2sec, sec_rows = locate_sections(doc, layout, toc, oi, skip=toc_units)
    page2sec2, starts = scan_headings(doc, layout, oi, skip=toc_units)
    # 三档，按可信度从高到低，用哪档如实记下来：
    #   1) 目录标题回正文找  —— 有逐节实测吻合（嘉友 8/8 全对上）
    #   2) 正文扫「第X节」   —— 有实测，但只有印了节号的文档可用（国货航 8 节）
    #   3) 直接采信目录页码  —— **没有任何实测**，只因正文不印标题（申通）
    got1 = len([r for r in sec_rows if r["found_phys"]])
    got2 = len(starts)
    need = max(4, int(0.6 * len(sec_rows))) if sec_rows else 99
    if got1 >= need:
        sec_method = "toc-title"
    elif got2 >= 4:
        page2sec, sec_method = page2sec2, "heading"
    elif sec_rows and any(r["toc_page"] for r in sec_rows):
        page2sec, sec_rows = sections_from_toc(doc, layout, toc, oi)
        sec_method = "toc-page-only"
    else:
        sec_method = "none"
    if verbose:
        print("   目录页: p%s %s  单调=%-5s  条目 %d" %
              (toc["toc_phys_page"], toc.get("toc_half"), toc.get("monotonic"), len(toc["items"])))
        print("   章节来源: %s  （档1实测 %d/%d，档2实测 %d）" %
              (SEC_NAME.get(sec_method, sec_method), got1, len(sec_rows), got2))
        print("   目录页单元(节号扫描时跳过): %s" % sorted(toc_units))
        print("   -- 章节定位（主用：按目录标题回正文找）--")
        for r in sec_rows:
            flag = "" if r["delta"] in (0, None) else "  <<< 与目录页码差 %+d" % r["delta"]
            print("     %-4s %-22s 目录印页 %-5s 实测 p%-4s %s 印%s%s" %
                  (r["sec"], r["title"][:20], r["toc_page"], r["found_phys"],
                   r["found_half"], r["found_printed"], flag))
        print("   -- 交叉校验（扫正文里的 第X节，仅部分文档有效）: %d 个 --" % len(starts))
        for s in starts:
            print("     p%-4s %s 印%s  %s" % (s["phys"], s["half"], s["printed"], s["sec"]))

    rng = range(npages)
    if pages:
        rng = range(max(0, pages[0] - 1), min(npages, pages[1]))

    recs = []
    tcache = load_tcache(name) if want_tables else {}
    dirty = False
    t0 = time.time()
    for i in rng:
        units = apply_offset(page_units(doc, i, layout), offset)
        for u in units:
            u["section"] = page2sec.get((u["phys"], u["half"]))
            u["chars"] = len(re.sub(r"\s+", "", u["text"]))
            if want_tables:
                key = "%d_%s" % (u["phys"], u["half"])
                if key in tcache:
                    tb = tcache[key]
                else:
                    tb = find_tables_both(doc[i], u["clip"])
                    tcache[key] = tb
                    dirty = True
                for k in ("lines", "text", "stream"):
                    u["n_" + k] = len([t for t in tb.get(k, []) if "rows" in t])
                    u["rows_" + k] = sum(t.get("rows", 0) for t in tb.get(k, []))
                u["tables"] = tb
            recs.append(u)
        if verbose and (i + 1) % 50 == 0:
            print("   ... %d/%d 页  %.1fs" % (i + 1, npages, time.time() - t0))
    if want_tables and dirty:
        save_tcache(name, tcache)
        if verbose:
            print("   表格缓存已存 %s（%d 条）" % (_cache_path(name), len(tcache)))

    doc.close()
    return {"name": name, "code": comp["code"], "exchange": comp["exchange"],
            "layout": layout, "layout_info": linfo, "offset": offset,
            "pages": npages, "sec_method": sec_method,
            "toc": toc, "section_rows": sec_rows, "section_starts": starts, "units": recs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--pages", help="如 1-60")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, r in load_companies().items():
            print("%-8s %-8s %-5s %4s 页  偏移 %-6s  %s" %
                  (n, r["code"], r["exchange"], r["pages"], r["page_offset"], r["layout"]))
        sys.exit(0)
    rng = None
    if a.pages:
        lo, _, hi = a.pages.partition("-")
        rng = (int(lo), int(hi or lo))
    res = extract_company(a.file, rng, want_tables=not a.no_tables)
    print("目录页: p%s  条目 %d" % (res["toc"]["toc_phys_page"], len(res["toc"]["items"])))
    for it in res["toc"]["items"][:12]:
        print("   %s %s -> 印页 %s" % it)
    print("正文扫描到的节: %d 个" % len(res["section_starts"]))
    for s in res["section_starts"][:12]:
        print("   p%-4d %s %s  %s" % (s["phys"], s["half"], s["printed"], s["sec"]))
