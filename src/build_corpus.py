# -*- coding: utf-8 -*-
"""阶段〇：建档

1. 建 finrag 目录树
2. 按**显式白名单**把 13 份年报从 Downloads 复制到 data/pdf（原件不动，多余文件不删）
3. 从每份年报前 3 页抽股票代码候选，打印出来供人工核对
4. 打印页码偏移的脚本推断值（**只作参考，须人工数**）
5. 写 data/companies.csv

用法：
  python build_corpus.py                     # 全量：从 ~/Downloads 复制 + 扫描 + 写 csv
  python build_corpus.py --scan              # 只扫描已存在的 pdf（不复制）
  python build_corpus.py --src D:\\年报        # 换原件的来源目录（默认 ~/Downloads）
"""
import csv
import io
import os
import re
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pymupdf as fitz

# 项目根从本文件位置推（src/ 的上一级），不写死绝对路径。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 年报原件的来源目录：默认 ~/Downloads，用 --src 换。
SRC = os.path.expanduser(os.environ.get("FINRAG_SRC") or "~/Downloads")
for _i, _a in enumerate(sys.argv):
    if _a == "--src" and _i + 1 < len(sys.argv):
        SRC = sys.argv[_i + 1]
PDFDIR = os.path.join(ROOT, "data", "pdf")
CSVPATH = os.path.join(ROOT, "data", "companies.csv")

# ---------------------------------------------------------------- 显式白名单
# 只认这张表。Downloads 里那份国货航重复件和嘉友国际摘要件**不删**，
# 但只要不在这张表里，就永远不会被读进来。
WHITELIST = [
    ("顺丰控股", "顺丰控股：2025年年度报告.pdf"),
    ("圆通速递", "圆通速递：圆通速递股份有限公司2025年年度报告.pdf"),
    ("韵达股份", "韵达股份：2025年年度报告.pdf"),
    ("申通快递", "申通快递：2025年年度报告.pdf"),
    ("中国外运", "中国外运：2025年年度报告.pdf"),
    ("华贸物流", "华贸物流：港中旅华贸国际物流股份有限公司2025年年度报告.pdf"),
    ("中储股份", "中储股份：中储发展股份有限公司2025年年度报告.pdf"),
    ("国货航", "国货航：2025年年度报告.pdf"),
    ("东航物流", "东航物流：东航物流2025年年度报告.pdf"),
    ("中谷物流", "中谷物流：2025年年度报告.pdf"),
    ("辽港股份", "辽港股份：辽宁港口股份有限公司2025年年度报告全文.pdf"),
    ("铁龙物流", "铁龙物流：中铁铁龙集装箱物流股份有限公司2025年年度报告.pdf"),
    ("嘉友国际", "嘉友国际：2025年年度报告.pdf"),
]

# 只有**确实抽不出**的项才写在这里，不许脚本瞎猜一个数上去。
# 圆通速递曾经被误列在这里（写成 "unknown"），实际人工数出来是 3：
#   p20/45/60 页脚交替印 017 / 042 / 057 → 物理页 - 印页 = 3，脚本的自动推断是对的，
#   是这张表冤枉了它。已删除该条，避免重跑阶段〇时把 companies.csv 又改坏。
KNOWN_BAD = {
    "国货航": ("twoup", "双页拼版：97/99 页两页并排，页码印在页眉，不走偏移"),
}

LAYOUT_HINT = {
    "国货航": "twoup",
}


def page_no(text):
    """从一页文字里找报告自己印的页码。返回候选列表，取不到返回空。"""
    t = text
    cands = []
    # 形如 "1 / 334" 或 "059"
    m = re.search(r"(?<!\d)(\d{1,3})\s*/\s*(\d{2,4})(?!\d)", t[:300] + " " + t[-300:])
    if m:
        cands.append(("slash", int(m.group(1))))
    head = " ".join(t[:110].split())
    tail = " ".join(t[-110:].split())
    for where, s in (("head", head), ("tail", tail)):
        for mm in re.finditer(r"(?<!\d)(\d{1,4})(?!\d)", s):
            cands.append((where, int(mm.group(1))))
    return cands


def infer_offset(doc):
    """对若干页投票推断 物理页 - 印的页码。样本不足或票数不集中则判为抽不出。"""
    votes = {}
    for phys in (20, 45, 60):
        if phys > doc.page_count:
            continue
        for where, num in page_no(doc[phys - 1].get_text()):
            if 1 <= num <= doc.page_count:
                d = phys - num
                if 0 <= d <= 5:
                    votes[d] = votes.get(d, 0) + 1
    if not votes:
        return None, []
    best, n = max(votes.items(), key=lambda kv: kv[1])
    # 要求至少 2 票，且是唯一最高票
    tied = [k for k, v in votes.items() if v == n]
    if n < 2 or len(tied) > 1:
        return None, sorted(votes.items())
    return best, sorted(votes.items())


def scan(path):
    d = fitz.open(path)
    npages = d.page_count
    chars = 0
    for i in range(npages):
        # 口径统一：**去掉所有空白字符后再计数**，与 out/pages.jsonl 的 chars、
        # 与切块阶段的 n_chars 同一个定义。
        # 阶段一这里写的是 len(get_text().strip())，只削首尾、不删中间空白，
        # 于是同一批 PDF 有三个字数：308 万（这里）/ 271.7 万（去空白，权威）。
        # 三个数并存的后果是"字数"这个指标没法横向比，所以阶段二统一到去空白。
        chars += len(re.sub(r"\s+", "", d[i].get_text()))
    # 封面/前3页的 6 位数字 —— 股票代码候选
    front = "\n".join(d[i].get_text() for i in range(min(3, npages)))
    code_hits = {}
    for m in re.finditer(r"(?<!\d)(\d{6})(?!\d)", front):
        c = m.group(1)
        code_hits[c] = code_hits.get(c, 0) + 1
    offset, votes = infer_offset(d)
    d.close()
    return npages, chars, code_hits, offset, votes


# 这些列**人工核对过**，重跑脚本一律沿用 CSV 里的旧值，只在脚本推断不同时把差异打出来。
# 为什么要保护：KNOWN_BAD 只登记了国货航一家，其余 12 家的备注全是手填的；
# 股票代码、交易所、页码偏移也都是逐家数出来的。重跑若不保护，
# 会把 SZSE/SSE、以及圆通/申通"含 9 页纯图"这类核对结论一起冲成脚本的粗推断值。
KEEP_COLS = ("code", "exchange", "pages", "page_offset", "layout", "note")


def load_existing():
    """读回现有 companies.csv（若存在），键=公司名。用于保护人工核对过的列。"""
    if not os.path.exists(CSVPATH):
        return {}
    with open(CSVPATH, encoding="utf-8-sig") as f:
        return {r["name"]: r for r in csv.DictReader(f)}


def main():
    do_copy = "--scan" not in sys.argv
    for sub in ("data/pdf", "src", "out/pairs", "out/tables"):
        os.makedirs(os.path.join(ROOT, sub.replace("/", os.sep)), exist_ok=True)

    rows = []
    existing = load_existing()
    if existing:
        print("读回现有 companies.csv %d 家：%s 列沿用旧值，chars 一律重算"
              % (len(existing), "/".join(KEEP_COLS)))
    print("=" * 78)
    print("阶段〇 建档：%d 家" % len(WHITELIST))
    print("=" * 78)

    for name, srcfile in WHITELIST:
        s = os.path.join(SRC, srcfile)
        t = os.path.join(PDFDIR, name + ".pdf")
        if not os.path.exists(s):
            print("!! 源文件不存在：%s" % srcfile)
            continue
        if do_copy:
            need = True
            if os.path.exists(t) and os.path.getsize(t) == os.path.getsize(s):
                need = False
            if need:
                shutil.copy2(s, t)
        if not os.path.exists(t):
            print("!! 未复制到：%s" % t)
            continue

        npages, chars, codes, offset, votes = scan(t)
        layout, note = KNOWN_BAD.get(name, (LAYOUT_HINT.get(name, "normal"), ""))
        if name not in KNOWN_BAD:
            if offset is None:
                page_offset = "unknown"
                note = "脚本推断失败（票数 %s），须人工数" % (votes or "空")
                layout = "normal"
            else:
                page_offset = offset
                note = "脚本推断（票数 %s），须人工数复核" % votes
                layout = LAYOUT_HINT.get(name, "normal")
        else:
            page_offset = layout  # twoup / unknown 直接占位

        code = max(codes.items(), key=lambda kv: kv[1])[0] if codes else ""
        row = {
            "code": code,
            "name": name,
            "exchange": "",
            "pdf_file": name + ".pdf",
            "pages": npages,
            "chars": chars,
            "page_offset": page_offset,
            "layout": layout,
            "note": note,
        }

        # 人工核对过的列优先用 CSV 旧值；chars / pages / pdf_file 用本次实算
        old = existing.get(name) or {}
        kept, diffs = [], []
        for k in KEEP_COLS:
            ov = (old.get(k) or "").strip()
            if not ov:
                continue
            if ov != str(row[k]):
                diffs.append((k, str(row[k]), ov))
            row[k] = ov
            kept.append(k)
        row["chars"] = chars
        rows.append(row)

        print("\n【%s】%d 页 / %d 字（去空白）" % (name, npages, chars))
        if kept:
            print("   [保护] 沿用 CSV 旧值：%s" % "、".join(kept))
        for k, new, ov in diffs:
            print("   [差异] %s：脚本推断 %r ≠ CSV %r → **采用 CSV 值**" % (k, new, ov))
        if old and old.get("chars"):
            try:
                oc = int(old["chars"])
                print("   [字数] 旧值 %d（含空白）→ 新值 %d（去空白），差 %+d"
                      % (oc, chars, chars - oc))
            except ValueError:
                pass
        if not row["exchange"]:
            print("   [注意] exchange 为空 —— 旧 CSV 里也没有，需人工补")
        print("   源文件: %s" % srcfile)
        print("   代码候选(前3页 6位数字, 次数): %s" % sorted(codes.items(), key=lambda kv: -kv[1]))
        print("   偏移: %s   票数: %s" % (page_offset, votes))
        print("   备注: %s" % note)

    with open(CSVPATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["code", "name", "exchange", "pdf_file",
                                          "pages", "chars", "page_offset", "layout", "note"])
        w.writeheader()
        w.writerows(rows)

    tp = sum(int(r["pages"]) for r in rows)
    tc = sum(r["chars"] for r in rows)
    print("\n" + "=" * 78)
    print("合计 %d 家 / %d 页 / %.1f 万字" % (len(rows), tp, tc / 10000.0))
    print("已写 %s" % CSVPATH)


if __name__ == "__main__":
    main()
