# -*- coding: utf-8 -*-
r"""把「每道题召回了哪些块」落成记录——阶段四报告的证据，阶段五直接复用。

**为什么不直接 import ask 跑一遍**：作业要的是「页面这条路」的记录。走 HTTP 打到
`/api/ask` 拿到的东西，与用户在页面上看到的是同一份（同一个 `answer_question`），
所以**记录即页面、页面即记录**，不会出现「报告里写召回第 3，页面上其实第 5」这种事。

用法：
    PY=/c/Users/yexinyu/AppData/Local/Python/bin/python.exe
    $PY -X utf8 -u src/collect.py --url http://127.0.0.1:8000 --q 问题1 --q 问题2 …
    $PY -X utf8 -u src/collect.py --stdin        # 一行一题（脚本里带空格/引号时省事）
落盘：`out/ask_json/<序号>_<题目前12字>.json`（每题一份原始 JSON）+ 屏上打一张紧凑表。
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "ask_json")


def slug(q, n=12):
    """文件名用题目前 n 字，去掉 Windows 不允许的字符。宁可截短也不要写失败。"""
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", q)[:n]
    return s or "q"


def ask(url, q, timeout=300):
    req = urllib.request.Request(
        url.rstrip("/") + "/api/ask",
        data=json.dumps({"q": q}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def show(i, q, d):
    """紧凑表：每块一行（编号 / 公司 / 块号 / 印页 / 物理页 / 分），再看角标与告警。"""
    print("=" * 78)
    print("[%d] %s" % (i, q))
    print("    用没用大模型：%s ｜ 模型 %s ｜ 用时 %ss ｜ k=%d"
          % ("用了" if d["used_llm"] else "**没用（模板回退）**",
             d.get("model") or "—", d.get("elapsed"), d["k"]))
    print("    子查询：%s" % " ； ".join(
        "%s→%s" % (s["company"] or "·", s["query"]) for s in d["sub_queries"]))
    for e in d["evidence"]:
        mark = "✔" if e["n"] in d["citations"] else " "
        print("     %s #%-2d %-6s %-10s 印%-9s 物理%-5s RRF %.5f"
              % (mark, e["n"], e["company"] or "—", e["chunk_id"],
                 e["printed"] or e["printed_note"] or "—", e["phys"], e["score"]))
    print("    角标：%s%s" % (d["citations"] or "**一个都没有**",
                            "（越界丢弃 %s）" % d["dropped_citations"]
                            if d["dropped_citations"] else ""))
    print("    no_citation：%s" % d["no_citation"])
    for w in d["warnings"]:
        print("    ⚠ %s" % w)
    if d.get("usage"):
        print("    usage：%s" % d["usage"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--q", action="append", default=[])
    ap.add_argument("--stdin", action="store_true", help="从 stdin 一行读一题")
    ap.add_argument("--no-save", action="store_true")
    # 阶段五的 10 道题接着阶段四那 7 份往下编（11 起），**不覆盖**——阶段四的记录是
    # 「没有 key」那一版的证据，两份都要留着对照。
    ap.add_argument("--start", type=int, default=1, help="起始编号（默认 1）")
    a = ap.parse_args()

    qs = list(a.q)
    if a.stdin:
        qs += [ln.strip() for ln in sys.stdin if ln.strip()]
    if not qs:
        ap.error("至少要一题：--q 或 --stdin")

    if not a.no_save:
        os.makedirs(OUT, exist_ok=True)
    bad = 0
    for i, q in enumerate(qs, a.start):
        try:
            d = ask(a.url, q)
        except urllib.error.HTTPError as e:
            print("[%d] %s → HTTP %s：%s" % (i, q, e.code, e.read()[:300]))
            bad += 1
            continue
        except Exception as e:
            print("[%d] %s → %s: %s" % (i, q, type(e).__name__, e))
            bad += 1
            continue
        show(i, q, d)
        if not a.no_save:
            p = os.path.join(OUT, "%02d_%s.json" % (i, slug(q)))
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=1)
            print("    已存 %s" % os.path.relpath(p, ROOT))
    print("=" * 78)
    print("共 %d 题，失败 %d 题" % (len(qs), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
