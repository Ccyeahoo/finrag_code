# finrag —— 财报知识库问答

物流运输行业 **13 家 A 股上市公司 2025 年年度报告全文**（非摘要）的知识库问答：
PDF → 提取（**表格按行列还原进正文**）→ 切块（每块带 公司 · 章节 · 印页 · 物理页）→
**向量 ＋ BM25 混合检索** → 本地问答页面，**答案带出处角标，角标可回跳 PDF 原页**。

> 本仓库只放代码。切块、索引、报告、截图都是可重跑出来的生成物，不进仓库。

---

## 一、跑起来（四步）

```bash
# 下面的 $PY 换成你自己的 python。Windows 上注意：裸 `python` 常常是
# Microsoft Store 的占位程序，装了也不干活，用全路径最稳。
PY=/c/Users/yexinyu/AppData/Local/Python/bin/python.exe

$PY -m pip install -r requirements.txt

# 0) 备料：把 13 份年报放进 data/pdf/，并生成/校验 data/companies.csv
#    默认从 ~/Downloads 找（白名单里的文件名），用 --src 换目录
$PY -X utf8 -u src/build_corpus.py --scan

# 1) 切块 → out/chunks.jsonl（4503 块）。首次要抽表，是耗时的大头，慢慢跑
$PY -X utf8 -u src/chunk.py --all

# 2) 建索引 → out/index/（向量 ＋ BM25）。本机实测 1670.2 秒 ≈ 27.8 分钟
$PY -X utf8 -u src/search.py --build

# 3) 起服务（会问你要 key，key 只进那个窗口的环境变量，不落盘）
./start_server.bat          # 或：$PY -X utf8 -u src/serve.py --port 8000
```

打开 <http://127.0.0.1:8000> 即可提问。**服务只绑 `127.0.0.1`，不对外。**

第一次跑第 2 步会从 HuggingFace 下模型（`BAAI/bge-small-zh-v1.5`，约 95 MB）。
国内直连 `huggingface.co` 常常超时，换成镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com    # 必须在 import huggingface_hub 之前设好
```

不想联网就先把模型下到本地，再用 `--model <本地目录>`（`search.py` / `ask.py` / `serve.py` 都认）。

### 目录结构

```
src/build_corpus.py     阶段〇 建档：年报 → data/pdf/，写 companies.csv
src/extract.py          提取核心（库；也可单独试跑一个文件）
src/chunk.py            阶段二 切块 → out/chunks.jsonl
src/search.py           阶段三 建索引 ＋ 混合检索
src/ask.py              阶段四 检索 → 组证据 → 大模型成文 → 回填角标
src/serve.py            阶段四 本地问答页面
src/collect.py          批量提问（走 /api/ask），把「每题召回了哪些块」落成记录
start_server.bat        起服务（脚本里**没有**密钥，运行时才问你要）
data/companies.csv      13 家公司清单 ＋ 页码 offset（**输入，不是生成物**）
data/text_patches.json  少数乱码页的人工整页替换（同上）
```

### 几个常用命令

```bash
$PY -X utf8 -u src/extract.py --list                     # 看公司清单
$PY -X utf8 -u src/extract.py --file 嘉友国际 --pages 1-60   # 单文件试跑
$PY -X utf8 -u src/search.py --selftest                  # 检索自检（不需要 key）
$PY -X utf8 -u src/ask.py --q "顺丰控股2025年营业收入是多少" --no-llm   # 只检索不调模型
$PY -X utf8 -u src/ask.py --selftest                     # 端到端（无 key 则走模板回退）
$PY -X utf8 -u src/collect.py --stdin < 题目.txt          # 批量录题
curl -s http://127.0.0.1:8000/api/health                 # 看 llm_configured 是否 true
```

`/api/ask` 也支持 `?q=...&view=ev`，只回证据块、不出答案——查「这句话到底出自哪一块」很好用。

---

## 二、语料与规模（本机实测）

| 项 | 数 |
|---|---|
| 公司 / 年报 | 13 家 / 13 份全文 |
| 切块 | **4503 块**（p50 773 字；>2000 字 0 块，<200 字 0.2%） |
| 含表块 | 3203（**71.1%**） |
| 表格恢复 | `stream` 5663 ／ `lines` 341 ／ `text` 0 |
| 向量 | `BAAI/bge-small-zh-v1.5`（512 维）× **8631** 条（4115 块超 480 token 被切窗） |
| 融合 | RRF，k=60；每路召回 20 条；默认取前 10 块进 prompt |

## 三、几处不是默认选择的决定（都附原因）

- **表格三策略 `stream` / `lines` / `text`，生产只用前两个。**
  一页优先 `stream`，`stream` 没抓到退 `lines`，`text` 不进生产（它会把标题、页眉一并吞进表里）。
- **不加 reranker。** 课件实测加重排反而更差（0.30 vs 0.90）——半年报表格题尤其明显，故整个链路不做精排。
- **不用演示里的 Qwen3-Embedding，用 bge-small。** 本机无 GPU、CPU 无 AVX512-BF16，
  该模型跑 bf16 会让 torch **静默走软件模拟**：21 s/块且**不报任何错**（fp32 是 8.12 s/块）。
  换 bge-small 后端到端 **0.19 s/向量**。这是个**静默失败**，值得单独提一句。
- **页码口径 `印页 = 物理页 − offset`**，offset 逐家不同（顺丰 1、圆通 3、韵达 0…），存在 `data/companies.csv`。
  **拼 PDF 跳转链接一律用物理页**，用印页会跳错。
- **`temperature=0` 不等于逐字可复现。** 本机用的这个端点实测：同题重问**召回块和数字不变**，
  但**措辞、角标选择、甚至某些句子出不出**都会变。检索是确定性的，生成不是。
- **拒答是模型的选择，不是代码强制。** 语料里没有的问题，模型会说「资料里没有」——
  但代码层只发一句 `no_citation` 警告，**没有硬拦截**。

## 四、已知偏差（不可靠处，直说）

| 项 | 实情 |
|---|---|
| **国货航双页拼版** | 99 个 PDF 页 → 196 个页单元。4503 块里有 2 块**无印页**，代码如实标 `page_note`，**不猜**一个数填上 |
| **圆通 9 页 / 申通 9 页整页图** | 章节分隔页文字层是空的（内容**只存在于图片里**）→ **申通章节可信度最弱**，只能采信目录 |
| **表格还原** | 仍有 `stream` 和 `lines` 都抓不到的表；量已在 `out/chunks_stats.md` 量化（需自己重跑生成） |
| **页码** | 见上，逐家 offset 不同；国货航用 `twoup` 标记 |

## 五、安全边界

- key **只从环境变量读**（`FINRAG_API_KEY` / `FINRAG_BASE_URL` / `FINRAG_MODEL`），
  **不写入任何文件、日志、截图、前端**。
- 服务**只绑 `127.0.0.1`**，**不发 CORS 头**，前端拿不到 key；响应出站前会断言响应体里不含 key。
- key 的连通性用 `GET /api/health` 的 `llm_configured` 验证，**从不打印 key 本身**。
- `start_server.bat` 里**没有密钥**，运行时才问你，且只进那个窗口的环境变量。
