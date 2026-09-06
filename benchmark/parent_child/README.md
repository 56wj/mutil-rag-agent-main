# Parent-Child Chunking Evaluation

参考 Sandia National Laboratories + NVIDIA 的 Kokkos Coding Assistant 评测方法：
**Parent Size 与 Child Size 成对评估**，而非固定 parent 只调 child。
[原始实验说明](https://developer.nvidia.com/blog/advanced-ai-and-retrieval-augmented-generation-for-code-development-in-high-performance-computing/)。
这里采用面向中英文运维语料的新配对，未复制 Kokkos 的具体尺寸。

## 1. 编码前审阅：当前数据流与模块

已审阅 `scripts/ingest_kb_corpus.py`、`app/core/splitter.py`、`app/core/embedding.py`、
`app/core/vector_store.py`、`app/core/hybrid_retriever.py`、`app/core/reranker.py`、
`app/rag/retrieval.py` 和 `benchmark/run_benchmark.py`。

```text
3 份 docs/sop + data/kb_corpus/**/*.md（本次冻结 958 文件）
 → H1/H2/H3 自然节作为 parent 候选
 → 超长 parent 按 Parent Size 二切（代码/表格等结构保护）
 → 各 parent 内按 Child Size 切 child，注入 chapter 前缀
 → child 保存 source/chapter/parent_id/parent_content/chunk_index
 → BGE-M3 → Milvus HNSW/COSINE（每个 child 一行）

query → dense Retrieve K + BM25 Retrieve K → weighted RRF（保留 Retrieve K）
 → BGE reranker（选 Final Top-K × 3 个 child）
 → 按 parent_id 去重 → Final Top-K parent
 → 正文按 Parent Size 截断 → 最终 context
```

BM25 使用项目原有英文 token/CJK 单字分词器与 BM25Okapi。生产内存单例首次从 Milvus
拉取，旧 loader 上限 16384；评测每组独立从该组冻结 children 构建，不污染生产 singleton。
本次最大 5618 children，未触及旧 loader 上限。评测保持相同公式及检索返回数量，固定
语料入库顺序以稳定 BM25 同分排序。

**现有 Parent Size 是自然标题节上限，而非目标长度。** 当前多数节约 100–200 字符。
本轮保留 H1/H2/H3 边界，不通过合并标题人为放大 parent：否则将同时改变标题层级和尺寸，
混淆归因。结构保护的代码块恢复后可能超过名义尺寸；最终截断行为也与生产一致。

模块改造：

| 模块 | 变化 |
|---|---|
| `app/core/splitter.py` | 新增显式 `ChunkingConfig`；既有调用保留 settings 解析、默认 overlap 及切分行为 |
| `app/core/vector_store.py` | 抽取 collection factory，共用原 HNSW/COSINE 参数；生产缓存入口不变 |
| `benchmark/parent_child_eval.py` | 新增 plan/run/report/cleanup、隔离索引、栈指纹、断点续跑、逐题计分与报告 |
| `benchmark/parent_child_production_baseline.py` | 对现有 production collection 做只读同栈回放，并用冻结语料反证其实际切分配置 |
| `benchmark/parent_child/evidence.py` | 正文证据校验及 AND/OR 事实组覆盖计分 |
| `configs.json` / `hardcases_20.jsonl` / `base_case_links.json` | 配对配置、20 道题、历史题语义关联 |
| `benchmark/tests/test_parent_child_eval.py` | 切分兼容、BM25/RRF 一致性、证据计分、隔离与完整性回归 |

## 2. 第一阶段 coarse sweep

字符单位；Parent/Child 比例 4–5.33；父、子 overlap 均为 0，标题层级固定为 3。

| ID | Scale | Parent | Child | Ratio |
|---|---|---:|---:|---:|
| S1 | small | 512 | 128 | 4.00 |
| S2 | small | 768 | 192 | 4.00 |
| M1 | medium | 1024 | 256 | 4.00 |
| M2 | medium | 1536 | 320 | 4.80 |
| M3 | medium | 2048 | 384 | 5.33 |
| L1 | large | 2560 | 512 | 5.00 |
| L2 | large | 3200 | 800 | 4.00 |

加载时检查 5–8 组、覆盖三档、唯一 ID、Parent 至少 3×Child，且不触发 splitter 尺寸钳制。

## 3. 50 + 20 题及计分

- 现有 `benchmark/ragas_qa_50.jsonl` 的 50 个 question/ground_truth **原样保留**。
- 对应 source/chapter gold 来自 `retrieval_rk_50.jsonl`。
- 使用审核后的 `base_case_links.json` 映射，**不是按 ID 后缀 join**：Redis 两套编号错位，
  例如 RAGAS 的 OOM/碎片率/CPU 对应 retrieval 03/04/05。两份原始题集保持原样。
- 新增四类各 5 道：精确阈值/命令、跨段证据、多步骤排障、相似章节干扰。
- 每题 hardcase 带正文 `evidence_groups`，运行前验证所有原文锚点存在于冻结 source。
  组间 AND（独立知识点），组内 OR（等价证据来源）；计分只看最终去重并截断的 parent 正文，
  不用注入 chapter、child preview 或 ground truth 充当证据。
- `Hard evidence recall` = 每题命中事实组比例的宏平均；`complete` = 全事实组覆盖题比例。
  另保留原 chapter Hit/Recall/MRR、四类证据覆盖、干扰章节占最终 parents 的比例。

证据覆盖是**检索质量代理指标**，不等于生成答案正确率。较难多事实题可能需要超过固定
Top-K 的自然标题节，应结合逐题命中正文解释缺失，不暗中扩大 K。

## 4. 固定条件与隔离

`run` 读取现有环境，不修改 `settings`、`.env` 或生产 collection：

- 当前 Ollama BGE-M3 模型、dimension、endpoint、embedding batch；
- Milvus HNSW/COSINE：M=8、efConstruction=64、当前 search ef；
- BM25 weight、vector weight、RRF k；
- 本地 BGE reranker 模型、device、backend、parent context cap、max length、batch；
- 当前 Retrieve K、Final Top-K、既有 child overfetch=3；
- 语料**内容**快照、题集、配置、代码及关键依赖版本指纹；
- warm-up 数、串行题目顺序、入库 batch、重复次数（coarse 为一次）。

`fixed_stack.json` 与 `measurement_protocol.json` 在续跑前核对，禁止不同条件覆盖记录。
如环境仍是仓库默认的其他模型，runner 会报错而非自动替换成 BGE。Retrieve K 若小于
Final Top-K×3，也要求先核对真实基线，不自动改 K。

collection 命名包含 snapshot、配置值、run 绝对路径摘要。drop/rebuild/cleanup 既检查评测
前缀，也校验本 run 的精确 ownership；cleanup 额外比对原 endpoint。线上 collection 只读
检查行数及 index 参数，绝不写入。每组入库 flush 后等待行数匹配，核对实际 index；组间
release 该组索引，避免累计加载。中断的半成品索引用同一 run 的 `--rebuild` 重建。

生产行数 guard 只用于发现并发变动，不代表已验证外部生产 collection 与本地 corpus 完全
相同；实验比较的依据是本 run 的冻结内容。建议在空闲实例执行，减小资源竞争对延迟的影响。

## 5. 执行

在已有项目依赖环境中运行；本地 reranker 还需 `FlagEmbedding`、`torch`。模型/服务与 `.env`
应来自现有运行环境，**不要为了跑实验覆盖生产配置为 `.env.example`**。

```bash
# 已有环境缺少依赖时安装到该评测环境，而非更新生产环境
python -m pip install -r requirements.txt
python -m pip install FlagEmbedding torch

# 无模型调用、无 DB 写入：冻结语料并测实际切分
python benchmark/parent_child_eval.py plan --run-dir benchmark/parent_child/runs/coarse-01

# 同一快照，7 组 × 70 题，默认每组预热 3 题
python benchmark/parent_child_eval.py run --run-dir benchmark/parent_child/runs/coarse-01

# 断点续跑指定组；只有半成品索引或需要重建时才加 --rebuild
python benchmark/parent_child_eval.py run --run-dir benchmark/parent_child/runs/coarse-01 --configs M2,M3,L1,L2

# 可选：完整 coarse 后，对候选追加端到端 RAGAS（额外生成/评委调用）
python benchmark/parent_child_eval.py run --run-dir benchmark/parent_child/runs/coarse-01 --configs M2,M3 --with-ragas

python benchmark/parent_child_eval.py report --run-dir benchmark/parent_child/runs/coarse-01

# 可选：只读回放现有 production collection；严格核对 fixed stack 和前后行数
python benchmark/parent_child_production_baseline.py \
  --run-dir benchmark/parent_child/runs/coarse-01

python benchmark/parent_child_eval.py cleanup --run-dir benchmark/parent_child/runs/coarse-01

python -m unittest discover -s benchmark/tests -v
```

`run` 的配置/题集版本应与 snapshot 一致；历史 `report` 使用 run 内冻结资产，不依赖当前
题集。推荐只在 **7 组各 70 题**、同 snapshot/stack/protocol 的完整结果通过校验后生成。
结果在 stack 与生产行数 guard 通过后原子落盘；失败/半组不计入比较。

RAGAS 复用现有 answer prompt、LLM/embedding 构造器和四项指标；单题失败记录 error，
不以 0 冒充有效分数，报告包含成功/失败题数。它只作端到端确认，不进入第一阶段 composite。
追加 RAGAS 会重新测该组检索；跨时段延迟应复测确认，不据单次小差异决策。

## 6. 输出和 trade-off

- `structural_plan.{json,md}`：实际 parent/child 数、长度分布、相同 parent 分区、payload 估算。
- `comparison.{json,md}`：逐配置质量、索引规模、延迟、最终上下文和推荐。
- `results/<ID>.json`：70 题命中正文、证据组、分段延迟、上下文成本，可复核计分。
- `snapshot_manifest.json` / `corpus_snapshot.jsonl` / `eval_cases.jsonl` / `configs.snapshot.json`：冻结输入。

索引 payload = float32 向量 + child UTF-8 文本 + metadata JSON。parent 在每个 child 冗余存储，
故小 child 不仅增加向量，也增加父正文副本。**这是逻辑下界，不是 Milvus 磁盘占用**。
延迟记录 vector（含 query embedding）、BM25+RRF、reranker、total 的 mean/P50/P95/max。
最终 context 统计 chars 及 `CJK chars + ceil(non-CJK chars / 4)` 估算 tokens，不冒充 tokenizer
精确账单或生成 TTFT。长 parent 的跨段收益与无关正文成本均由最终真实命中衡量。

推荐规则：`Q = 0.4 × Base chapter recall + 0.4 × Hard evidence recall + 0.2 × Overall MRR`；
先要求 Base recall、Hard evidence recall、Hard complete 均不低于 0.99，再最大化 Q，完全同分时
依次最小化平均延迟、P95、上下文和索引 payload。旧的“Q 距最佳 0.01 后先最小化 token”规则会
让约 1 个估算 token 的噪声覆盖质量和延迟，已由实际结果否决。这仍是 coarse 候选规则，不是
统计显著性或全局最优证明；推荐只写报告，不自动上线。

## 7. 本机已验证结果（2026-09-01）

完整 run：`runs/prod-20260831/`。958 文件、7 组 × 70 题；使用 BGE-M3、Milvus 2.4.10
HNSW/COSINE M=8/efConstruction=64/search ef=128、BM25 0.4 + Vector 0.6、RRF k=60、
BAAI/bge-reranker-v2-m3、Retrieve K=30、Final Top-K=3。Apple M3 Pro/MPS，串行、每组 3 次
warm-up。所有配置均为隔离 collection，production collection 行数在只读回放前后保持 4113。

| ID | Parent/Child | Base R | Hard evidence R/C | Q | Children | MinIO 实占 MiB | Total mean/P95 ms | Context avg tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 512/128 | 1.000 | 0.911/0.750 | 0.956 | 5618 | 43.40 | 5812/7372 | 544 |
| S2 | 768/192 | 0.980 | 0.955/0.850 | 0.960 | 4290 | 33.29 | 6389/8077 | 678 |
| M1 | 1024/256 | 1.000 | 1.000/1.000 | 0.990 | 4157 | 32.27 | 6142/8000 | 712 |
| **M2** | **1536/320** | **1.000** | **1.000/1.000** | **0.991** | **4141** | **32.15** | **5833/7554** | **712** |
| M3 | 2048/384 | 1.000 | 1.000/1.000 | 0.989 | 4129 | 32.06 | 5911/7338 | 705 |
| L1 | 2560/512 | 1.000 | 1.000/1.000 | 0.989 | 4121 | 32.01 | 8283/10321 | 704 |
| L2 | 3200/800 | 0.960 | 1.000/1.000 | 0.968 | 4113 | 31.94 | 7834/9492 | 688 |

推荐 **M2 = Parent 1536 / Child 320 / overlap 0**；M3 作为 tail-latency challenger。M2 与 M3 的
quality 差只有 0.00286，paired bootstrap 95% CI `[0, 0.00714]`，因此应通过 shadow/canary
继续确认，不把 coarse 结果包装成显著性结论。M2 延迟中 reranker 占约 95%，瓶颈不在 Milvus
或 BM25；本轮按要求未改变 reranker/Retrieve K 等固定项。

actual parent P50/P95/max 为 121/204/1026 字符，M1/M2/M3/L1/L2 的 parent 分区完全相同；
本轮主要识别到 Child 320–384 的甜点区，不代表 1536 对未来长文档永久最优。

现有 `multi_agent_kb` 经冻结语料 multiset 精确反推为 2400/800/overlap 100。只读 70 题基线
Q=0.903、Hard evidence R/C=0.850/0.850，但其同文本向量与 fresh-build 并非逐 float 相等且
HNSW 图构建时点不同，因此 PROD 与 fresh sweep 的差值不能全部归因于 chunk。完整证据：

- `runs/prod-20260831/comparison.md`：七组主报告；
- `runs/prod-20260831/production_analysis.md`：生产成熟度、M2/M3 权衡和上线边界；
- `runs/prod-20260831/production_baseline.json`：生产只读逐题基线；
- `runs/prod-20260831/storage_measurement.json`：MinIO insert/HNSW/stats 实际占用；
- `runs/prod-20260831/production_vector_audit.json`：生产与 fresh-build 文本/向量审计。
