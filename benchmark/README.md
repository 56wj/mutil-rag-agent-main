# Benchmark

这个目录放 RAG 评测与 Agent 策略评测：

- `ragas_qa_50.jsonl`: 50 条端到端 RAGAS QA, 每个场景 5 条。
- `retrieval_rk_50.jsonl`: 50 条检索侧 R@K 题, 每个场景 5 条。
- `run_benchmark.py`: 支持 retrieval / ragas 两种模式, 逐题打印滚动指标。
- `parent_child_eval.py`: 隔离 collection 的 Parent/Child 成对 coarse sweep（现有 50 题 + 20 hardcases）。
- `parent_child_production_baseline.py`: 对当前 production collection 做只读 70 题同栈基线回放。
- `parent_child/README.md`: Parent-Child 评测设计、固定条件、指标、运行与清理说明。
- `agent_strategy_cases.jsonl`: fast/deep 共用的固定故障集与可机评 Gold。
- `agent_strategy_eval.py`: 只对比 fast 与 deep，输出质量、时延、Token 和工具调用差值。

## fast vs deep Agent 评测

先用同一份本地数据成对执行，两种策略的输出会写入同一份报告：

```bash
python benchmark/agent_strategy_eval.py local --limit 2
# 全量 6 条
python benchmark/agent_strategy_eval.py local
```

报告位置：`benchmark/reports/agent_strategy_<timestamp>.json`。核心指标包括：

- `diagnosis_success`
- `root_cause_recall`
- `evidence_recall`
- `remediation_recall`
- mean/P50/P95 latency
- mean total tokens / tool calls
- `deep_minus_fast` 成对差值

Langfuse 正式 Dataset Experiment：

```bash
# .env 中先配置 LANGFUSE_ENABLED/PUBLIC_KEY/SECRET_KEY/BASE_URL
python benchmark/agent_strategy_eval.py langfuse \
  --dataset-name aiops/fast-deep-diagnosis \
  --experiment-name aiops-fast-vs-deep
```

命令会幂等 upsert 固定集，然后生成 `<run_id>-fast` 与 `<run_id>-deep` 两个实验运行。
两边使用同一 Dataset、同一评测器，不加入 Single-Agent 第三基线。
评测调用会关闭可变 Wiki 的读取和回写，防止先运行的 fast 报告泄漏给随后运行的 deep。
若 `deep` 因配置关闭而回落到 `fast`，该样本会标记为失败，避免生成伪对比数据。

本地评测入口会显式初始化/关闭 MCP Client，与 API/Worker lifespan 保持一致，避免
CLI 静默退化成仅本地工具。Deep 图会把原始告警作为 `alert_payload` Evidence：其
`trust_level=reported`、基础分低于现场 metric/log，既保留入口事实，又不会压过本轮取证。

Parent-Child sweep 快速入口：

```bash
python benchmark/parent_child_eval.py plan
python benchmark/parent_child_eval.py run --rebuild

# sweep 完成后，可选只读生产基线（不写/不重建 production collection）
python benchmark/parent_child_production_baseline.py --run-dir benchmark/parent_child/runs/<run>
```

## 前置条件

先确保 Docker 里的 Milvus 已启动, 并且已用当前 embedding 配置重建知识库:

```bash
docker compose up -d
python scripts/ingest_kb_corpus.py --reset --batch 8
```

## 检索侧 R@K

推荐先跑这个, 它最快, 能实时观察检索参数变化:

```bash
python benchmark/run_benchmark.py retrieval --k 3
```

常用参数:

```bash
# 看 R@5
python benchmark/run_benchmark.py retrieval --k 5

# 只跑某个场景
python benchmark/run_benchmark.py retrieval --scenario Kafka --k 3

# 关闭 rerank 做 A/B
python benchmark/run_benchmark.py retrieval --k 3 --no-rerank

# 关闭 hybrid 做 A/B
python benchmark/run_benchmark.py retrieval --k 3 --no-hybrid
```

输出指标:

- `hit@k`: top-k 中是否命中任意 gold。
- `mrr@k`: 第一个命中位置的倒数。
- `recall@k`: top-k 覆盖的知识点组比例。

Gold 规则:

- 旧格式 `relevant: [A, B, C]` 表示 A/B/C 是同一知识点的替代来源，命中任意一个即可。
- 多个独立知识点使用 `relevant_groups: [[A, B], [C, D]]`，组内是 OR，组间按覆盖率计算 recall。
- 这样 awesome 告警、自建 runbook、SOP 是替代答案时，不会因为未同时进入 top-k 而错误扣分。

## RAGAS 端到端

这个会调用 LLM 生成答案, 再用 RAGAS judge 打分, 会比较慢:

```bash
python benchmark/run_benchmark.py ragas --limit 5
```

默认同时运行 OpenEvals:

- `groundedness`: 回答是否由检索上下文支持。
- `helpfulness`: 回答是否真正解决用户问题。

如只想运行原来的 RAGAS 四项:

```bash
python benchmark/run_benchmark.py ragas --limit 5 --no-openevals
```

加 `--verbose` 会打印 OpenEvals 的扣分原因，并写入 JSON 报告。

全量 50 条:

```bash
python benchmark/run_benchmark.py ragas
```

结果会逐题打印滚动均值, 并写入:

- `benchmark/reports/retrieval_*.json`
- `benchmark/reports/ragas_*.json`
