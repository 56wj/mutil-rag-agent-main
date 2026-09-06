# AgentOps 可观测性与真实取证

这一层解决两个生产问题：诊断 Agent 要读取真实指标/日志；平台自身也要能回答“哪一段慢、
哪个工具失败、Kafka 是否堆积、一次 RCA 花了多少 token”。

## 数据流

```text
HTTP / Alertmanager
  └─ W3C traceparent
      └─ Kafka W3C headers（JSON trace_context 兼容重放）
          └─ Diagnosis Worker consumer span
              └─ LangGraph diagnosis span
                  ├─ LLM round spans + token metrics
                  └─ Tool call spans + latency/result metrics

Diagnosis Runner ─> Langfuse Agent / Chain / Generation / Tool trace

API /metrics ─────────────┐
Worker :9910/metrics ─────┼─> Prometheus ─> Grafana / alerts
Docker stdout ─> Alloy ───┴─> Loki ───────> LogAgent (LogQL)
OTLP/HTTP ─> Collector ─────> Tempo ──────> Grafana trace view
```

职责边界：Langfuse 用于 Agent 语义 Trace、Dataset Experiment 和质量分数；Prometheus
用于 API/Worker/Kafka 的 SLI；Loki 是诊断取证数据源。平台可只启用 Langfuse +
Prometheus，Tempo 保留为通用基础设施 Trace 的可选后端。

## Langfuse 与 fast/deep 实验

Deep 诊断的根 Observation 下同时记录入口 `alert_payload`、专业 Agent、LLM 与 Tool
子树。`alert_payload` 带 `provenance` 和 `trust_level=reported`，RCAJudge 可在实时
Prometheus/Loki 暂时无结果时保留告警事实，同时仍优先采用本轮现场 metric/log 证据。

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=staging
LANGFUSE_RELEASE=1.0.0
LANGFUSE_SAMPLE_RATE=1.0
```

Runner 会给 Trace 写入 `requested_mode`、`effective_mode`、`session_id`、release 及
benchmark 的 dataset/case/run 维度。SDK、密钥或后端异常时自动退化，不改变诊断结果。

```bash
# 本地成对报告
python benchmark/agent_strategy_eval.py local

# 同步 Langfuse Dataset，并各跑一个 fast/deep Experiment
python benchmark/agent_strategy_eval.py langfuse \
  --dataset-name aiops/fast-deep-diagnosis
```

HTTP 响应包含 `X-Request-ID` 和 `X-Trace-ID`。审计 Evidence metadata 同时保存
`trace_id`，可以从 Incident/AgentRun 事实记录跳到 Tempo，再从日志中的
`trace=<32 hex>` 跳回同一条 Trace。

## 启动与验收

```bash
docker compose --profile app up -d --build
docker compose ps

curl -fsS http://localhost:9900/metrics | grep '^aiops_'
curl -fsS http://localhost:9090/-/ready
curl -fsS http://localhost:3100/ready
curl -fsS http://localhost:3200/ready
```

提交一个任务并检查链路：

```bash
curl -i -X POST http://localhost:9900/api/v1/aiops/diagnose/submit \
  -H 'Content-Type: application/json' \
  -d '{"query":"checkout 5xx 激增，检查指标与错误日志","mode":"deep","service":"checkout"}'

curl -G http://localhost:3100/loki/api/v1/query_range \
  --data-urlencode 'query={platform="multi-agent-aiops"} |= "diagnosis-worker"' \
  --data-urlencode 'limit=20'
```

Grafana 打开 `http://localhost:3000/d/multi-agent-aiops-agentops`。Explore 中选择
Tempo 可按 service/name 搜索 Trace；选择 Loki 可执行：

```logql
{platform="multi-agent-aiops", service=~"api|worker-.*"} |= "trace="
```

## 指标目录

| Metric | 维度 | 用途 |
|---|---|---|
| `aiops_http_requests_total` | method, route, status | API 流量、错误率 |
| `aiops_http_request_duration_seconds` | method, route | API P95/P99 |
| `aiops_kafka_messages_total` | action, result, priority | enqueue/consume/ack/DLQ |
| `aiops_kafka_consumer_lag` | priority | 各优先级 backlog |
| `aiops_kafka_dlq_depth` | - | 死信积压 |
| `aiops_diagnosis_runs_total` | mode, status | fast/deep 成功率 |
| `aiops_diagnosis_duration_seconds` | mode, status | 端到端诊断延迟 |
| `aiops_diagnosis_in_progress` | mode | 当前运行数 |
| `aiops_agent_tool_calls_total` | tool, status | 工具可靠性 |
| `aiops_agent_tool_duration_seconds` | tool, status | 工具延迟 |
| `aiops_llm_calls_total` | model, status | 模型调用可靠性 |
| `aiops_llm_call_duration_seconds` | model, status | 模型调用延迟 |
| `aiops_llm_tokens_total` | model, direction | token 吞吐与成本代理 |

指标刻意排除 `task_id`、`trace_id`、query、service 实例等高基数 label；这些字段属于
Trace、日志和 Postgres 审计库，避免 Prometheus cardinality 爆炸。

## Agent 真实数据源

配置后，`MetricAgent` 优先使用 `prom_query` / `prom_query_range`；`LogAgent` 优先使用
`loki_label_values` / `loki_query_range`，再用 RAG 中的日志模板和 SOP 解释观测结果。

```env
PROMETHEUS_URL=https://PROMETHEUS_HOST
PROMETHEUS_TENANT_ID=TENANT
PROMETHEUS_BEARER_TOKEN=TOKEN
LOKI_URL=https://LOKI_HOST
LOKI_TENANT_ID=TENANT
LOKI_BEARER_TOKEN=TOKEN
```

典型工具输入：

```text
prom_query_range('sum by (service) (rate(http_requests_total{status=~"5.."}[5m]))', 1800, 30)
loki_query_range('{service="checkout"} |~ "ERROR|Exception|timeout"', 1800, 100)
```

查询窗口上限为 24 小时，日志条数受 `LOKI_MAX_ENTRIES` 限制，防止一次 Agent 调用把
Loki 和 LLM context 打满。`LOKI_REDACT_SECRETS=true` 默认在日志进入 LLM 前遮盖
Bearer、JWT、password、token、api_key 和 secret；LogAgent 同时把日志视为不可信证据，
忽略其中可能混入的指令文本。

## 告警与 SLO

`deploy/observability/alerts.yml` 已包含：

- 10 分钟诊断失败率 > 10%；
- 诊断 P95 > 120 秒；
- Kafka lag > 20；
- DLQ 非空；
- Tool 调用失败率 > 15%。

这些是本地基线，不是所有业务的固定 SLO。生产部署应根据任务量调整窗口/阈值，接入
Alertmanager，并给 Prometheus/Loki/Tempo 使用对象存储、鉴权、TLS、HA、副本和容量策略。
Compose 中 Alloy 通过只读 Docker socket 采日志，适合单机验收；Kubernetes 应替换为
DaemonSet/Operator 采集，并限制 RBAC 和租户边界。

## Rollback

业务旁路开关：

```env
OBSERVABILITY_ENABLED=false
METRICS_ENABLED=false
OTEL_ENABLED=false
PROMETHEUS_URL=
LOKI_URL=
```

停止本地观测栈但保留业务数据卷：

```bash
docker compose --profile observability stop grafana prometheus loki tempo otel-collector alloy
```
