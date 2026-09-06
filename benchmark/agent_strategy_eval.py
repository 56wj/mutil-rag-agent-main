#!/usr/bin/env python3
"""同一固定故障集上的 fast/deep Agent 策略评测。

两种执行方式：
  local    本地逐条跑 fast 与 deep，生成成对 JSON 报告；启用 Langfuse 时每条诊断仍会追踪。
  langfuse 把固定集 upsert 到 Langfuse Dataset，再各跑一个正式 Experiment，UI 可直接比较。

不引入 Single-Agent 第三基线，避免扩大本轮范围。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = ROOT / "benchmark" / "agent_strategy_cases.jsonl"
REPORTS_DIR = ROOT / "benchmark" / "reports"
STRATEGIES = ("fast", "deep")


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    scenario: str
    input: dict[str, Any]
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


def load_cases(path: Path, *, limit: int | None = None) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: JSON 无效: {exc}") from exc
        case_id = str(item.get("id") or "").strip()
        query = str((item.get("input") or {}).get("query") or "").strip()
        if not case_id or not query:
            raise ValueError(f"{path}:{line_no}: id 和 input.query 必填")
        if case_id in seen:
            raise ValueError(f"{path}:{line_no}: case id 重复: {case_id}")
        seen.add(case_id)
        cases.append(
            EvaluationCase(
                id=case_id,
                scenario=str(item.get("scenario") or "unknown"),
                input={"query": query},
                expected_output=dict(item.get("expected_output") or {}),
                metadata=dict(item.get("metadata") or {}),
            )
        )
        if limit and len(cases) >= limit:
            break
    if not cases:
        raise ValueError(f"评测集为空: {path}")
    return cases


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def keyword_group_recall(text: str, groups: Sequence[Sequence[str]]) -> float:
    """组间 AND、组内 OR：返回命中的独立知识点比例。"""
    if not groups:
        return 1.0
    normalized = _normalize_text(text)
    hit = 0
    for group in groups:
        terms = [_normalize_text(term) for term in group if _normalize_text(term)]
        if terms and any(term in normalized for term in terms):
            hit += 1
    return round(hit / len(groups), 4)


def score_output(
    output: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    input_text: str = "",
) -> dict[str, float]:
    # A graph may deliberately degrade to a human-readable fallback report.  That is
    # useful for the API, but it is not a successful benchmark sample.  Gate every
    # quality metric on the validated run status so an auth/provider failure cannot
    # receive recall credit merely by echoing keywords from the incident input.
    if output.get("status") != "succeeded":
        return {
            "diagnosis_success": 0.0,
            "root_cause_recall": 0.0,
            "evidence_recall": 0.0,
            "remediation_recall": 0.0,
        }

    report = str(output.get("report") or "")
    if input_text:
        report = report.replace(input_text, "")
    evidence = output.get("evidence") or []
    rca = output.get("rca") or {}
    searchable = "\n".join(
        [
            report,
            json.dumps(evidence, ensure_ascii=False),
            json.dumps(rca, ensure_ascii=False),
        ]
    )
    return {
        "diagnosis_success": (
            1.0 if output.get("status") == "succeeded" and report.strip() else 0.0
        ),
        "root_cause_recall": keyword_group_recall(
            searchable, expected.get("root_cause_groups") or []
        ),
        "evidence_recall": keyword_group_recall(
            searchable, expected.get("evidence_groups") or []
        ),
        "remediation_recall": keyword_group_recall(
            searchable, expected.get("remediation_groups") or []
        ),
    }


def detect_run_failures(
    *,
    report: str,
    evidence: Sequence[Mapping[str, Any]],
    rca: Mapping[str, Any],
    usage: Mapping[str, Any],
    event_errors: Sequence[str],
) -> list[str]:
    """Reject completed-but-invalid fallback output before it pollutes comparisons."""
    reasons = [str(item) for item in event_errors if str(item).strip()]
    searchable = _normalize_text(
        "\n".join(
            [
                report,
                json.dumps(list(evidence), ensure_ascii=False),
                json.dumps(dict(rca), ensure_ascii=False),
            ]
        )
    )
    provider_failure_markers = (
        "authenticationerror",
        "authentication fails",
        "invalid api key",
        "incorrect api key",
        "permissiondeniederror",
    )
    if any(marker in searchable for marker in provider_failure_markers):
        reasons.append("model_provider_authentication_failed")

    if str(rca.get("via") or "").lower() == "empty":
        reasons.append("rca_has_no_candidates")

    total_tokens = int(usage.get("total_tokens") or 0)
    tool_calls = int(usage.get("tool_calls") or 0)
    if "执行失败:" in report and total_tokens == 0 and tool_calls == 0:
        reasons.append("fallback_report_without_successful_llm_or_tool_call")

    # Stable de-duplication keeps JSON reports diff-friendly.
    return list(dict.fromkeys(reasons))


async def run_case(
    case: EvaluationCase,
    strategy: str,
    *,
    experiment_name: str,
    run_id: str,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy 必须为 fast/deep，收到: {strategy}")
    from app.orchestration.diagnosis_runner import run_diagnosis_graph

    report = ""
    evidences: list[dict[str, Any]] = []
    rca: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    errors: list[str] = []
    trace_id = ""
    effective_strategy = ""
    completed = False
    started = time.perf_counter()
    session_id = f"eval-{run_id}-{strategy}-{case.id}"

    async for event in run_diagnosis_graph(
        case.input["query"],
        session_id=session_id,
        diagnosis_mode=strategy,
        cache_reports=False,
        experience_recall_enabled=False,
        persist_experience=False,
        trace_metadata={
            "experiment_name": experiment_name,
            "experiment_run_id": run_id,
            "case_id": case.id,
            "scenario": case.scenario,
            "strategy": strategy,
        },
    ):
        event_type = str(event.get("type") or "")
        stage = str(event.get("stage") or "")
        data = dict(event.get("data") or {})
        if event_type == "mode_selected":
            effective_strategy = str(data.get("effective_mode") or "")
        elif event_type == "report" and data.get("report"):
            report = str(data["report"])
        elif event_type == "evidence":
            evidences.append(data)
        elif event_type == "rca" and isinstance(data.get("rca"), dict):
            rca = dict(data["rca"])
        elif event_type == "progress" and stage == "stats":
            usage = data
        elif event_type == "error":
            errors.append(
                str(event.get("message") or data.get("error_type") or "unknown")
            )
        elif event_type == "complete":
            completed = True
            trace_id = str(data.get("langfuse_trace_id") or "")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if effective_strategy != strategy:
        errors.append(
            f"strategy mismatch: requested={strategy}, effective={effective_strategy or 'missing'}"
        )
    failures = detect_run_failures(
        report=report,
        evidence=evidences,
        rca=rca,
        usage=usage,
        event_errors=errors,
    )
    output: dict[str, Any] = {
        "case_id": case.id,
        "scenario": case.scenario,
        "strategy": strategy,
        "effective_strategy": effective_strategy,
        "status": (
            "succeeded"
            if completed and report and effective_strategy == strategy and not failures
            else "failed"
        ),
        "report": report,
        "evidence": evidences,
        "rca": rca,
        "usage": usage,
        "elapsed_ms": elapsed_ms,
        "langfuse_trace_id": trace_id,
        "errors": failures,
    }
    output["scores"] = score_output(
        output,
        case.expected_output,
        input_text=case.input["query"],
    )
    return output


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"runs": 0}
    score_names = sorted({name for row in rows for name in (row.get("scores") or {})})
    elapsed = [float(row.get("elapsed_ms") or 0) for row in rows]
    tokens = [float((row.get("usage") or {}).get("total_tokens") or 0) for row in rows]
    tools = [float((row.get("usage") or {}).get("tool_calls") or 0) for row in rows]
    return {
        "runs": len(rows),
        "scores": {
            name: round(
                statistics.fmean(
                    float((row.get("scores") or {}).get(name, 0)) for row in rows
                ),
                4,
            )
            for name in score_names
        },
        "latency_ms": {
            "mean": round(statistics.fmean(elapsed), 1),
            "p50": round(statistics.median(elapsed), 1),
            "p95": round(_percentile(elapsed, 0.95), 1),
        },
        "mean_total_tokens": round(statistics.fmean(tokens), 1),
        "mean_tool_calls": round(statistics.fmean(tools), 2),
    }


def build_report(
    cases_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_name: str,
    run_id: str,
) -> dict[str, Any]:
    by_strategy = {
        strategy: summarize([row for row in rows if row.get("strategy") == strategy])
        for strategy in STRATEGIES
    }
    fast = by_strategy["fast"]
    deep = by_strategy["deep"]
    score_delta = {
        name: round(
            float((deep.get("scores") or {}).get(name, 0))
            - float((fast.get("scores") or {}).get(name, 0)),
            4,
        )
        for name in sorted(
            set((fast.get("scores") or {})) | set((deep.get("scores") or {}))
        )
    }
    return {
        "mode": "agent_strategy",
        "strategies": list(STRATEGIES),
        "experiment_name": experiment_name,
        "run_id": run_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(cases_path),
        "dataset_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "cases": len({str(row.get("case_id")) for row in rows}),
        "summary": by_strategy,
        "deep_minus_fast": {
            "scores": score_delta,
            "mean_latency_ms": round(
                float((deep.get("latency_ms") or {}).get("mean", 0))
                - float((fast.get("latency_ms") or {}).get("mean", 0)),
                1,
            ),
            "mean_total_tokens": round(
                float(deep.get("mean_total_tokens") or 0)
                - float(fast.get("mean_total_tokens") or 0),
                1,
            ),
            "mean_tool_calls": round(
                float(deep.get("mean_tool_calls") or 0)
                - float(fast.get("mean_tool_calls") or 0),
                2,
            ),
        },
        "details": list(rows),
    }


async def run_local(args: argparse.Namespace) -> Path:
    # CLI benchmark 不经过 FastAPI/Worker lifespan；显式初始化 MCP，避免评测时
    # 悄悄退化成“仅本地工具”，使结果与真实 API/Worker 运行路径不一致。
    from app.core.mcp_client import mcp_client_manager
    from app.observability.langfuse_client import flush_langfuse

    cases_path = Path(args.cases).resolve()
    cases = load_cases(cases_path, limit=args.limit)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    rows: list[dict[str, Any]] = []
    await mcp_client_manager.connect(fail_silently=True)
    print(
        f"[benchmark] MCP initialized={mcp_client_manager.is_connected} "
        f"tools={len(mcp_client_manager.tools)}",
        flush=True,
    )
    try:
        for index, case in enumerate(cases, 1):
            for strategy in STRATEGIES:
                print(f"[{index}/{len(cases)}] {case.id} strategy={strategy}", flush=True)
                row = await run_case(
                    case,
                    strategy,
                    experiment_name=args.experiment_name,
                    run_id=run_id,
                )
                rows.append(row)
                print(
                    f"  status={row['status']} latency={row['elapsed_ms']}ms "
                    f"scores={json.dumps(row['scores'], ensure_ascii=False)}",
                    flush=True,
                )

        report = build_report(
            cases_path,
            rows,
            experiment_name=args.experiment_name,
            run_id=run_id,
        )
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output = (
            Path(args.output).resolve()
            if args.output
            else REPORTS_DIR / f"agent_strategy_{run_id}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        print(f"report={output}")
        return output
    finally:
        flush_langfuse()
        await mcp_client_manager.close()


def _evaluation_factory(metric: str):
    def evaluator(*, output: Mapping[str, Any], **_: Any):
        from langfuse import Evaluation

        scores = output.get("scores") if isinstance(output, Mapping) else {}
        value = float((scores or {}).get(metric, 0.0))
        return Evaluation(name=metric, value=value)

    evaluator.__name__ = f"evaluate_{metric}"
    return evaluator


def _experiment_task(strategy: str, experiment_name: str, run_id: str):
    async def task(*, item: Any, **_: Any) -> dict[str, Any]:
        case = EvaluationCase(
            id=str(getattr(item, "id", "case")),
            scenario=str(
                (getattr(item, "metadata", None) or {}).get("scenario") or "unknown"
            ),
            input=dict(getattr(item, "input", None) or {}),
            expected_output=dict(getattr(item, "expected_output", None) or {}),
            metadata=dict(getattr(item, "metadata", None) or {}),
        )
        return await run_case(
            case,
            strategy,
            experiment_name=experiment_name,
            run_id=run_id,
        )

    return task


def run_langfuse_experiments(args: argparse.Namespace) -> None:
    from app.observability.langfuse_client import flush_langfuse, get_langfuse_client

    client = get_langfuse_client()
    if client is None:
        raise RuntimeError(
            "请设置 LANGFUSE_ENABLED=true 及 LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY"
        )

    cases = load_cases(Path(args.cases).resolve(), limit=args.limit)
    try:
        client.create_dataset(
            name=args.dataset_name,
            description="Multi-Agent AIOps fast/deep paired diagnosis evaluation",
            metadata={
                "strategies": list(STRATEGIES),
                "source": "benchmark/agent_strategy_cases.jsonl",
            },
        )
    except Exception as exc:
        # create_dataset 非幂等；已存在时后续 item upsert + get_dataset 仍是正确路径。
        print(f"dataset create skipped: {type(exc).__name__}: {exc}")

    for case in cases:
        client.create_dataset_item(
            dataset_name=args.dataset_name,
            id=case.id,
            input=case.input,
            expected_output=case.expected_output,
            metadata={"scenario": case.scenario, **case.metadata},
        )
    dataset = client.get_dataset(args.dataset_name)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    evaluators = [
        _evaluation_factory(name)
        for name in (
            "diagnosis_success",
            "root_cause_recall",
            "evidence_recall",
            "remediation_recall",
        )
    ]
    for strategy in STRATEGIES:
        result = dataset.run_experiment(
            name=args.experiment_name,
            run_name=f"{run_id}-{strategy}",
            description=f"AIOps diagnosis strategy={strategy}; paired dataset={args.dataset_name}",
            task=_experiment_task(strategy, args.experiment_name, run_id),
            evaluators=evaluators,
            max_concurrency=max(1, args.concurrency),
            metadata={"strategy": strategy, "run_id": run_id},
        )
        print(result.format())
    flush_langfuse()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="fast/deep Agent diagnosis paired evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("local", "langfuse"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--cases", default=str(DEFAULT_CASES))
        cmd.add_argument("--limit", type=int, default=None)
        cmd.add_argument("--run-id", default="")
        cmd.add_argument("--experiment-name", default="aiops-fast-vs-deep")
    local = sub.choices["local"]
    local.add_argument("--output", default="")
    remote = sub.choices["langfuse"]
    remote.add_argument("--dataset-name", default="aiops/fast-deep-diagnosis")
    remote.add_argument("--concurrency", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "local":
        asyncio.run(run_local(args))
    else:
        run_langfuse_experiments(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
