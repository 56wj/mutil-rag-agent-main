"""Parent/Child hierarchical chunking coarse-sweep evaluation.

Design invariant: this runner never mutates ``app.config.settings`` and never writes to the
production Milvus collection. Every sweep point gets an isolated collection whose name contains
the frozen corpus fingerprint and configuration ID.

Examples:
  python benchmark/parent_child_eval.py plan
  python benchmark/parent_child_eval.py run --rebuild
  python benchmark/parent_child_eval.py run --configs M1,M2,M3 --with-ragas
  python benchmark/parent_child_eval.py report --run-dir benchmark/parent_child/runs/<run>
  python benchmark/parent_child_eval.py cleanup --run-dir benchmark/parent_child/runs/<run>
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
from importlib import metadata as package_metadata
import json
import math
import re
import statistics
import sqlite3
import struct
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "benchmark" / "parent_child"
CONFIG_PATH = ASSET_DIR / "configs.json"
HARDCASE_PATH = ASSET_DIR / "hardcases_20.jsonl"
LINKS_PATH = ASSET_DIR / "base_case_links.json"
BASE_RETRIEVAL_PATH = ROOT / "benchmark" / "retrieval_rk_50.jsonl"
BASE_RAGAS_PATH = ROOT / "benchmark" / "ragas_qa_50.jsonl"
DEFAULT_RUNS_DIR = ASSET_DIR / "runs"
EVAL_COLLECTION_PREFIX = "pc_chunk_eval_"
SCHEMA_VERSION = 2

from benchmark.parent_child.evidence import score_evidence, validate_evidence


@dataclass(frozen=True)
class SweepPoint:
    id: str
    scale: str
    parent_size: int
    child_size: int
    child_overlap: int
    parent_overlap: int
    parent_header_depth: int

    @property
    def ratio(self) -> float:
        return self.parent_size / self.child_size


@dataclass(frozen=True)
class RetrievalScore:
    hit: float
    mrr: float
    recall: float
    first_rank: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_sweep_points(path: Path = CONFIG_PATH) -> list[SweepPoint]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    depth = int(payload.get("parent_header_depth", 3))
    if depth != 3:
        raise ValueError("本轮固定生产 H1/H2/H3 标题边界，parent_header_depth 必须为 3")
    points: list[SweepPoint] = []
    seen: set[str] = set()
    for raw in payload.get("configurations") or []:
        point = SweepPoint(
            id=str(raw["id"]),
            scale=str(raw["scale"]),
            parent_size=int(raw["parent_size"]),
            child_size=int(raw["child_size"]),
            child_overlap=int(raw.get("child_overlap") or 0),
            parent_overlap=int(raw.get("parent_overlap") or 0),
            parent_header_depth=depth,
        )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,15}", point.id):
            raise ValueError(f"非法配置 ID: {point.id!r}")
        if point.id in seen:
            raise ValueError(f"重复配置 ID: {point.id}")
        if point.scale not in {"small", "medium", "large"}:
            raise ValueError(f"{point.id}: scale 必须是 small/medium/large")
        if point.child_overlap != 0 or point.parent_overlap != 0:
            raise ValueError(f"{point.id}: coarse sweep 的 parent/child overlap 必须固定为 0")
        if point.parent_size < 500 or point.child_size < 80:
            raise ValueError(f"{point.id}: Parent >= 500、Child >= 80，避免 splitter 隐式钳制")
        if point.parent_size <= point.child_size or point.ratio < 3.0:
            raise ValueError(f"{point.id}: Parent 必须明显大于 Child（至少 3x）")
        seen.add(point.id)
        points.append(point)
    if not 5 <= len(points) <= 8:
        raise ValueError(f"coarse sweep 应包含 5-8 组，实际 {len(points)} 组")
    if {point.scale for point in points} != {"small", "medium", "large"}:
        raise ValueError("coarse sweep 必须覆盖 small/medium/large")
    return points


def _case_key(case_id: str) -> str:
    return re.sub(r"^(?:rk|ragas)-", "", case_id)


def load_eval_cases() -> list[dict[str, Any]]:
    """合并现有 50 道 RAG Eval 与新增 20 道 hardcase。

    两套历史题目的编号不是语义对齐的。使用人工审核的 base_case_links.json 关联
    RAGAS question/ground_truth 与对应 retrieval gold，原始两份 50 题不作修改。
    """
    retrieval = {_case_key(str(row["id"])): row for row in load_jsonl(BASE_RETRIEVAL_PATH)}
    ragas = {_case_key(str(row["id"])): row for row in load_jsonl(BASE_RAGAS_PATH)}
    links = json.loads(LINKS_PATH.read_text(encoding="utf-8"))["links"]
    if len(retrieval) != 50 or len(ragas) != 50 or set(links) != {row["id"] for row in ragas.values()}:
        raise ValueError("现有 50 题或语义映射不完整")

    cases: list[dict[str, Any]] = []
    for key, qa in ragas.items():
        gold = retrieval[_case_key(links[qa["id"]])]
        if qa["scenario"] != gold["scenario"]:
            raise ValueError(f"跨场景 gold 映射: {qa['id']}")
        cases.append(
            {
                "id": str(qa["id"]),
                "dataset": "base_50",
                "category": "base",
                "scenario": qa.get("scenario"),
                "query": qa["question"],
                "ground_truth": qa["ground_truth"],
                "gold_from_id": gold["id"],
                "relevant": gold.get("relevant") or [],
                "relevant_groups": gold.get("relevant_groups"),
            }
        )

    hardcases = load_jsonl(HARDCASE_PATH)
    expected_categories = {
        "exact_fact",
        "cross_paragraph",
        "multi_step",
        "similar_section_distractor",
    }
    counts: dict[str, int] = defaultdict(int)
    for row in hardcases:
        category = str(row.get("category") or "")
        counts[category] += 1
        if not row.get("relevant") and not row.get("relevant_groups"):
            raise ValueError(f"hardcase {row.get('id')} 缺少 relevant/relevant_groups")
        cases.append({**row, "dataset": "hard_20"})
    if len(hardcases) != 20 or set(counts) != expected_categories or any(v != 5 for v in counts.values()):
        raise ValueError(f"hardcase 必须四类各 5 题，实际 {dict(counts)}")
    return cases


def collect_corpus_files() -> list[tuple[Path, str]]:
    """与 ``scripts/ingest_kb_corpus.py`` 保持同一文件范围和 source 命名。"""
    files: list[tuple[Path, str]] = []
    sop_dir = ROOT / "docs" / "sop"
    for name in ("redis_oncall_sop.md", "mysql_oncall_sop.md", "common_alerts.md"):
        path = sop_dir / name
        if path.exists():
            files.append((path, name))
    kb_dir = ROOT / "data" / "kb_corpus"
    if kb_dir.exists():
        for path in sorted(kb_dir.rglob("*.md")):
            files.append((path, path.relative_to(kb_dir).as_posix()))
    if not files:
        raise ValueError("语料为空：docs/sop 与 data/kb_corpus 中未找到评测语料")
    return files


def current_eval_input_hashes() -> dict[str, str]:
    return {
        "base_retrieval_sha256": hashlib.sha256(BASE_RETRIEVAL_PATH.read_bytes()).hexdigest(),
        "base_ragas_sha256": hashlib.sha256(BASE_RAGAS_PATH.read_bytes()).hexdigest(),
        "hardcases_sha256": hashlib.sha256(HARDCASE_PATH.read_bytes()).hexdigest(),
        "configs_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "case_links_sha256": hashlib.sha256(LINKS_PATH.read_bytes()).hexdigest(),
    }


def verify_eval_input_hashes(manifest: dict[str, Any]) -> None:
    expected = manifest.get("eval_inputs") or {}
    actual = current_eval_input_hashes()
    if expected != actual:
        raise RuntimeError(
            "评测配置/题集与冻结 snapshot manifest 不一致；创建新 run，避免跨版本续跑。"
        )


def create_frozen_snapshot(run_dir: Path) -> dict[str, Any]:
    """冻结语料内容到 run 目录，而不只是记录活动文件路径。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "configs.snapshot.json").write_bytes(CONFIG_PATH.read_bytes())
    (run_dir / "eval_cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in load_eval_cases()), encoding="utf-8"
    )
    records: list[dict[str, Any]] = []
    corpus_path = run_dir / "corpus_snapshot.jsonl"
    combined = hashlib.sha256()
    total_bytes = 0
    with corpus_path.open("w", encoding="utf-8") as handle:
        for path, source in collect_corpus_files():
            content = path.read_text(encoding="utf-8")
            raw = content.encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            relative_path = path.relative_to(ROOT).as_posix()
            record = {
                "source": source,
                "relative_path": relative_path,
                "sha256": digest,
                "bytes": len(raw),
                "content": content,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append({key: value for key, value in record.items() if key != "content"})
            combined.update(relative_path.encode("utf-8"))
            combined.update(b"\0")
            combined.update(raw)
            combined.update(b"\0")
            total_bytes += len(raw)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "snapshot_sha256": combined.hexdigest(),
        "corpus_file": corpus_path.name,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
        "eval_inputs": current_eval_input_hashes(),
        "frozen_cases_sha256": hashlib.sha256((run_dir / "eval_cases.jsonl").read_bytes()).hexdigest(),
    }
    (run_dir / "snapshot_manifest.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return snapshot


def load_frozen_snapshot(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"缺少 snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(run_dir / str(manifest["corpus_file"]))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("run schema 已更新；使用新 run，保留旧产物作历史记录")
    if len(rows) != int(manifest["file_count"]):
        raise ValueError("冻结语料文件数与 manifest 不一致")
    if ([{key: value for key, value in row.items() if key != "content"} for row in rows]
            != manifest["files"] or len({row["source"] for row in rows}) != len(rows)):
        raise ValueError("冻结语料 source/路径/元数据与 manifest 不一致")
    combined = hashlib.sha256()
    total_bytes = 0
    for row in rows:
        content = str(row["content"])
        raw = content.encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"冻结语料内容哈希不一致: {row['relative_path']}")
        combined.update(str(row["relative_path"]).encode("utf-8"))
        combined.update(b"\0")
        combined.update(raw)
        combined.update(b"\0")
        total_bytes += len(raw)
    if combined.hexdigest() != manifest["snapshot_sha256"] or total_bytes != manifest["total_bytes"]:
        raise ValueError("冻结语料总指纹不一致")
    return manifest, rows


def load_frozen_cases(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_dir / "eval_cases.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("frozen_cases_sha256"):
        raise ValueError("冻结题集指纹不一致")
    cases = load_jsonl(path)
    if len(cases) != 70 or len({row["id"] for row in cases}) != 70:
        raise ValueError("冻结题集需要 70 个唯一 case")
    return cases


def load_frozen_points(run_dir: Path, manifest: dict[str, Any]) -> list[SweepPoint]:
    path = run_dir / "configs.snapshot.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["eval_inputs"]["configs_sha256"]:
        raise ValueError("冻结配置指纹不一致")
    return load_sweep_points(path)


def freeze_record(path: Path, value: dict[str, Any]) -> None:
    """Resume checks before writes; never overwrite a different experiment contract."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"冻结条件发生变化，使用新 run: {path.name}")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def owned_collection(state: dict[str, Any], run_dir: Path, manifest: dict[str, Any],
                     stack: dict[str, Any], points: Sequence[SweepPoint]) -> str:
    point = next((p for p in points if asdict(p) == state.get("config")), None)
    if point is None:
        raise RuntimeError("collection state 配置不属于本 run")
    expected = collection_name(manifest["snapshot_sha256"], point, str(run_dir.resolve()))
    if (state.get("collection") != expected
            or state.get("snapshot_sha256") != manifest["snapshot_sha256"]
            or state.get("stack_fingerprint") != stack["fingerprint"]):
        raise RuntimeError("collection state 与本 run 的名称/快照/stack 不符")
    assert_eval_collection(expected, stack["milvus"]["production_collection"])
    return expected


def select_points(points: Sequence[SweepPoint], selected: str | None) -> list[SweepPoint]:
    if not selected:
        return list(points)
    wanted = {part.strip() for part in selected.split(",") if part.strip()}
    found = [point for point in points if point.id in wanted]
    missing = wanted - {point.id for point in found}
    if missing:
        raise ValueError(f"未知配置: {sorted(missing)}")
    return found


def require_runtime_dependencies() -> None:
    required = {
        "loguru": "loguru",
        "pymilvus": "pymilvus",
        "langchain_milvus": "langchain-milvus",
        "rank_bm25": "rank-bm25",
        "FlagEmbedding": "FlagEmbedding",
        "torch": "torch",
    }
    missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            "Parent/Child full run 缺少运行依赖: "
            + ", ".join(missing)
            + ". 先执行 pip install -r requirements.txt && pip install FlagEmbedding torch"
        )


def split_snapshot(rows: Sequence[dict[str, Any]], point: SweepPoint) -> list[Any]:
    from app.core.splitter import ChunkingConfig, split_markdown

    config = ChunkingConfig(
        parent_size=point.parent_size,
        child_size=point.child_size,
        child_overlap=point.child_overlap,
        parent_overlap=point.parent_overlap,
    ).validated()
    chunks: list[Any] = []
    for row in rows:
        chunks.extend(split_markdown(str(row["content"]), str(row["source"]), config=config))
    return chunks


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Sequence[float]) -> dict[str, float]:
    vals = [float(value) for value in values]
    return {
        "mean": statistics.fmean(vals) if vals else 0.0,
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "max": max(vals, default=0.0),
    }


def estimate_tokens(text: str) -> int:
    """无模型依赖的确定性 token 成本估算；报告中同时保留 chars，避免伪装精确值。"""
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    non_cjk = max(0, len(text) - cjk)
    return cjk + math.ceil(non_cjk / 4)


def chunk_statistics(chunks: Sequence[Any], embedding_dim: int) -> dict[str, Any]:
    parents: dict[str, str] = {}
    children_per_parent: dict[str, int] = defaultdict(int)
    child_chars: list[int] = []
    child_bytes = parent_metadata_bytes = metadata_bytes = 0
    for chunk in chunks:
        meta = chunk.metadata or {}
        source = str(meta.get("source") or "")
        parent_id = str(meta.get("parent_id") or "")
        key = f"{source}\0{parent_id}"
        parent = str(meta.get("parent_content") or "")
        parents.setdefault(key, parent)
        children_per_parent[key] += 1
        content = str(chunk.page_content)
        child_chars.append(len(content))
        child_bytes += len(content.encode("utf-8"))
        parent_metadata_bytes += len(parent.encode("utf-8"))
        metadata_bytes += len(canonical_json(meta).encode("utf-8"))
    parent_chars = [len(parent) for parent in parents.values()]
    vector_bytes = len(chunks) * max(1, embedding_dim) * 4
    # HNSW graph/allocator overhead depends on Milvus internals; this is a comparable payload floor.
    comparable_payload = vector_bytes + child_bytes + metadata_bytes
    return {
        "parent_count": len(parents),
        "child_count": len(chunks),
        "children_per_parent": distribution(list(children_per_parent.values())),
        "actual_parent_chars": distribution(parent_chars),
        "embedded_child_chars": distribution(child_chars),
        "vector_payload_bytes": vector_bytes,
        "child_text_bytes": child_bytes,
        "duplicated_parent_metadata_bytes": parent_metadata_bytes,
        "metadata_bytes": metadata_bytes,
        "comparable_payload_bytes": comparable_payload,
        "comparable_payload_mib": comparable_payload / 1024 / 1024,
        "parent_partition_fingerprint": sha256_text(canonical_json(sorted(parents.items()))),
        "note": "comparable_payload 是 vector+child+metadata 的下界，不含 HNSW 图、segment/allocator 开销",
    }


def is_relevant(hit: dict[str, Any], gold: dict[str, Any]) -> bool:
    source_ok = not gold.get("source") or hit.get("source") == gold.get("source")
    chapter_need = str(gold.get("chapter_contains") or "")
    chapter_ok = not chapter_need or chapter_need in str(hit.get("chapter") or "")
    return source_ok and chapter_ok


def score_hits(
    hits: Sequence[dict[str, Any]],
    relevant: Sequence[dict[str, Any]],
    k: int,
    relevant_groups: Sequence[Sequence[dict[str, Any]]] | None = None,
) -> RetrievalScore:
    top = list(hits[:k])
    groups = list(relevant_groups) if relevant_groups else ([list(relevant)] if relevant else [])
    matched: set[int] = set()
    first_rank: int | None = None
    for rank, hit in enumerate(top, 1):
        for group_index, alternatives in enumerate(groups):
            if group_index in matched:
                continue
            if any(is_relevant(hit, gold) for gold in alternatives):
                matched.add(group_index)
                first_rank = first_rank or rank
    return RetrievalScore(
        hit=1.0 if first_rank else 0.0,
        mrr=1.0 / first_rank if first_rank else 0.0,
        recall=len(matched) / len(groups) if groups else 0.0,
        first_rank=first_rank,
    )


def capture_stack() -> dict[str, Any]:
    from app.config import settings
    from app.core.vector_store import MILVUS_INDEX_PARAMS, milvus_search_params
    from app.core.reranker import _resolve_local_device
    from app.rag.retrieval import _CHILD_OVERFETCH
    import platform

    embedding_model = (
        settings.ollama_embedding_model
        if settings.embedding_provider == "ollama"
        else settings.dashscope_embedding_model
    )
    embedding_dim = (
        settings.ollama_embedding_dim
        if settings.embedding_provider == "ollama"
        else settings.dashscope_embedding_dim
    )
    model_path = Path(str(settings.rag_rerank_model)).expanduser()
    local_model_identity = None
    if model_path.is_dir():
        weights = model_path / "model.safetensors"
        checksum = model_path / "model.safetensors.sha256"
        if not weights.exists() or not checksum.exists():
            raise RuntimeError("本地 reranker 目录缺少权重或 model.safetensors.sha256")
        local_model_identity = {
            "path": str(model_path.resolve()),
            "weights_bytes": weights.stat().st_size,
            "weights_sha256": checksum.read_text(encoding="utf-8").strip().split()[0],
        }
    stack = {
        "embedding": {
            "provider": settings.embedding_provider,
            "model": embedding_model,
            "dimension": embedding_dim,
            "base_url": (
                settings.ollama_base_url
                if settings.embedding_provider == "ollama"
                else settings.dashscope_base_url
            ),
            "batch_size": (
                settings.ollama_embedding_batch_size
                if settings.embedding_provider == "ollama"
                else 10
            ),
            "timeout_sec": (
                settings.ollama_embedding_timeout_sec
                if settings.embedding_provider == "ollama"
                else None
            ),
        },
        "milvus": {
            "host": settings.milvus_host,
            "port": settings.milvus_port,
            "timeout_ms": settings.milvus_timeout_ms,
            "production_collection": settings.milvus_collection,
            "index_params": MILVUS_INDEX_PARAMS,
            "search_params": milvus_search_params(),
        },
        "hybrid": {
            "enabled": settings.rag_hybrid_enabled,
            "bm25_weight": settings.rag_hybrid_bm25_weight,
            "vector_weight": 1.0 - settings.rag_hybrid_bm25_weight,
            "rrf_k": max(1, int(settings.rag_hybrid_rrf_k or 60)),
        },
        "reranker": {
            "enabled": settings.rag_rerank_enabled,
            "provider": settings.rag_rerank_provider,
            "model": settings.rag_rerank_model,
            "backend": settings.rag_local_rerank_backend,
            "device": settings.rag_local_rerank_device,
            "resolved_device": _resolve_local_device(),
            "use_parent_context": settings.rag_rerank_use_parent_context,
            "parent_max_chars": settings.rag_rerank_parent_max_chars,
            "max_length": settings.rag_local_rerank_max_length,
            "batch_size": settings.rag_local_rerank_batch_size,
            "timeout_sec": settings.rag_rerank_timeout_sec,
            "local_model_identity": local_model_identity,
        },
        "retrieval": {
            "retrieve_k": settings.rag_retrieve_k,
            "final_top_k": settings.rag_top_k,
            "child_overfetch": _CHILD_OVERFETCH,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "code_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "app/core/splitter.py", "app/core/vector_store.py", "app/core/hybrid_retriever.py",
                "app/core/embedding.py", "app/core/reranker.py", "app/rag/retrieval.py",
                "benchmark/parent_child_eval.py", "benchmark/parent_child/evidence.py",
            )
        },
        "packages": {
            name: package_metadata.version(name)
            for name in ("langchain-core", "langchain-text-splitters", "langchain-milvus", "pymilvus",
                         "rank-bm25", "FlagEmbedding", "torch")
        },
    }
    stack["fingerprint"] = sha256_text(canonical_json(stack))
    return stack


def validate_required_stack(stack: dict[str, Any]) -> None:
    failures: list[str] = []
    embedding = stack["embedding"]
    reranker = stack["reranker"]
    if embedding["provider"] != "ollama" or "bge-m3" not in embedding["model"].lower():
        failures.append(f"embedding 不是当前要求的 Ollama BGE-M3: {embedding}")
    if not stack["hybrid"]["enabled"]:
        failures.append("Hybrid Search 未启用")
    if not reranker["enabled"]:
        failures.append("Reranker 未启用")
    if reranker["provider"] != "local" or "bge-reranker" not in reranker["model"].lower():
        failures.append(f"reranker 不是当前要求的本地 BGE Reranker: {reranker}")
    retrieval = stack["retrieval"]
    if retrieval["final_top_k"] < 1 or retrieval["retrieve_k"] < retrieval["final_top_k"] * retrieval["child_overfetch"]:
        failures.append("Retrieve K 须覆盖既有 Final Top-K × 3 overfetch；请核对现有配置，不自动调参")
    if failures:
        raise RuntimeError("评测栈与固定实验条件不一致:\n- " + "\n- ".join(failures))


def collection_name(snapshot_sha: str, point: SweepPoint, run_key: str = "") -> str:
    suffix = sha256_text(canonical_json(asdict(point)) + run_key)[:12]
    name = f"{EVAL_COLLECTION_PREFIX}{snapshot_sha[:12]}_{point.id.lower()}_{suffix}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"非法 Milvus collection name: {name}")
    return name


def assert_eval_collection(name: str, production_collection: str) -> None:
    if name == production_collection or not name.startswith(EVAL_COLLECTION_PREFIX):
        raise RuntimeError(f"生产 collection 保护触发: {name!r}")


def collection_entity_count(client: Any, name: str) -> int | None:
    """读取 Milvus collection 行数；不存在时返回 None。"""
    if not client.has_collection(name):
        return None
    stats = client.get_collection_stats(collection_name=name)
    for key in ("row_count", "num_entities", "rows"):
        if key in stats:
            return int(stats[key])
    raise RuntimeError(f"Milvus collection stats 缺少 row count: {name} -> {stats}")


def wait_for_entity_count(client: Any, name: str, expected: int, timeout_sec: float = 120.0) -> int:
    deadline = time.monotonic() + timeout_sec
    last: int | None = None
    while time.monotonic() < deadline:
        last = collection_entity_count(client, name)
        if last == expected:
            return last
        time.sleep(0.5)
    raise RuntimeError(
        f"Milvus entity count 未稳定到预期值: collection={name}, expected={expected}, actual={last}"
    )


def verify_vector_index(client: Any, name: str, expected: dict[str, Any]) -> dict[str, Any]:
    for index_name in client.list_indexes(collection_name=name):
        info = client.describe_index(collection_name=name, index_name=index_name)
        if info.get("field_name") != "vector":
            continue
        for key in ("index_type", "metric_type"):
            if info.get(key) != expected[key]:
                raise RuntimeError(f"{name}: index {key} 漂移: {info}")
        params = info.get("params") or info
        for key, value in expected["params"].items():
            if str(params.get(key)) != str(value):
                raise RuntimeError(f"{name}: index {key} 漂移: {info}")
        return info
    raise RuntimeError(f"{name}: vector index 缺失")


def verify_vector_dimension(client: Any, name: str, expected: int) -> None:
    description = client.describe_collection(collection_name=name)
    dimension = next((field.get("params", {}).get("dim") for field in description["fields"]
                      if field.get("name") == "vector"), None)
    if dimension is None or int(dimension) != expected:
        raise RuntimeError(f"{name}: embedding dimension 漂移: expected={expected}, actual={dimension}")


class BM25Corpus:
    def __init__(self, docs: Sequence[Any]) -> None:
        from rank_bm25 import BM25Okapi
        from app.core.hybrid_retriever import _tokenize

        self.docs = list(docs)
        self._tokenize = _tokenize
        self._bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in self.docs])

    def search(self, query: str, k: int) -> list[tuple[Any, float]]:
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
        return [(self.docs[index], float(score)) for index, score in ranked if score > 0]


class DocumentEmbeddingCache:
    """Persistent, exact-text cache for ingestion only; query latency always embeds live.

    Reusing identical child vectors removes 22k duplicate BGE-M3 computations across paired
    configs without changing vectors, index parameters, queries, or measured retrieval latency.
    """
    def __init__(self, path: Path, base: Any, identity: str, dimension: int) -> None:
        self.base, self.dimension = base, dimension
        self.hits = self.misses = 0
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        existing = self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if existing and existing[0] != identity:
            raise RuntimeError("document embedding cache 与固定 embedding stack 不一致")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('identity', ?)", (identity,))
        self.db.commit()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}
        for index, value in enumerate(texts):
            key = self._key(value)
            row = self.db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
            if row:
                output[index] = list(struct.unpack(f"<{self.dimension}f", row[0]))
                self.hits += 1
            elif key in missing:
                missing[key][1].append(index)
            else:
                missing[key] = (value, [index])
        if missing:
            keys = list(missing)
            vectors = self.base.embed_documents([missing[key][0] for key in keys])
            if len(vectors) != len(keys) or any(len(vector) != self.dimension for vector in vectors):
                raise RuntimeError("BGE-M3 文档 embedding 数量或维度不一致")
            for key, vector in zip(keys, vectors):
                packed = struct.pack(f"<{self.dimension}f", *vector)
                self.db.execute("INSERT INTO vectors VALUES (?, ?)", (key, packed))
                unpacked = list(struct.unpack(f"<{self.dimension}f", packed))
                for index in missing[key][1]:
                    output[index] = unpacked
                self.misses += 1
            self.db.commit()
        if any(vector is None for vector in output):
            raise AssertionError("embedding cache output incomplete")
        return output  # type: ignore[return-value]

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(text)

    def snapshot(self) -> dict[str, int]:
        rows = int(self.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        return {"rows": rows, "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        self.db.close()


def _doc_key(doc: Any) -> str:
    meta = doc.metadata or {}
    raw = f"{meta.get('source', '')}\0{meta.get('chapter', '')}\0{doc.page_content}"
    return sha256_text(raw)


def rrf_fuse(
    vector_docs: Sequence[Any],
    bm25_docs: Sequence[tuple[Any, float]],
    *,
    k: int,
    rrf_k: int,
    bm25_weight: float,
) -> list[Any]:
    if not bm25_docs:
        return list(vector_docs[:k])
    scores: dict[str, float] = {}
    docs: dict[str, Any] = {}
    vector_weight = 1.0 - bm25_weight
    for rank, doc in enumerate(vector_docs, 1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + vector_weight / (rrf_k + rank)
        docs.setdefault(key, doc)
    for rank, (doc, _score) in enumerate(bm25_docs, 1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + bm25_weight / (rrf_k + rank)
        docs.setdefault(key, doc)
    return [docs[key] for key, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]


def parents_from_docs(docs: Sequence[Any], final_k: int, parent_max: int) -> tuple[list[Any], str]:
    selected: list[Any] = []
    seen: set[str] = set()
    context_parts: list[str] = []
    for doc in docs:
        meta = doc.metadata or {}
        parent_id = str(meta.get("parent_id") or "")
        # Same deduplication as app.rag.retrieval.build_context, including legacy fallback.
        key = parent_id or f"__legacy:{hash(doc.page_content)}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(doc)
        parent = str(meta.get("parent_content") or doc.page_content).strip()
        truncated = parent[:parent_max]
        if len(parent) > parent_max:
            truncated += "... (已截断)"
        context_parts.append(
            f"## 来源 {len(selected)} | {meta.get('source') or '未知'}"
            + (f" | 章节: {meta.get('chapter')}" if meta.get("chapter") else "")
            + f"\n{truncated}"
        )
        if len(selected) >= final_k:
            break
    return selected, "\n\n".join(context_parts)


def hits_meta(docs: Sequence[Any], parent_max: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        score = meta.get("rerank_score")
        output.append(
            {
                "source": str(meta.get("source") or "未知"),
                "chapter": str(meta.get("chapter") or ""),
                "parent_id": str(meta.get("parent_id") or ""),
                "score": round(float(score), 6) if score is not None else None,
                "preview": " ".join(str(doc.page_content).split())[:240],
                "parent_text": str(meta.get("parent_content") or doc.page_content).strip()[:parent_max],
                "parent_truncated": len(str(meta.get("parent_content") or doc.page_content).strip()) > parent_max,
            }
        )
    return output


async def retrieve_one(
    *,
    query: str,
    store: Any,
    bm25: BM25Corpus,
    stack: dict[str, Any],
    strict_reranker: bool,
    parent_max: int,
) -> dict[str, Any]:
    from app.core.reranker import rerank_docs

    retrieve_k = int(stack["retrieval"]["retrieve_k"])
    final_k = int(stack["retrieval"]["final_top_k"])
    rerank_top_n = final_k * int(stack["retrieval"]["child_overfetch"])

    total_start = time.perf_counter()
    vector_start = time.perf_counter()
    vector_docs = store.similarity_search(query, k=retrieve_k)
    vector_ms = (time.perf_counter() - vector_start) * 1000
    if not vector_docs:
        raise RuntimeError(f"Milvus dense 路无召回: {query}")

    fusion_start = time.perf_counter()
    sparse_docs = bm25.search(query, retrieve_k)
    candidates = rrf_fuse(
        vector_docs,
        sparse_docs,
        k=retrieve_k,
        rrf_k=int(stack["hybrid"]["rrf_k"]),
        bm25_weight=float(stack["hybrid"]["bm25_weight"]),
    )
    fusion_ms = (time.perf_counter() - fusion_start) * 1000

    rerank_start = time.perf_counter()
    rerank_applied = len(candidates) > rerank_top_n
    reranked = (
        await rerank_docs(query, candidates, top_n=rerank_top_n)
        if rerank_applied else candidates[:rerank_top_n]
    )
    rerank_ms = (time.perf_counter() - rerank_start) * 1000
    scored = sum(1 for doc in reranked if (doc.metadata or {}).get("rerank_score") is not None)
    if strict_reranker and rerank_applied and (
        scored != len(reranked) or len(reranked) != min(rerank_top_n, len(candidates))
        or any(not math.isfinite(float(doc.metadata["rerank_score"])) for doc in reranked)
    ):
        raise RuntimeError("BGE reranker 发生静默降级；停止实验，避免混入非同栈结果")

    parents, context = parents_from_docs(reranked, final_k, parent_max)
    total_ms = (time.perf_counter() - total_start) * 1000
    return {
        "hits": hits_meta(parents, parent_max),
        "contexts": [
            chunk if chunk.startswith("## 来源 ") else "## 来源 " + chunk
            for chunk in context.split("\n\n## 来源 ") if chunk
        ],
        "context": context,
        "context_chars": len(context),
        "context_estimated_tokens": estimate_tokens(context),
        "candidate_children": len(candidates),
        "reranked_children": len(reranked),
        "unique_parents": len(parents),
        "reranker_scored": scored,
        "reranker_applied": rerank_applied,
        "latency_ms": {
            "vector": vector_ms,
            "bm25_rrf": fusion_ms,
            "reranker": rerank_ms,
            "total": total_ms,
        },
    }


def aggregate_scores(details: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(float(row["retrieval_score"][metric]) for row in details)
        if details
        else 0.0
        for metric in ("hit", "mrr", "recall")
    }


def aggregate_config(
    point: SweepPoint,
    index_stats: dict[str, Any],
    details: Sequence[dict[str, Any]],
    *,
    collection: str,
    ingest_seconds: float,
    ragas_averages: dict[str, float] | None,
) -> dict[str, Any]:
    base = [row for row in details if row["dataset"] == "base_50"]
    hard = [row for row in details if row["dataset"] == "hard_20"]
    categories: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in hard}):
        subset = [row for row in hard if row["category"] == category]
        categories[category] = aggregate_scores(subset)
        categories[category]["evidence_recall"] = statistics.fmean(
            row["evidence_score"]["recall"] for row in subset
        )
        categories[category]["evidence_complete"] = statistics.fmean(
            row["evidence_score"]["complete"] for row in subset
        )

    latency = {
        stage: distribution([row["latency_ms"][stage] for row in details])
        for stage in ("vector", "bm25_rrf", "reranker", "total")
    }
    context = {
        "chars": distribution([row["context_chars"] for row in details]),
        "estimated_tokens": distribution([row["context_estimated_tokens"] for row in details]),
        "total_estimated_tokens": sum(row["context_estimated_tokens"] for row in details),
        "estimator": "CJK char + ceil(non-CJK chars / 4); comparative estimate, not tokenizer billing",
    }
    quality = {
        "overall_70": aggregate_scores(details),
        "base_50": aggregate_scores(base),
        "hard_20": aggregate_scores(hard),
        "hard_by_category": categories,
        "hard_evidence_recall": statistics.fmean(row["evidence_score"]["recall"] for row in hard),
        "hard_evidence_complete": statistics.fmean(row["evidence_score"]["complete"] for row in hard),
        "distractor_parent_fraction": statistics.fmean(
            row.get("distractor_parent_fraction", 0.0) for row in hard
            if row["category"] == "similar_section_distractor"
        ),
    }
    base_recall = quality["base_50"]["recall"]
    quality_score = (
        0.40 * base_recall
        + 0.40 * quality["hard_evidence_recall"]
        + 0.20 * quality["overall_70"]["mrr"]
    )
    quality["composite_score"] = quality_score
    if ragas_averages:
        quality["ragas"] = ragas_averages
        quality["ragas_confirmation_mean"] = statistics.fmean(ragas_averages.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(point) | {"parent_child_ratio": point.ratio},
        "collection": collection,
        "quality": quality,
        "index": index_stats | {"ingest_seconds": ingest_seconds},
        "latency_ms": latency,
        "context_cost": context,
        "case_count": len(details),
        "details": list(details),
    }


def recommend(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    best_quality = max(row["quality"]["composite_score"] for row in results)
    # A one-token mean difference is estimator noise and must not override retrieval quality.
    # First enforce production-facing quality SLOs, then choose accuracy-first; operational
    # metrics break only exact quality ties. This also makes a failed hardcase class ineligible.
    gates = {
        "base_recall": 0.99,
        "hard_evidence_recall": 0.99,
        "hard_evidence_complete": 0.99,
    }
    eligible = [
        row for row in results
        if row["quality"]["base_50"]["recall"] >= gates["base_recall"]
        and row["quality"]["hard_evidence_recall"] >= gates["hard_evidence_recall"]
        and row["quality"]["hard_evidence_complete"] >= gates["hard_evidence_complete"]
    ]
    pool = eligible or list(results)
    selected = max(
        pool,
        key=lambda row: (
            row["quality"]["composite_score"],
            -row["latency_ms"]["total"]["mean"],
            -row["latency_ms"]["total"]["p95"],
            -row["context_cost"]["estimated_tokens"]["mean"],
            -row["index"]["comparable_payload_bytes"],
        ),
    )

    return {
        "config_id": selected["config"]["id"],
        "rule": (
            "先满足 Base recall、Hard evidence recall、Hard complete 均不低于 0.99；"
            "再最大化 composite，完全同分时依次最小化平均延迟、P95、上下文和索引 payload"
        ),
        "quality_gates": gates,
        "eligible_configs": [row["config"]["id"] for row in eligible],
        "best_quality": best_quality,
        "selected_quality": selected["quality"]["composite_score"],
        "basis": {
            "base_recall": selected["quality"]["base_50"]["recall"],
            "hard_recall": selected["quality"]["hard_20"]["recall"],
            "hard_evidence_recall": selected["quality"]["hard_evidence_recall"],
            "hard_evidence_complete": selected["quality"]["hard_evidence_complete"],
            "mrr": selected["quality"]["overall_70"]["mrr"],
            "retrieval_mean_ms": selected["latency_ms"]["total"]["mean"],
            "context_tokens_mean": selected["context_cost"]["estimated_tokens"]["mean"],
            "retrieval_p95_ms": selected["latency_ms"]["total"]["p95"],
            "index_payload_mib": selected["index"]["comparable_payload_mib"],
        },
        "production_change": "仅为评测后建议；runner 未修改生产 settings 或 production collection",
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_report(payload: dict[str, Any]) -> str:
    results = payload.get("results") or []
    lines = [
        "# Parent-Child Chunking Evaluation",
        "",
        f"- 生成时间: `{payload.get('generated_at')}`",
        f"- 语料快照: `{payload.get('snapshot_sha256')}`",
        f"- 题集: 现有 50 道 RAG Eval + 20 道 chunk-sensitive hardcases",
        "- 固定项: BGE-M3、Milvus HNSW/COSINE、BM25 Hybrid、RRF、BGE Reranker、Retrieve K、Final Top-K",
        "- overlap: parent=0，child=0",
        "",
        "## 对比表",
        "",
        "| 配置 | Scale | Parent/Child(chars) | 实际 Parent P50/P95 | Children | "
        "Payload MiB | Base R | Hard evidence R/C | MRR | Exact/Cross/Multi/Distractor evidence R | "
        "Stage mean V/BM25/Rerank ms | Total mean/P95 ms | Context avg est. tokens | Quality |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        config = row["config"]
        index = row["index"]
        quality = row["quality"]
        categories = quality["hard_by_category"]
        category_text = "/".join(
            fmt(categories.get(name, {}).get("evidence_recall"))
            for name in (
                "exact_fact",
                "cross_paragraph",
                "multi_step",
                "similar_section_distractor",
            )
        )
        lines.append(
            f"| {config['id']} | {config['scale']} | {config['parent_size']}/{config['child_size']} "
            f"| {index['actual_parent_chars']['p50']:.0f}/{index['actual_parent_chars']['p95']:.0f} "
            f"| {index['child_count']} | {index['comparable_payload_mib']:.2f} "
            f"| {fmt(quality['base_50']['recall'])} "
            f"| {fmt(quality['hard_evidence_recall'])}/{fmt(quality['hard_evidence_complete'])} "
            f"| {fmt(quality['overall_70']['mrr'])} | {category_text} "
            f"| {row['latency_ms']['vector']['mean']:.1f}/"
            f"{row['latency_ms']['bm25_rrf']['mean']:.1f}/"
            f"{row['latency_ms']['reranker']['mean']:.1f} "
            f"| {row['latency_ms']['total']['mean']:.1f}/{row['latency_ms']['total']['p95']:.1f} "
            f"| {row['context_cost']['estimated_tokens']['mean']:.0f} "
            f"| {fmt(quality['composite_score'])} |"
        )

    lines.extend(["", "## Trade-off", ""])
    if results:
        smallest = min(results, key=lambda row: row["config"]["parent_size"])
        largest = max(results, key=lambda row: row["config"]["parent_size"])
        lines.extend(
            [
                "- **质量**：composite = 0.4 × Base chapter recall + 0.4 × Hard 正文证据 recall "
                "+ 0.2 × 全题 MRR；最高分为 "
                f"`{max(row['quality']['composite_score'] for row in results):.3f}`。",
                f"- **索引规模**：小块配置 `{smallest['config']['id']}` 有 "
                f"{smallest['index']['child_count']} 个向量；大块配置 "
                f"`{largest['config']['id']}` 有 {largest['index']['child_count']} 个向量。"
                "当前 schema 在每个 child 中冗余 parent_content，表内 payload 不只由向量数决定。",
                "- **检索延迟**：P50/P95 包含 query embedding、Milvus ANN、BM25+RRF 与 BGE reranker；"
                "各题一次串行采样（coarse），warm-up 不计入。复测候选后再判断细小差异。",
                "- **上下文成本**：按最终去重、截断后的真实 context 统计 chars 和估算 tokens；"
                "扩大实际 parent 可能增加跨段覆盖，也可能增加输入长度及无关正文。未直接测量 TTFT 或账单。",
                "- **名义值与实际值**：Parent Size 是上限；报告同时展示实际 parent P50/P95，防止标题自然边界让多个名义配置退化成同一种切分。",
                "- **证据**：Hard complete、干扰 parent 比例、逐题命中正文和分段延迟保存在 comparison.json。"
                "证据 recall 是检索代理指标，不等同生成答案正确率；可选 RAGAS 单列，失败计数同样披露。",
            ]
        )
    else:
        lines.append("尚无完整检索结果；先执行 `run`。")

    recommendation = payload.get("recommendation")
    lines.extend(["", "## 评测后推荐", ""])
    if recommendation:
        lines.extend(
            [
                f"推荐 `{recommendation['config_id']}`。",
                "",
                f"选择规则：{recommendation['rule']}。",
                "",
                f"依据：Base recall={recommendation['basis']['base_recall']:.3f}，"
                f"Hard recall={recommendation['basis']['hard_recall']:.3f}，"
                f"Hard evidence recall={recommendation['basis']['hard_evidence_recall']:.3f}，"
                f"Hard complete={recommendation['basis']['hard_evidence_complete']:.3f}，"
                f"MRR={recommendation['basis']['mrr']:.3f}，"
                f"平均上下文≈{recommendation['basis']['context_tokens_mean']:.0f} tokens，"
                f"检索 mean/P95={recommendation['basis']['retrieval_mean_ms']:.1f}/"
                f"{recommendation['basis']['retrieval_p95_ms']:.1f} ms，"
                f"payload={recommendation['basis']['index_payload_mib']:.2f} MiB。",
                "",
                "该结论只写入评测报告；应用到生产仍需单独变更和重建 production collection。",
            ]
        )
    else:
        lines.append("完整 sweep 结束后生成；当前未改生产配置。")
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path, manifest: dict[str, Any], stack: dict[str, Any] | None) -> dict[str, Any]:
    result_files = sorted((run_dir / "results").glob("*.json")) if (run_dir / "results").exists() else []
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
    points = load_frozen_points(run_dir, manifest)
    cases = load_frozen_cases(run_dir, manifest)
    method_path = run_dir / "measurement_protocol.json"
    method = json.loads(method_path.read_text(encoding="utf-8")) if method_path.exists() else None
    validate_results(results, points, cases, manifest, stack, method, run_dir)
    order = {point.id: index for index, point in enumerate(points)}
    results.sort(key=lambda row: order.get(row["config"]["id"], 999))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "stack": stack,
        "measurement_protocol": method,
        "results": results,
        "pending_configs": [p.id for p in points if p.id not in {r['config']['id'] for r in results}],
        "recommendation": recommend(results) if len(results) == len(points) else None,
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "comparison.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def validate_results(results: Sequence[dict[str, Any]], points: Sequence[SweepPoint],
                     cases: Sequence[dict[str, Any]], manifest: dict[str, Any],
                     stack: dict[str, Any] | None, method: dict[str, Any] | None,
                     run_dir: Path) -> None:
    """Only comparable, complete configurations can enter the report/recommendation."""
    expected = {p.id: p for p in points}
    expected_cases = {c["id"]: c for c in cases}
    seen: set[str] = set()
    for result in results:
        config = result.get("config") or {}
        point = expected.get(config.get("id"))
        if point is None or point.id in seen or config != asdict(point) | {"parent_child_ratio": point.ratio}:
            raise RuntimeError("混合/重复/未知配置结果")
        seen.add(point.id)
        if (result.get("schema_version") != SCHEMA_VERSION
                or result.get("snapshot_sha256") != manifest["snapshot_sha256"]
                or result.get("eval_inputs") != manifest["eval_inputs"]
                or not stack or result.get("stack_fingerprint") != stack["fingerprint"]
                or not method or result.get("measurement_protocol") != method
                or result.get("collection") != collection_name(manifest["snapshot_sha256"], point, str(run_dir.resolve()))):
            raise RuntimeError(f"{point.id}: 结果的快照/stack/测量条件不一致")
        details = result.get("details") or []
        if (result.get("case_count") != 70 or len(details) != 70
                or {r['id'] for r in details} != set(expected_cases)):
            raise RuntimeError(f"{point.id}: 结果并非完整 70 题")
        for detail in details:
            case = expected_cases[detail["id"]]
            if any(detail.get(key) != case.get(key) for key in ("query", "dataset", "category")):
                raise RuntimeError(f"{point.id}: case 内容错配 {case['id']}")


async def maybe_run_ragas(
    case: dict[str, Any],
    retrieved: dict[str, Any],
    judge: Any,
    embeddings: Any,
) -> dict[str, Any]:
    from benchmark.run_benchmark import generate_answer
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    answer = await generate_answer(str(case["query"]), str(retrieved["context"]))
    sample = {
        "user_input": str(case["query"]),
        "retrieved_contexts": retrieved["contexts"] or ["(无召回)"],
        "response": answer,
        "reference": str(case["ground_truth"]),
    }
    # Same metrics as the existing evaluator, but a failed judge is not silently scored as zero.
    def evaluate_sample() -> dict[str, float]:
        result = evaluate(
            dataset=EvaluationDataset.from_list([sample]),
            metrics=[Faithfulness(llm=judge), AnswerRelevancy(llm=judge, embeddings=embeddings),
                     ContextPrecision(llm=judge), ContextRecall(llm=judge)],
            llm=judge, embeddings=embeddings, show_progress=False,
        )
        row = result.to_pandas().iloc[0].to_dict()
        scores = {name: float(row[name]) for name in (
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        )}
        if any(not math.isfinite(value) for value in scores.values()):
            raise ValueError("RAGAS 返回非有限分数")
        return scores

    return {"scores": await asyncio.to_thread(evaluate_sample), "answer": answer}


async def evaluate_point(
    *,
    point: SweepPoint,
    chunks: Sequence[Any],
    cases: Sequence[dict[str, Any]],
    run_dir: Path,
    manifest: dict[str, Any],
    stack: dict[str, Any],
    batch_size: int,
    rebuild: bool,
    warmup: int,
    with_ragas: bool,
    embedding_cache: DocumentEmbeddingCache,
) -> dict[str, Any]:
    from pymilvus import MilvusClient
    from app.core.vector_store import create_vector_store

    production = str(stack["milvus"]["production_collection"])
    name = collection_name(str(manifest["snapshot_sha256"]), point, str(run_dir.resolve()))
    assert_eval_collection(name, production)
    uri = f"http://{stack['milvus']['host']}:{stack['milvus']['port']}"
    client = MilvusClient(uri=uri)
    exists = bool(client.has_collection(name))
    if exists and not rebuild:
        state_path = run_dir / "collection_states" / f"{point.id}.json"
        if not state_path.exists():
            raise RuntimeError(f"隔离 collection 已存在但缺少本 run 的 state，使用 --rebuild: {name}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        owned_collection(state, run_dir, manifest, stack, [point])
        if (
            state.get("snapshot_sha256") != manifest["snapshot_sha256"]
            or state.get("config") != asdict(point)
            or state.get("stack_fingerprint") != stack["fingerprint"]
            or int(state.get("child_count") or -1) != len(chunks)
        ):
            raise RuntimeError(f"collection state 与当前 snapshot/config 不匹配: {name}")

    cache_before = embedding_cache.snapshot()
    store = create_vector_store(
        name, drop_old=bool(exists and rebuild), embedding_function=embedding_cache
    )
    ingest_start = time.perf_counter()
    if rebuild or not exists:
        for start in range(0, len(chunks), batch_size):
            store.add_documents(list(chunks[start : start + batch_size]))
        client.flush(collection_name=name)
    actual_entities = wait_for_entity_count(client, name, len(chunks))
    index_description = verify_vector_index(client, name, stack["milvus"]["index_params"])
    verify_vector_dimension(client, name, int(stack["embedding"]["dimension"]))
    ingest_seconds = time.perf_counter() - ingest_start
    client.load_collection(collection_name=name)

    state_dir = run_dir / "collection_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{point.id}.json").write_text(
        json.dumps(
            {
                "collection": name,
                "snapshot_sha256": manifest["snapshot_sha256"],
                "stack_fingerprint": stack["fingerprint"],
                "config": asdict(point),
                "child_count": len(chunks),
                "created_at": utc_now(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bm25 = BM25Corpus(chunks)
    for case in cases[: max(0, warmup)]:
        await retrieve_one(
            query=str(case["query"]), store=store, bm25=bm25, stack=stack, strict_reranker=True,
            parent_max=point.parent_size,
        )

    judge = embeddings = None
    ragas_values: dict[str, list[float]] = defaultdict(list)
    if with_ragas:
        from benchmark.run_benchmark import make_ragas_judge_and_embeddings

        judge, embeddings = make_ragas_judge_and_embeddings()

    details: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        retrieved = await retrieve_one(
            query=str(case["query"]), store=store, bm25=bm25, stack=stack, strict_reranker=True,
            parent_max=point.parent_size,
        )
        score = score_hits(
            retrieved["hits"],
            case.get("relevant") or [],
            int(stack["retrieval"]["final_top_k"]),
            case.get("relevant_groups"),
        )
        evidence_score = score_evidence(retrieved["hits"], case.get("evidence_groups") or [])
        distractor_hits = sum(
            any(is_relevant(hit, gold) for gold in case.get("distractors", []))
            for hit in retrieved["hits"]
        )
        ragas_result = None
        ragas_error = None
        if judge is not None and embeddings is not None:
            try:
                ragas_result = await maybe_run_ragas(case, retrieved, judge, embeddings)
                for metric, value in ragas_result["scores"].items():
                    ragas_values[metric].append(float(value))
            except Exception as exc:
                ragas_error = f"{type(exc).__name__}: {exc}"
        details.append(
            {
                "id": case["id"],
                "dataset": case["dataset"],
                "category": case["category"],
                "scenario": case.get("scenario"),
                "query": case["query"],
                "retrieval_score": asdict(score),
                "evidence_score": evidence_score,
                "distractor_parent_fraction": distractor_hits / max(1, len(retrieved["hits"])),
                "hits": retrieved["hits"],
                "context_chars": retrieved["context_chars"],
                "context_estimated_tokens": retrieved["context_estimated_tokens"],
                "candidate_children": retrieved["candidate_children"],
                "reranked_children": retrieved["reranked_children"],
                "unique_parents": retrieved["unique_parents"],
                "reranker_scored": retrieved["reranker_scored"],
                "reranker_applied": retrieved["reranker_applied"],
                "latency_ms": retrieved["latency_ms"],
                "ragas": ragas_result,
                "ragas_error": ragas_error,
            }
        )
        print(
            f"[{point.id} {index:02d}/{len(cases):02d}] "
            f"hit={score.hit:.0f} recall={score.recall:.2f} "
            f"latency={retrieved['latency_ms']['total']:.1f}ms "
            f"context≈{retrieved['context_estimated_tokens']}tok | {case['id']}"
        )

    index_stats = chunk_statistics(chunks, int(stack["embedding"]["dimension"]))
    index_stats["milvus_num_entities"] = actual_entities
    index_stats["milvus_index_description"] = index_description
    index_stats["ingestion_reused"] = exists and not rebuild
    cache_after = embedding_cache.snapshot()
    index_stats["document_embedding_cache"] = {
        key: cache_after[key] - cache_before[key] for key in ("hits", "misses")
    } | {"rows_after": cache_after["rows"]}
    ragas_averages = (
        {metric: statistics.fmean(values) for metric, values in ragas_values.items()}
        if ragas_values
        else None
    )
    result = aggregate_config(
        point,
        index_stats,
        details,
        collection=name,
        ingest_seconds=ingest_seconds,
        ragas_averages=ragas_averages,
    )
    result["snapshot_sha256"] = manifest["snapshot_sha256"]
    result["eval_inputs"] = manifest["eval_inputs"]
    result["stack_fingerprint"] = stack["fingerprint"]
    result["ragas_coverage"] = {
        "requested": with_ragas,
        "successful_cases": sum(row["ragas"] is not None for row in details),
        "failed_cases": sum(row["ragas_error"] is not None for row in details),
    }
    # Only this run's collection; avoid accumulating loaded indexes over the sweep.
    client.release_collection(collection_name=name)
    client.close()
    return result


def create_run_dir(base: Path, explicit: str | None) -> Path:
    run_dir = Path(explicit).expanduser().resolve() if explicit else (base / now_tag()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def structural_plan(run_dir: Path, points: Sequence[SweepPoint]) -> dict[str, Any]:
    manifest_path = run_dir / "snapshot_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else create_frozen_snapshot(run_dir)
    )
    manifest, rows = load_frozen_snapshot(run_dir)
    verify_eval_input_hashes(manifest)
    validate_evidence(load_frozen_cases(run_dir, manifest), rows)
    results: list[dict[str, Any]] = []
    for point in points:
        print(f"[plan] split {point.id}: parent={point.parent_size}, child={point.child_size}")
        chunks = split_snapshot(rows, point)
        stats = chunk_statistics(chunks, embedding_dim=1024)
        results.append({"config": asdict(point) | {"parent_child_ratio": point.ratio}, "index": stats})
    partitions: dict[str, list[str]] = defaultdict(list)
    for row in results:
        partitions[row["index"]["parent_partition_fingerprint"]].append(row["config"]["id"])
    equivalent = [group for group in partitions.values() if len(group) > 1]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "file_count": manifest["file_count"],
        "case_count": len(load_frozen_cases(run_dir, manifest)),
        "parent_equivalent_groups": equivalent,
        "results": results,
        "note": "离线结构计划；质量和延迟必须通过 run 使用固定 BGE/Milvus/BM25/RRF/BGE-Reranker 实测",
    }
    (run_dir / "structural_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Parent-Child Structural Plan",
        "",
        f"语料 `{manifest['snapshot_sha256']}`，{manifest['file_count']} files，70 cases。",
        "",
        "固定生产 H1/H2/H3 边界、结构保护、父/子 overlap=0。单位为字符，向量 payload 按 BGE-M3 的 1024×float32 估算。",
        "",
        "| Config | Scale | Parent/Child | Parents | Actual Parent P50/P95/Max | Children | Payload MiB | Quality / Latency / Context |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        config, index = row["config"], row["index"]
        lines.append(
            f"| {config['id']} | {config['scale']} | {config['parent_size']}/{config['child_size']} "
            f"| {index['parent_count']} "
            f"| {index['actual_parent_chars']['p50']:.0f}/{index['actual_parent_chars']['p95']:.0f}/{index['actual_parent_chars']['max']:.0f} "
            f"| {index['child_count']} | {index['comparable_payload_mib']:.2f} | 待实测 |"
        )
    lines.extend([
        "", "## 结构性发现（非检索质量结论）", "",
        "- 相同 parent 分区（按 source、parent_id、正文指纹确认）：" + repr(equivalent),
        "- Parent 参数是自然标题节的上限，并非目标长度。上限饱和时，大档配置主要改变 child，"
        "应据此解释结果，避免将收益错误归因于 parent 扩大。",
        "- 小 child 通常增加向量、BM25 文档数及重复 parent metadata。Payload 是可比较逻辑下界，"
        "未包含 HNSW 图、实际 segment/索引文件占用；不等同 Milvus 磁盘实测。",
        "- 结构保护可能让代码块/表格恢复后超过名义上限；最终 parent 按相应 cap 截断，证据计分只看实际交付正文。",
        "- 检索质量、延迟和最终上下文取决于真实命中；本表不填模拟值，也不据此推荐生产配置。",
    ])
    (run_dir / "structural_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


async def run_command(args: argparse.Namespace) -> None:
    require_runtime_dependencies()
    points = select_points(load_sweep_points(), args.configs)
    run_dir = create_run_dir(DEFAULT_RUNS_DIR, args.run_dir)
    manifest_path = run_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        create_frozen_snapshot(run_dir)
    manifest, rows = load_frozen_snapshot(run_dir)
    verify_eval_input_hashes(manifest)
    cases = load_frozen_cases(run_dir, manifest)
    validate_evidence(cases, rows)
    stack = capture_stack()
    validate_required_stack(stack)
    from app.core.embedding import get_embeddings
    embedding_cache = DocumentEmbeddingCache(
        run_dir / "document_embeddings.sqlite3", get_embeddings(),
        sha256_text(canonical_json(stack["embedding"])), int(stack["embedding"]["dimension"]),
    )
    freeze_record(run_dir / "fixed_stack.json", stack)
    method = {
        "batch_size": max(1, args.batch_size),
        "warmup_queries": max(0, args.warmup),
        "query_order": [case["id"] for case in cases],
        "repetitions": 1,
        "concurrency": 1,
        "bm25_corpus": "all frozen children, ingestion order",
        "context_tokens": "CJK + ceil(non-CJK/4)",
    }
    freeze_record(run_dir / "measurement_protocol.json", method)
    from pymilvus import MilvusClient

    guard_client = MilvusClient(uri=f"http://{stack['milvus']['host']}:{stack['milvus']['port']}")
    production_collection = str(stack["milvus"]["production_collection"])
    production_count_before = collection_entity_count(guard_client, production_collection)
    if production_count_before:
        verify_vector_index(guard_client, production_collection, stack["milvus"]["index_params"])
    production_guard = {
        "collection": production_collection,
        "entity_count_before": production_count_before,
        "captured_at": utc_now(),
    }
    (run_dir / "production_guard.json").write_text(
        json.dumps(production_guard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(run_dir, manifest, stack)  # Validate existing results before any index writes.
    settings_before = canonical_json(stack)
    for point in points:
        print(
            f"\n=== {point.id} {point.scale}: parent={point.parent_size}, child={point.child_size}, "
            f"overlap=0, snapshot={manifest['snapshot_sha256'][:12]} ==="
        )
        chunks = split_snapshot(rows, point)
        result = await evaluate_point(
            point=point,
            chunks=chunks,
            cases=cases,
            run_dir=run_dir,
            manifest=manifest,
            stack=stack,
            batch_size=method["batch_size"],
            rebuild=args.rebuild,
            warmup=method["warmup_queries"],
            with_ragas=args.with_ragas,
            embedding_cache=embedding_cache,
        )
        if canonical_json(capture_stack()) != settings_before:
            raise RuntimeError("评测期间固定 stack 发生变化，结果作废")
        production_count_after = collection_entity_count(guard_client, production_collection)
        if production_count_after != production_count_before:
            raise RuntimeError(
                "生产 collection 行数在评测期间发生变化: "
                f"before={production_count_before}, after={production_count_after}"
            )
        result["measurement_protocol"] = method
        validate_results([result], points, cases, manifest, stack, method, run_dir)
        result_dir = run_dir / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        temporary = result_dir / f"{point.id}.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(result_dir / f"{point.id}.json")
        write_summary(run_dir, manifest, stack)
    summary = write_summary(run_dir, manifest, stack)
    print(f"\ncomparison: {run_dir / 'comparison.md'}")
    if summary.get("recommendation"):
        print(f"recommendation: {summary['recommendation']['config_id']}")
    guard_client.close()
    embedding_cache.close()


def report_command(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest, _rows = load_frozen_snapshot(run_dir)
    stack_path = run_dir / "fixed_stack.json"
    stack = json.loads(stack_path.read_text(encoding="utf-8")) if stack_path.exists() else None
    payload = write_summary(run_dir, manifest, stack)
    print(run_dir / "comparison.md")
    if payload.get("recommendation"):
        print(f"recommendation: {payload['recommendation']['config_id']}")


def cleanup_command(args: argparse.Namespace) -> None:
    from pymilvus import MilvusClient
    from app.config import settings

    run_dir = Path(args.run_dir).expanduser().resolve()
    states = sorted((run_dir / "collection_states").glob("*.json"))
    if not states:
        print("no evaluation collections recorded")
        return
    manifest, _rows = load_frozen_snapshot(run_dir)
    points = load_frozen_points(run_dir, manifest)
    stack = json.loads((run_dir / "fixed_stack.json").read_text(encoding="utf-8"))
    recorded = stack["milvus"]
    if (str(recorded["host"]) != str(settings.milvus_host)
            or str(recorded["port"]) != str(settings.milvus_port)
            or recorded["production_collection"] != settings.milvus_collection):
        raise RuntimeError("cleanup endpoint/生产 collection 与 run 记录不一致")
    # Validate every ownership record before the first destructive call.
    names = [owned_collection(json.loads(path.read_text(encoding="utf-8")),
                              run_dir, manifest, stack, points) for path in states]
    client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
    for name in names:
        if client.has_collection(name):
            client.drop_collection(name)
            print(f"dropped {name}")
    client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parent-Child chunking evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="freeze corpus and compute offline structural/index estimates")
    plan.add_argument("--run-dir", type=str, default=None)
    plan.add_argument("--configs", type=str, default=None, help="comma-separated config IDs")

    run = sub.add_parser("run", help="build isolated indexes and run 50+20 evaluation")
    run.add_argument("--run-dir", type=str, default=None)
    run.add_argument("--configs", type=str, default=None, help="comma-separated config IDs")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--rebuild", action="store_true", help="drop/rebuild only guarded eval collections")
    run.add_argument("--with-ragas", action="store_true", help="also run end-to-end RAGAS confirmation")

    report = sub.add_parser("report", help="regenerate comparison from completed per-config results")
    report.add_argument("--run-dir", type=str, required=True)

    cleanup = sub.add_parser("cleanup", help="drop only guarded eval collections recorded by this run")
    cleanup.add_argument("--run-dir", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        points = select_points(load_sweep_points(), args.configs)
        run_dir = create_run_dir(DEFAULT_RUNS_DIR, args.run_dir)
        structural_plan(run_dir, points)
        print(run_dir / "structural_plan.md")
    elif args.command == "run":
        asyncio.run(run_command(args))
    elif args.command == "report":
        report_command(args)
    elif args.command == "cleanup":
        cleanup_command(args)


if __name__ == "__main__":
    main()
