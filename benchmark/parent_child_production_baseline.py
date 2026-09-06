"""Read-only replay of the current production Parent/Child Milvus collection.

This is deliberately separate from the coarse sweep runner: the sweep may only write
guarded ``pc_chunk_eval_*`` collections, while this command only queries the configured
production collection and writes a JSON result under an existing frozen run directory.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.parent_child.evidence import score_evidence
from benchmark.parent_child_eval import (
    BM25Corpus,
    SweepPoint,
    aggregate_config,
    canonical_json,
    capture_stack,
    chunk_statistics,
    collection_entity_count,
    is_relevant,
    load_frozen_cases,
    load_frozen_snapshot,
    retrieve_one,
    score_hits,
    split_snapshot,
    verify_vector_dimension,
    verify_vector_index,
)


def _document_multiset(docs: list[Any]) -> Counter[tuple[str, str, str, str, int]]:
    return Counter(
        (
            str((doc.metadata or {}).get("source") or ""),
            str((doc.metadata or {}).get("chapter") or ""),
            str((doc.metadata or {}).get("parent_id") or ""),
            str(doc.page_content),
            int((doc.metadata or {}).get("chunk_index") or 0),
        )
        for doc in docs
    )


async def run(run_dir: Path, *, warmup: int) -> Path:
    from app.core.hybrid_retriever import _load_all_chunks_from_milvus
    from app.core.vector_store import create_vector_store
    from pymilvus import MilvusClient

    manifest, frozen_rows = load_frozen_snapshot(run_dir)
    cases = load_frozen_cases(run_dir, manifest)
    recorded_stack = json.loads((run_dir / "fixed_stack.json").read_text(encoding="utf-8"))
    current_stack = capture_stack()
    if canonical_json(current_stack) != canonical_json(recorded_stack):
        raise RuntimeError("当前检索 stack 与 coarse sweep 的 fixed_stack.json 不一致")

    production = str(recorded_stack["milvus"]["production_collection"])
    uri = f"http://{recorded_stack['milvus']['host']}:{recorded_stack['milvus']['port']}"
    client = MilvusClient(uri=uri)
    count_before = collection_entity_count(client, production)
    if not count_before:
        raise RuntimeError(f"生产 collection 不存在或为空: {production}")
    index_description = verify_vector_index(
        client, production, recorded_stack["milvus"]["index_params"]
    )
    verify_vector_dimension(client, production, int(recorded_stack["embedding"]["dimension"]))

    docs = _load_all_chunks_from_milvus()
    if len(docs) != count_before:
        raise RuntimeError(f"BM25 全量读取不完整: expected={count_before}, actual={len(docs)}")

    # The collection predates the sweep manifest. Reconstruct and prove the exact split instead
    # of trusting a comment or current .env value.
    observed = SweepPoint("PROD", "production", 2400, 800, 100, 0, 3)
    reconstructed = split_snapshot(frozen_rows, observed)
    actual_multiset = _document_multiset(docs)
    expected_multiset = _document_multiset(reconstructed)
    if actual_multiset != expected_multiset:
        raise RuntimeError(
            "生产 collection 不是 frozen corpus 的 2400/800/100 精确重建结果: "
            f"missing={sum((expected_multiset-actual_multiset).values())}, "
            f"extra={sum((actual_multiset-expected_multiset).values())}"
        )

    store = create_vector_store(production, drop_old=False)
    bm25 = BM25Corpus(docs)
    for case in cases[: max(0, warmup)]:
        await retrieve_one(
            query=str(case["query"]), store=store, bm25=bm25, stack=recorded_stack,
            strict_reranker=True, parent_max=observed.parent_size,
        )

    details: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        retrieved = await retrieve_one(
            query=str(case["query"]), store=store, bm25=bm25, stack=recorded_stack,
            strict_reranker=True, parent_max=observed.parent_size,
        )
        score = score_hits(
            retrieved["hits"], case.get("relevant") or [],
            int(recorded_stack["retrieval"]["final_top_k"]), case.get("relevant_groups"),
        )
        evidence = score_evidence(retrieved["hits"], case.get("evidence_groups") or [])
        distractor_hits = sum(
            any(is_relevant(hit, gold) for gold in case.get("distractors", []))
            for hit in retrieved["hits"]
        )
        details.append(
            {
                "id": case["id"],
                "dataset": case["dataset"],
                "category": case["category"],
                "scenario": case.get("scenario"),
                "query": case["query"],
                "retrieval_score": asdict(score),
                "evidence_score": evidence,
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
                "ragas": None,
                "ragas_error": None,
            }
        )
        print(
            f"[PROD {index:02d}/{len(cases):02d}] hit={score.hit:.0f} "
            f"recall={score.recall:.2f} latency={retrieved['latency_ms']['total']:.1f}ms "
            f"context≈{retrieved['context_estimated_tokens']}tok | {case['id']}"
        )

    index_stats = chunk_statistics(docs, int(recorded_stack["embedding"]["dimension"]))
    index_stats.update(
        {
            "milvus_num_entities": count_before,
            "milvus_index_description": index_description,
            "ingestion_reused": True,
            "document_embedding_cache": {"hits": 0, "misses": 0, "rows_after": 0},
            "split_provenance": "exact frozen-corpus multiset match for 2400/800/100",
        }
    )
    result = aggregate_config(
        observed, index_stats, details, collection=production, ingest_seconds=0.0,
        ragas_averages=None,
    )
    result.update(
        {
            "snapshot_sha256": manifest["snapshot_sha256"],
            "stack_fingerprint": recorded_stack["fingerprint"],
            "read_only_production_baseline": True,
            "measurement_protocol": {
                "warmup_queries": max(0, warmup),
                "repetitions": 1,
                "concurrency": 1,
                "query_order": [case["id"] for case in cases],
            },
        }
    )

    count_after = collection_entity_count(client, production)
    if count_after != count_before:
        raise RuntimeError(
            f"生产 collection 行数发生变化: before={count_before}, after={count_after}"
        )
    result["production_guard"] = {
        "entity_count_before": count_before,
        "entity_count_after": count_after,
    }
    client.close()

    output = run_dir / "production_baseline.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only production collection baseline replay")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    output = asyncio.run(run(args.run_dir.expanduser().resolve(), warmup=args.warmup))
    print(output)


if __name__ == "__main__":
    main()
