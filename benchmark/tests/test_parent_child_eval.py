from __future__ import annotations

import asyncio
import copy
import json
import tempfile
import sys
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock, patch
import unittest
from collections import Counter

from benchmark.parent_child_eval import (
    EVAL_COLLECTION_PREFIX,
    aggregate_config,
    assert_eval_collection,
    chunk_statistics,
    collection_name,
    collect_corpus_files,
    freeze_record,
    hits_meta,
    owned_collection,
    parents_from_docs,
    retrieve_one,
    rrf_fuse,
    validate_results,
    verify_vector_index,
    evaluate_point,
    DocumentEmbeddingCache,
    SCHEMA_VERSION,
    load_eval_cases,
    load_sweep_points,
    recommend,
    render_report,
    score_hits,
)
from benchmark.parent_child.evidence import score_evidence, validate_evidence


class ParentChildEvalContractTests(unittest.TestCase):
    def test_sweep_contract(self) -> None:
        points = load_sweep_points()
        self.assertGreaterEqual(len(points), 5)
        self.assertLessEqual(len(points), 8)
        self.assertEqual({point.scale for point in points}, {"small", "medium", "large"})
        for point in points:
            self.assertEqual(point.child_overlap, 0)
            self.assertEqual(point.parent_overlap, 0)
            self.assertGreaterEqual(point.ratio, 3.0)

    def test_dataset_contract(self) -> None:
        cases = load_eval_cases()
        self.assertEqual(len(cases), 70)
        self.assertEqual(sum(row["dataset"] == "base_50" for row in cases), 50)
        hard = [row for row in cases if row["dataset"] == "hard_20"]
        self.assertEqual(
            Counter(row["category"] for row in hard),
            Counter(
                {
                    "exact_fact": 5,
                    "cross_paragraph": 5,
                    "multi_step": 5,
                    "similar_section_distractor": 5,
                }
            ),
        )

    def test_group_recall_requires_independent_knowledge_points(self) -> None:
        groups = [
            [{"source": "runbook.md", "chapter_contains": "A"}],
            [{"source": "runbook.md", "chapter_contains": "B"}],
        ]
        score = score_hits(
            [{"source": "runbook.md", "chapter": "A"}],
            [],
            3,
            groups,
        )
        self.assertEqual(score.hit, 1.0)
        self.assertEqual(score.recall, 0.5)

    def test_collection_guard(self) -> None:
        assert_eval_collection(EVAL_COLLECTION_PREFIX + "abc_s1", "multi_agent_kb")
        with self.assertRaises(RuntimeError):
            assert_eval_collection("multi_agent_kb", "multi_agent_kb")
        with self.assertRaises(RuntimeError):
            assert_eval_collection("unrelated", "multi_agent_kb")

    def test_aggregate_report_and_recommendation_contract(self) -> None:
        point = load_sweep_points()[0]
        details = []
        categories = [
            ("base_50", "base"),
            ("hard_20", "exact_fact"),
            ("hard_20", "cross_paragraph"),
            ("hard_20", "multi_step"),
            ("hard_20", "similar_section_distractor"),
        ]
        for dataset, category in categories:
            details.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "retrieval_score": {"hit": 1.0, "mrr": 1.0, "recall": 1.0},
                    "evidence_score": {"recall": 1.0, "complete": 1.0},
                    "latency_ms": {"vector": 1.0, "bm25_rrf": 2.0, "reranker": 3.0, "total": 6.0},
                    "context_chars": 100,
                    "context_estimated_tokens": 50,
                }
            )
        result = aggregate_config(
            point,
            chunk_statistics([], embedding_dim=1024),
            details,
            collection=EVAL_COLLECTION_PREFIX + "fixture_s1",
            ingest_seconds=0.1,
            ragas_averages=None,
        )
        recommendation = recommend([result])
        self.assertEqual(recommendation["config_id"], point.id)
        markdown = render_report(
            {
                "generated_at": "fixture",
                "snapshot_sha256": "abc",
                "results": [result],
                "recommendation": recommendation,
            }
        )
        self.assertIn("Parent-Child Chunking Evaluation", markdown)
        self.assertIn(f"推荐 `{point.id}`", markdown)

    def test_recommendation_does_not_trade_quality_for_one_token(self) -> None:
        point = load_sweep_points()[0]
        details = []
        for dataset, category in [
            ("base_50", "base"),
            ("hard_20", "exact_fact"),
            ("hard_20", "cross_paragraph"),
            ("hard_20", "multi_step"),
            ("hard_20", "similar_section_distractor"),
        ]:
            details.append({
                "dataset": dataset, "category": category,
                "retrieval_score": {"hit": 1.0, "mrr": 1.0, "recall": 1.0},
                "evidence_score": {"recall": 1.0, "complete": 1.0},
                "latency_ms": {"vector": 1.0, "bm25_rrf": 1.0, "reranker": 8.0, "total": 10.0},
                "context_chars": 100, "context_estimated_tokens": 51,
            })
        better = aggregate_config(point, chunk_statistics([], 1024), details,
                                  collection=EVAL_COLLECTION_PREFIX + "better",
                                  ingest_seconds=0.0, ragas_averages=None)
        worse = copy.deepcopy(better)
        worse["config"] = worse["config"] | {"id": "CHEAP"}
        worse["quality"]["overall_70"]["mrr"] = 0.95
        worse["quality"]["composite_score"] = 0.99
        worse["context_cost"]["estimated_tokens"]["mean"] = 50
        self.assertEqual(recommend([worse, better])["config_id"], point.id)


class SplitterIsolationTests(unittest.TestCase):
    def test_explicit_production_config_preserves_default_output(self) -> None:
        from app.core.splitter import ChunkingConfig, split_markdown

        text = "# A\n\n## B\n\n### C\n\n" + ("Redis timeout。" * 100)
        implicit = split_markdown(text, "x.md")
        explicit = split_markdown(text, "x.md", config=ChunkingConfig.production())
        self.assertEqual(
            [(doc.page_content, doc.metadata) for doc in implicit],
            [(doc.page_content, doc.metadata) for doc in explicit],
        )

    def test_sweep_parent_size_changes_real_parent_count(self) -> None:
        from app.core.splitter import ChunkingConfig, split_markdown

        text = "# Ops\n\n" + "\n\n".join(
            f"## Runbook {index}\n\n" + "\n".join(f"step-{j} of runbook-{index} 诊断动作。" for j in range(90)) for index in range(1, 7)
        )
        small = split_markdown(
            text,
            "ops.md",
            config=ChunkingConfig(512, 128, 0, 0),
        )
        large = split_markdown(
            text,
            "ops.md",
            config=ChunkingConfig(3200, 800, 0, 0),
        )
        small_parents = {doc.metadata["parent_id"] for doc in small}
        large_parents = {doc.metadata["parent_id"] for doc in large}
        self.assertGreater(len(small_parents), len(large_parents))
        self.assertTrue(any("Runbook 1" in doc.metadata["chapter"] for doc in small))


    def test_header_saturation_is_reported_not_merged(self) -> None:
        from app.core.splitter import ChunkingConfig, split_markdown
        text = "# Ops\n\n## Redis\n\n### Memory\n" + "memory。" * 20
        text += "\n\n### CPU\n" + "cpu。" * 20
        small = split_markdown(text, "x", config=ChunkingConfig(1024, 256))
        large = split_markdown(text, "x", config=ChunkingConfig(3200, 800))
        self.assertEqual({d.metadata["parent_id"] for d in small},
                         {d.metadata["parent_id"] for d in large})

    def test_legacy_settings_ratio_stays_compatible(self) -> None:
        from app.config import settings
        from app.core.splitter import split_markdown
        with patch.object(settings, "rag_parent_max_chars", 500), patch.object(settings, "rag_chunk_size", 800):
            self.assertTrue(split_markdown("# Ops\n" + "step。" * 200, "x"))

    def test_explicit_overlap_zero_and_protected_block(self) -> None:
        from app.core.splitter import ChunkingConfig, split_markdown
        config = ChunkingConfig(512, 128, 0, 0)
        self.assertEqual(config.validated().child_overlap, 0)
        block = "```bash\n" + "unique_command\n" * 80 + "```"
        chunks = split_markdown("# Ops\n" + block, "x", config=config)
        self.assertTrue(any(block in d.page_content for d in chunks))
        self.assertTrue(any(h["parent_truncated"] for h in hits_meta(chunks, 512)))


class EvidenceTests(unittest.TestCase):
    def test_all_anchors_exist_in_actual_corpus(self) -> None:
        validate_evidence(load_eval_cases(), [{"source": source, "content": path.read_text()}
                                             for path, source in collect_corpus_files()])

    def test_header_and_child_preview_do_not_count_as_context_evidence(self) -> None:
        groups = [[{"source": "x", "text": "maxmemory 80%"}]]
        hits = [{"source": "x", "chapter": "maxmemory 80%", "preview": "maxmemory 80%",
                 "parent_text": "unrelated"}]
        self.assertEqual(score_evidence(hits, groups)["recall"], 0)

    def test_partial_cross_paragraph_and_source_constraints(self) -> None:
        groups = [[{"source": "x", "text": "INFO memory"}], [{"source": "x", "text": "UNLINK"}]]
        hits = [{"source": "x", "parent_text": "info   memory"}, {"source": "y", "parent_text": "UNLINK"}]
        self.assertEqual(score_evidence(hits, groups)["recall"], 0.5)
        self.assertEqual(score_evidence(hits, groups)["complete"], 0)
        hits[1]["source"] = "x"
        self.assertEqual(score_evidence(hits, groups)["complete"], 1)

    def test_bad_anchor_fails_before_run(self) -> None:
        with self.assertRaises(ValueError):
            validate_evidence([{"id": "hard", "dataset": "hard_20", "evidence_groups":
                                [[{"source": "x", "text": "missing"}]]}], [{"source": "x", "content": "fact"}])

    def test_redis_semantic_mapping_not_suffix_join(self) -> None:
        cases = {c["id"]: c for c in load_eval_cases()}
        self.assertEqual(cases["ragas-redis-02"]["gold_from_id"], "rk-redis-03")
        self.assertEqual(cases["ragas-redis-03"]["gold_from_id"], "rk-redis-04")
        self.assertEqual(cases["ragas-redis-04"]["gold_from_id"], "rk-redis-05")


class RetrievalParityTests(unittest.TestCase):
    def test_bm25_and_rrf_match_existing_hybrid(self) -> None:
        from langchain_core.documents import Document
        from app.core import hybrid_retriever as hybrid
        from benchmark.parent_child_eval import BM25Corpus
        docs = [Document(page_content=f"topic{i} redis maxmemory" if i == 3 else f"topic{i} JVM cpu",
                         metadata={"source": "x", "chapter": str(i)}) for i in range(12)]
        local = BM25Corpus(docs)
        production_index = hybrid._BM25Index()
        production_index.build(docs)
        self.assertEqual(local.search("maxmemory", 10), production_index.search("maxmemory", 10))
        before = hybrid._bm25_index
        with patch.object(hybrid, "_bm25_index", production_index):
            expected = hybrid.hybrid_search("maxmemory", docs[::-1], k=10, retrieve_k=10)
            actual = rrf_fuse(docs[::-1], local.search("maxmemory", 10), k=10,
                              rrf_k=max(1, int(hybrid.settings.rag_hybrid_rrf_k or 60)),
                              bm25_weight=hybrid.settings.rag_hybrid_bm25_weight)
            self.assertEqual(expected, actual)
        self.assertIs(hybrid._bm25_index, before)
        self.assertEqual(rrf_fuse([docs[0], docs[0]], [], k=2, rrf_k=60, bm25_weight=.4), [docs[0], docs[0]])

    def test_parent_dedup_and_truncation_match_production(self) -> None:
        from langchain_core.documents import Document
        docs = [Document(page_content="child", metadata={"source": source, "parent_id": pid,
                "parent_content": "x" * 600}) for source, pid in [("a", "same"), ("b", "same"), ("c", "other")]]
        parents, context = parents_from_docs(docs, 3, 512)
        self.assertEqual(len(parents), 2)  # production deduplicates parent_id even across sources
        self.assertEqual(context.count("... (已截断)"), 2)
        self.assertEqual(len(hits_meta(parents, 512)[0]["parent_text"]), 512)
        self.assertNotIn("来源 3", context)

    def test_silent_reranker_fallback_is_rejected(self) -> None:
        from langchain_core.documents import Document
        from types import SimpleNamespace
        docs = [Document(page_content=str(i), metadata={"parent_id": str(i)}) for i in range(12)]
        stack = {"retrieval": {"retrieve_k": 12, "final_top_k": 3, "child_overfetch": 3},
                 "hybrid": {"rrf_k": 60, "bm25_weight": .4}}
        with patch("app.core.reranker.rerank_docs", new=AsyncMock(return_value=docs[:9])):
            with self.assertRaisesRegex(RuntimeError, "静默降级"):
                asyncio.run(retrieve_one(query="q", store=SimpleNamespace(similarity_search=lambda *a, **k: docs),
                            bm25=SimpleNamespace(search=lambda *a: []), stack=stack,
                            strict_reranker=True, parent_max=512))


class IntegrityTests(unittest.TestCase):
    def test_freeze_never_overwrites_changed_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stack.json"
            freeze_record(path, {"K": 30})
            freeze_record(path, {"K": 30})
            with self.assertRaises(RuntimeError):
                freeze_record(path, {"K": 20})
            self.assertEqual(json.loads(path.read_text()), {"K": 30})

    def test_collection_ownership_includes_run_path_and_config(self) -> None:
        point = load_sweep_points()[0]
        root = Path("/tmp/eval-fixture").resolve()
        manifest = {"snapshot_sha256": "abc"}
        stack = {"fingerprint": "stack", "milvus": {"production_collection": "prod"}}
        state = {"config": asdict(point), "snapshot_sha256": "abc", "stack_fingerprint": "stack",
                 "collection": collection_name("abc", point, str(root))}
        self.assertEqual(owned_collection(state, root, manifest, stack, [point]), state["collection"])
        with self.assertRaises(RuntimeError):
            owned_collection(state, Path("/tmp/other-run"), manifest, stack, [point])
        state["collection"] = EVAL_COLLECTION_PREFIX + "foreign"
        with self.assertRaises(RuntimeError):
            owned_collection(state, root, manifest, stack, [point])

    def test_result_integrity_rejects_partial_and_mixed_runs(self) -> None:
        point = load_sweep_points()[0]
        cases = load_eval_cases()
        root = Path("/tmp/fixture").resolve()
        manifest = {"snapshot_sha256": "abc", "eval_inputs": {"hash": "fixed"}}
        stack = {"fingerprint": "stack"}
        method = {"warmup_queries": 3}
        result = {"schema_version": SCHEMA_VERSION,
                  "config": asdict(point) | {"parent_child_ratio": point.ratio},
                  "snapshot_sha256": "abc", "eval_inputs": manifest["eval_inputs"],
                  "stack_fingerprint": "stack", "measurement_protocol": method,
                  "collection": collection_name("abc", point, str(root)),
                  "case_count": 70, "details": cases}
        validate_results([result], [point], cases, manifest, stack, method, root)
        for field, value in [("stack_fingerprint", "other"), ("case_count", 69), ("details", cases[:-1])]:
            bad = copy.deepcopy(result)
            bad[field] = value
            with self.assertRaises(RuntimeError):
                validate_results([bad], [point], cases, manifest, stack, method, root)
        with self.assertRaises(RuntimeError):
            validate_results([result, result], [point], cases, manifest, stack, method, root)

    def test_actual_index_parameters_checked(self) -> None:
        from types import SimpleNamespace
        info = {"field_name": "vector", "index_type": "HNSW", "metric_type": "COSINE",
                "M": "8", "efConstruction": "64"}
        client = SimpleNamespace(list_indexes=lambda **k: ["vector"], describe_index=lambda **k: info)
        expected = {"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 8, "efConstruction": 64}}
        self.assertEqual(verify_vector_index(client, "eval", expected), info)
        info["M"] = "16"
        with self.assertRaises(RuntimeError):
            verify_vector_index(client, "eval", expected)


class IsolatedRunnerSmokeTests(unittest.TestCase):
    def test_seventy_case_pipeline_with_fake_external_services(self) -> None:
        """Exercise orchestration, not real retrieval accuracy or service performance."""
        from langchain_core.documents import Document
        from unittest.mock import Mock
        from contextlib import redirect_stdout
        import io
        # Load native dependencies before patch.dict restores sys.modules (NumPy is not reloadable).
        import rank_bm25  # noqa: F401
        from app.core import hybrid_retriever, reranker  # noqa: F401

        cases = load_eval_cases()
        point = load_sweep_points()[0]
        docs = [Document(page_content=f"redis memory fact{i}", metadata={
            "source": "fixture.md", "chapter": f"chapter{i}", "parent_id": str(i),
            "parent_content": f"fixture parent {i}", "chunk_index": i,
        }) for i in range(12)]
        index = {"field_name": "vector", "index_type": "HNSW", "metric_type": "COSINE",
                 "M": "8", "efConstruction": "64"}
        client = Mock()
        client.has_collection.side_effect = [False, True]
        client.get_collection_stats.return_value = {"row_count": 12}
        client.list_indexes.return_value = ["vector"]
        client.describe_index.return_value = index
        client.describe_collection.return_value = {"fields": [{"name": "vector", "params": {"dim": 1024}}]}
        store = Mock()
        store.similarity_search.return_value = docs
        factory = Mock(return_value=store)
        async def rerank(query, candidates, *, top_n):
            return [Document(page_content=d.page_content, metadata=d.metadata | {"rerank_score": 1.0})
                    for d in candidates[:top_n]]
        stack = {"fingerprint": "fake-runtime-only", "embedding": {"dimension": 1024},
                 "milvus": {"production_collection": "prod", "host": "localhost", "port": 19530,
                             "index_params": {"index_type": "HNSW", "metric_type": "COSINE",
                                              "params": {"M": 8, "efConstruction": 64}}},
                 "retrieval": {"retrieve_k": 12, "final_top_k": 3, "child_overfetch": 3},
                 "hybrid": {"rrf_k": 60, "bm25_weight": .4}}
        modules = {"pymilvus": types.SimpleNamespace(MilvusClient=Mock(return_value=client)),
                   "app.core.vector_store": types.SimpleNamespace(create_vector_store=factory)}
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, modules), \
                patch("app.core.reranker.rerank_docs", side_effect=rerank), redirect_stdout(io.StringIO()):
            root = Path(temp)
            base_embeddings = Mock()
            base_embeddings.embed_documents.side_effect = lambda texts: [[0.0] * 1024 for _ in texts]
            cache = DocumentEmbeddingCache(root / "embeddings.sqlite3", base_embeddings, "fixture", 1024)
            result = asyncio.run(evaluate_point(
                point=point, chunks=docs, cases=cases, run_dir=root,
                manifest={"snapshot_sha256": "fixture", "eval_inputs": {}}, stack=stack,
                batch_size=5, rebuild=False, warmup=3, with_ragas=False,
                embedding_cache=cache,
            ))
            self.assertEqual(result["case_count"], 70)
            self.assertEqual(len(result["details"]), 70)
            self.assertEqual(store.add_documents.call_count, 3)
            self.assertEqual(store.similarity_search.call_count, 73)
            self.assertTrue(all(d["unique_parents"] == 3 for d in result["details"]))
            self.assertFalse((root / "results").exists())  # saved only after outer guards
            factory.assert_called_once_with(result["collection"], drop_old=False, embedding_function=cache)
            client.release_collection.assert_called_once_with(collection_name=result["collection"])
            client.drop_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
