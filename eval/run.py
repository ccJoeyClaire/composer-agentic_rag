"""CLI entry point for the eval pipeline.

Usage (from repo root)::

    python -m eval.run index --doc-slug codex --source path/to/doc.md
    python -m eval.run infer  --doc-slug codex --profile baseline --arm direct_rag
    python -m eval.run assemble --doc-slug codex --profile baseline --arm direct_rag
    python -m eval.run score   --doc-slug codex --profile baseline --arm direct_rag \\
        --extractor <TBD> --checker <TBD>
    python -m eval.run compare --doc-slug codex --baseline-profile token \\
        --baseline-arm direct_rag --candidate-profile baseline --candidate-arm crag_self_rag
    python -m eval.run full    --doc-slug codex --source path/to/doc.md \\
        --extractor <TBD> --checker <TBD>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from eval.compare import compute_deltas, load_all_run_metrics
from eval.gold import load_eval_gold
from eval.pipeline import (
    run_assemble_stage,
    run_full_matrix,
    run_index_stage,
    run_infer_stage,
    run_score_stage,
)
from eval.run_config import EvalRunConfig, build_run_matrix
from eval.score import RAGCheckerConfig
from eval.types import AGENT_ARMS, DIRECT_RAG_ARM, EVAL_PROFILES, EvalArm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SOURCE = (
    _REPO_ROOT
    / "_eval_/datasets/Easy-Dataset/file"
    / "工程技术：在智能体优先的世界中利用 Codex.md"
)
_ALL_ARMS: tuple[EvalArm, ...] = (DIRECT_RAG_ARM,) + AGENT_ARMS


def _build_config(
    doc_slug: str,
    profile_id: str,
    arm: EvalArm,
    collection_prefix: str,
) -> EvalRunConfig:
    return EvalRunConfig(
        doc_slug=doc_slug,
        profile_id=profile_id,
        arm=arm,
        collection_prefix=collection_prefix,
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--doc-slug", default="codex")
    parser.add_argument(
        "--collection-prefix",
        default="eval_codex",
        help="Qdrant collection prefix (default: eval_codex)",
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=None,
        help="Override gold_rubric.jsonl path",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=EVAL_PROFILES,
        required=True,
        help="RAG profile ID",
    )
    parser.add_argument(
        "--arm",
        choices=_ALL_ARMS,
        required=True,
        help="Inference arm",
    )


def _add_checker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--extractor", required=True, help="RAGChecker extractor model name")
    parser.add_argument("--checker", required=True, help="RAGChecker checker model name")
    parser.add_argument("--batch-size-extractor", type=int, default=32)
    parser.add_argument("--batch-size-checker", type=int, default=32)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval.run",
        description="Eval pipeline CLI (Easy Dataset + RAGChecker).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = sub.add_parser("index", help="Index source doc for all profiles")
    _add_common_args(p_index)
    p_index.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    p_index.add_argument(
        "--profiles",
        nargs="+",
        choices=EVAL_PROFILES,
        default=list(EVAL_PROFILES),
    )

    # --- infer ---
    p_infer = sub.add_parser("infer", help="Run one arm for all gold queries")
    _add_common_args(p_infer)
    _add_run_args(p_infer)

    # --- assemble ---
    p_assemble = sub.add_parser("assemble", help="Build RAGChecker input JSON")
    _add_common_args(p_assemble)
    _add_run_args(p_assemble)

    # --- score ---
    p_score = sub.add_parser("score", help="Run RAGChecker on assembled input")
    _add_common_args(p_score)
    _add_run_args(p_score)
    _add_checker_args(p_score)

    # --- compare ---
    p_compare = sub.add_parser("compare", help="Show metric deltas between two runs")
    _add_common_args(p_compare)
    p_compare.add_argument("--baseline-profile", choices=EVAL_PROFILES, required=True)
    p_compare.add_argument("--baseline-arm", choices=_ALL_ARMS, required=True)
    p_compare.add_argument("--candidate-profile", choices=EVAL_PROFILES, required=True)
    p_compare.add_argument("--candidate-arm", choices=_ALL_ARMS, required=True)

    # --- full ---
    p_full = sub.add_parser("full", help="Run complete pipeline for all runs")
    _add_common_args(p_full)
    p_full.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    _add_checker_args(p_full)
    p_full.add_argument(
        "--profiles",
        nargs="+",
        choices=EVAL_PROFILES,
        default=list(EVAL_PROFILES),
    )

    return parser


async def _cmd_index(args: argparse.Namespace) -> None:
    matrix = build_run_matrix(
        args.doc_slug,
        profiles=tuple(args.profiles),
        collection_prefix=args.collection_prefix,
    )
    results = await run_index_stage(matrix, args.source)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"[{status}] profile={r.config.profile_id} chunks={r.chunk_count}")


async def _cmd_infer(args: argparse.Namespace) -> None:
    config = _build_config(args.doc_slug, args.profile, args.arm, args.collection_prefix)
    gold = load_eval_gold(args.gold_path)
    results = await run_infer_stage(config, gold)
    print(f"infer done: run_id={config.run_id} queries={len(results)}")


async def _cmd_assemble(args: argparse.Namespace) -> None:
    config = _build_config(args.doc_slug, args.profile, args.arm, args.collection_prefix)
    gold = load_eval_gold(args.gold_path)
    out = await run_assemble_stage(config, gold)
    print(f"assembled: {out}")


async def _cmd_score(args: argparse.Namespace) -> None:
    config = _build_config(args.doc_slug, args.profile, args.arm, args.collection_prefix)
    gold = load_eval_gold(args.gold_path)
    checker_config = RAGCheckerConfig(
        extractor_name=args.extractor,
        checker_name=args.checker,
        batch_size_extractor=args.batch_size_extractor,
        batch_size_checker=args.batch_size_checker,
    )
    out = await run_score_stage(config, checker_config, gold)
    print(f"scored: {out}")


async def _cmd_compare(args: argparse.Namespace) -> None:
    baseline_cfg = _build_config(
        args.doc_slug, args.baseline_profile, args.baseline_arm, args.collection_prefix
    )
    candidate_cfg = _build_config(
        args.doc_slug, args.candidate_profile, args.candidate_arm, args.collection_prefix
    )
    all_metrics = load_all_run_metrics([baseline_cfg, candidate_cfg])
    by_id = {m["run_id"]: m for m in all_metrics}

    baseline = by_id.get(baseline_cfg.run_id)
    candidate = by_id.get(candidate_cfg.run_id)
    if baseline is None or candidate is None:
        print("Missing checker output for one or both runs.", file=sys.stderr)
        sys.exit(1)

    deltas = compute_deltas(baseline, candidate)
    print(f"\nBaseline : {baseline_cfg.run_id}")
    print(f"Candidate: {candidate_cfg.run_id}\n")
    for d in deltas:
        arrow = "↑" if d["improved"] else "↓"
        print(
            f"  {d['metric']:40s}  {d['baseline_val']:6.1f} → {d['candidate_val']:6.1f}"
            f"  ({d['delta']:+.1f} {arrow})"
        )


async def _cmd_full(args: argparse.Namespace) -> None:
    checker_config = RAGCheckerConfig(
        extractor_name=args.extractor,
        checker_name=args.checker,
        batch_size_extractor=args.batch_size_extractor,
        batch_size_checker=args.batch_size_checker,
    )
    matrix = await run_full_matrix(
        args.doc_slug,
        args.source,
        checker_config,
        gold_path=args.gold_path,
        profiles=tuple(args.profiles),
    )
    print(f"full pipeline done: {len(matrix)} runs")


async def _main(args: argparse.Namespace) -> None:
    dispatch = {
        "index": _cmd_index,
        "infer": _cmd_infer,
        "assemble": _cmd_assemble,
        "score": _cmd_score,
        "compare": _cmd_compare,
        "full": _cmd_full,
    }
    await dispatch[args.command](args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
