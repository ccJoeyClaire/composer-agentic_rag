"""Standalone eval pipelines built around BEIR-style datasets.

Layout (lifecycle)::

    configs/          — per-dataset rag_eval_arg_config_*.yaml overrides
    data_preparing/   — corpus, queries, qrels, pooled subset
    scoring/          — Recall@k / MRR@k / nDCG@k
    rag_eval/         — RAG profile comparison (pipeline + run)
    agent_eval/       — agent pattern comparison (pipeline + judge + run)
    reflection_eval/  — per-pattern reflection eval (crag, self_rag, …)
    analysis/         — post-eval: load_results + compare, visual/, replay/

Run RAG eval (config: ``rag_eval_arg_config.yaml``)::

    python -m _eval_.rag_eval.run

Run agent eval::

    python -m _eval_.agent_eval.run

Run reflection eval (baseline vs one pattern)::

    python -m _eval_.reflection_eval.crag.run
    python -m _eval_.reflection_eval.self_rag.run

Post-eval analysis::

    # Offline metrics (JSON artifacts)
    from _eval_.analysis import load_run_result, winrate_matrix

    # Live replay (existing Qdrant collections)
    python -m _eval_.analysis.replay.query --profile … --query-id …
    python -m _eval_.analysis.replay.inspect_collection --collection …
"""
