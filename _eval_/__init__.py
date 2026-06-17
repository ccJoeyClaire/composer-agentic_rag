"""Standalone eval pipelines built around BEIR-style datasets.

Layout::

    data_preparing/   — corpus, queries, qrels, pooled subset
    scoring/          — Recall@k / MRR@k / nDCG@k
    rag_eval/         — RAG profile comparison (pipeline + run)
    agent_eval/       — agent pattern comparison (pipeline + judge + run)
    reflection_eval/  — per-pattern reflection eval (crag, self_rag, …)

Run RAG eval (config: ``rag_eval_arg_config.yaml``)::

    python -m _eval_.rag_eval.run

Run agent eval::

    python -m _eval_.agent_eval.run

Run reflection eval (baseline vs one pattern)::

    python -m _eval_.reflection_eval.crag.run
    python -m _eval_.reflection_eval.self_rag.run
"""
