"""Per-pattern reflection eval pipelines (CRAG, Self-RAG, Feedback).

Layout::

    shared.py       — RecordingRetriever, BEIR agent scoring, index helpers
    beir_runner.py  — pooled BEIR setup shared by CRAG / Self-RAG
    crag/           — react vs react_crag (+ CRAG signal aggregates)
    self_rag/       — react vs react_self_rag (+ Self-RAG signal aggregates)

Run one pattern::

    python -m _eval_.reflection_eval.crag.run
    python -m _eval_.reflection_eval.self_rag.run

Compare all patterns in one table (legacy entry point)::

    python -m _eval_.agent_eval.run
"""
