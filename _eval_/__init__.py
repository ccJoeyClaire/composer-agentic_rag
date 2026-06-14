"""Standalone RAG eval pipeline built around BEIR-style datasets.

Pipeline stages (each in its own module, composed by :mod:`_eval_.run`):

1. :mod:`_eval_.beir`     — stream corpus / queries / qrels (graded relevance).
2. :mod:`_eval_.pooling`  — pick a query subset and the pooled doc set to index.
3. :mod:`_eval_.pipeline` — index the pool per RAG profile, then run queries.
4. :mod:`_eval_.metrics`  — Recall@k / MRR@k / nDCG@k on the ranked doc ids.

Run it with ``python -m _eval_.run --dataset trec-covid``.
"""
