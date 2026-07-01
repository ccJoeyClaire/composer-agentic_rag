"""Batch agent-graph infer for ``EvalRunner(mode='agent')``."""

from __future__ import annotations

from eval.base import AgentInferArtifact, EvalRunner, GoldSample


async def run_agent_infer(
    runner: EvalRunner,
    gold: list[GoldSample],
) -> list[AgentInferArtifact]:
    """Invoke ``user_question`` for each gold row (see ``get_start.agent_example``).

    TODO: build_graph from ``runner.agent_config`` + ``runner.collection``;
    extract ``retrieved_context`` from state metadata and strip Think/Action fences
    from the final AIMessage.
    """
    raise NotImplementedError
