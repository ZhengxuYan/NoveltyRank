"""Prompt assembly helpers for similarity report generation."""
from __future__ import annotations

import math
from typing import Any, List

from tinker_cookbook import renderers

from .metadata import SimilarPaper


def build_similarity_prompt(target: dict[str, Any], similars: List[SimilarPaper]) -> List[renderers.Message]:
    """Construct chat messages guiding the model to craft a similarity report."""
    target_title = target.get("Title", "")
    target_abstract = target.get("Abstract", "")
    target_arxiv_id = target.get("arXiv ID", "")

    similar_sections: List[str] = []
    actual_count = len(similars)
    for idx, paper in enumerate(similars, start=1):
        score_display = "N/A" if math.isnan(paper.similarity_score) else f"{paper.similarity_score:.4f}"
        similar_sections.append(
            "\n".join(
                [
                    f"{idx}. arXiv ID: {paper.arxiv_id}",
                    f"   Title: {paper.title or '[missing title]'}",
                    f"   Abstract: {paper.abstract or '[missing abstract]'}",
                    f"   Similarity score: {score_display}",
                ]
            )
        )

    similar_block = "\n".join(similar_sections) if similar_sections else "(No comparable papers found)."

    instruction = (
        "You are assisting a research team in summarising how a target arXiv paper relates to prior work. "
        f"Using the provided set of {actual_count} similar papers, craft an overall similarity report with the following sections (no per-paper breakdowns):\n"
        "1) Shared Themes:\n"
        "   - Bullet list describing common ideas.\n"
        "   - Each bullet: {{theme}} — cite supporting phrases and list relevant arXiv IDs in parentheses.\n"
        "   - Focus on topics, methods, tasks, data types, or motivations explicitly present in the texts.\n\n"
        "2) Overlap Snapshot:\n"
        "   - Summarise key overlap areas across the similar papers in 2-3 sentences.\n"
        "   - Mention which arXiv IDs drive each overlap signal, but do not analyse papers individually.\n\n"
        "3) Distinctive Aspects:\n"
        "   - Bullet list of target-specific ideas or claims not covered by the similar papers.\n"
        "   - Ground every bullet in explicit differences from the abstracts.\n\n"
        "4) Novelty Verdict:\n"
        "   - Conclude with High/Medium/Low and a one-sentence justification tied to the aggregated evidence.\n\n"
        "Do not invent details beyond the supplied abstracts. Keep the tone analytical."
    )

    user_prompt = (
        f"--- TARGET PAPER ({target_arxiv_id}) ---\n"
        f"Title: {target_title}\n"
        f"Abstract: {target_abstract}\n\n"
        "--- TOP 10 MOST SIMILAR PAPERS ---\n"
        f"{similar_block}\n\n"
        "--- OUTPUT FORMAT ---\n"
        "Shared Themes:\n"
        "Overlap Snapshot:\n"
        "Distinctive Aspects:\n"
        "Novelty Verdict: High/Medium/Low - justification\n"
    )

    return [
        renderers.Message(role="system", content=instruction),
        renderers.Message(role="user", content=user_prompt),
    ]


__all__ = ["build_similarity_prompt"]
