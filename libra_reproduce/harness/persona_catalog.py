from __future__ import annotations

from ..schemas import Persona

DEFAULT_PERSONA_SPECS: list[dict[str, object]] = [
    {"persona_id": "persona_main", "role": "main", "label": "Main", "is_main": True},
    {"persona_id": "persona_paper", "role": "paper", "label": "Paper", "is_main": False},
    {"persona_id": "persona_data", "role": "data", "label": "Data", "is_main": False},
    {"persona_id": "persona_analysis", "role": "analysis", "label": "Analysis", "is_main": False},
    {"persona_id": "persona_validation", "role": "validation", "label": "Validation", "is_main": False},
    {"persona_id": "persona_support", "role": "support", "label": "Support", "is_main": False},
]


def build_initial_personas() -> list[Persona]:
    """
    Build the configured visual roster (L1.5 §2).
    All personas start `idle` with `runtime_agent_id=None` and `task_ids=[]`;
    visual presence NEVER implies an active backend LLM subprocess.
    """
    return [
        Persona(
            persona_id=str(spec["persona_id"]),
            role=str(spec["role"]),
            label=str(spec["label"]),
            runtime_agent_id=None,
            state="idle",
            is_main=bool(spec["is_main"]),
            task_ids=[],
        )
        for spec in DEFAULT_PERSONA_SPECS
    ]
