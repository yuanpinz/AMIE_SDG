from __future__ import annotations

import json

from .llm import ChatMessage
from .models import CritiqueState, DDxResult, DialogueMessage, RoundState, Vignette
from .prompt_config import PromptCatalog, load_prompt_catalog
from .rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
    CriterionDefinition,
    prompt_rubric,
)


DEFAULT_PROMPTS = load_prompt_catalog()
DOCTOR_OPENING = DEFAULT_PROMPTS.doctor_opening
VIGNETTE_ONE_SHOT = DEFAULT_PROMPTS.example("vignette_one_shot")


def rubric_schema(rubric: tuple[CriterionDefinition, ...]) -> str:
    entries = []
    for item in rubric:
        allowed = "|".join(str(option.score) for option in item.options)
        entries.append(
            {
                "criterion_id": item.criterion_id,
                "score": f"{allowed}|null",
                "evidence": "one short evidence sentence",
            }
        )
    return json.dumps({"scores": entries}, ensure_ascii=False, indent=2)


VIGNETTE_SCHEMA = DEFAULT_PROMPTS.schema("vignette")
MODERATOR_SCHEMA = DEFAULT_PROMPTS.schema("moderator")
DDX_SCHEMA = DEFAULT_PROMPTS.schema("ddx")
ACCURACY_SCHEMA = DEFAULT_PROMPTS.schema("accuracy")
PATIENT_ACTOR_SCHEMA = DEFAULT_PROMPTS.schema(
    "patient_actor", patient_actor_rubric_schema=rubric_schema(PATIENT_ACTOR_RUBRIC)
)
SPECIALIST_SCHEMA = DEFAULT_PROMPTS.schema(
    "specialist", specialist_rubric_schema=rubric_schema(SPECIALIST_RUBRIC)
)
AUTO_PACES_SCHEMA = DEFAULT_PROMPTS.schema(
    "auto_paces", auto_paces_rubric_schema=rubric_schema(AUTO_PACES_RUBRIC)
)


def _catalog(prompts: PromptCatalog | None) -> PromptCatalog:
    return prompts or DEFAULT_PROMPTS


def transcript(
    messages: list[DialogueMessage], prompts: PromptCatalog | None = None
) -> str:
    p = _catalog(prompts)
    labels = {
        "doctor": p.meta("doctor_label"),
        "patient": p.meta("patient_label"),
    }
    return "\n".join(f"{labels[item.role]}：{item.content}" for item in messages)


def improvement_context(
    prior_rounds: list[RoundState],
    critiques: list[CritiqueState],
    prompts: PromptCatalog | None = None,
) -> str:
    p = _catalog(prompts)
    if not prior_rounds:
        return ""
    critique_by_round = {item.round_number: item.content for item in critiques}
    sections: list[str] = []
    for item in prior_rounds:
        sections.append(
            p.meta(
                "prior_round",
                round=item.round_number,
                transcript=transcript(item.messages, p),
            )
        )
        if item.round_number in critique_by_round:
            sections.append(
                p.meta(
                    "critique",
                    round=item.round_number,
                    content=critique_by_round[item.round_number],
                )
            )
    return "\n\n".join(sections)


def vignette_messages(
    condition: str, prompts: PromptCatalog | None = None
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {"role": "system", "content": p.render("vignette", "system")},
        {
            "role": "user",
            "content": p.render(
                "vignette",
                "user",
                condition=condition,
                vignette_one_shot=p.example("vignette_one_shot"),
                schema=p.schema("vignette"),
            ),
        },
    ]


def patient_messages(
    vignette: Vignette,
    current_dialogue: list[DialogueMessage],
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render(
                "patient",
                "system",
                vignette_json=json.dumps(
                    vignette.model_dump(), ensure_ascii=False, indent=2
                ),
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "patient", "user", transcript=transcript(current_dialogue, p)
            ),
        },
    ]


def doctor_messages(
    current_dialogue: list[DialogueMessage],
    prior_rounds: list[RoundState],
    critiques: list[CritiqueState],
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    prior = improvement_context(prior_rounds, critiques, p)
    improvement = p.render("doctor", "improvement", prior=prior) if prior else ""
    return [
        {
            "role": "system",
            "content": p.render(
                "doctor", "system", improvement_context=improvement
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "doctor", "user", transcript=transcript(current_dialogue, p)
            ),
        },
    ]


def moderator_messages(
    current_dialogue: list[DialogueMessage], prompts: PromptCatalog | None = None
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render(
                "moderator", "system", schema=p.schema("moderator")
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "moderator", "user", transcript=transcript(current_dialogue, p)
            ),
        },
    ]


def ddx_messages(
    current_dialogue: list[DialogueMessage], prompts: PromptCatalog | None = None
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render("ddx", "system", schema=p.schema("ddx")),
        },
        {
            "role": "user",
            "content": p.render("ddx", "user", transcript=transcript(current_dialogue, p)),
        },
    ]


def critic_messages(
    condition: str,
    vignette: Vignette,
    current_dialogue: list[DialogueMessage],
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {"role": "system", "content": p.render("critic", "system")},
        {
            "role": "user",
            "content": p.render(
                "critic",
                "user",
                condition=condition,
                vignette_json=json.dumps(
                    vignette.model_dump(), ensure_ascii=False, indent=2
                ),
                transcript=transcript(current_dialogue, p),
            ),
        },
    ]


def accuracy_rater_messages(
    vignette: Vignette,
    ddx: DDxResult,
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    reference = {
        "ground_truth_diagnosis": vignette.ground_truth_diagnosis,
        "accepted_differential_diagnoses": vignette.accepted_differential_diagnoses,
        "differential_diagnoses": ddx.differential_diagnoses,
    }
    return [
        {
            "role": "system",
            "content": p.render(
                "accuracy_rater", "system", schema=p.schema("accuracy")
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "accuracy_rater",
                "user",
                reference_json=json.dumps(reference, ensure_ascii=False, indent=2),
            ),
        },
    ]


def patient_actor_rater_messages(
    vignette: Vignette,
    current_dialogue: list[DialogueMessage],
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render(
                "patient_actor_rater",
                "system",
                rubric=prompt_rubric(PATIENT_ACTOR_RUBRIC),
                schema=p.schema(
                    "patient_actor",
                    patient_actor_rubric_schema=rubric_schema(PATIENT_ACTOR_RUBRIC),
                ),
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "patient_actor_rater",
                "user",
                vignette_json=json.dumps(
                    vignette.model_dump(), ensure_ascii=False, indent=2
                ),
                transcript=transcript(current_dialogue, p),
            ),
        },
    ]


def specialist_rater_messages(
    vignette: Vignette,
    current_dialogue: list[DialogueMessage],
    ddx: DDxResult,
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    materials = {
        "vignette": vignette.model_dump(),
        "reference_management_plan": vignette.reference_management_plan,
        "accepted_differential_diagnoses": vignette.accepted_differential_diagnoses,
        "doctor_ddx": ddx.differential_diagnoses,
        "transcript": transcript(current_dialogue, p),
    }
    return [
        {
            "role": "system",
            "content": p.render(
                "specialist_rater",
                "system",
                rubric=prompt_rubric(SPECIALIST_RUBRIC),
                schema=p.schema(
                    "specialist",
                    specialist_rubric_schema=rubric_schema(SPECIALIST_RUBRIC),
                ),
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "specialist_rater",
                "user",
                materials_json=json.dumps(materials, ensure_ascii=False, indent=2),
            ),
        },
    ]


def auto_paces_rater_messages(
    current_dialogue: list[DialogueMessage], prompts: PromptCatalog | None = None
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render(
                "auto_paces_rater",
                "system",
                rubric=prompt_rubric(AUTO_PACES_RUBRIC),
                schema=p.schema(
                    "auto_paces", auto_paces_rubric_schema=rubric_schema(AUTO_PACES_RUBRIC)
                ),
            ),
        },
        {
            "role": "user",
            "content": p.render(
                "auto_paces_rater", "user", transcript=transcript(current_dialogue, p)
            ),
        },
    ]


def json_repair_messages(
    raw: str,
    schema: str,
    kind: str,
    validation_error: str = "",
    prompts: PromptCatalog | None = None,
) -> list[ChatMessage]:
    p = _catalog(prompts)
    return [
        {
            "role": "system",
            "content": p.render("json_repair", "system", kind=kind),
        },
        {
            "role": "user",
            "content": p.render(
                "json_repair",
                "user",
                validation_error=validation_error or "未提供",
                schema=schema,
                raw=raw,
            ),
        },
    ]
