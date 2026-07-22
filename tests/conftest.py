from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from amie_self_play.rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
)


def vignette_payload(condition: str = "SECRET_CONDITION") -> dict[str, Any]:
    return {
        "condition": condition,
        "summary": "SECRET_SUMMARY",
        "demographics": "40 岁，女性",
        "symptoms": ["右手麻木"],
        "past_medical_history": "无",
        "past_surgical_history": "无",
        "past_social_history": "办公室工作",
        "medication": "无",
        "allergy": "无",
        "family_history": "SECRET_FAMILY_HISTORY",
        "patient_questions": ["是什么原因？"],
        "ground_truth_diagnosis": "SECRET_GROUND_TRUTH",
        "accepted_differential_diagnoses": [
            "SECRET_GROUND_TRUTH",
            "SECRET_ACCEPTED_DIFFERENTIAL",
            "颈椎神经根病",
        ],
        "reference_management_plan": ["SECRET_MANAGEMENT_PLAN"],
    }


def moderator_payload(
    ended: bool = True, reason: str = "诊断、治疗建议和患者问题均已完成"
) -> dict[str, Any]:
    return {
        "ended": ended,
        "reason": reason,
        "diagnosis_complete": ended,
        "treatment_plan_complete": ended,
        "patient_questions_resolved": ended,
        "farewell_detected": False,
    }


def ddx_payload(*diagnoses: str) -> dict[str, Any]:
    return {
        "differential_diagnoses": list(
            diagnoses or ("腕管综合征", "颈椎神经根病", "糖尿病周围神经病变")
        )
    }


def accuracy_payload(
    diagnoses: tuple[str, ...] = ("腕管综合征", "颈椎神经根病", "糖尿病周围神经病变"),
    *,
    ground_truth: str = "SECRET_GROUND_TRUTH",
) -> dict[str, Any]:
    candidates = []
    for rank, diagnosis in enumerate(diagnoses, start=1):
        accepted_match = diagnosis if diagnosis == "颈椎神经根病" else None
        candidates.append(
            {
                "rank": rank,
                "candidate": diagnosis,
                "ground_truth": {
                    "level": "exact" if diagnosis == ground_truth else "no_match",
                    "matched_diagnosis": ground_truth if diagnosis == ground_truth else None,
                    "rationale": "与 ground truth 比较。",
                },
                "accepted_differential": {
                    "level": "exact" if accepted_match else "no_match",
                    "matched_diagnosis": accepted_match,
                    "rationale": "与 accepted differential 比较。",
                },
            }
        )
    return {"candidates": candidates}


def quality_payload(rubric, *, na_id: str | None = None) -> dict[str, Any]:
    scores = []
    for item in rubric:
        scores.append(
            {
                "criterion_id": item.criterion_id,
                "score": None if item.criterion_id == na_id else item.options[-1].score,
                "evidence": f"{item.label} 的 transcript 证据。",
            }
        )
    return {"scores": scores}


def evaluation_responses(rounds: int = 1) -> dict[str, list[str]]:
    return {
        "ACCURACY_RATER": [json_text(accuracy_payload())] * rounds,
        "PATIENT_ACTOR_RATER": [
            json_text(quality_payload(PATIENT_ACTOR_RUBRIC))
        ]
        * rounds,
        "SPECIALIST_RATER": [json_text(quality_payload(SPECIALIST_RUBRIC))]
        * rounds,
        "AUTO_PACES_RATER": [json_text(quality_payload(AUTO_PACES_RUBRIC))]
        * rounds,
    }


class FakeLLM:
    def __init__(self, **responses: list[str]) -> None:
        self.responses = defaultdict(deque)
        for key, values in responses.items():
            self.responses[key].extend(values)
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def role_for(messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        marker = system.split("\n", 1)[0]
        if marker.startswith("[ROLE:JSON_REPAIR:"):
            return "JSON_REPAIR_" + marker.removeprefix("[ROLE:JSON_REPAIR:").removesuffix("]")
        return marker.removeprefix("[ROLE:").removesuffix("]")

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        model_name: str | None = None,
    ) -> str:
        role = self.role_for(messages)
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model_name": model_name,
            }
        )
        if not self.responses[role]:
            raise AssertionError(f"No fake response queued for {role}")
        return self.responses[role].popleft()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)
