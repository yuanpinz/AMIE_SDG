from __future__ import annotations

from pydantic import ValidationError
import pytest

from amie_self_play.models import (
    CandidateSemanticRating,
    PatientActorRaterOutput,
    SemanticMatch,
    Vignette,
)
from amie_self_play.rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
)
from amie_self_play.simulation import SimulationSession

from conftest import (
    FakeLLM,
    accuracy_payload,
    ddx_payload,
    evaluation_responses,
    json_text,
    moderator_payload,
    quality_payload,
    vignette_payload,
)


async def collect_events():
    events: list[dict] = []

    async def send(event: dict) -> None:
        events.append(event)

    return events, send


def test_vignette_requires_three_to_ten_unique_accepted_diagnoses_including_truth() -> None:
    payload = vignette_payload()
    payload["accepted_differential_diagnoses"] = [
        "SECRET_GROUND_TRUTH",
        " 颈椎神经根病 ",
        "颈椎神经根病",
        "糖尿病周围神经病变",
    ]
    vignette = Vignette.model_validate(payload)
    assert vignette.accepted_differential_diagnoses == [
        "SECRET_GROUND_TRUTH",
        "颈椎神经根病",
        "糖尿病周围神经病变",
    ]

    payload["accepted_differential_diagnoses"] = ["诊断甲", "诊断乙", "诊断丙"]
    with pytest.raises(ValidationError, match="ground_truth_diagnosis"):
        Vignette.model_validate(payload)


def test_rubric_validator_rejects_missing_duplicate_and_out_of_range_scores() -> None:
    valid = quality_payload(PATIENT_ACTOR_RUBRIC)
    valid["scores"][0]["criterion_id"] = "  gmcpq_being_polite  "
    valid["scores"][0]["label"] = "harmless display field"
    valid["scores"][17]["score"] = "N/A"
    normalized = PatientActorRaterOutput.model_validate(valid)
    assert len(normalized.scores) == 26
    assert normalized.scores[0].criterion_id == "gmcpq_being_polite"
    assert normalized.scores[17].score is None

    missing = {"scores": valid["scores"][:-1]}
    with pytest.raises(ValidationError, match="criterion IDs do not match"):
        PatientActorRaterOutput.model_validate(missing)

    duplicate = {"scores": [*valid["scores"][:-1], valid["scores"][0]]}
    with pytest.raises(ValidationError, match="duplicate criterion IDs"):
        PatientActorRaterOutput.model_validate(duplicate)

    invalid_score = quality_payload(PATIENT_ACTOR_RUBRIC)
    invalid_score["scores"][0]["score"] = 6
    with pytest.raises(ValidationError, match="outside its scale"):
        PatientActorRaterOutput.model_validate(invalid_score)

    with_na = quality_payload(
        PATIENT_ACTOR_RUBRIC, na_id="pccbp_acknowledging_mistakes"
    )
    assert PatientActorRaterOutput.model_validate(with_na).scores[17].score is None


def test_service_computes_binary_top_k_from_all_supported_semantic_levels() -> None:
    candidates = [
        CandidateSemanticRating(
            rank=1,
            candidate="腕部神经卡压",
            ground_truth=SemanticMatch(
                level="highly_related",
                matched_diagnosis="腕管综合征",
                rationale="相关但不够明确。",
            ),
            accepted_differential=SemanticMatch(
                level="no_match", matched_diagnosis=None, rationale="不匹配。"
            ),
        ),
        CandidateSemanticRating(
            rank=2,
            candidate="C6 cervical radiculopathy",
            ground_truth=SemanticMatch(
                level="no_match", matched_diagnosis=None, rationale="不匹配。"
            ),
            accepted_differential=SemanticMatch(
                level="more_specific",
                matched_diagnosis="颈椎神经根病",
                rationale="是更具体的节段诊断。",
            ),
        ),
        CandidateSemanticRating(
            rank=3,
            candidate="DM neuropathy",
            ground_truth=SemanticMatch(
                level="no_match", matched_diagnosis=None, rationale="不匹配。"
            ),
            accepted_differential=SemanticMatch(
                level="synonym",
                matched_diagnosis="糖尿病周围神经病变",
                rationale="是明确缩写和同义表达。",
            ),
        ),
    ]
    result = SimulationSession._build_accuracy_evaluation(candidates)
    assert result.ground_truth_hits.model_dump() == {
        "top_1": True,
        "top_3": True,
        "top_10": True,
    }
    assert result.accepted_differential_hits.model_dump() == {
        "top_1": False,
        "top_3": True,
        "top_10": True,
    }


@pytest.mark.asyncio
async def test_one_evaluation_group_failure_is_partial_and_does_not_block_review() -> None:
    responses = evaluation_responses()
    responses["PATIENT_ACTOR_RATER"] = ["not json"]
    responses["JSON_REPAIR_PATIENT_ACTOR_RATER"] = ["still not json"]
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["右手麻木。"],
        DOCTOR=["考虑腕管综合征，建议面诊。"],
        MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text(ddx_payload())],
        CRITIC=["补充建议。"],
        **responses,
    )
    events, send = await collect_events()

    session = SimulationSession(fake, send)
    await session.start("腕管综合征")

    evaluation = next(
        event for event in events if event["type"] == "evaluation_completed"
    )
    assert evaluation["status"] == "partial"
    assert evaluation["patient_actor"] is None
    assert evaluation["accuracy"] is not None
    assert evaluation["errors"]["patient_actor"]["category"] == "invalid_output"
    assert events[-1]["type"] == "round_review_completed"
    assert events[-1]["can_refine"] is True


@pytest.mark.asyncio
async def test_all_evaluation_groups_can_fail_without_blocking_round_completion() -> None:
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["右手麻木。"],
        DOCTOR=["考虑腕管综合征，建议面诊。"],
        MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text(ddx_payload())],
        CRITIC=["补充建议。"],
    )
    events, send = await collect_events()

    session = SimulationSession(fake, send)
    await session.start("腕管综合征")

    evaluation = next(
        event for event in events if event["type"] == "evaluation_completed"
    )
    assert evaluation["status"] == "failed"
    assert set(evaluation["errors"]) == {
        "accuracy",
        "patient_actor",
        "specialist",
        "auto_paces",
    }
    assert events[-1]["type"] == "round_review_completed"
    assert session.rounds[0].evaluation is not None


@pytest.mark.asyncio
async def test_invalid_complete_rubric_gets_exactly_one_json_repair() -> None:
    responses = evaluation_responses()
    responses["PATIENT_ACTOR_RATER"] = [
        json_text({"scores": quality_payload(PATIENT_ACTOR_RUBRIC)["scores"][:-1]})
    ]
    responses["JSON_REPAIR_PATIENT_ACTOR_RATER"] = [
        json_text(quality_payload(PATIENT_ACTOR_RUBRIC))
    ]
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["右手麻木。"],
        DOCTOR=["考虑腕管综合征，建议面诊。"],
        MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text(ddx_payload())],
        CRITIC=["补充建议。"],
        **responses,
    )
    events, send = await collect_events()

    session = SimulationSession(fake, send)
    await session.start("腕管综合征")

    evaluation = next(
        event for event in events if event["type"] == "evaluation_completed"
    )
    assert evaluation["status"] == "complete"
    assert [call["role"] for call in fake.calls].count(
        "JSON_REPAIR_PATIENT_ACTOR_RATER"
    ) == 1
    repair_call = next(
        call
        for call in fake.calls
        if call["role"] == "JSON_REPAIR_PATIENT_ACTOR_RATER"
    )
    repair_prompt = "\n".join(
        message["content"] for message in repair_call["messages"]
    )
    assert "首次校验错误" in repair_prompt
    assert "paces_maintaining_patient_welfare" in repair_prompt
    criteria = evaluation["patient_actor"]["criteria"]
    assert len(criteria) == 26
    assert {item["polarity"] for item in criteria} == {"positive"}


def test_paper_axis_counts_are_fixed() -> None:
    assert len(PATIENT_ACTOR_RUBRIC) == 26
    assert len(SPECIALIST_RUBRIC) == 32
    assert len(AUTO_PACES_RUBRIC) == 4
