from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from amie_self_play.llm import ModelAPIError
from amie_self_play.prompts import DOCTOR_OPENING
from amie_self_play.simulation import SimulationSession, SimulationStateError

from conftest import (
    FakeLLM,
    ddx_payload,
    evaluation_responses,
    json_text,
    moderator_payload,
    vignette_payload,
)


async def collect_events():
    events: list[dict] = []

    async def send(event: dict) -> None:
        events.append(event)

    return events, send


@pytest.mark.asyncio
async def test_strict_baseline_order_and_context_boundaries() -> None:
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["我最近手有些麻。", "这一轮还是手麻。"],
        DOCTOR=["ROUND_ONE_SENTINEL：建议线下面诊。", "我会针对性改进问诊。"],
        MODERATOR=[json_text(moderator_payload()), json_text(moderator_payload())],
        DDX=[json_text(ddx_payload()), json_text(ddx_payload())],
        CRITIC=[
            "CRITIQUE_SENTINEL：下一轮更具体地询问感觉分布。",
            "第二轮评价。",
        ],
        **evaluation_responses(2),
    )
    events, send = await collect_events()
    session = SimulationSession(fake, send)

    await session.start("SECRET_CONDITION", "model-beta")

    assert [call["role"] for call in fake.calls] == [
        "VIGNETTE",
        "PATIENT",
        "DOCTOR",
        "MODERATOR",
        "DDX",
        "CRITIC",
        "ACCURACY_RATER",
        "PATIENT_ACTOR_RATER",
        "SPECIALIST_RATER",
        "AUTO_PACES_RATER",
    ]
    assert all(call["model_name"] == "model-beta" for call in fake.calls)
    dialogue_events = [
        event for event in events if event["type"] == "message_completed"
    ]
    assert [(event["role"], event["content"]) for event in dialogue_events] == [
        ("doctor", DOCTOR_OPENING),
        ("patient", "我最近手有些麻。"),
        ("doctor", "ROUND_ONE_SENTINEL：建议线下面诊。"),
    ]

    doctor_round_1 = next(call for call in fake.calls if call["role"] == "DOCTOR")
    doctor_round_1_text = "\n".join(
        message["content"] for message in doctor_round_1["messages"]
    )
    assert "SECRET_CONDITION" not in doctor_round_1_text
    assert "SECRET_GROUND_TRUTH" not in doctor_round_1_text
    assert "SECRET_FAMILY_HISTORY" not in doctor_round_1_text
    assert "SECRET_GROUND_TRUTH" not in doctor_round_1_text
    assert "SECRET_ACCEPTED_DIFFERENTIAL" not in doctor_round_1_text
    assert "至少两个" not in doctor_round_1_text
    assert "即时医患聊天" in doctor_round_1_text
    assert "只输出自然、可直接发送给患者的纯文本" in doctor_round_1_text
    assert "Markdown 标题、强调符号、代码块、表格或项目符号" in doctor_round_1_text

    patient_call = next(call for call in fake.calls if call["role"] == "PATIENT")
    assert "SECRET_GROUND_TRUTH" in patient_call["messages"][0]["content"]
    moderator_call = next(call for call in fake.calls if call["role"] == "MODERATOR")
    moderator_text = "\n".join(
        message["content"] for message in moderator_call["messages"]
    )
    assert "SECRET_CONDITION" not in moderator_text
    assert "SECRET_GROUND_TRUTH" not in moderator_text
    assert "SECRET_ACCEPTED_DIFFERENTIAL" not in moderator_text

    critic_call = next(call for call in fake.calls if call["role"] == "CRITIC")
    critic_text = "\n".join(message["content"] for message in critic_call["messages"])
    assert "SECRET_CONDITION" in critic_text
    assert "SECRET_GROUND_TRUTH" in critic_text
    assert "至少两个" in critic_text

    ddx_call = next(call for call in fake.calls if call["role"] == "DDX")
    ddx_text = "\n".join(message["content"] for message in ddx_call["messages"])
    assert "SECRET_CONDITION" not in ddx_text
    assert "SECRET_GROUND_TRUTH" not in ddx_text
    assert "SECRET_FAMILY_HISTORY" not in ddx_text
    assert "SECRET_ACCEPTED_DIFFERENTIAL" not in ddx_text
    assert "3–10" in ddx_text

    critique_event = next(event for event in events if event["type"] == "critique_completed")
    evaluation_event = next(
        event for event in events if event["type"] == "evaluation_completed"
    )
    review_event = next(event for event in events if event["type"] == "round_review_completed")
    ready_event = next(event for event in events if event["type"] == "round_review_ready")
    assert critique_event["critique"].startswith("CRITIQUE_SENTINEL")
    assert evaluation_event["status"] == "complete"
    assert len(evaluation_event["patient_actor"]["criteria"]) == 26
    assert len(evaluation_event["specialist"]["criteria"]) == 32
    assert len(evaluation_event["auto_paces"]["criteria"]) == 4
    assert ready_event["can_refine"] is True
    assert review_event["can_refine"] is True

    event_types = [event["type"] for event in events]
    assert event_types.index("ddx_completed") < event_types.index("critique_completed")
    assert event_types.index("critique_completed") < event_types.index(
        "round_review_ready"
    )
    assert event_types.index("round_review_ready") < event_types.index(
        "evaluation_completed"
    )
    assert event_types.index("evaluation_completed") < event_types.index(
        "round_review_completed"
    )

    patient_rater = next(
        call for call in fake.calls if call["role"] == "PATIENT_ACTOR_RATER"
    )
    patient_rater_text = "\n".join(
        message["content"] for message in patient_rater["messages"]
    )
    assert "SECRET_GROUND_TRUTH" in patient_rater_text
    assert "CRITIQUE_SENTINEL" not in patient_rater_text
    specialist_rater = next(
        call for call in fake.calls if call["role"] == "SPECIALIST_RATER"
    )
    specialist_text = "\n".join(
        message["content"] for message in specialist_rater["messages"]
    )
    assert "SECRET_MANAGEMENT_PLAN" in specialist_text
    assert "CRITIQUE_SENTINEL" not in specialist_text
    auto_rater = next(
        call for call in fake.calls if call["role"] == "AUTO_PACES_RATER"
    )
    auto_text = "\n".join(
        message["content"] for message in auto_rater["messages"]
    )
    assert "SECRET_GROUND_TRUTH" not in auto_text
    assert "CRITIQUE_SENTINEL" not in auto_text

    await session.refine()

    patient_calls = [call for call in fake.calls if call["role"] == "PATIENT"]
    second_patient_text = "\n".join(
        message["content"] for message in patient_calls[1]["messages"]
    )
    assert "ROUND_ONE_SENTINEL" not in second_patient_text
    assert "CRITIQUE_SENTINEL" not in second_patient_text
    assert "gmcpq_being_polite" not in second_patient_text

    doctor_calls = [call for call in fake.calls if call["role"] == "DOCTOR"]
    second_doctor_text = "\n".join(
        message["content"] for message in doctor_calls[1]["messages"]
    )
    assert "ROUND_ONE_SENTINEL" in second_doctor_text
    assert "CRITIQUE_SENTINEL" in second_doctor_text
    assert "gmcpq_being_polite" not in second_doctor_text
    assert "不得照搬其中的 Markdown" in second_doctor_text


class BlockingEvaluationLLM(FakeLLM):
    evaluation_roles = {
        "ACCURACY_RATER",
        "PATIENT_ACTOR_RATER",
        "SPECIALIST_RATER",
        "AUTO_PACES_RATER",
    }

    def __init__(self, **responses: list[str]) -> None:
        super().__init__(**responses)
        self.evaluation_started = asyncio.Event()
        self.release_evaluation = asyncio.Event()

    async def complete(self, messages, **kwargs):
        role = self.role_for(messages)
        if role in self.evaluation_roles:
            self.evaluation_started.set()
            await self.release_evaluation.wait()
        return await super().complete(messages, **kwargs)


@pytest.mark.asyncio
async def test_refine_can_start_while_previous_evaluation_runs_in_background() -> None:
    fake = BlockingEvaluationLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["第一轮患者回复。", "第二轮患者回复。"],
        DOCTOR=["第一轮医生回复。", "第二轮医生回复。"],
        MODERATOR=[json_text(moderator_payload())] * 2,
        DDX=[json_text(ddx_payload())] * 2,
        CRITIC=["第一轮 Critic。", "第二轮 Critic。"],
        **evaluation_responses(2),
    )
    events, send = await collect_events()
    first_ready = asyncio.Event()
    second_ready = asyncio.Event()
    evaluations_completed = asyncio.Event()

    async def tracked_send(event: dict) -> None:
        await send(event)
        if event["type"] == "round_review_ready":
            (first_ready if event["round"] == 1 else second_ready).set()
        if len(
            [item for item in events if item["type"] == "evaluation_completed"]
        ) == 2:
            evaluations_completed.set()

    session = SimulationSession(fake, tracked_send)
    await session.launch_start("腕管综合征")
    await asyncio.wait_for(first_ready.wait(), 1)
    await asyncio.wait_for(fake.evaluation_started.wait(), 1)

    assert session.busy is False
    assert session.rounds[0].evaluation is None

    await session.launch_refine()
    await asyncio.wait_for(second_ready.wait(), 1)
    assert len(session.rounds) == 2

    fake.release_evaluation.set()
    await asyncio.wait_for(evaluations_completed.wait(), 1)
    assert all(round_state.evaluation is not None for round_state in session.rounds)
    await session.disconnect()


@pytest.mark.asyncio
async def test_three_round_limit_and_each_round_restarts_from_opening() -> None:
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["患者回复 1", "患者回复 2", "患者回复 3"],
        DOCTOR=["医生回复 1", "医生回复 2", "医生回复 3"],
        MODERATOR=[json_text(moderator_payload())] * 3,
        DDX=[json_text(ddx_payload())] * 3,
        CRITIC=["反馈 1", "反馈 2", "最终轮评价"],
        **evaluation_responses(3),
    )
    events, send = await collect_events()
    session = SimulationSession(fake, send)

    await session.start("腕管综合征")
    await session.refine()
    await session.refine()

    assert len(session.rounds) == 3
    assert len(session.critiques) == 3
    assert all(item.messages[0].content == DOCTOR_OPENING for item in session.rounds)
    assert [item.round_number for item in session.rounds] == [1, 2, 3]
    assert [item.round_number for item in session.critiques] == [1, 2, 3]
    assert all(item.ddx is not None for item in session.rounds)
    doctor_calls = [call for call in fake.calls if call["role"] == "DOCTOR"]
    third_doctor_context = "\n".join(
        message["content"] for message in doctor_calls[2]["messages"]
    )
    assert "医生回复 1" in third_doctor_context
    assert "医生回复 2" in third_doctor_context
    assert "反馈 1" in third_doctor_context
    assert "反馈 2" in third_doctor_context
    patient_calls = [call for call in fake.calls if call["role"] == "PATIENT"]
    third_patient_context = "\n".join(
        message["content"] for message in patient_calls[2]["messages"]
    )
    assert "医生回复 1" not in third_patient_context
    assert "医生回复 2" not in third_patient_context
    assert "反馈 1" not in third_patient_context
    with pytest.raises(SimulationStateError, match="最多三轮"):
        await session.refine()


@pytest.mark.asyncio
async def test_thirty_utterance_guard_truncates_without_counting_moderator() -> None:
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=[f"患者 {index}" for index in range(15)],
        DOCTOR=[f"医生 {index}" for index in range(14)],
        MODERATOR=[json_text(moderator_payload(False, "继续"))] * 14,
        DDX=[json_text(ddx_payload())],
        CRITIC=["截断轮评价"],
        **evaluation_responses(),
    )
    events, send = await collect_events()
    session = SimulationSession(fake, send)

    await session.start("腕管综合征")

    assert len(session.rounds) == 1
    assert session.rounds[0].status == "truncated"
    assert len(session.rounds[0].messages) == 30
    assert session.rounds[0].messages[-1].role == "patient"
    completed = next(event for event in events if event["type"] == "dialogue_completed")
    assert completed["utterance_count"] == 30
    assert completed["can_refine"] is False
    review = next(event for event in events if event["type"] == "round_review_completed")
    assert review["can_refine"] is False
    assert session.rounds[0].ddx is not None
    assert len([call for call in fake.calls if call["role"] == "MODERATOR"]) == 14


@pytest.mark.asyncio
async def test_structured_outputs_are_repaired_once() -> None:
    fake = FakeLLM(
        VIGNETTE=["not json"],
        JSON_REPAIR_VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["手麻。"],
        DOCTOR=["初步判断并给出治疗建议。"],
        MODERATOR=["```json\n{broken}\n```"],
        JSON_REPAIR_MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text({"differential_diagnoses": ["只有一个"]})],
        JSON_REPAIR_DDX=[json_text(ddx_payload())],
        CRITIC=["评价完成"],
        **evaluation_responses(),
    )
    events, send = await collect_events()
    session = SimulationSession(fake, send)

    await session.start("腕管综合征")

    assert [call["role"] for call in fake.calls] == [
        "VIGNETTE",
        "JSON_REPAIR_VIGNETTE",
        "PATIENT",
        "DOCTOR",
        "MODERATOR",
        "JSON_REPAIR_MODERATOR",
        "DDX",
        "JSON_REPAIR_DDX",
        "CRITIC",
        "ACCURACY_RATER",
        "PATIENT_ACTOR_RATER",
        "SPECIALIST_RATER",
        "AUTO_PACES_RATER",
    ]
    assert session.rounds[0].status == "completed"


@pytest.mark.asyncio
async def test_moderator_enforces_completion_flags_and_farewell() -> None:
    incomplete = moderator_payload(True, "错误地提前结束")
    incomplete["treatment_plan_complete"] = False
    farewell = moderator_payload(False, "患者告别")
    farewell.update(
        {
            "diagnosis_complete": False,
            "treatment_plan_complete": False,
            "patient_questions_resolved": False,
            "farewell_detected": True,
        }
    )
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["我想了解原因。", "谢谢医生，再见。"],
        DOCTOR=["先继续了解。", "再见。"],
        MODERATOR=[json_text(incomplete), json_text(farewell)],
        DDX=[json_text(ddx_payload())],
        CRITIC=["评价完成"],
        **evaluation_responses(),
    )
    events, send = await collect_events()
    session = SimulationSession(fake, send)

    await session.start("腕管综合征")

    assert len(session.rounds[0].messages) == 5
    assert session.rounds[0].moderator is not None
    assert session.rounds[0].moderator.farewell_detected is True
    moderator_events = [event for event in events if event["type"] == "moderator_result"]
    assert moderator_events[0]["ended"] is False
    assert moderator_events[1]["ended"] is True


class BlockingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def complete(self, messages, **kwargs):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_user_stop_cancels_background_generation_and_emits_stopped() -> None:
    llm = BlockingLLM()
    events, send = await collect_events()
    session = SimulationSession(llm, send)
    await session.launch_start("腕管综合征")
    await asyncio.wait_for(llm.started.wait(), 1)

    await session.stop()

    assert llm.cancelled is True
    assert session.busy is False
    assert events[-1]["type"] == "stopped"


@pytest.mark.asyncio
async def test_disconnect_cancels_without_trying_to_send_stopped() -> None:
    llm = BlockingLLM()
    events, send = await collect_events()
    session = SimulationSession(llm, send)
    await session.launch_start("腕管综合征")
    await asyncio.wait_for(llm.started.wait(), 1)

    await session.disconnect()

    assert llm.cancelled is True
    assert all(event["type"] != "stopped" for event in events)


class FailingLLM:
    async def complete(self, messages, **kwargs):
        raise ModelAPIError("网关超时", code="network_error", retryable=True)


@pytest.mark.asyncio
async def test_api_failure_is_sent_as_user_visible_error_event() -> None:
    events, send = await collect_events()
    session = SimulationSession(FailingLLM(), send)
    await session.launch_start("腕管综合征")
    task = session._task
    assert task is not None
    with suppress(asyncio.CancelledError):
        await task

    error = next(event for event in events if event["type"] == "error")
    assert error["code"] == "network_error"
    assert error["retryable"] is True
    assert "网关超时" in error["message"]
