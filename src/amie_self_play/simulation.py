from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM, ModelAPIError
from .models import (
    AccuracyEvaluation,
    AccuracyRaterOutput,
    AutoPacesRaterOutput,
    CandidateSemanticRating,
    CritiqueState,
    CriterionScore,
    DDxResult,
    DialogueMessage,
    EvaluationError,
    EvaluationState,
    ModeratorResult,
    PatientActorRaterOutput,
    QualityEvaluation,
    RubricRaterOutput,
    RoundState,
    SpecialistRaterOutput,
    TopKHits,
    Vignette,
)
from .prompts import (
    ACCURACY_SCHEMA,
    AUTO_PACES_SCHEMA,
    DDX_SCHEMA,
    DOCTOR_OPENING,
    MODERATOR_SCHEMA,
    PATIENT_ACTOR_SCHEMA,
    SPECIALIST_SCHEMA,
    VIGNETTE_SCHEMA,
    accuracy_rater_messages,
    auto_paces_rater_messages,
    critic_messages,
    ddx_messages,
    doctor_messages,
    json_repair_messages,
    moderator_messages,
    patient_actor_rater_messages,
    patient_messages,
    specialist_rater_messages,
    vignette_messages,
)
from .rendering import render_doctor_content
from .rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
    CriterionDefinition,
    rubric_by_id,
)


SendEvent = Callable[[dict[str, Any]], Awaitable[None]]
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class SimulationStateError(RuntimeError):
    pass


def parse_json_object(raw: str) -> object:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class SimulationSession:
    max_rounds = 3

    def __init__(
        self,
        llm: LLM,
        send_event: SendEvent,
        *,
        max_utterances: int = 30,
        model_name: str | None = None,
    ) -> None:
        self.llm = llm
        self._send_event = send_event
        self._send_lock = asyncio.Lock()
        self.max_utterances = max_utterances
        self.model_name = model_name
        self.condition: str | None = None
        self.vignette: Vignette | None = None
        self.rounds: list[RoundState] = []
        self.critiques: list[CritiqueState] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def emit(self, event_type: str, **payload: Any) -> None:
        async with self._send_lock:
            await self._send_event({"type": event_type, **payload})

    async def start(self, condition: str, model_name: str | None = None) -> None:
        condition = condition.strip()
        if not condition:
            raise SimulationStateError("请输入 medical condition")
        self.condition = condition
        if model_name:
            self.model_name = model_name
        self.vignette = None
        self.rounds = []
        self.critiques = []
        await self.emit(
            "simulation_started", condition=condition, model=self.model_name
        )
        await self.emit("phase_started", phase="vignette", round=0)
        self.vignette = await self._structured(
            vignette_messages(condition),
            Vignette,
            schema=VIGNETTE_SCHEMA,
            repair_kind="VIGNETTE",
            max_tokens=1800,
        )
        await self.emit("vignette_completed", vignette=self.vignette.model_dump())
        await self._run_dialogue(1)
        await self._review_round(self.rounds[-1])

    async def refine(self) -> None:
        if self.vignette is None or self.condition is None or not self.rounds:
            raise SimulationStateError("请先完成第一轮对话")
        if len(self.rounds) >= self.max_rounds:
            raise SimulationStateError("已达到最多三轮对话")
        current = self.rounds[-1]
        if current.status != "completed":
            raise SimulationStateError("只有自然结束的对话才能生成改进轮")
        if not any(item.round_number == current.round_number for item in self.critiques):
            raise SimulationStateError("请等待当前轮 Critic 评价完成")
        if current.evaluation is None:
            raise SimulationStateError("请等待当前轮 Evaluation 完成")
        await self._run_dialogue(current.round_number + 1)
        await self._review_round(self.rounds[-1])

    async def _run_dialogue(self, round_number: int) -> None:
        messages: list[DialogueMessage] = []
        await self.emit("phase_started", phase="dialogue", round=round_number)
        await self.emit("message_started", role="doctor", round=round_number)
        opening = DialogueMessage(role="doctor", content=DOCTOR_OPENING)
        messages.append(opening)
        await self.emit(
            "message_completed",
            role="doctor",
            round=round_number,
            content=opening.content,
            content_html=render_doctor_content(opening.content),
            index=1,
        )

        final_moderator: ModeratorResult | None = None
        while True:
            if len(messages) >= self.max_utterances:
                await self._complete_round(
                    round_number,
                    messages,
                    final_moderator,
                    status="truncated",
                    reason=f"达到 {self.max_utterances} 条医患消息上限",
                )
                return

            patient_text = await self._agent_turn(
                "patient",
                round_number,
                patient_messages(self._require_vignette(), messages),
            )
            messages.append(DialogueMessage(role="patient", content=patient_text))
            await self.emit(
                "message_completed",
                role="patient",
                round=round_number,
                content=patient_text,
                index=len(messages),
            )
            if len(messages) >= self.max_utterances:
                await self._complete_round(
                    round_number,
                    messages,
                    final_moderator,
                    status="truncated",
                    reason=f"达到 {self.max_utterances} 条医患消息上限",
                )
                return


            doctor_text = await self._agent_turn(
                "doctor",
                round_number,
                doctor_messages(messages, self.rounds, self.critiques),
            )
            messages.append(DialogueMessage(role="doctor", content=doctor_text))
            await self.emit(
                "message_completed",
                role="doctor",
                round=round_number,
                content=doctor_text,
                content_html=render_doctor_content(doctor_text),
                index=len(messages),
            )

            await self.emit(
                "phase_started", phase="moderator", round=round_number
            )
            final_moderator = await self._structured(
                moderator_messages(messages),
                ModeratorResult,
                schema=MODERATOR_SCHEMA,
                repair_kind="MODERATOR",
                max_tokens=500,
            )
            final_moderator = self._normalize_moderator(final_moderator)
            await self.emit(
                "moderator_result",
                round=round_number,
                **final_moderator.model_dump(),
            )
            if final_moderator.ended:
                await self._complete_round(
                    round_number,
                    messages,
                    final_moderator,
                    status="completed",
                    reason=final_moderator.reason,
                )
                return

    async def _agent_turn(
        self,
        role: str,
        round_number: int,
        prompt: list[dict[str, str]],
    ) -> str:
        await self.emit("phase_started", phase=role, round=round_number)
        await self.emit("message_started", role=role, round=round_number)
        result = await self._complete(prompt, max_tokens=900, temperature=0.35)
        if not result.strip():
            raise ModelAPIError("角色生成了空回复", code="empty_response")
        return result.strip()

    async def _complete_round(
        self,
        round_number: int,
        messages: list[DialogueMessage],
        moderator: ModeratorResult | None,
        *,
        status: str,
        reason: str,
    ) -> None:
        round_state = RoundState(
            round_number=round_number,
            messages=list(messages),
            moderator=moderator,
            status=status,
            completion_reason=reason,
        )
        self.rounds.append(round_state)
        await self.emit(
            "dialogue_completed",
            round=round_number,
            status=status,
            utterance_count=len(messages),
            reason=reason,
            can_refine=False,
            review_pending=True,
            max_rounds=self.max_rounds,
        )

    async def _review_round(self, round_state: RoundState) -> None:
        round_number = round_state.round_number
        await self.emit("phase_started", phase="ddx", round=round_number)
        ddx = await self._structured(
            ddx_messages(round_state.messages),
            DDxResult,
            schema=DDX_SCHEMA,
            repair_kind="DDX",
            max_tokens=700,
        )
        round_state.ddx = ddx
        await self.emit(
            "ddx_completed",
            round=round_number,
            differential_diagnoses=ddx.differential_diagnoses,
        )

        await self.emit("phase_started", phase="critic", round=round_number)
        critique = await self._complete(
            critic_messages(
                self.condition or "", self._require_vignette(), round_state.messages
            ),
            max_tokens=1200,
            temperature=0.2,
        )
        critique_state = CritiqueState(
            round_number=round_number, content=critique.strip()
        )
        self.critiques = [
            item for item in self.critiques if item.round_number != round_number
        ]
        self.critiques.append(critique_state)
        can_refine = (
            round_state.status == "completed" and round_number < self.max_rounds
        )
        await self.emit(
            "critique_completed",
            round=round_number,
            next_round=round_number + 1 if can_refine else None,
            can_refine=can_refine,
            critique=critique_state.content,
        )

        await self._evaluate_round(round_state)
        await self.emit(
            "round_review_completed",
            round=round_number,
            status=round_state.status,
            can_refine=can_refine,
            next_round=round_number + 1 if can_refine else None,
        )

    async def _evaluate_round(self, round_state: RoundState) -> None:
        round_number = round_state.round_number
        vignette = self._require_vignette()
        if round_state.ddx is None:
            raise SimulationStateError("DDx 尚未生成，无法开始 Evaluation")

        await self.emit("phase_started", phase="evaluation", round=round_number)
        jobs = {
            "accuracy": self._rate_accuracy(vignette, round_state.ddx),
            "patient_actor": self._rate_quality(
                patient_actor_rater_messages(vignette, round_state.messages),
                PatientActorRaterOutput,
                PATIENT_ACTOR_RUBRIC,
                schema=PATIENT_ACTOR_SCHEMA,
                repair_kind="PATIENT_ACTOR_RATER",
                max_tokens=6500,
            ),
            "specialist": self._rate_quality(
                specialist_rater_messages(
                    vignette, round_state.messages, round_state.ddx
                ),
                SpecialistRaterOutput,
                SPECIALIST_RUBRIC,
                schema=SPECIALIST_SCHEMA,
                repair_kind="SPECIALIST_RATER",
                max_tokens=6500,
            ),
            "auto_paces": self._rate_quality(
                auto_paces_rater_messages(round_state.messages),
                AutoPacesRaterOutput,
                AUTO_PACES_RUBRIC,
                schema=AUTO_PACES_SCHEMA,
                repair_kind="AUTO_PACES_RATER",
                max_tokens=1400,
            ),
        }
        outcomes = await asyncio.gather(*jobs.values(), return_exceptions=True)
        completed: dict[str, AccuracyEvaluation | QualityEvaluation] = {}
        errors: dict[str, EvaluationError] = {}
        for name, outcome in zip(jobs, outcomes, strict=True):
            if isinstance(outcome, Exception):
                errors[name] = self._evaluation_error(outcome)
            else:
                completed[name] = outcome

        if not errors:
            status = "complete"
        elif not completed:
            status = "failed"
        else:
            status = "partial"
        evaluation = EvaluationState(
            status=status,
            accuracy=completed.get("accuracy"),
            patient_actor=completed.get("patient_actor"),
            specialist=completed.get("specialist"),
            auto_paces=completed.get("auto_paces"),
            errors=errors,
        )
        round_state.evaluation = evaluation
        await self.emit(
            "evaluation_completed",
            round=round_number,
            **evaluation.model_dump(),
        )

    async def _rate_accuracy(
        self, vignette: Vignette, ddx: DDxResult
    ) -> AccuracyEvaluation:
        output = await self._structured(
            accuracy_rater_messages(vignette, ddx),
            AccuracyRaterOutput,
            schema=ACCURACY_SCHEMA,
            repair_kind="ACCURACY_RATER",
            max_tokens=2200,
            validation_context={
                "differential_diagnoses": ddx.differential_diagnoses,
                "ground_truth_diagnosis": vignette.ground_truth_diagnosis,
                "accepted_differential_diagnoses": (
                    vignette.accepted_differential_diagnoses
                ),
            },
        )
        return self._build_accuracy_evaluation(output.candidates)

    async def _rate_quality(
        self,
        messages: list[dict[str, str]],
        model_type: type[RubricRaterOutput],
        rubric: tuple[CriterionDefinition, ...],
        *,
        schema: str,
        repair_kind: str,
        max_tokens: int,
    ) -> QualityEvaluation:
        output = await self._structured(
            messages,
            model_type,
            schema=schema,
            repair_kind=repair_kind,
            max_tokens=max_tokens,
        )
        definitions = rubric_by_id(rubric)
        raw_by_id = {item.criterion_id: item for item in output.scores}
        criteria: list[CriterionScore] = []
        for criterion in rubric:
            raw = raw_by_id[criterion.criterion_id]
            if raw.score is None:
                response_label = "N/A"
                polarity = "na"
            else:
                option = next(
                    item for item in criterion.options if item.score == raw.score
                )
                response_label = option.label
                polarity = option.polarity
            criteria.append(
                CriterionScore(
                    criterion_id=criterion.criterion_id,
                    group=definitions[criterion.criterion_id].group,
                    label=definitions[criterion.criterion_id].label,
                    raw_score=raw.score,
                    scale_max=criterion.scale_max,
                    scale_name=criterion.scale_name,
                    response_label=response_label,
                    polarity=polarity,
                    evidence=raw.evidence,
                )
            )
        return QualityEvaluation(criteria=criteria)

    @staticmethod
    def _build_accuracy_evaluation(
        candidates: list[CandidateSemanticRating],
    ) -> AccuracyEvaluation:
        matching = {"exact", "synonym", "more_specific", "highly_related"}

        def hits(attribute: str) -> TopKHits:
            def has_match(limit: int) -> bool:
                return any(
                    getattr(item, attribute).level in matching
                    for item in candidates[:limit]
                )

            return TopKHits(
                top_1=has_match(1),
                top_3=has_match(3),
                top_10=has_match(10),
            )

        return AccuracyEvaluation(
            candidates=candidates,
            ground_truth_hits=hits("ground_truth"),
            accepted_differential_hits=hits("accepted_differential"),
        )

    @staticmethod
    def _evaluation_error(error: Exception) -> EvaluationError:
        if isinstance(error, ModelAPIError):
            category = (
                "invalid_output" if error.code == "invalid_json" else "model_error"
            )
            return EvaluationError(
                category=category,
                code=error.code,
                message=str(error)[:500],
            )
        return EvaluationError(
            category="internal_error",
            code=type(error).__name__,
            message=str(error)[:500] or "Evaluation group failed",
        )

    async def _structured(
        self,
        messages: list[dict[str, str]],
        model_type: type[StructuredModel],
        *,
        schema: str,
        repair_kind: str,
        max_tokens: int,
        validation_context: dict[str, Any] | None = None,
    ) -> StructuredModel:
        raw = await self._complete(
            messages, max_tokens=max_tokens, temperature=0.0
        )
        try:
            return model_type.model_validate(
                parse_json_object(raw), context=validation_context
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as initial_error:
            initial_error_summary = self._structured_error_summary(initial_error)
            repaired = await self._complete(
                json_repair_messages(
                    raw, schema, repair_kind, initial_error_summary
                ),
                max_tokens=max_tokens,
                temperature=0.0,
            )
            try:
                return model_type.model_validate(
                    parse_json_object(repaired), context=validation_context
                )
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                final_error_summary = self._structured_error_summary(exc)
                raise ModelAPIError(
                    f"{repair_kind} 返回的 JSON 在一次修复后仍无效："
                    f"{final_error_summary}",
                    code="invalid_json",
                ) from exc

    @staticmethod
    def _structured_error_summary(error: Exception) -> str:
        if isinstance(error, ValidationError):
            details: list[str] = []
            for item in error.errors(include_url=False):
                location = ".".join(str(part) for part in item.get("loc", ()))
                message = str(item.get("msg", "validation failed"))
                details.append(f"{location}: {message}" if location else message)
            return "; ".join(details)[:1200]
        if isinstance(error, json.JSONDecodeError):
            return f"JSON 语法错误（line {error.lineno}, column {error.colno}）：{error.msg}"
        return str(error)[:1200] or type(error).__name__

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        return await self.llm.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model_name=self.model_name,
        )

    def _require_vignette(self) -> Vignette:
        if self.vignette is None:
            raise SimulationStateError("病例尚未生成")
        return self.vignette

    @staticmethod
    def _normalize_moderator(result: ModeratorResult) -> ModeratorResult:
        ended = result.farewell_detected or (
            result.diagnosis_complete
            and result.treatment_plan_complete
            and result.patient_questions_resolved
        )
        if ended == result.ended:
            return result
        reason = result.reason
        if not ended:
            reason = "诊断、治疗计划和患者问题尚未同时完成，继续对话"
        return result.model_copy(update={"ended": ended, "reason": reason})

    async def launch_start(
        self, condition: str, model_name: str | None = None
    ) -> None:
        await self._launch(self.start(condition, model_name))

    async def launch_refine(self) -> None:
        await self._launch(self.refine())

    async def _launch(self, action: Coroutine[Any, Any, None]) -> None:
        if self.busy:
            action.close()
            await self.emit("error", code="busy", message="当前角色仍在生成，请稍候")
            return
        self._task = asyncio.create_task(self._guard(action))

    async def _guard(self, action: Coroutine[Any, Any, None]) -> None:
        try:
            await action
        except asyncio.CancelledError:
            raise
        except SimulationStateError as exc:
            with suppress(Exception):
                await self.emit("error", code="invalid_state", message=str(exc))
        except ModelAPIError as exc:
            with suppress(Exception):
                await self.emit(
                    "error",
                    code=exc.code,
                    message=str(exc),
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                )
        except Exception as exc:
            with suppress(Exception):
                await self.emit(
                    "error", code="internal_error", message=f"运行失败：{exc}"
                )
        finally:
            if self._task is asyncio.current_task():
                self._task = None

    async def stop(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._task = None
        await self.emit(
            "stopped",
            round=len(self.rounds) + 1 if self.condition else 0,
            reason="用户已停止模拟",
        )

    async def disconnect(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._task = None
