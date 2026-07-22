from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
    CriterionDefinition,
    rubric_by_id,
)


class Vignette(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    summary: str
    demographics: str
    symptoms: list[str]
    past_medical_history: str
    past_surgical_history: str
    past_social_history: str
    medication: str
    allergy: str
    family_history: str
    patient_questions: list[str]
    ground_truth_diagnosis: str
    accepted_differential_diagnoses: list[str]
    reference_management_plan: list[str]

    @field_validator(
        "symptoms",
        "patient_questions",
        "accepted_differential_diagnoses",
        "reference_management_plan",
        mode="before",
    )
    @classmethod
    def coerce_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @model_validator(mode="after")
    def validate_accepted_differentials(self) -> Vignette:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in self.accepted_differential_diagnoses:
            diagnosis = item.strip()
            key = diagnosis.casefold()
            if diagnosis and key not in seen:
                seen.add(key)
                cleaned.append(diagnosis)
        if not 3 <= len(cleaned) <= 10:
            raise ValueError(
                "accepted_differential_diagnoses must contain 3 to 10 unique diagnoses"
            )
        if self.ground_truth_diagnosis.strip().casefold() not in seen:
            raise ValueError(
                "accepted_differential_diagnoses must include ground_truth_diagnosis"
            )
        self.accepted_differential_diagnoses = cleaned
        return self


class DialogueMessage(BaseModel):
    role: Literal["doctor", "patient"]
    content: str


class ModeratorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ended: bool
    reason: str
    diagnosis_complete: bool
    treatment_plan_complete: bool
    patient_questions_resolved: bool
    farewell_detected: bool


class DDxResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    differential_diagnoses: list[str]

    @field_validator("differential_diagnoses")
    @classmethod
    def validate_ranked_list(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            diagnosis = item.strip()
            key = diagnosis.casefold()
            if diagnosis and key not in seen:
                seen.add(key)
                cleaned.append(diagnosis)
        if not 3 <= len(cleaned) <= 10:
            raise ValueError("DDx must contain 3 to 10 unique diagnoses")
        return cleaned


SemanticMatchLevel = Literal[
    "exact", "synonym", "more_specific", "highly_related", "no_match"
]


class SemanticMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: SemanticMatchLevel
    matched_diagnosis: str | None
    rationale: str

    @model_validator(mode="after")
    def validate_target(self) -> SemanticMatch:
        if self.level == "no_match" and self.matched_diagnosis is not None:
            raise ValueError("no_match must use null matched_diagnosis")
        if self.level != "no_match" and not self.matched_diagnosis:
            raise ValueError("a semantic match must name matched_diagnosis")
        if not self.rationale.strip():
            raise ValueError("semantic match rationale cannot be empty")
        return self


class CandidateSemanticRating(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=10)
    candidate: str
    ground_truth: SemanticMatch
    accepted_differential: SemanticMatch


class AccuracyRaterOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateSemanticRating]

    @model_validator(mode="after")
    def validate_candidates(self, info: ValidationInfo) -> AccuracyRaterOutput:
        context = info.context or {}
        expected = context.get("differential_diagnoses")
        if not isinstance(expected, list):
            return self
        if len(self.candidates) != len(expected):
            raise ValueError("accuracy rater must return one item for every DDx candidate")
        ground_truth = str(context.get("ground_truth_diagnosis", "")).casefold()
        accepted = {
            str(item).casefold()
            for item in context.get("accepted_differential_diagnoses", [])
        }
        for index, (rating, diagnosis) in enumerate(
            zip(self.candidates, expected, strict=True), start=1
        ):
            if rating.rank != index or rating.candidate.casefold() != diagnosis.casefold():
                raise ValueError("accuracy rater ranks and candidates must match the DDx")
            ground_match = rating.ground_truth.matched_diagnosis
            if ground_match is not None and ground_match.casefold() != ground_truth:
                raise ValueError("ground-truth match must copy the supplied diagnosis")
            accepted_match = rating.accepted_differential.matched_diagnosis
            if accepted_match is not None and accepted_match.casefold() not in accepted:
                raise ValueError("accepted match must copy a supplied accepted diagnosis")
        return self


class TopKHits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_1: bool
    top_3: bool
    top_10: bool


class AccuracyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateSemanticRating]
    ground_truth_hits: TopKHits
    accepted_differential_hits: TopKHits


class RawCriterionScore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    criterion_id: str
    score: int | None
    evidence: str

    @field_validator("criterion_id")
    @classmethod
    def normalize_criterion_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("score", mode="before")
    @classmethod
    def normalize_na_score(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().casefold() in {
            "n/a",
            "na",
            "null",
            "cannot rate",
            "does not apply",
        }:
            return None
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("criterion evidence cannot be empty")
        return value


class RubricRaterOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rubric: ClassVar[tuple[CriterionDefinition, ...]] = ()
    scores: list[RawCriterionScore]

    @model_validator(mode="after")
    def validate_complete_rubric(self) -> RubricRaterOutput:
        definitions = rubric_by_id(self.rubric)
        ids = [item.criterion_id for item in self.scores]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric contains duplicate criterion IDs")
        if set(ids) != set(definitions):
            missing = sorted(set(definitions) - set(ids))
            unexpected = sorted(set(ids) - set(definitions))
            raise ValueError(
                f"rubric criterion IDs do not match; missing={missing}, unexpected={unexpected}"
            )
        for item in self.scores:
            if item.score is None:
                continue
            allowed = {option.score for option in definitions[item.criterion_id].options}
            if item.score not in allowed:
                raise ValueError(
                    f"{item.criterion_id} score {item.score} is outside its scale"
                )
        return self


class PatientActorRaterOutput(RubricRaterOutput):
    rubric = PATIENT_ACTOR_RUBRIC


class SpecialistRaterOutput(RubricRaterOutput):
    rubric = SPECIALIST_RUBRIC


class AutoPacesRaterOutput(RubricRaterOutput):
    rubric = AUTO_PACES_RUBRIC


class CriterionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    group: str
    label: str
    raw_score: int | None
    scale_max: int
    scale_name: str
    response_label: str
    polarity: Literal["positive", "neutral", "negative", "na"]
    evidence: str


class QualityEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: list[CriterionScore]


class EvaluationError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["invalid_output", "model_error", "internal_error"]
    code: str
    message: str


class EvaluationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["complete", "partial", "failed"]
    accuracy: AccuracyEvaluation | None = None
    patient_actor: QualityEvaluation | None = None
    specialist: QualityEvaluation | None = None
    auto_paces: QualityEvaluation | None = None
    errors: dict[str, EvaluationError] = Field(default_factory=dict)


class RoundState(BaseModel):
    round_number: int
    messages: list[DialogueMessage] = Field(default_factory=list)
    moderator: ModeratorResult | None = None
    ddx: DDxResult | None = None
    evaluation: EvaluationState | None = None
    status: Literal["completed", "truncated"]
    completion_reason: str


class CritiqueState(BaseModel):
    round_number: int
    content: str
