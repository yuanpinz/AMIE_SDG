from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Polarity = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class ResponseOption:
    score: int
    label: str
    polarity: Polarity


@dataclass(frozen=True)
class CriterionDefinition:
    criterion_id: str
    group: str
    label: str
    scale_name: str
    options: tuple[ResponseOption, ...]

    @property
    def scale_max(self) -> int:
        return max(option.score for option in self.options)


FIVE_POINT = (
    ResponseOption(1, "Very unfavourable", "negative"),
    ResponseOption(2, "Unfavourable", "negative"),
    ResponseOption(3, "Neither favourable nor unfavourable", "neutral"),
    ResponseOption(4, "Favourable", "positive"),
    ResponseOption(5, "Very favourable", "positive"),
)

BINARY = (
    ResponseOption(0, "No", "negative"),
    ResponseOption(1, "Yes", "positive"),
)


GROUP_TRANSLATIONS = {
    "GMCPQ": "GMCPQ 患者问卷",
    "PCCBP Relationship Fostering": "PCCBP 关系建立",
    "PACES": "PACES 临床沟通",
    "PACES: Eliciting Information": "PACES：采集信息",
    "PACES: Explaining Clinical Information": "PACES：解释临床信息",
    "PACES: Differential Diagnosis": "PACES：鉴别诊断",
    "PACES: Clinical Judgement": "PACES：临床判断",
    "PACES: Managing Patient Concerns": "PACES：处理患者关切",
    "PACES: Maintaining Patient Welfare": "PACES：维护患者福祉",
    "PCCBP": "PCCBP 以患者为中心的沟通",
    "Diagnosis & Management": "诊断与管理",
    "Auto PACES": "自动 PACES",
}

LABEL_TRANSLATIONS = {
    "Being polite": "礼貌待人",
    "Making the patient feel at ease": "让患者感到自在",
    "Listening to the patient": "倾听患者",
    "Assessing the medical condition": "评估患者病情",
    "Explaining the condition and treatment": "解释病情与治疗",
    "Involving the patient in treatment decisions": "让患者参与治疗决策",
    "Providing an appropriate treatment plan": "提供恰当的治疗计划",
    "Patient trusts that information is confidential": "患者相信其信息会被保密",
    "Appearing honest and trustworthy": "表现得诚实可信",
    "Patient is confident about the care provided": "患者对所获照护有信心",
    "Patient would be happy to return in future": "患者愿意再次就诊",
    "Building rapport and connection": "建立融洽关系与连接",
    "Appearing open and honest": "表现得开放坦诚",
    "Discussing roles and responsibilities": "讨论角色与责任",
    "Respecting the patient's privacy": "尊重患者隐私",
    "Engaging in partnership building": "建立伙伴关系",
    "Expressing care and commitment": "表达关怀与承诺",
    "Acknowledging mistakes": "承认错误",
    "Greeting the patient appropriately": "恰当地问候患者",
    "Using appropriate language": "使用恰当语言",
    "Encouraging patient participation": "鼓励患者参与",
    "Valuing the patient as a person": "将患者作为独立个体予以重视",
    "Addressing patient concerns": "回应患者关切",
    "Understanding patient concerns": "理解患者关切",
    "Showing empathy": "展现共情",
    "Maintaining patient welfare": "维护患者福祉",
    "Presenting complaint": "主诉",
    "Systems review": "系统回顾",
    "Past medical history": "既往病史",
    "Family history": "家族史",
    "Medication history": "用药史",
    "Explaining relevant clinical information accurately": "准确解释相关临床信息",
    "Explaining relevant clinical information clearly": "清晰解释相关临床信息",
    "Explaining relevant clinical information with structure": "有条理地解释相关临床信息",
    "Explaining relevant clinical information comprehensively": "全面解释相关临床信息",
    "Explaining relevant clinical information professionally": "专业地解释相关临床信息",
    "Constructing a sensible differential diagnosis": "构建合理的鉴别诊断",
    "Selecting a comprehensive, sensible and appropriate management plan": "选择全面、合理且恰当的管理计划",
    "Confirming patient knowledge and understanding": "确认患者的知识与理解",
    "Fostering the relationship": "促进医患关系",
    "Gathering information": "收集信息",
    "Providing information": "提供信息",
    "Decision making with the patient": "与患者共同决策",
    "Enabling disease and treatment-related behavior": "促进疾病及治疗相关行为",
    "Responding to emotions": "回应患者情绪",
    "DDx appropriateness": "鉴别诊断恰当性",
    "DDx comprehensiveness": "鉴别诊断全面性",
    "Management plan appropriateness": "管理计划恰当性",
    "Escalation recommendation appropriate": "升级就诊建议恰当性",
    "Appropriate investigations recommended": "推荐恰当检查",
    "Inappropriate investigations avoided": "避免不恰当检查",
    "Appropriate treatments recommended": "推荐恰当治疗",
    "Inappropriate treatments avoided": "避免不恰当治疗",
    "Follow-up recommendation appropriate": "随访建议恰当性",
    "Confabulation absent": "无虚构信息",
    "Addressing concerns": "回应关切",
    "Understanding concerns": "理解关切",
    "Empathy": "共情",
    "Maintaining welfare": "维护福祉",
}


def criterion(
    criterion_id: str,
    group: str,
    label: str,
    scale_name: str = "5-point",
    options: tuple[ResponseOption, ...] = FIVE_POINT,
) -> CriterionDefinition:
    bilingual_group = f"{GROUP_TRANSLATIONS[group]} / {group}"
    bilingual_label = f"{LABEL_TRANSLATIONS[label]} / {label}"
    return CriterionDefinition(
        criterion_id, bilingual_group, bilingual_label, scale_name, options
    )


PATIENT_ACTOR_RUBRIC = (
    criterion("gmcpq_being_polite", "GMCPQ", "Being polite"),
    criterion(
        "gmcpq_making_patient_feel_at_ease",
        "GMCPQ",
        "Making the patient feel at ease",
    ),
    criterion("gmcpq_listening_to_patient", "GMCPQ", "Listening to the patient"),
    criterion(
        "gmcpq_assessing_medical_condition",
        "GMCPQ",
        "Assessing the medical condition",
    ),
    criterion(
        "gmcpq_explaining_condition_and_treatment",
        "GMCPQ",
        "Explaining the condition and treatment",
    ),
    criterion(
        "gmcpq_involving_patient_in_treatment_decisions",
        "GMCPQ",
        "Involving the patient in treatment decisions",
    ),
    criterion(
        "gmcpq_providing_appropriate_treatment_plan",
        "GMCPQ",
        "Providing an appropriate treatment plan",
    ),
    criterion(
        "gmcpq_information_confidential",
        "GMCPQ",
        "Patient trusts that information is confidential",
    ),
    criterion(
        "gmcpq_honest_and_trustworthy",
        "GMCPQ",
        "Appearing honest and trustworthy",
    ),
    criterion(
        "gmcpq_confident_about_care",
        "GMCPQ",
        "Patient is confident about the care provided",
        "binary",
        BINARY,
    ),
    criterion(
        "gmcpq_happy_to_return",
        "GMCPQ",
        "Patient would be happy to return in future",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_building_rapport_and_connection",
        "PCCBP Relationship Fostering",
        "Building rapport and connection",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_appearing_open_and_honest",
        "PCCBP Relationship Fostering",
        "Appearing open and honest",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_discussing_roles_and_responsibilities",
        "PCCBP Relationship Fostering",
        "Discussing roles and responsibilities",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_respecting_patient_privacy",
        "PCCBP Relationship Fostering",
        "Respecting the patient's privacy",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_partnership_building",
        "PCCBP Relationship Fostering",
        "Engaging in partnership building",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_expressing_care_and_commitment",
        "PCCBP Relationship Fostering",
        "Expressing care and commitment",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_acknowledging_mistakes",
        "PCCBP Relationship Fostering",
        "Acknowledging mistakes",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_greeting_patient_appropriately",
        "PCCBP Relationship Fostering",
        "Greeting the patient appropriately",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_using_appropriate_language",
        "PCCBP Relationship Fostering",
        "Using appropriate language",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_encouraging_patient_participation",
        "PCCBP Relationship Fostering",
        "Encouraging patient participation",
        "binary",
        BINARY,
    ),
    criterion(
        "pccbp_valuing_patient_as_person",
        "PCCBP Relationship Fostering",
        "Valuing the patient as a person",
        "binary",
        BINARY,
    ),
    criterion(
        "paces_addressing_patient_concerns",
        "PACES",
        "Addressing patient concerns",
    ),
    criterion(
        "paces_understanding_patient_concerns",
        "PACES",
        "Understanding patient concerns",
    ),
    criterion("paces_showing_empathy", "PACES", "Showing empathy"),
    criterion(
        "paces_maintaining_patient_welfare",
        "PACES",
        "Maintaining patient welfare",
    ),
)


SPECIALIST_RUBRIC = (
    criterion("paces_presenting_complaint", "PACES: Eliciting Information", "Presenting complaint"),
    criterion("paces_systems_review", "PACES: Eliciting Information", "Systems review"),
    criterion("paces_past_medical_history", "PACES: Eliciting Information", "Past medical history"),
    criterion("paces_family_history", "PACES: Eliciting Information", "Family history"),
    criterion("paces_medication_history", "PACES: Eliciting Information", "Medication history"),
    criterion(
        "paces_explaining_accurately",
        "PACES: Explaining Clinical Information",
        "Explaining relevant clinical information accurately",
    ),
    criterion(
        "paces_explaining_clearly",
        "PACES: Explaining Clinical Information",
        "Explaining relevant clinical information clearly",
    ),
    criterion(
        "paces_explaining_with_structure",
        "PACES: Explaining Clinical Information",
        "Explaining relevant clinical information with structure",
    ),
    criterion(
        "paces_explaining_comprehensively",
        "PACES: Explaining Clinical Information",
        "Explaining relevant clinical information comprehensively",
    ),
    criterion(
        "paces_explaining_professionally",
        "PACES: Explaining Clinical Information",
        "Explaining relevant clinical information professionally",
    ),
    criterion(
        "paces_sensible_differential_diagnosis",
        "PACES: Differential Diagnosis",
        "Constructing a sensible differential diagnosis",
    ),
    criterion(
        "paces_clinical_judgement",
        "PACES: Clinical Judgement",
        "Selecting a comprehensive, sensible and appropriate management plan",
    ),
    criterion(
        "paces_addressing_patient_concerns",
        "PACES: Managing Patient Concerns",
        "Addressing patient concerns",
    ),
    criterion(
        "paces_understanding_patient_concerns",
        "PACES: Managing Patient Concerns",
        "Confirming patient knowledge and understanding",
    ),
    criterion(
        "paces_showing_empathy",
        "PACES: Managing Patient Concerns",
        "Showing empathy",
    ),
    criterion(
        "paces_maintaining_patient_welfare",
        "PACES: Maintaining Patient Welfare",
        "Maintaining patient welfare",
    ),
    criterion("pccbp_fostering_relationship", "PCCBP", "Fostering the relationship"),
    criterion("pccbp_gathering_information", "PCCBP", "Gathering information"),
    criterion("pccbp_providing_information", "PCCBP", "Providing information"),
    criterion("pccbp_decision_making", "PCCBP", "Decision making with the patient"),
    criterion(
        "pccbp_enabling_disease_treatment_behavior",
        "PCCBP",
        "Enabling disease and treatment-related behavior",
    ),
    criterion("pccbp_responding_to_emotions", "PCCBP", "Responding to emotions"),
    criterion("dm_ddx_appropriateness", "Diagnosis & Management", "DDx appropriateness"),
    criterion(
        "dm_ddx_comprehensiveness",
        "Diagnosis & Management",
        "DDx comprehensiveness",
        "4-point",
        (
            ResponseOption(1, "Major candidates are missing", "negative"),
            ResponseOption(2, "Some candidates are present, but several are missing", "negative"),
            ResponseOption(3, "Most candidates are present, but some are missing", "positive"),
            ResponseOption(4, "All candidates that are reasonable are present", "positive"),
        ),
    ),
    criterion(
        "dm_management_plan_appropriateness",
        "Diagnosis & Management",
        "Management plan appropriateness",
    ),
    criterion(
        "dm_escalation_recommendation_appropriate",
        "Diagnosis & Management",
        "Escalation recommendation appropriate",
        "4-point",
        (
            ResponseOption(1, "Required escalation was omitted and could cause harm", "negative"),
            ResponseOption(2, "Escalation was performed unnecessarily", "negative"),
            ResponseOption(3, "Escalation was required and performed", "positive"),
            ResponseOption(4, "Escalation was not required and not performed", "positive"),
        ),
    ),
    criterion(
        "dm_appropriate_investigations_recommended",
        "Diagnosis & Management",
        "Appropriate investigations recommended",
        "3-point",
        (
            ResponseOption(1, "Required investigations were not recommended", "negative"),
            ResponseOption(2, "Recommended investigations were incomplete", "neutral"),
            ResponseOption(3, "A comprehensive and appropriate investigation strategy was recommended", "positive"),
        ),
    ),
    criterion(
        "dm_inappropriate_investigations_avoided",
        "Diagnosis & Management",
        "Inappropriate investigations avoided",
        "binary",
        BINARY,
    ),
    criterion(
        "dm_appropriate_treatments_recommended",
        "Diagnosis & Management",
        "Appropriate treatments recommended",
        "3-point",
        (
            ResponseOption(1, "Required treatments were not recommended", "negative"),
            ResponseOption(2, "Recommended treatments were incomplete", "neutral"),
            ResponseOption(3, "A comprehensive and appropriate treatment strategy was recommended", "positive"),
        ),
    ),
    criterion(
        "dm_inappropriate_treatments_avoided",
        "Diagnosis & Management",
        "Inappropriate treatments avoided",
        "binary",
        BINARY,
    ),
    criterion(
        "dm_follow_up_recommendation_appropriate",
        "Diagnosis & Management",
        "Follow-up recommendation appropriate",
        "4-point",
        (
            ResponseOption(1, "Needed follow-up was omitted", "negative"),
            ResponseOption(2, "Unnecessary follow-up was recommended", "negative"),
            ResponseOption(3, "Needed follow-up was appropriately recommended", "positive"),
            ResponseOption(4, "Follow-up was not needed and was not recommended", "positive"),
        ),
    ),
    criterion(
        "dm_confabulation_absent",
        "Diagnosis & Management",
        "Confabulation absent",
        "binary",
        BINARY,
    ),
)


AUTO_PACES_RUBRIC = (
    criterion("auto_paces_addressing_concerns", "Auto PACES", "Addressing concerns"),
    criterion("auto_paces_understanding_concerns", "Auto PACES", "Understanding concerns"),
    criterion("auto_paces_empathy", "Auto PACES", "Empathy"),
    criterion("auto_paces_maintaining_welfare", "Auto PACES", "Maintaining welfare"),
)


def rubric_by_id(
    rubric: tuple[CriterionDefinition, ...],
) -> dict[str, CriterionDefinition]:
    return {item.criterion_id: item for item in rubric}


def prompt_rubric(rubric: tuple[CriterionDefinition, ...]) -> str:
    lines: list[str] = []
    for item in rubric:
        options = "; ".join(
            f"{option.score}={option.label}" for option in item.options
        )
        lines.append(
            f"- {item.criterion_id} | {item.group} | {item.label} | "
            f"{item.scale_name}: {options}; null=N/A"
        )
    return "\n".join(lines)


assert len(PATIENT_ACTOR_RUBRIC) == 26
assert len(SPECIALIST_RUBRIC) == 32
assert len(AUTO_PACES_RUBRIC) == 4
assert len(rubric_by_id(PATIENT_ACTOR_RUBRIC)) == 26
assert len(rubric_by_id(SPECIALIST_RUBRIC)) == 32
assert len(rubric_by_id(AUTO_PACES_RUBRIC)) == 4
assert all(
    " / " in item.group and " / " in item.label
    for rubric in (PATIENT_ACTOR_RUBRIC, SPECIALIST_RUBRIC, AUTO_PACES_RUBRIC)
    for item in rubric
)
