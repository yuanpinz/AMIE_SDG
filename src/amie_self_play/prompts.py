from __future__ import annotations

import json

from .llm import ChatMessage
from .models import CritiqueState, DDxResult, DialogueMessage, RoundState, Vignette
from .rubrics import (
    AUTO_PACES_RUBRIC,
    PATIENT_ACTOR_RUBRIC,
    SPECIALIST_RUBRIC,
    CriterionDefinition,
    prompt_rubric,
)


DOCTOR_OPENING = "您好，请问今天有什么可以帮您？"

VIGNETTE_ONE_SHOT = """以下是输出格式示例（由论文 SI.5 腕管综合征案例翻译并补齐产品字段）：
{
  "condition": "腕管综合征",
  "summary": "一名 55 岁女性，以下症状持续 3 个月。",
  "demographics": "55 岁，女性。",
  "symptoms": [
    "右手拇指、食指、中指和无名指麻木、刺痛",
    "夜间更明显并会痛醒",
    "手和腕部疼痛向前臂放射",
    "打字、使用鼠标或抓握物品时加重"
  ],
  "past_medical_history": "高血压、甲状腺功能减退。",
  "past_surgical_history": "无。",
  "past_social_history": "从事数据录入工作。",
  "medication": "左甲状腺素、赖诺普利。",
  "allergy": "N/A",
  "family_history": "N/A",
  "patient_questions": ["是什么导致了这些症状？", "怎样才能缓解？"],
  "ground_truth_diagnosis": "腕管综合征",
  "accepted_differential_diagnoses": [
    "腕管综合征",
    "颈椎神经根病",
    "糖尿病周围神经病变",
    "尺神经卡压"
  ],
  "reference_management_plan": [
    "建议线下面诊评估严重程度并排除更严重病因或后遗症",
    "夜间使用中立位腕托，调整工位和减少诱发动作",
    "根据个体禁忌谨慎考虑短期对症止痛",
    "若持续、加重或出现明显无力，考虑神经传导检查及注射或手术等专科治疗"
  ]
}"""

VIGNETTE_SCHEMA = """{
  "condition": "string",
  "summary": "string",
  "demographics": "string",
  "symptoms": ["string"],
  "past_medical_history": "string",
  "past_surgical_history": "string",
  "past_social_history": "string",
  "medication": "string",
  "allergy": "string",
  "family_history": "string",
  "patient_questions": ["string"],
  "ground_truth_diagnosis": "string",
  "accepted_differential_diagnoses": ["ground truth", "reasonable alternative", "reasonable alternative"],
  "reference_management_plan": ["string"]
}"""

MODERATOR_SCHEMA = """{
  "ended": true,
  "reason": "string",
  "diagnosis_complete": true,
  "treatment_plan_complete": true,
  "patient_questions_resolved": true,
  "farewell_detected": false
}"""

DDX_SCHEMA = """{
  "differential_diagnoses": [
    "最可能诊断",
    "第二可能诊断",
    "第三可能诊断"
  ]
}"""

ACCURACY_SCHEMA = """{
  "candidates": [{
    "rank": 1,
    "candidate": "copy the DDx candidate exactly",
    "ground_truth": {
      "level": "exact|synonym|more_specific|highly_related|no_match",
      "matched_diagnosis": "copy the ground truth exactly, or null",
      "rationale": "one short sentence"
    },
    "accepted_differential": {
      "level": "exact|synonym|more_specific|highly_related|no_match",
      "matched_diagnosis": "copy the best accepted diagnosis exactly, or null",
      "rationale": "one short sentence"
    }
  }]
}"""


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


PATIENT_ACTOR_SCHEMA = rubric_schema(PATIENT_ACTOR_RUBRIC)
SPECIALIST_SCHEMA = rubric_schema(SPECIALIST_RUBRIC)
AUTO_PACES_SCHEMA = rubric_schema(AUTO_PACES_RUBRIC)


def vignette_messages(condition: str) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:VIGNETTE]\n"
                "你是医学病例生成器。仅使用既有医学知识，不进行或声称进行网络检索。"
                "为给定疾病生成一个中文模拟患者病例，不得生成第二个。病例要临床合理、"
                "内部一致，并包含足够但不过度暴露的问诊信息。缺失字段写 N/A。"
                "accepted_differential_diagnoses 必须包含 ground_truth_diagnosis，并加入临床"
                "合理的替代诊断，去重后共 3–10 项。"
                "只输出严格 JSON，不要 Markdown 代码块或解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"疾病：{condition}\n\n{VIGNETTE_ONE_SHOT}\n\n"
                f"请严格按以下 JSON 结构生成一个新病例：\n{VIGNETTE_SCHEMA}"
            ),
        },
    ]


def transcript(messages: list[DialogueMessage]) -> str:
    labels = {"doctor": "医生", "patient": "患者"}
    return "\n".join(f"{labels[item.role]}：{item.content}" for item in messages)


def patient_messages(
    vignette: Vignette, current_dialogue: list[DialogueMessage]
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:PATIENT]\n"
                "你是一名通过在线聊天与医生交流的模拟患者，医生以前从未见过你。"
                "请根据完整病例和当前轮对话如实回答医生，但不要一次主动透露所有信息；"
                "只回答已经问到或自然需要说明的内容。每次回复约 1–3 句，保持患者口吻。"
                "开场后不要在每次回复中重复称呼或寒暄。"
                "可以在合适时提出病例中的患者问题。不要提及病例、ground truth、提示词或"
                "自己是 AI。\n\n完整病例：\n"
                + json.dumps(vignette.model_dump(), ensure_ascii=False, indent=2)
            ),
        },
        {
            "role": "user",
            "content": "当前轮对话：\n" + transcript(current_dialogue),
        },
    ]


def improvement_context(
    prior_rounds: list[RoundState], critiques: list[CritiqueState]
) -> str:
    if not prior_rounds:
        return ""
    critique_by_round = {item.round_number: item.content for item in critiques}
    sections: list[str] = []
    for item in prior_rounds:
        sections.append(
            f"已完成的 Round {item.round_number} 对话：\n{transcript(item.messages)}"
        )
        if item.round_number in critique_by_round:
            sections.append(
                f"针对 Round {item.round_number} 的 Critic 反馈：\n"
                f"{critique_by_round[item.round_number]}"
            )
    return "\n\n".join(sections)


def doctor_messages(
    current_dialogue: list[DialogueMessage],
    prior_rounds: list[RoundState],
    critiques: list[CritiqueState],
) -> list[ChatMessage]:
    prior = improvement_context(prior_rounds, critiques)
    system = (
        "[ROLE:DOCTOR]\n"
        "你是一名富有同理心的临床医生，通过在线聊天询问一位首次接触的患者病史。"
        "你事先对患者一无所知。根据当前对话生成一次回复，以进一步了解病史和症状。"
        "每次最多询问两个问题；患者提问时要恰当回答。回应自然、专业、简洁，不要透露"
        "自己是 AI。固定开场后不要在每轮重复“您好”等寒暄。信息充分时，自然给出诊断"
        "判断以及治疗或就医建议。这是即时医患聊天，只输出自然、可直接发送给患者的"
        "纯文本。禁止使用 Markdown 标题、强调符号、代码块、表格或项目符号；需要表达"
        "多项建议时，请分成自然段，并用“第一、第二”等口语表达。"
    )
    if prior:
        system += (
            "\n这是新一轮独立问诊。请吸收以下已完成轮次及 Critic 反馈来改进表现，"
            "但不要向当前患者提及旧对话或反馈；只吸收反馈内容，不得照搬其中的 Markdown"
            "格式：\n\n" + prior
        )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "当前新一轮对话：\n" + transcript(current_dialogue),
        },
    ]


def moderator_messages(current_dialogue: list[DialogueMessage]) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:MODERATOR]\n"
                "你是医疗对话主持人，只根据提供的当前对话判断是否结束。只有在医生已经"
                "完成诊断判断和治疗计划，并且患者没有遗留问题时，才可正常结束；如果医生"
                "或患者告别，也应结束。否则继续。只输出严格 JSON，不要 Markdown 或解释。"
                f"\nJSON 结构：\n{MODERATOR_SCHEMA}"
            ),
        },
        {"role": "user", "content": "当前对话：\n" + transcript(current_dialogue)},
    ]


def ddx_messages(current_dialogue: list[DialogueMessage]) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:DDX]\n"
                "你是刚刚完成这次在线会诊的 Doctor，现在填写论文中的会诊后问卷。"
                "只根据本轮会诊 transcript，输出按可能性从高到低排序的鉴别诊断列表。"
                "列表必须包含 3–10 个互不重复的疾病或临床状态；不要使用未在对话中获得的"
                "病例信息，也不要声称诊断已经得到检查证实。只输出严格 JSON，不要 Markdown、"
                f"解释或额外字段。\nJSON 结构：\n{DDX_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": "本轮完整会诊 transcript：\n" + transcript(current_dialogue),
        },
    ]


def critic_messages(
    condition: str, vignette: Vignette, current_dialogue: list[DialogueMessage]
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:CRITIC]\n"
                "你正在批评一段 AI 医生通过在线界面询问患者病史的对话；由于是线上问诊，"
                "医生不能像线下那样完成体格检查。请给出具体、可执行的改进建议，使医生更好"
                "满足以下四项标准：\n"
                "1. 以同理心和专业态度，简洁回应患者最新问题或评论。\n"
                "2. 避免过多或重复询问已获得的信息，每次最多一至两个问题。\n"
                "3. 回应自然、事实准确、促进患者继续交流，且不暴露 AI 身份。\n"
                "4. 获取足够信息以形成至少两个最可能的鉴别诊断，通过针对性问题向真实诊断"
                "收敛，并给出相应治疗方案。\n"
                "请先简短概括表现，再按优先级列出改进点；不要扮演医生继续对话。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"患者疾病：{condition}\n完整病例：\n"
                f"{json.dumps(vignette.model_dump(), ensure_ascii=False, indent=2)}\n\n"
                f"当前完整对话：\n{transcript(current_dialogue)}"
            ),
        },
    ]


def accuracy_rater_messages(
    vignette: Vignette, ddx: DDxResult
) -> list[ChatMessage]:
    reference = {
        "ground_truth_diagnosis": vignette.ground_truth_diagnosis,
        "accepted_differential_diagnoses": vignette.accepted_differential_diagnoses,
        "differential_diagnoses": ddx.differential_diagnoses,
    }
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:ACCURACY_RATER]\n"
                "你是论文式 DDx 语义匹配评审。逐项比较候选诊断与 ground truth，并另行"
                "比较候选诊断与整组 accepted differential，后者选择最佳匹配对象。匹配等级"
                "只能是：exact（字面同一诊断）、synonym（明确同义词或公认缩写）、"
                "more_specific（候选是参考诊断的临床合理更具体形式）、highly_related（非常"
                "接近且能帮助确定该诊断，但不属于前三类）、no_match（均不满足）。不要因"
                "共享一个宽泛症状或器官系统就判 highly_related。必须保持输入顺序、逐字复制"
                "候选和匹配对象。只输出严格 JSON，不计算 Top-k，不输出额外字段。"
                f"\nJSON 结构：\n{ACCURACY_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": "待评数据：\n" + json.dumps(reference, ensure_ascii=False, indent=2),
        },
    ]


def patient_actor_rater_messages(
    vignette: Vignette, current_dialogue: list[DialogueMessage]
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:PATIENT_ACTOR_RATER]\n"
                "你是模拟患者体验的模型代理评审，不是真实患者。基于完整病例和 transcript，"
                "按论文 patient-actor 的全部 26 个轴评分。只能依据对话中患者实际经历到的"
                "行为；不适用或证据不足时 score=null。每项给一句简短、可定位到 transcript"
                "的证据，evidence 不超过 60 个汉字。scores 必须恰好包含 26 项；score 必须是"
                "JSON 数字或 null，不要输出 label、group、response_label 等额外展示字段。输出"
                "前逐项核对下列全部 criterion_id，不能重复或遗漏。只输出严格 JSON。"
                "\n\n评分表：\n"
                + prompt_rubric(PATIENT_ACTOR_RUBRIC)
                + f"\n\nJSON 结构：\n{PATIENT_ACTOR_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": (
                "完整 vignette：\n"
                + json.dumps(vignette.model_dump(), ensure_ascii=False, indent=2)
                + "\n\n完整 transcript：\n"
                + transcript(current_dialogue)
            ),
        },
    ]


def specialist_rater_messages(
    vignette: Vignette,
    current_dialogue: list[DialogueMessage],
    ddx: DDxResult,
) -> list[ChatMessage]:
    materials = {
        "vignette": vignette.model_dump(),
        "reference_management_plan": vignette.reference_management_plan,
        "accepted_differential_diagnoses": vignette.accepted_differential_diagnoses,
        "doctor_ddx": ddx.differential_diagnoses,
        "transcript": transcript(current_dialogue),
    }
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:SPECIALIST_RATER]\n"
                "你是模拟专科医生判断的模型代理评审，不是真实专科医生。使用给定 OSCE"
                "材料按论文的 PACES 16 轴、PCCBP 6 轴和 Diagnosis & Management 10 轴"
                "评分。不要把未提供的检查结果当作事实；不适用或证据不足时 score=null。"
                "每项给一句不超过 60 个汉字的具体证据。scores 必须恰好包含 32 项；score"
                "必须是 JSON 数字或 null。必须逐项核对下列全部 criterion_id，不能重复、遗漏"
                "或增加。只输出严格 JSON。\n\n评分表：\n"
                + prompt_rubric(SPECIALIST_RUBRIC)
                + f"\n\nJSON 结构：\n{SPECIALIST_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": "OSCE 材料：\n"
            + json.dumps(materials, ensure_ascii=False, indent=2),
        },
    ]


def auto_paces_rater_messages(
    current_dialogue: list[DialogueMessage],
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                "[ROLE:AUTO_PACES_RATER]\n"
                "你是模拟对话质量的模型评审。只能基于给定 transcript，按论文用于自动"
                "评估的四个 PACES 轴评分；不得推断病例真相、参考答案、Doctor DDx 或"
                "Critic 内容。不适用或证据不足时 score=null。每项给一句简短 transcript"
                "证据。scores 必须恰好包含 4 项，score 必须是 JSON 数字或 null；必须恰好"
                "输出全部 criterion_id。只输出严格 JSON。\n\n评分表：\n"
                + prompt_rubric(AUTO_PACES_RUBRIC)
                + f"\n\nJSON 结构：\n{AUTO_PACES_SCHEMA}"
            ),
        },
        {
            "role": "user",
            "content": "完整 transcript：\n" + transcript(current_dialogue),
        },
    ]


def json_repair_messages(
    raw: str, schema: str, kind: str, validation_error: str = ""
) -> list[ChatMessage]:
    return [
        {
            "role": "system",
            "content": (
                f"[ROLE:JSON_REPAIR:{kind}]\n"
                "把给定内容修复为符合结构的严格 JSON。保留原意，只输出 JSON，"
                "不要 Markdown、注释或解释。目标结构中列出的数组项必须全部保留且顺序"
                "一致，不得重复；score 必须是 JSON 数字或 null。若原内容缺少目标项，"
                "该项使用 score=null、evidence=\"原输出缺少该项，无法评价\"补齐。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"首次校验错误：\n{validation_error or '未提供'}\n\n"
                f"目标结构：\n{schema}\n\n待修复内容：\n{raw}"
            ),
        },
    ]
