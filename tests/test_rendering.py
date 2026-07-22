from __future__ import annotations

import re

import pytest

from amie_self_play.rendering import render_doctor_content
from amie_self_play.simulation import SimulationSession

from conftest import (
    FakeLLM,
    ddx_payload,
    evaluation_responses,
    json_text,
    moderator_payload,
    vignette_payload,
)


def test_doctor_markdown_is_rendered_with_a_small_safe_allowlist() -> None:
    content = """# 问诊建议

请注意 **夜间症状**，也可记录 *诱发动作* 和 `持续时间`。

- 第一项
- 第二项

> 症状加重时请及时就医。

[安全链接](https://example.com/guide "指南")
[危险链接](javascript:alert(1))

<script>alert(1)</script>
<span onclick="alert(1)">不要执行</span>
"""

    rendered = render_doctor_content(content)

    assert "# 问诊建议" not in rendered
    assert "<strong>夜间症状</strong>" in rendered
    assert "<em>诱发动作</em>" in rendered
    assert "<code>持续时间</code>" in rendered
    assert "<ul>" in rendered
    assert "<blockquote>" in rendered
    assert 'href="https://example.com/guide"' in rendered
    assert 'rel="noopener noreferrer"' in rendered
    assert 'href="javascript:' not in rendered
    assert "javascript:" not in rendered
    assert "[危险链接]" not in rendered
    assert "<script" not in rendered
    assert "<span" not in rendered
    assert "&lt;script&gt;" in rendered

    tags = set(re.findall(r"</?([a-z][a-z0-9]*)\b", rendered))
    assert tags <= {"a", "blockquote", "code", "em", "li", "ol", "p", "strong", "ul"}


@pytest.mark.asyncio
async def test_doctor_event_adds_html_without_rewriting_transcript() -> None:
    doctor_content = """# 建议

**第一**，夜间佩戴腕托。

- 如果无力加重，请线下面诊。
""".strip()
    fake = FakeLLM(
        VIGNETTE=[json_text(vignette_payload())],
        PATIENT=["右手夜间麻木。"],
        DOCTOR=[doctor_content],
        MODERATOR=[json_text(moderator_payload())],
        DDX=[json_text(ddx_payload())],
        CRITIC=["问诊完整。"],
        **evaluation_responses(),
    )
    events: list[dict] = []

    async def send(event: dict) -> None:
        events.append(event)

    session = SimulationSession(fake, send)
    await session.start("腕管综合征")

    doctor_events = [
        event
        for event in events
        if event["type"] == "message_completed" and event["role"] == "doctor"
    ]
    patient_event = next(
        event
        for event in events
        if event["type"] == "message_completed" and event["role"] == "patient"
    )

    assert "content_html" in doctor_events[0]
    assert doctor_events[-1]["content"] == doctor_content
    assert "<strong>第一</strong>" in doctor_events[-1]["content_html"]
    assert "<ul>" in doctor_events[-1]["content_html"]
    assert "content_html" not in patient_event
    assert session.rounds[0].messages[-1].content == doctor_content
