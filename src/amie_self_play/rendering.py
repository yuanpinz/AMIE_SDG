from __future__ import annotations

from markdown_it import MarkdownIt
import nh3


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False})
# Let the sanitizer remove unsafe destinations without leaving raw Markdown syntax visible.
_MARKDOWN.validateLink = lambda _: True
_ALLOWED_TAGS = {
    "a",
    "blockquote",
    "code",
    "em",
    "li",
    "ol",
    "p",
    "strong",
    "ul",
}
_ALLOWED_ATTRIBUTES = {"a": {"href", "title"}}
_ALLOWED_URL_SCHEMES = {"http", "https", "mailto"}


def render_doctor_content(content: str) -> str:
    """Render model Markdown into the small, sanitized subset used by Doctor bubbles."""
    rendered = _MARKDOWN.render(content)
    return nh3.clean(
        rendered,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRIBUTES,
        url_schemes=_ALLOWED_URL_SCHEMES,
        link_rel="noopener noreferrer",
    )
