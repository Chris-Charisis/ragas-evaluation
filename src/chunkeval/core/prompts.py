"""Prompt files: prompts/*.md split into `## <name>` sections, checked on load.

Why files: a prompt can be changed without touching code, and the stages record a hash of the file
in their provenance, so outputs made with another prompt are never mixed with new ones.

File format (see prompts/qa_generation.md for a full example): everything above the first
`## <name>` line is notes and is never sent; each section runs until the next one. Template
sections use `string.Template` placeholders ($name); a literal dollar sign is written `$$` there.

Used by: qa/generation.py, qa/filtering.py (later evaluation/answering.py).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from string import Template

_SECTION_LINE = re.compile(r"^## ([a-z_]+)[ \t]*$", re.MULTILINE)


def load_sections(path: Path, expected: Mapping[str, Iterable[str] | None]) -> dict[str, str]:
    """The `## <name>` sections of a prompt file, by name, stripped.

    `expected` maps each section name to the placeholders its template must use, or to None for a
    section sent as is (not a template, so a plain "$" is fine there). Raises ValueError, so a typo
    fails before any LLM call instead of producing odd prompts, unless the file has exactly these
    sections and each template uses exactly its placeholders.
    """
    parts = _SECTION_LINE.split(path.read_text(encoding="utf-8"))
    names, bodies = parts[1::2], [body.strip() for body in parts[2::2]]
    if sorted(names) != sorted(expected):
        raise ValueError(f"{path}: sections must be {sorted(expected)}, found {names}")
    sections = dict(zip(names, bodies))
    for name, placeholders in expected.items():
        if placeholders is None:
            continue
        template = Template(sections[name])
        if not template.is_valid():
            raise ValueError(f"{path}: section {name!r} has a bad placeholder (a literal $ is written $$)")
        if set(template.get_identifiers()) != set(placeholders):
            raise ValueError(f"{path}: section {name!r} must use the placeholders {sorted(placeholders)}, "
                             f"found {sorted(template.get_identifiers())}")
    return sections
