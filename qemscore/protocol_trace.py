"""Parser for the machine-readable protocol trace table."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BEGIN = "<!-- protocol-trace:begin -->"
END = "<!-- protocol-trace:end -->"
HEADER = (
    "Rule ID",
    "Authority",
    "Protocol Rule",
    "Implementation",
    "Guarding Test",
)
_RULE_ID = re.compile(r"QEM-P[0-9]{3}")
_SEPARATOR = re.compile(r":?-{3,}:?")


@dataclass(frozen=True)
class ProtocolRule:
    rule_id: str
    authority: str
    statement: str
    implementations: tuple[str, ...]
    guards: tuple[str, ...]


def _cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _references(cell: str) -> tuple[str, ...]:
    return tuple(
        reference.strip().strip("`")
        for reference in re.split(r"<br\s*/?>", cell)
        if reference.strip()
    )


def load_protocol_rules(path: Path) -> tuple[ProtocolRule, ...]:
    text = path.read_text(encoding="utf-8")
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("PROTOCOL-TRACE.md must contain one begin marker and one end marker")
    body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 2 or _cells(lines[0]) != HEADER:
        raise ValueError("protocol trace table header does not match the five-column contract")
    separators = _cells(lines[1])
    if len(separators) != len(HEADER) or any(
        _SEPARATOR.fullmatch(cell) is None for cell in separators
    ):
        raise ValueError("protocol trace table separator is malformed")

    rules = []
    seen = set()
    for line in lines[2:]:
        cells = _cells(line)
        if len(cells) != len(HEADER) or any(not cell for cell in cells):
            raise ValueError(f"malformed protocol trace row: {line}")
        rule_id = cells[0]
        if _RULE_ID.fullmatch(rule_id) is None:
            raise ValueError(f"malformed protocol rule ID: {rule_id}")
        if rule_id in seen:
            raise ValueError(f"duplicate protocol rule ID: {rule_id}")
        seen.add(rule_id)
        implementations = _references(cells[3])
        guards = _references(cells[4])
        if not implementations or not guards:
            raise ValueError(f"protocol rule {rule_id} needs implementation and guard references")
        if any("::" not in guard for guard in guards):
            raise ValueError(f"protocol rule {rule_id} has a non-node guard reference")
        rules.append(
            ProtocolRule(
                rule_id=rule_id,
                authority=cells[1],
                statement=cells[2],
                implementations=implementations,
                guards=guards,
            )
        )
    return tuple(rules)
