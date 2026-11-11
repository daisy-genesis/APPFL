"""The campaign schema, generated from the problem's own region space.

The planner's whole output is one object: a region, a rule, a budget, a hypothesis, a reason
and a test. Everything in it is constrained -- the rule and the metric are enumerations, the
budget is bounded, the region's coordinates are exactly the ones the problem declared -- so a
served model under guided decoding cannot answer with something the loop has to interpret.
Narrowing happens here and clipping happens in ``RegionSpace.region``; by the time a design
reaches an agent it is already legal.

The schema is built per problem rather than written out, because the coordinates differ
(molecule properties, composition fractions, Suzuki's categorical options) and a hand-written
schema would drift from the space it describes. Every coordinate is nullable and required:
``null`` is how the planner says "leave this one free", which is a meaningful choice and a
narrower schema than making properties optional, which some guided-decoding backends ignore.

Version 3 (:func:`campaign_format_v3`, :func:`parse_v3`) asks for a design in **parts** -- up to
a few regions, each with a rule and a count -- plus, when the arm has them, an anchor per part,
a class claim, and a ``workspace`` object whose schema the harness supplies. The same stance
holds: every field required, every choice an enumeration, every count bounded.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

# METRICS_V3 (version 3 leads with the signal metrics) lives with the tests, in ``campaign``.
from appfl.decentralized.algorithm.adko_h.campaign import (  # noqa: F401
    BUDGET_RANGE,
    METRICS_V3,
    RULES,
    Test,
    clamp_counts,
)
from appfl.decentralized.algorithm.adko_h.region import KIND_CAT, RegionSpace

#: Tests the planner may write. ``novel_distance`` and ``hull_growth`` are in descriptor units,
#: which the prompt states, so a threshold is meaningful rather than guessed.
METRICS = ("new_families", "novel_distance", "hull_growth", "max_objective")
#: Version 2 adds the hit metrics, so a test can be about the endpoint itself.
METRICS_V2 = ("new_hit_families", "hits") + METRICS
#: What an anchor may name: a surprise id from the prompt.
_ANCHOR = re.compile(r"^s[0-9]+$")


def _region_schema(space: RegionSpace) -> Dict[str, Any]:
    """The region object: every coordinate required and nullable (``null`` = leave it free)."""
    props: Dict[str, Any] = {}
    for d in space.dims:
        if d.kind == KIND_CAT:
            props[d.name] = {
                "type": ["array", "null"],
                "items": {"type": "integer", "minimum": 0, "maximum": len(d.options) - 1},
                "description": f"option indices to allow for {d.name}, or null to leave free",
            }
        else:
            props[d.name] = {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "description": f"[low, high] for {d.name} within [{d.lo:g}, {d.hi:g}], or null",
            }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }


def _test_schema(metrics: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["metric", "op", "threshold"],
        "properties": {
            "metric": {"type": "string", "enum": list(metrics)},
            "op": {"type": "string", "enum": [">=", "<="]},
            "threshold": {"type": "number"},
        },
    }


_NOTE_SCHEMA = {
    "type": "string",
    "description": (
        "one sentence to your neighbours: what your last campaign taught you about "
        "where hits are, in your own words"
    ),
}


def campaign_format(
    space: RegionSpace, metrics: tuple = METRICS, budget_range=None, note: bool = False
) -> Dict[str, Any]:
    """A ``json_schema`` response format for one campaign design.

    ``note`` adds the one field the detailed channel needs: a sentence addressed to the
    neighbours rather than to the agent itself. It is asked for in the same call as the design,
    so a richer channel costs output tokens and not a second request.
    """
    lo, hi = tuple(budget_range) if budget_range else BUDGET_RANGE
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["region", "rule", "budget", "hypothesis", "rationale", "test", "p_success"],
        "properties": {
            "region": _region_schema(space),
            "rule": {"type": "string", "enum": list(RULES)},
            "budget": {
                "type": "integer",
                "minimum": lo,
                "maximum": hi,
            },
            "hypothesis": {
                "type": "string",
                "description": "one sentence, stated before the data exists",
            },
            "rationale": {"type": "string", "description": "why this region and this rule"},
            "test": _test_schema(metrics),
            "p_success": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "your probability that the test passes",
            },
        },
    }
    if note:
        schema["required"].append("note")
        schema["properties"]["note"] = dict(_NOTE_SCHEMA)
    return {"type": "json_schema", "name": "campaign", "schema": schema, "strict": True}


def with_workspace(fmt: Dict[str, Any], workspace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``fmt`` with a ``workspace`` property added to the design object (and required, as every
    field here is: the harness's own schema decides what an empty answer looks like)."""
    if not workspace:
        return fmt
    schema = dict(fmt["schema"])
    schema["properties"] = {**schema["properties"], "workspace": workspace}
    schema["required"] = list(schema["required"]) + ["workspace"]
    return {**fmt, "schema": schema}


def campaign_format_v3(
    space: RegionSpace,
    metrics: Sequence[str] = METRICS_V3,
    budget_range=None,
    rules: Sequence[str] = RULES,
    max_parts: int = 3,
    classes: Optional[Sequence[str]] = None,
    note: bool = False,
    workspace: Optional[Dict[str, Any]] = None,
    anchors: bool = False,
    text_limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """The version-3 response format: a design in 1 to ``max_parts`` parts.

    Each part is a region, a rule (``rules``: the built-in five and any the problem adds) and a
    count; ``anchors`` adds a nullable surprise id per part. ``classes`` (what a class claim may
    name) adds a nullable ``claim``; ``note`` the detailed channel's sentence; ``workspace`` the
    harness's object. The total of the counts is not expressible in JSON schema, so the parser
    clamps it to ``budget_range``; each count is bounded here by the largest total.

    ``text_limits`` ({hypothesis, rationale, note}: characters) go into the field descriptions, as the
    prompt states them; they are not ``maxLength`` keywords, which guided-decoding backends do not all
    support, and the agent cuts what runs past them.
    """
    lo, hi = tuple(budget_range) if budget_range else BUDGET_RANGE
    lim = {"hypothesis": 300, "rationale": 600, "note": 240, **{k: int(v) for k, v in (text_limits or {}).items()}}
    max_parts = max(int(max_parts), 1)
    part_props: Dict[str, Any] = {
        "region": _region_schema(space),
        "rule": {"type": "string", "enum": list(rules)},
        "n": {
            "type": "integer",
            "minimum": 1,
            "maximum": int(hi),
            "description": f"experiments in this part; all parts together {lo} to {hi}",
        },
    }
    if anchors:
        part_props["anchor"] = {
            "type": ["string", "null"],
            "description": "a surprise id (s<k>) this part must contain, or null",
        }
    props: Dict[str, Any] = {
        "parts": {
            "type": "array",
            "minItems": 1,
            "maxItems": max_parts,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(part_props),
                "properties": part_props,
            },
        },
        "hypothesis": {"type": "string",
                       "description": f"what you expect to find, stated before the data exists; at most {lim['hypothesis']} characters"},
        "rationale": {"type": "string", "description": f"why these regions and rules; at most {lim['rationale']} characters"},
        "test": _test_schema(metrics),
        "p_success": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "your probability that the test passes",
        },
    }
    if classes:
        props["claim"] = {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["part", "class"],
            "properties": {
                "part": {"type": "integer", "minimum": 0, "maximum": max_parts - 1},
                "class": {"type": "string", "enum": list(classes)},
            },
            "description": "the class you claim forms in one part (0-based), or null",
        }
    if note:
        props["note"] = {"type": "string", "description": (
            f"to your neighbours, at most {lim['note']} characters: what your last campaign taught you about "
            "where the surprises are, in your own words")}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(props),
        "properties": props,
    }
    return with_workspace({"type": "json_schema", "name": "campaign_v3", "schema": schema, "strict": True},
                          workspace)


def _region(space: RegionSpace, raw: Any):
    """A region from the model's object, malformed intervals dropped; and whether it said anything."""
    raw = raw if isinstance(raw, dict) else {}
    bounds: Dict[str, Any] = {}
    picks: Dict[str, Any] = {}
    for d in space.dims:
        v = raw.get(d.name)
        if v is None or not isinstance(v, (list, tuple)):
            continue
        if d.kind == KIND_CAT:
            picks[d.name] = [int(i) for i in v if isinstance(i, (int, float))]
        elif len(v) >= 2:
            try:
                bounds[d.name] = [float(v[0]), float(v[1])]
            except (TypeError, ValueError):
                continue
    return space.region(bounds=bounds, picks=picks), bool(bounds or picks)


def parse_v3(
    space: RegionSpace,
    data: Optional[Dict[str, Any]],
    budget_range=None,
    rules: Sequence[str] = RULES,
    max_parts: int = 3,
    claim_classes: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """A version-3 answer turned into a design, or ``None`` if there is nothing usable.

    Clipped, never trusted: at most ``max_parts`` parts, an unknown rule becomes ``explore``,
    each count at least 1 and the total clamped to ``budget_range`` (excess taken from the last
    parts), an anchor kept only if it looks like a surprise id (the agent drops one it does not
    know), a claim kept only if it names a part and a listed class. A reply with no usable part
    and no test is refused, as in version 2. The result also carries ``region``, ``rule`` and
    ``budget`` (first part's region and rule, total count) for code that reads a legacy design.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("parts")
    raw = [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []
    raw = raw[: max(int(max_parts), 1)]
    t = data.get("test") if isinstance(data.get("test"), dict) else {}
    parts: List[Dict[str, Any]] = []
    said = False
    for p in raw:
        region, any_ = _region(space, p.get("region"))
        said = said or any_
        rule = str(p.get("rule", "explore"))
        try:
            n = int(p.get("n", 1))
        except (TypeError, ValueError):
            n = 1
        anchor = p.get("anchor")
        parts.append({
            "region": region,
            "rule": rule if rule in tuple(rules) else "explore",
            "n": n,
            "anchor": str(anchor) if isinstance(anchor, str) and _ANCHOR.match(anchor) else None,
        })
    if not parts or (not said and not t):
        return None
    counts = clamp_counts([p["n"] for p in parts], tuple(budget_range) if budget_range else BUDGET_RANGE)
    parts = parts[: len(counts)]
    for p, n in zip(parts, counts):
        p["n"] = int(n)
    try:
        test = Test(
            metric=str(t.get("metric", "mismatch_rate")),
            op=str(t.get("op", ">=")),
            threshold=float(t.get("threshold", 0.5)),
        )
    except (ValueError, TypeError):
        test = Test("mismatch_rate", ">=", 0.5)
    claim = None
    c = data.get("claim")
    if isinstance(c, dict) and claim_classes:
        try:
            j = int(c.get("part", 0))
        except (TypeError, ValueError):
            j = -1
        if 0 <= j < len(parts) and str(c.get("class", "")) in tuple(claim_classes):
            claim = {"part": j, "class": str(c["class"])}
    try:
        p_success = float(data.get("p_success", 0.5))
    except (TypeError, ValueError):
        p_success = 0.5
    return {
        "parts": parts,
        "region": parts[0]["region"],
        "rule": parts[0]["rule"],
        "budget": sum(p["n"] for p in parts),
        "hypothesis": str(data.get("hypothesis", "")),
        "rationale": str(data.get("rationale", "")),
        "note": str(data.get("note", "")),
        "claim": claim,
        "test": test,
        "p_success": min(max(p_success, 0.0), 1.0),
        "source": "llm",
        "workspace": data.get("workspace"),
    }


def parse(space: RegionSpace, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A model's answer turned into a design, or ``None`` if there is nothing usable.

    Forgiving about shape and strict about meaning: a malformed interval is dropped (leaving
    that coordinate free), an unknown rule becomes the settled one, and a budget outside the
    range is clipped -- but a reply with no region and no test at all is refused, so a broken
    call is counted as an error instead of quietly becoming a random campaign.
    """
    if not isinstance(data, dict):
        return None
    raw = data.get("region")
    if not isinstance(raw, dict):
        raw = {}
    bounds: Dict[str, Any] = {}
    picks: Dict[str, Any] = {}
    for d in space.dims:
        v = raw.get(d.name)
        if v is None or not isinstance(v, (list, tuple)):
            continue
        if d.kind == KIND_CAT:
            picks[d.name] = [int(i) for i in v if isinstance(i, (int, float))]
        elif len(v) >= 2:
            bounds[d.name] = [float(v[0]), float(v[1])]
    t = data.get("test") if isinstance(data.get("test"), dict) else {}
    if not bounds and not picks and not t:
        return None
    try:
        test = Test(
            metric=str(t.get("metric", "new_families")),
            op=str(t.get("op", ">=")),
            threshold=float(t.get("threshold", 1.0)),
        )
    except (ValueError, TypeError):
        test = Test()
    return {
        "region": space.region(bounds=bounds, picks=picks),
        "rule": str(data.get("rule", "adko")),
        "budget": int(data.get("budget", 5)) if str(data.get("budget", "")).strip() != "" else 5,
        "hypothesis": str(data.get("hypothesis", "")),
        "rationale": str(data.get("rationale", "")),
        "note": str(data.get("note", "")),
        "test": test,
        "p_success": float(data.get("p_success", 0.5)),
        "source": "llm",
    }
