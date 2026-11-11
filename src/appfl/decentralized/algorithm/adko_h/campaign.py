"""Campaigns: a region, a rule, a budget, and a test the planner can fail.

ADKO-H moves the language model up one level. In L3 it replaced a single pick, competing with
a calibrated score on the score's own ground, and the screen found it costs more than it adds.
A campaign is the unit where a model can plausibly beat a score instead: *where to look, how
to look there, for how long, and what would count as being right*. Every individual experiment
inside the campaign is still chosen by a numeric rule.

The test is what makes this more than prompting. The planner states a hypothesis, a budget and
a threshold up front, the campaign runs, and the verdict is computed from the data -- not from
the model's opinion of its own work. The verdict travels to neighbours, and the stated
``p_success`` makes the planner scoreable (Brier against a constant predictor), so "the LLM
helped" becomes a measurement rather than a vibe.

The tests are novelty-first, because the applications this is built for reward *distinct*
outcomes rather than one optimum: how many unseen family keys a region yields, how far its best
point lands from everything already known, how much it grows the descriptor hull. Objective
tests are kept for yield-style proxies.

A campaign may also be cut into **parts** (``Part``): up to a few regions, each with its own rule
and count, run one after the other and judged together. A legacy design -- one region, one rule,
one budget -- is a campaign of one part, so everything that reads ``region``, ``rule`` and
``budget`` keeps its meaning. When the application reports signals (``signals.py``), the tests
can be about them -- mismatches, surprises, novelty -- and a result whose label is still pending
counts neither for nor against a test. A **ticket** is the code's own follow-up of a surprise:
a few draws in a small box around it, owned by no planner, judged on whether the surprise came
back.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h import signals as hsignals

#: Rules a campaign may ask for; see ``rules.py``.
RULES = ("adko", "indep", "exploit", "explore", "novelty")
#: Default smallest and largest budget a campaign may claim, in experiments. A campaign is
#: the unit a language model is paid for, so this is what sets the token cost of a run: at a
#: 10,000-experiment budget, campaigns of 5 mean 2,000 calls and campaigns of 40 mean 250.
#: Arms set it from the run's own size (``AgentConfig.budget_range``).
BUDGET_RANGE = (2, 10)


@dataclass
class Evidence:
    """What a test is judged on: the campaign's own runs, against what was known at its start.

    The agent fills this in; a test never reaches into the agent. ``known_*`` and
    ``hull_before`` are snapshotted when the campaign starts, so a test cannot be satisfied by
    something a neighbour found while the campaign was running.
    """

    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    families: List[str] = field(default_factory=list)
    known_descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    known_families: Set[str] = field(default_factory=set)
    hull_before: float = 0.0
    hull_after: float = 0.0
    #: What counts as a hit, for the hit metrics; ``None`` means nothing does.
    hit_threshold: Optional[float] = None
    maximize: bool = True
    #: The signals of every run (pending ones included) and the part each ran in, row-aligned
    #: with the campaign's runs. When signals are present, ``scores``, ``families`` and
    #: ``descriptors`` hold the settled runs only, so a pending label cannot pass or fail a test.
    signals: List[Dict[str, Any]] = field(default_factory=list)
    parts: List[int] = field(default_factory=list)

    def settled(self) -> List[Dict[str, Any]]:
        """The signals of the runs whose labels are final."""
        return [s for s in self.signals if not hsignals.pending(s)]

    def n_pending(self) -> int:
        return sum(1 for s in self.signals if hsignals.pending(s))

    def hit_mask(self) -> np.ndarray:
        """Which of the campaign's outcomes are hits: at or past the threshold."""
        s = np.asarray(self.scores, dtype=float)
        if self.hit_threshold is None or s.size == 0:
            return np.zeros(s.size, dtype=bool)
        return s >= self.hit_threshold if self.maximize else s <= self.hit_threshold


def _new_families(ev: Evidence) -> float:
    return float(len({f for f in ev.families if f} - ev.known_families))


def _hits(ev: Evidence) -> float:
    return float(np.count_nonzero(ev.hit_mask()))


def _new_hit_families(ev: Evidence) -> float:
    """Families among the campaign's hits that nobody had reported when it started.

    The campaign's own increment to a diverse-hits endpoint, which ``new_families`` is not:
    on a large pool almost every experiment finds a new family, and almost none is a hit.
    """
    hit = ev.hit_mask()
    return float(len({f for f, h in zip(ev.families, hit) if h and f} - ev.known_families))


def _novel_distance(ev: Evidence) -> float:
    """How far the campaign's most isolated point sits from everything known at its start."""
    Z, K = np.atleast_2d(ev.descriptors), np.atleast_2d(ev.known_descriptors)
    if Z.size == 0:
        return 0.0
    if K.size == 0 or K.shape[1] != Z.shape[1]:
        return float("inf")
    d = np.sqrt(((Z[:, None, :] - K[None, :, :]) ** 2).sum(axis=2))
    return float(np.max(np.min(d, axis=1)))


def _hull_growth(ev: Evidence) -> float:
    return float(ev.hull_after - ev.hull_before)


def _max_objective(ev: Evidence) -> float:
    return float(np.max(ev.scores)) if np.size(ev.scores) else 0.0


def _mean_objective(ev: Evidence) -> float:
    return float(np.mean(ev.scores)) if np.size(ev.scores) else 0.0


def _mismatches(ev: Evidence) -> float:
    """Settled runs whose observed class differs from the predicted one."""
    return float(sum(1 for s in ev.settled() if s.get("mismatch") is True))


def _mismatch_rate(ev: Evidence) -> float:
    """Mismatches over settled runs; 0 when nothing has settled, so pending runs cannot pass."""
    settled = ev.settled()
    return _mismatches(ev) / len(settled) if settled else 0.0


def _surprises(ev: Evidence) -> float:
    return float(sum(1 for s in ev.signals if hsignals.is_surprise(s)))


def _max_novelty(ev: Evidence) -> float:
    """The largest novelty among settled runs; 0 when none is known."""
    vals = [float(s["novelty"]) for s in ev.settled() if s.get("novelty") is not None]
    return max(vals) if vals else 0.0


#: Metric name -> how it is computed from :class:`Evidence`.
TESTS: Dict[str, Callable[[Evidence], float]] = {
    "new_families": _new_families,
    "novel_distance": _novel_distance,
    "hull_growth": _hull_growth,
    "max_objective": _max_objective,
    "mean_objective": _mean_objective,
    "hits": _hits,
    "new_hit_families": _new_hit_families,
    "mismatches": _mismatches,
    "mismatch_rate": _mismatch_rate,
    "surprises": _surprises,
    "max_novelty": _max_novelty,
}
#: The metrics computed from signals, which the version-3 schema offers first.
SIGNAL_METRICS = ("mismatches", "mismatch_rate", "surprises", "max_novelty")
#: What a version-3 test may use: the signal metrics, then the descriptor ones. The hit metrics
#: are left out: the arms that use signals have no hit threshold.
METRICS_V3 = SIGNAL_METRICS + ("new_families", "novel_distance", "hull_growth", "max_objective")
OPS = (">=", "<=")


@dataclass(frozen=True)
class Test:
    """``metric op threshold`` -- the whole of what the planner commits to.

    Deliberately tiny. A richer language would let the planner write tests it cannot fail,
    which is the failure mode this design exists to avoid.
    """

    metric: str = "new_families"
    op: str = ">="
    threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.metric not in TESTS:
            raise ValueError(f"metric must be one of {sorted(TESTS)}, got {self.metric!r}")
        if self.op not in OPS:
            raise ValueError(f"op must be one of {OPS}, got {self.op!r}")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")

    def value(self, ev: Evidence) -> float:
        return TESTS[self.metric](ev)

    def passed(self, ev: Evidence) -> bool:
        v = self.value(ev)
        return v >= self.threshold if self.op == ">=" else v <= self.threshold

    def label(self) -> str:
        return f"{self.metric} {self.op} {self.threshold:g}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clamp_counts(counts: Sequence[int], budget_range: Tuple[int, int]) -> List[int]:
    """Part counts with each at least 1 and the total inside ``budget_range``.

    Excess is taken from the last parts first and a shortfall added to the last part, so the
    planner's first part -- usually its main bet -- keeps what it asked for. More parts than
    the largest budget are dropped from the end.
    """
    lo, hi = int(budget_range[0]), int(budget_range[1])
    ns = [max(int(n), 1) for n in counts][: max(hi, 1)]
    if not ns:
        return []
    excess = sum(ns) - hi
    for j in range(len(ns) - 1, -1, -1):
        if excess <= 0:
            break
        cut = min(ns[j] - 1, excess)
        ns[j] -= cut
        excess -= cut
    if sum(ns) < lo:
        ns[-1] += lo - sum(ns)
    return ns


def claim_status(ev: Evidence, part: int, cls: str) -> Dict[str, Any]:
    """A class claim's status, set by code from part ``part``'s runs.

    ``supported`` when at least one run of the part has settled and at least half of the settled
    ones show ``cls``; ``refuted`` when some have settled and fewer than half show it;
    ``unsettled`` when none has.
    """
    rows = [(s, hsignals.pending(s)) for s, p in zip(ev.signals, ev.parts) if int(p) == int(part)]
    settled = [s for s, pend in rows if not pend]
    hit = sum(1 for s in settled if hsignals.shows(s, cls))
    if not settled:
        status = "unsettled"
    else:
        status = "supported" if hit >= 0.5 * len(settled) else "refuted"
    return {
        "status": status,
        "settled": len(settled),
        "shows": hit,
        "pending": sum(1 for _, pend in rows if pend),
    }


@dataclass
class Part:
    """One stretch of a multi-part campaign: a region, a rule, and how many picks it gets.

    ``anchor`` is the surprise id (``s<k>``) the region was widened to contain, if any;
    ``claim_id`` is what this part's claim is announced under (``a{i}c{n}p{j}``).
    """

    region: Any
    rule: str = "adko"
    n: int = 1
    spent: int = 0
    anchor: Optional[str] = None
    claim_id: str = ""

    @property
    def done(self) -> bool:
        return self.spent >= self.n

    def to_dict(self, space=None) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "region": self.region.to_dict() if hasattr(self.region, "to_dict") else {},
            "region_label": self.region.label(space) if space is not None else "",
            "rule": self.rule,
            "n": self.n,
            "spent": self.spent,
            "anchor": self.anchor,
        }


@dataclass
class Ticket:
    """The code's own follow-up of one surprise: ``draws`` picks in a small box around it.

    The swarm's new states came mostly from confirmation draws its code made around an in-loop
    surprise, not from anything its model planned, so here that step belongs to code: while a
    ticket has draws left the agent's picks are its draws (the nearest unrun candidate to the
    origin), its campaign waits, and the ticket is judged on whether the surprise came back.

    Args:
        id: ``a{i}t{k}``; what the ticket's verdict cites.
        surprise: the surprise id (``s<k>``) it follows up.
        origin_key: the surprising run's key.
        F: the surprising run's region coordinates.
        kind: the surprise's kind; a draw reproduces it by showing a surprise of the same kind.
        draws: how many follow-up picks.
        region: the box the draws come from.
        opened: round the ticket opened.
    """

    id: str
    surprise: str
    origin_key: int
    F: List[float]
    kind: str
    draws: int
    region: Any = None
    opened: int = 0
    spent: int = 0
    runs: List[Dict[str, Any]] = field(default_factory=list)
    status: Optional[str] = None
    closed: Optional[int] = None

    @property
    def drawing(self) -> bool:
        """Open with draws left: the agent's next pick is one of them."""
        return self.status is None and self.spent < self.draws

    def reproduced(self) -> int:
        return sum(
            1
            for r in self.runs
            if hsignals.is_surprise(r.get("signals")) and (r.get("signals") or {}).get("kind") == self.kind
        )

    def outcome(self, final: bool = False) -> Optional[str]:
        """The ticket's status once it can be closed, else ``None``.

        It closes when its draws are spent and their labels have settled, or when the run ends
        (``final``): ``reproduced`` if a draw shows a surprise of the same kind,
        ``not_reproduced`` if every draw settled without one, ``unsettled`` otherwise.
        """
        sigs = [r.get("signals") or {} for r in self.runs]
        complete = self.spent >= self.draws and not any(hsignals.pending(s) for s in sigs)
        if not (complete or final):
            return None
        if self.reproduced():
            return "reproduced"
        return "not_reproduced" if complete else "unsettled"

    def to_dict(self, space=None) -> Dict[str, Any]:
        sigs = [r.get("signals") or {} for r in self.runs]
        return {
            "id": self.id,
            "surprise": self.surprise,
            "origin_key": self.origin_key,
            "kind": self.kind,
            "region_label": self.region.label(space) if space is not None and self.region is not None else "",
            "draws": self.draws,
            "spent": self.spent,
            "reproduced": self.reproduced(),
            "pending": sum(1 for s in sigs if hsignals.pending(s)),
            "opened": self.opened,
            "status": self.status or "open",
            "closed": self.closed,
        }


@dataclass
class Campaign:
    """One planned stretch of a run: what was committed to, and what came of it.

    Args:
        region: where to look.
        rule: how to pick inside it (one of :data:`RULES`).
        budget: how many experiments to spend before judging.
        hypothesis: one line, stated before the data exists.
        rationale: why this region and rule, in the planner's own words.
        note: one sentence for the neighbours, written when this campaign opened: what the
            previous one taught. Only the detailed channel sends it.
        test: what would count as being right.
        p_success: the planner's stated probability that the test passes.
        claim_id: this agent's id for the campaign, ``a{source}c{n}``; what a verdict cites.
        started: round the campaign began.
        parts: the campaign's parts in order. Left empty, the campaign is one part built from
            ``region``, ``rule`` and ``budget``; given, ``region`` and ``rule`` follow the part
            being run and ``budget`` is the sum of the counts (clamped to ``budget_range``).
        extra_rules: rule names a problem supplies, valid here besides :data:`RULES`.
        claim: a class claim, ``{"part": j, "class": C, "claim_id": ...}``; its status is set
            by :meth:`judge`.
        early: why the campaign closed before its budget was spent (``"surprise"``), if it did.
    """

    region: Any
    rule: str = "adko"
    budget: int = 5
    hypothesis: str = ""
    rationale: str = ""
    note: str = ""
    test: Test = field(default_factory=Test)
    p_success: float = 0.5
    claim_id: str = ""
    started: int = 0
    runs: List[Dict[str, Any]] = field(default_factory=list)
    baseline: Dict[str, Any] = field(default_factory=dict)
    verdict: Optional[Dict[str, Any]] = None
    source: str = "llm"
    budget_range: Tuple[int, int] = BUDGET_RANGE
    parts: List[Part] = field(default_factory=list)
    extra_rules: Tuple[str, ...] = ()
    claim: Optional[Dict[str, Any]] = None
    early: Optional[str] = None
    anchors_dropped: List[str] = field(default_factory=list)
    #: Whether the design came in parts; only such a campaign records them.
    multipart: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        allowed = RULES + tuple(self.extra_rules)
        if self.parts:
            self.multipart = True
            for p in self.parts:
                p.rule = p.rule if p.rule in allowed else "adko"
            counts = (
                [max(int(p.n), 1) for p in self.parts]
                if self.source == "none"
                else clamp_counts([p.n for p in self.parts], self.budget_range)
            )
            self.parts = self.parts[: len(counts)]
            for p, n in zip(self.parts, counts):
                p.n = int(n)
            self.budget = sum(p.n for p in self.parts)
            self._sync()
        else:
            self.rule = self.rule if self.rule in allowed else "adko"
            if self.source != "none":
                # A planner's budget is clamped; the no-planner arm's single campaign is
                # open-ended by design -- it is plain ADKO, which never stops to be judged.
                lo, hi = self.budget_range
                self.budget = int(min(max(int(self.budget), lo), hi))
            self.parts = [Part(self.region, self.rule, self.budget, claim_id=self.claim_id)]
        self.p_success = float(min(max(float(self.p_success), 0.0), 1.0))

    @property
    def part_index(self) -> int:
        """The part being run: the first with picks left, or the last once all are spent."""
        for j, p in enumerate(self.parts):
            if not p.done:
                return j
        return len(self.parts) - 1

    def _sync(self) -> None:
        p = self.parts[self.part_index]
        self.region, self.rule = p.region, p.rule

    def add_run(self, run: Dict[str, Any]) -> int:
        """Record one of the campaign's experiments against the part being run; returns it."""
        j = self.part_index
        self.runs.append({**run, "part": j} if self.multipart else run)
        self.parts[j].spent += 1
        self._sync()
        return j

    def footprint(self, space):
        """The region the campaign covered: its region, or the hull of its parts' regions."""
        if not self.multipart:
            return self.region
        return space.hull([p.region for p in self.parts])

    @property
    def spent(self) -> int:
        return len(self.runs)

    @property
    def done(self) -> bool:
        return self.spent >= self.budget or self.early is not None

    @property
    def last(self) -> bool:
        """Whether the next experiment is the campaign's last -- when the verdict is written."""
        return self.spent + 1 >= self.budget

    def judge(self, ev: Evidence, early: Optional[str] = None) -> Dict[str, Any]:
        """Close the campaign: compute the metric, compare it, keep the result.

        Idempotent, so a replayed run does not re-judge. ``early`` closes it before its budget
        is spent (on what is known so far) and says why. With signals present the verdict also
        records how many of its runs were still pending, and a class claim gets its status.
        """
        if self.verdict is None:
            if early:
                self.early = str(early)
            v = self.test.value(ev)
            self.verdict = {
                "claim_id": self.claim_id,
                "metric": self.test.metric,
                "value": None if not math.isfinite(v) else round(float(v), 6),
                "op": self.test.op,
                "threshold": float(self.test.threshold),
                "success": bool(self.test.passed(ev)),
                "p_success": self.p_success,
                "spent": self.spent,
                "hypothesis": self.hypothesis,
            }
            if any(ev.signals):
                self.verdict["n_pending"] = ev.n_pending()
            if self.early:
                self.verdict["early"] = self.early
            if self.claim is not None:
                self.claim.update(claim_status(ev, self.claim["part"], self.claim["class"]))
                self.verdict["claim"] = dict(self.claim)
        return self.verdict

    def brier(self) -> Optional[float]:
        """``(p - outcome)^2`` once judged: the planner's calibration, one campaign at a time."""
        if self.verdict is None:
            return None
        return float((self.p_success - (1.0 if self.verdict["success"] else 0.0)) ** 2)

    def to_dict(self, space=None) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "region": self.region.to_dict() if hasattr(self.region, "to_dict") else {},
            "region_label": self.region.label(space) if space is not None else "",
            "share": round(self.region.share(space), 6) if space is not None else None,
            "rule": self.rule,
            "budget": self.budget,
            "spent": self.spent,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "note": self.note,
            "test": self.test.to_dict(),
            "p_success": self.p_success,
            "started": self.started,
            "source": self.source,
            "verdict": self.verdict,
            **({"parts": [p.to_dict(space) for p in self.parts]} if self.multipart else {}),
            **({"claim": dict(self.claim)} if self.claim is not None else {}),
            **({"early": self.early} if self.early else {}),
            **({"anchors_dropped": list(self.anchors_dropped)} if self.anchors_dropped else {}),
        }


def random_design(
    space,
    rng: np.random.Generator,
    half_width: Tuple[float, float] = (0.1, 0.4),
    budget_range: Tuple[int, int] = BUDGET_RANGE,
    center: Optional[Sequence[float]] = None,
    widths: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """A campaign design drawn at random -- the control arm's planner.

    Matched to the LLM arm in *lifecycle*, not in content: same call sites, same budget range,
    same tests, same claims and verdicts. Whatever the LLM arm wins over this is the planning,
    not the machinery around it.

    By default the box sits anywhere in the unit cube with one random half-width, which on a
    clumped pool leaves many boxes nearly empty. ``center`` (a unit-cube point, e.g. a pool
    member's normalized coordinates) and ``widths`` (one half-width per coordinate) draw boxes
    where candidates exist and shaped like a planner's, which is the control a planner's
    regions should be compared with.
    """
    center = rng.random(len(space)) if center is None else np.asarray(center, dtype=float)
    hw = float(rng.uniform(*half_width)) if widths is None else widths
    region = space.box_around(center, hw)
    metric = str(rng.choice(["new_families", "novel_distance", "hull_growth"]))
    threshold = {
        "new_families": float(rng.integers(1, 4)),
        "novel_distance": float(rng.uniform(0.5, 2.0)),
        "hull_growth": float(rng.uniform(0.0, 1.0)),
    }[metric]
    return {
        "region": region,
        "rule": str(rng.choice(RULES)),
        "budget": int(rng.integers(budget_range[0], budget_range[1] + 1)),
        "hypothesis": f"random design around {region.label(space)}",
        "rationale": "random control",
        "test": Test(metric=metric, op=">=", threshold=threshold),
        "p_success": 0.5,
        "source": "random",
    }
