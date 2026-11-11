"""The ADKO-H agent: campaigns outside the loop, a numeric rule inside it.

One round, for one agent:

1. **plan if needed.** With no campaign, or with the last one spent, the agent assembles its
   context -- its own history, the active claims, the verdicts that reached it, its territory --
   and the driver asks the planner for a design. Every agent needing a design in a round is
   asked in one batched call, which is what keeps a 27B model serving a hundred agents.
2. **pick.** The problem draws a candidate pool; the agent scores it with its GP, the peer
   field and the round's ``match_rms`` factor; the campaign's rule turns that into values; the
   claim penalty and any novelty bonus adjust them; the argmax *inside the campaign's region*
   is the experiment. The planner is not consulted here, ever.
3. **complete.** The result updates the GP and the history, and counts against the campaign's
   budget. On the campaign's last experiment the test is evaluated and the verdict written --
   the verdict is computed from data, not asserted by the model.
4. **emit and observe.** One token goes out, carrying ``q``, optionally where the result landed,
   and this round's claims and verdicts. Incoming tokens update trust, memory, and the claim
   store, and new claims and verdicts are queued for relay.

What this does *not* do is pick experiments with a language model. The L3 screen measured that
and it lost to the score it was replacing; the same calls spent one level up buy a region, a
rule, a budget and a falsifiable test instead, and every experiment stays with the calibrated
rule that won.

**Agency** (every part off by default; ``AgentConfig`` turns each on). The snapshot read-out
showed what the planner lacked: the signal the read-out judges by, and the code-owned follow-up
that found the swarm's new states. So results may carry signals (``signals.py``) and late
labels (:meth:`ADKOHAgent.revise`); a design may come in parts, carry a class claim whose status
code sets, and anchor a part on a surprise; a surprise opens a ticket of follow-up draws and may
close the campaign early so the planner re-plans with it in view; a problem may add rules; and
everything that happens can be drained as events for a ledger. None of it puts the model back
into a pick.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from appfl.decentralized.algorithm.adko_h import metrics as hmetrics
from appfl.decentralized.algorithm.adko_h import rules as hrules
from appfl.decentralized.algorithm.adko_h import signals as hsignals
from appfl.decentralized.algorithm.adko_h.campaign import (
    BUDGET_RANGE,
    METRICS_V3,
    RULES,
    Campaign,
    Evidence,
    Part,
    Test,
    Ticket,
    random_design,
)
from appfl.decentralized.algorithm.adko_h.claims import (
    CLAIM_CHARS,
    CLAIM_SLACK,
    CLAIM_TTL,
    DETAIL_CHARS,
    NOTE_CHARS,
    VERDICT_CHARS,
    VERDICT_TTL,
    Claim,
    ClaimStore,
    Verdict,
    region_of,
)
from appfl.decentralized.algorithm.adko_h.problem import Candidates, Outcome, Pick, Problem, RuleView
from appfl.decentralized.algorithm.adko_h.region import KIND_CAT
from appfl.decentralized.algorithm.adko_h.token import HToken, encode, quantize_descriptor
from appfl.decentralized.algorithm.adko_v2.components import Surrogate
from appfl.decentralized.algorithm.adko_v2.knowledge_token import Quantizer, score_against
from appfl.decentralized.algorithm.adko_v2.memory import TokenMemory
from appfl.decentralized.algorithm.adko_v2.reasoning import (
    Calibrator,
    PeerField,
    ReasoningWeights,
    score,
)
from appfl.decentralized.algorithm.adko_v2.trust import TrustState, agreement, relative_sigma


@dataclass
class AgentConfig:
    """Everything an arm varies. The defaults are ADKO v2's settled method plus campaigns.

    Args:
        weights: the v2 score's ``beta``, ``lam``, ``gamma``, kernel and bandwidth.
        q_bits: token codebook; 8 is settled.
        memory_rounds: FIFO depth, in rounds times degree.
        trust: whether source weights follow the odds of learned trust.
        warmup_rounds: random picks before the first campaign, so the GP has data.
        pool_size: candidates drawn per agent-round. The reference set for ``match_rms`` too.
        claim_weight: strength of the steer-away penalty, in units of ``RMS(sigma)``.
        novelty_weight: novelty bonus applied whatever the rule -- the diversity-aware
            baseline is this with no planner attached.
        share_descriptor: put the quantized descriptor on the token.
        share_family: put the family key on the token.
        claims_on: announce claims and relay verdicts at all (off is the coordination ablation).
        share_text: carry the hypothesis on the wire. Off by default: a relayed claim's prose
            costs several times the result it rides on (measured), and only the verdict's text
            earns its place in a prompt, so verdicts keep a short form either way.
        territory: confine the first campaign to the agent's own Voronoi cell.
        default_budget: campaign length when there is no planner (the plain-ADKO arm runs one
            open-ended campaign instead).
        force_rule: the rule the no-planner arm uses for its open-ended campaign, and, when set,
            the rule of every planned campaign too. How much the endpoint moves across the five
            rules is how much headroom a planner that chooses rules has at all; if they all
            score alike, ADKO-H cannot win by choosing between them and the honest thing is to
            know that before spending a GPU on it. Fixing it under a planner keeps that planner's
            regions and takes the rule out of the comparison.
        budget_range: what a campaign's budget is clamped to, and what the planner is told it
            may ask for. Set it from the run's size: it decides how many language-model calls
            a run costs, since one campaign is one call.
        token_noise: Gaussian noise on the input a token carries, in units of the problem's
            per-coordinate ``input_scale``. v2's continuous channel adds 0.05 on a unit range,
            about 0.17 of a coordinate's spread. The agent's own model keeps the true input.
        skip_peer_inputs: leave out of the candidate pool anything a neighbour's token shows was
            already run. Without it agents re-run each other's experiments: the exclusion covers
            only their own history and the round's other picks.
        skip_radius: how close a candidate must be to a reported input to be skipped. ``0``
            means the same input; with ``token_noise`` the reported input has moved, so this
            should cover the noise.
        input_hops: how far a reported input travels. ``1`` is v2: an agent hears only its
            in-neighbours' own inputs, so the filter cannot see what anyone further away ran,
            while claims and verdicts are relayed several hops. ``h > 1`` has each token also
            carry the inputs its sender heard at fewer than ``h`` hops in the previous round,
            for the filter only -- not the peer field, trust, novelty or the planner -- so the
            arm changes nothing but the filter's reach. Costs 8 bits a coordinate per relayed
            input; needs ``skip_peer_inputs``.
        soft_radius: a soft ring around the hard filter. A candidate between ``skip_radius``
            and ``soft_radius`` of a reported input keeps only part of its value above the
            pool's lowest: ``v' = v_lo + (1 - K(d)) (v - v_lo)``, with the biweight
            ``K(d) = (1 - (d / soft_radius)^2)^2`` for ``d < soft_radius`` (1 at the report, 0 at
            the radius). Near-repeats of what others ran are discouraged, not forbidden: a
            candidate the rule rates far above the rest can still win. ``0`` turns it off;
            needs ``skip_peer_inputs``, whose hard ``skip_radius`` still removes the reported
            input itself.
        hit_threshold: the score that makes an outcome a hit, for the hit tests and what the
            planner is told. ``None`` leaves hits undefined.
        share_hits: verdicts carry what the campaign found (hits, new hit families, best score)
            as well as whether its own test passed.
        share_detail: widen the campaign channel to prose. Claims and verdicts carry the full
            hypothesis instead of a truncation, the planner is asked for one sentence to its
            neighbours and that sentence travels, and a verdict says which box its hits fell
            in. The arm that asks whether the words are worth the bits; implies ``share_text``.

    Agency (all off by default; with these at their defaults the agent is the one above):

        parts: accept multi-part designs (the version-3 schema): up to ``max_parts`` regions,
            each with a rule and a count, run in order and judged together. Off, a design in
            parts keeps its first part only.
        replan_on_surprise: a surprise closes the open campaign early (judged on its runs so
            far, verdict ``early: "surprise"``), so the planner is asked again once the
            agent's follow-up draws are spent.
        followup: follow-up draws per surprise (0 = off). A surprise opens a ticket: the next
            picks are the nearest unrun candidates to it inside a box of half-width
            ``followup_half_width`` (unit cube), and the campaign waits. At most
            ``max_open_tickets`` tickets draw at once; further surprises queue.
        anchors: a part may name a surprise (``s<k>``); its region is widened to contain it.
        class_claims: a design may claim ``{part: j, class: C}``; code sets its status.
        extra_rules: rule names the problem supplies (``Problem.rules()``), offered to the
            planner besides the built-in five.
        prompt: ``"v3"`` gives the planner the version-3 context (signals, surprises, tickets,
            parts); the planner's own ``prompt`` must say ``"v3"`` too. Earlier versions'
            contexts are unchanged.
        share_signals: tokens carry a result's signal bits (kind, mismatch, surprise; where a
            surprise sat) and revisions of earlier results, so neighbours see surprises.
        events: collect events (:meth:`ADKOHAgent.drain_events`) for a ledger.
    """

    weights: ReasoningWeights = field(default_factory=ReasoningWeights)
    q_bits: int = 8
    memory_rounds: int = 3
    trust: bool = True
    trust_eta: float = 0.1
    warmup_rounds: int = 3
    pool_size: int = 256
    claim_weight: float = 1.0
    novelty_weight: float = 0.0
    share_descriptor: bool = True
    share_family: bool = True
    claims_on: bool = True
    share_text: bool = False
    claim_ttl: int = CLAIM_TTL
    verdict_ttl: int = VERDICT_TTL
    territory: bool = True
    territory_half_width: float = 0.25
    default_budget: int = 5
    budget_range: Tuple[int, int] = BUDGET_RANGE
    force_rule: Optional[str] = None
    token_noise: float = 0.0
    skip_peer_inputs: bool = False
    skip_radius: float = 0.0
    input_hops: int = 1
    soft_radius: float = 0.0
    hit_threshold: Optional[float] = None
    share_hits: bool = False
    share_detail: bool = False
    parts: bool = False
    max_parts: int = 3
    replan_on_surprise: bool = False
    followup: int = 0
    followup_half_width: float = 0.05
    max_open_tickets: int = 2
    anchors: bool = False
    class_claims: bool = False
    extra_rules: Tuple[str, ...] = ()
    prompt: str = "v2"
    share_signals: bool = False
    events: bool = False
    # Text limits, in characters, of what the planner writes: what a campaign stores, and what the
    # detailed channel carries on the wire (claims and verdicts carry up to ``hypothesis_chars`` of
    # the hypothesis, never less than ``DETAIL_CHARS``; the insight line up to ``insight_chars``).
    # The defaults are the limits every earlier arm ran with, unchanged. In snap500 the note was cut
    # in 48 of 49 designs (median 61 % kept) because no prompt stated the limit; the v3 prompt states
    # these, so a model writes to fit instead of being cut, and an arm can raise them.
    note_chars: int = NOTE_CHARS
    hypothesis_chars: int = 300
    rationale_chars: int = 600
    insight_chars: int = 400


def _agency(cfg: AgentConfig) -> bool:
    """Whether any behaviour beyond the legacy agent is on. Events only watch, so not them."""
    return bool(
        cfg.parts
        or cfg.replan_on_surprise
        or cfg.followup > 0
        or cfg.anchors
        or cfg.class_claims
        or cfg.extra_rules
        or cfg.prompt == "v3"
        or cfg.share_signals
    )


#: Candidates a ``pool`` event lists, by the pick's final value.
N_TOP = 8


class ADKOHAgent:
    """One lab. Owns a private GP, a history, a campaign, and a view of its neighbours.

    Args:
        agent_id: APPFL node id, what the transport routes by.
        index: federation index, what memory, trust and claims key on.
        surrogate: this agent's private model.
        problem: shared, read-only; it draws pools and runs experiments.
        cfg: the arm's configuration.
        neighbor_indices: federation indices of in-neighbours, for trust.
        degree: out-degree, which sets the FIFO budget with ``memory_rounds``.
        planner: ``None`` for a no-LLM arm, a campaign planner otherwise.
        centers: the run's territory seeds in the unit cube; ``None`` disables territories.
        descriptor_scale: the run's frozen per-component descriptor scale, for quantization.
    """

    def __init__(
        self,
        agent_id: str,
        index: int,
        surrogate: Surrogate,
        problem: Problem,
        cfg: Optional[AgentConfig] = None,
        *,
        neighbor_indices: Sequence[int] = (),
        degree: int = 1,
        seed: int = 0,
        planner: Optional[Any] = None,
        centers: Optional[np.ndarray] = None,
        descriptor_scale: Optional[Sequence[float]] = None,
    ):
        self.agent_id = str(agent_id)
        self.index = int(index)
        self.surrogate = surrogate
        self.problem = problem
        self.cfg = cfg or AgentConfig()
        if self.cfg.input_hops > 1 and not self.cfg.skip_peer_inputs:
            raise ValueError("input_hops > 1 relays inputs only the peer-input filter reads; "
                             "set skip_peer_inputs too, or the relay costs bits for nothing")
        if self.cfg.soft_radius > 0 and not self.cfg.skip_peer_inputs:
            raise ValueError("soft_radius discounts candidates near reported inputs, which only "
                             "skip_peer_inputs keeps; set it too")
        if self.cfg.prompt not in ("v1", "v2", "v3"):
            raise ValueError(f"prompt must be 'v1', 'v2' or 'v3', got {self.cfg.prompt!r}")
        if self.cfg.followup < 0 or self.cfg.max_parts < 1 or self.cfg.max_open_tickets < 1:
            raise ValueError("followup must be >= 0, and max_parts and max_open_tickets >= 1")
        bad = [r for r in self.cfg.extra_rules if r in RULES]
        if bad:
            raise ValueError(f"extra_rules may not reuse a built-in rule's name: {bad}")
        self.space = problem.region_space
        self.planner = planner
        self.quantizer = Quantizer(self.cfg.q_bits)
        self.memory = TokenMemory(
            self.cfg.memory_rounds * max(int(degree), 1),
            self.cfg.weights.kernel,
            self.cfg.weights.bandwidth,
        )
        self.trust = TrustState(neighbor_indices, eta=self.cfg.trust_eta) if self.cfg.trust else None
        self.claims = ClaimStore(self.space, self.index)
        self.seed = int(seed)
        self.rng = np.random.default_rng([self.seed, self.index, 2025])
        self.token_rng = np.random.default_rng([self.seed, self.index, 2026])
        self.centers = None if centers is None else np.atleast_2d(np.asarray(centers, dtype=float))
        self.descriptor_scale = (
            None if descriptor_scale is None else np.asarray(descriptor_scale, dtype=float)
        )

        self.campaign: Optional[Campaign] = None
        self.campaigns: List[Campaign] = []
        self._pending: Optional[Dict[str, Any]] = None
        self._cands: Optional[Candidates] = None
        self.sigma_ref: Optional[float] = None

        # Own history. ``_u`` is the objective in maximize orientation, as v2 keeps it.
        self._keys: set = set()
        self._keylist: List[int] = []
        self._X: List[np.ndarray] = []
        self._u: List[float] = []
        self._desc: List[np.ndarray] = []
        self._fams: List[str] = []
        # What peers' tokens told me. The FIFO memory bounds how much a peer's *signal* can
        # pull the score -- that is a method choice the v2 findings settled. It must not also
        # decide how much of the team's footprint the novelty rule can see: forgetting that a
        # neighbour visited a point makes "furthest from everything anyone tried" mean
        # "furthest from what I tried recently", and ten agents rediscover the same thing.
        # These accumulate, and cost no communication -- the tokens already arrived.
        self._peer_X: List[np.ndarray] = []
        self._peer_desc: List[np.ndarray] = []
        self._peer_fams: set = set()
        # Inputs relayed from beyond the in-neighbours (``input_hops > 1``), kept apart from
        # ``_peer_X`` so that only the filter reads them; what this agent will relay next
        # round; and every input it has already taken in, so nothing is stored or relayed twice.
        self._relayed_X: List[np.ndarray] = []
        self._relay_out: List[Dict[str, Any]] = []
        self._seen_inputs: set = set()
        self.diag: Dict[str, Any] = {}
        self.records: List[Dict[str, Any]] = []
        #: History entries that came from shared initial data rather than from this run's
        #: budget. Endpoints that count effort must skip them, or every agent holding the same
        #: initial points looks like the federation repeating itself.
        self._n_loaded = 0

        # Agency. Row-aligned with the history above: each result's signals, its region
        # coordinates and the round it ran (None for shared initial points). Nothing here draws
        # a random number or changes a pick unless a config field turns it on.
        self._agency = _agency(self.cfg)
        self._signals: List[Dict[str, Any]] = []
        self._F: List[Optional[np.ndarray]] = []
        self._rounds: List[Optional[int]] = []
        #: Follow-up tickets in the order they opened, and surprises waiting for a slot.
        self.tickets: List[Ticket] = []
        self._queue: List[str] = []
        #: Surprises this agent knows of, its own and its neighbours', ``s<k>`` in arrival order.
        self.surprises: List[Dict[str, Any]] = []
        self._surprise_at: Dict[Tuple[Any, ...], int] = {}
        #: Revisions waiting for the next token, and a verdict line waiting for one.
        self._revised_out: List[Dict[str, Any]] = []
        self._insight_out: Optional[str] = None
        self._rule_fns: Optional[Dict[str, Any]] = None
        self._events: List[Dict[str, Any]] = []
        self._collect: Optional[List[Dict[str, Any]]] = None

    # -- shared initial data ---------------------------------------------------------------

    def load(self, cands: Candidates, outcomes: Sequence[Outcome]) -> None:
        """Take the shared initial points into the GP without spending budget."""
        for i, out in enumerate(outcomes):
            key, X, F, _meta = cands.row(i)
            self._remember(key, X, out, F)
            self.surrogate.update(X, self._u[-1])
        self._n_loaded = len(self._u)

    # -- planning --------------------------------------------------------------------------

    def needs_plan(self, rnd: int) -> bool:
        """Whether the driver should ask the planner for a design before this round."""
        if self.planner is None or rnd < self.cfg.warmup_rounds:
            return False
        if self._drawing_ticket() is not None:
            return False  # follow-up draws first; the planner then sees what they found
        return self.campaign is None or self.campaign.done

    def territory_region(self):
        """The agent's own box, or the whole space when territories are off."""
        if not self.cfg.territory or self.centers is None or self.index >= len(self.centers):
            return self.space.full()
        return self.space.box_around(self.centers[self.index], self.cfg.territory_half_width)

    def plan_context(self, rnd: int) -> Dict[str, Any]:
        """What the planner sees. Its own data, its neighbours' claims, settled verdicts.

        No raw peer outcomes appear here -- an agent cannot see them, which is the constraint
        the whole method is built around, and the prompt must not quietly assume otherwise.
        """
        desc = self.problem.describe()
        order = [int(i) for i in np.argsort(self._u)[::-1][:8]] if self._u else []
        best = [
            {
                "point": self.problem.label(self._keylist[i], None),
                "score": round(float(self._u[i]), 4),
                "family": self._fams[i],
            }
            for i in order
        ]
        ctx = {
            "agent": self.index,
            "round": rnd,
            "problem": desc,
            "dims": desc.get("dims", self.space.describe()),
            "rules": list(hrules.RULES),
            "budget_range": list(self.cfg.budget_range),
            "metrics": ["new_families", "novel_distance", "hull_growth", "max_objective"],
            "n_evaluated": len(self._u),
            "best": best,
            "my_families": sorted({f for f in self._fams if f})[:20],
            "peer_families": sorted(self._peer_fams)[:20],
            "territory": self.territory_region().label(self.space),
            "first_campaign": not self.campaigns,
            "active_claims": self.claims.claim_digest(rnd),
            "verdicts": self.claims.digest(),
            "history": [c.to_dict(self.space) for c in self.campaigns[-3:]],
            # What a planner needs to judge regions by the endpoint rather than by its own
            # test: what each recent campaign found, and the rates its tests should be set
            # against. Prompts that do not use these ignore them.
            "hit_threshold": self.cfg.hit_threshold,
            "claims_on": self.cfg.claims_on,
            "n_families": len({f for f in self._fams if f}),
            "n_peer_families": len(self._peer_fams),
            "base_rates": self._base_rates(),
            "share_detail": self.cfg.share_detail,
            "recent": [
                {**c.to_dict(self.space), "found": self._found(c)} for c in self.campaigns[-6:]
            ],
        }
        if self.cfg.prompt == "v3":
            ctx.update(self._context_v3(rnd, desc))
        return ctx

    def _context_v3(self, rnd: int, desc: Dict[str, Any]) -> Dict[str, Any]:
        """What the version-3 prompt adds: the target and role, signals, surprises, tickets.

        Every number here is computed from this agent's own records and what its tokens
        brought, as for the earlier prompts; nothing a neighbour did not send appears.
        """
        classes = [str(c) for c in (desc.get("classes") or ())]
        return {
            "prompt": "v3",
            "target": str(desc.get("target") or ""),
            "role": _role_of(desc.get("role"), self.index),
            "rules": list(RULES) + list(self.cfg.extra_rules),
            "rule_notes": dict(desc.get("rule_notes") or {}),
            "metrics": list(METRICS_V3),
            "parts_on": self.cfg.parts,
            "max_parts": self.cfg.max_parts if self.cfg.parts else 1,
            "anchors": self.cfg.anchors,
            "class_claims": self.cfg.class_claims,
            "classes": classes,
            "claim_classes": list(hsignals.claim_classes(classes)) if self.cfg.class_claims else [],
            "followup": self.cfg.followup,
            "replan_on_surprise": self.cfg.replan_on_surprise,
            "text_limits": {"hypothesis": self.cfg.hypothesis_chars, "rationale": self.cfg.rationale_chars,
                            "note": self.cfg.note_chars},
            "surprises": [self._surprise_row(s) for s in self.surprises[-12:]],
            "n_surprises": len(self.surprises),
            "tickets": [t.to_dict(self.space) for t in self.tickets[-8:]],
            "queued": len(self._queue),
            "recent_v3": [
                {**c.to_dict(self.space), "signals": self._signal_found(c.runs)}
                for c in self.campaigns[-6:]
            ],
            "signal_rates": self._signal_rates(),
        }

    def _signal_found(self, runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """What a set of runs found in signal terms: settled, pending, kinds, surprises."""
        sigs = [r.get("signals") or {} for r in runs]
        settled = [s for s in sigs if not hsignals.pending(s)]
        kinds: Dict[str, int] = {}
        for s in settled:
            if s.get("kind"):
                kinds[str(s["kind"])] = kinds.get(str(s["kind"]), 0) + 1
        nov = [float(s["novelty"]) for s in settled if s.get("novelty") is not None]
        return {
            "spent": len(runs),
            "settled": len(settled),
            "pending": len(sigs) - len(settled),
            "mismatches": sum(1 for s in settled if s.get("mismatch") is True),
            "kinds": dict(sorted(kinds.items())),
            "surprises": sum(1 for s in settled if hsignals.is_surprise(s)),
            "max_novelty": round(max(nov), 4) if nov else None,
        }

    def _signal_rates(self) -> Optional[Dict[str, Any]]:
        """Per settled experiment, over this agent's judged campaigns: what a test should beat."""
        done = [self._signal_found(c.runs) for c in self.campaigns if c.verdict is not None]
        n = sum(f["spent"] for f in done)
        settled = sum(f["settled"] for f in done)
        if not n:
            return None
        nov = [f["max_novelty"] for f in done if f["max_novelty"] is not None]
        return {
            "campaigns": len(done),
            "experiments": n,
            "settled": settled,
            "pending": n - settled,
            "mismatch_rate": (sum(f["mismatches"] for f in done) / settled) if settled else None,
            "surprise_rate": (sum(f["surprises"] for f in done) / settled) if settled else None,
            "mean_max_novelty": round(float(np.mean(nov)), 4) if nov else None,
        }

    def _surprise_row(self, s: Dict[str, Any]) -> Dict[str, Any]:
        """A surprise as the prompt lists it, with its follow-up's status as it now stands."""
        status = s.get("status")
        if s.get("agent") != self.index:
            status = self._peer_ticket_status(s) or status
        return {
            "id": s["id"],
            "source": "mine" if s.get("agent") == self.index else f"a{s.get('agent')}",
            "round": s.get("round"),
            "recipe": self._recipe(s.get("F")),
            "observed": s.get("observed"),
            "predicted": s.get("predicted"),
            "kind": s.get("kind"),
            "ticket": s.get("ticket"),
            "status": status,
        }

    def _recipe(self, F) -> str:
        """Region coordinates as one line of ``name=value``, the planner's own vocabulary."""
        if F is None:
            return "(unknown)"
        out = []
        for d, v in zip(self.space.dims, np.asarray(F, dtype=float).ravel()):
            if d.kind == KIND_CAT:
                k = int(round(float(v)))
                out.append(f"{d.name}={d.options[k] if 0 <= k < len(d.options) else k}")
            else:
                out.append(f"{d.name}={float(v):.4g}")
        return " ".join(out)

    def _peer_ticket_status(self, s: Dict[str, Any]) -> Optional[str]:
        """A neighbour's surprise's follow-up status, from a ticket verdict of the same origin
        whose box holds it -- the only way it can reach this agent."""
        if s.get("F") is None:
            return None
        F = np.asarray(s["F"], dtype=float)[None, :]
        for v in self.claims.recent_verdicts():
            if v.origin == s.get("agent") and v.metric == "ticket" and v.status:
                if region_of(self.space, v.region).mask(self.space, F)[0]:
                    return v.status
        return None

    def _is_hit(self, score: float) -> bool:
        t = self.cfg.hit_threshold
        if t is None:
            return False
        return score >= t if self.problem.maximize else score <= t

    def _found(self, c: Campaign) -> Dict[str, Any]:
        """What a campaign found, measured the way the endpoint is, against its own baseline."""
        known = c.baseline.get("known_families", set())
        scores = [float(r["score"]) for r in c.runs]
        fams = {r["family"] for r in c.runs if r["family"]}
        hit_fams = {r["family"] for r in c.runs if r["family"] and self._is_hit(float(r["score"]))}
        best = (max(scores) if self.problem.maximize else min(scores)) if scores else None
        return {
            "spent": len(c.runs),
            "hits": int(sum(self._is_hit(s) for s in scores)),
            "new_hit_families": len(hit_fams - known),
            "new_families": len(fams - known),
            "best": None if best is None else round(best, 4),
            "mean": round(float(np.mean(scores)), 4) if scores else None,
        }

    def _base_rates(self) -> Optional[Dict[str, float]]:
        """Per-experiment rates over this agent's finished campaigns: what a test should beat."""
        done = [self._found(c) for c in self.campaigns if c.verdict is not None]
        n = sum(f["spent"] for f in done)
        if not n:
            return None
        return {
            "campaigns": len(done),
            "experiments": n,
            "new_families": sum(f["new_families"] for f in done) / n,
            "hits": sum(f["hits"] for f in done) / n,
            "new_hit_families": sum(f["new_hit_families"] for f in done) / n,
        }

    def set_campaign(self, design: Optional[Dict[str, Any]], rnd: int) -> Campaign:
        """Open a campaign from a design, snapshot its baseline, and announce its claim.

        ``design`` of ``None`` means no planner: one open-ended campaign over the whole space
        with the settled rule, which is exactly ADKO v2 with a pool instead of a grid.

        A design with ``parts`` (the version-3 parser's output) opens a multi-part campaign:
        each part's region is confined to the territory on a first campaign, then widened to
        its anchor when anchors are on, and announced as its own claim. A design without
        ``parts`` is one part, exactly as before.
        """
        if design is not None and isinstance(design.get("parts"), list):
            return self._set_parts(design, rnd)
        if design is None:
            region = self.space.full()
            c = Campaign(
                region=region,
                rule=hrules.valid_rule(self.cfg.force_rule or "adko"),
                budget=10**9,
                test=Test("new_families", ">=", 1.0),
                claim_id=f"a{self.index}c0",
                started=rnd,
                source="none",
                budget_range=self.cfg.budget_range,
            )
        else:
            region = design.get("region") or self.space.full()
            if self.cfg.territory and not self.campaigns:
                region = self._intersect(region, self.territory_region())
            c = Campaign(
                region=region,
                rule=hrules.valid_rule(self.cfg.force_rule or design.get("rule", "adko"), self.cfg.extra_rules),
                budget=int(design.get("budget", self.cfg.default_budget)),
                hypothesis=str(design.get("hypothesis", ""))[: self.cfg.hypothesis_chars],
                rationale=str(design.get("rationale", ""))[: self.cfg.rationale_chars],
                note=str(design.get("note", ""))[: self.cfg.note_chars],
                test=design.get("test") or Test(),
                p_success=float(design.get("p_success", 0.5)),
                claim_id=f"a{self.index}c{len(self.campaigns)}",
                started=rnd,
                source=str(design.get("source", "llm")),
                budget_range=self.cfg.budget_range,
                extra_rules=tuple(self.cfg.extra_rules),
            )
        if design is not None and self.cfg.class_claims:
            c.claim = self._class_claim(design.get("claim"), [c.region], c.claim_id)
        c.baseline = self._snapshot()
        self.campaign = c
        self.campaigns.append(c)
        if self.cfg.claims_on and design is not None:
            self.claims.announce(
                Claim(
                    origin=self.index,
                    claim_id=c.claim_id,
                    region=c.region.pack(self.space),
                    rule=c.rule,
                    hypothesis=self._claim_text(c),
                    round=rnd,
                    expires=rnd + c.budget + CLAIM_SLACK,
                    ttl=self.cfg.claim_ttl,
                    note=c.note if self.cfg.share_detail else "",
                    predicts=c.claim["class"] if c.claim else "",
                )
            )
        self._opened(c, design, rnd)
        return c

    def _set_parts(self, design: Dict[str, Any], rnd: int) -> Campaign:
        """A multi-part campaign from a version-3 design: parts, anchors, a class claim."""
        cid = f"a{self.index}c{len(self.campaigns)}"
        raw = [p for p in design["parts"] if isinstance(p, dict)]
        raw = raw[: self.cfg.max_parts if self.cfg.parts else 1]
        if not raw:  # nothing usable in parts: the design's own region, as one part
            raw = [{"region": design.get("region"), "rule": design.get("rule"),
                    "n": design.get("budget", self.cfg.default_budget)}]
        parts, dropped = [], []
        for j, p in enumerate(raw):
            region = p.get("region") or self.space.full()
            if self.cfg.territory and not self.campaigns:
                region = self._intersect(region, self.territory_region())
            anchor = p.get("anchor") if self.cfg.anchors else None
            if anchor:
                found = self._surprise_by_id(str(anchor))
                if found is None or found.get("F") is None:
                    dropped.append(str(anchor))
                    anchor = None
                else:
                    region = self.space.widen(region, found["F"])
            rule = hrules.valid_rule(
                self.cfg.force_rule or p.get("rule", "explore"), self.cfg.extra_rules, default="explore"
            )
            parts.append(Part(region=region, rule=rule, n=int(p.get("n", 1) or 1),
                              anchor=str(anchor) if anchor else None, claim_id=f"{cid}p{j}"))
        c = Campaign(
            region=parts[0].region,
            rule=parts[0].rule,
            parts=parts,
            hypothesis=str(design.get("hypothesis", ""))[: self.cfg.hypothesis_chars],
            rationale=str(design.get("rationale", ""))[: self.cfg.rationale_chars],
            note=str(design.get("note", ""))[: self.cfg.note_chars],
            test=design.get("test") or Test(),
            p_success=float(design.get("p_success", 0.5)),
            claim_id=cid,
            started=rnd,
            source=str(design.get("source", "llm")),
            budget_range=self.cfg.budget_range,
            extra_rules=tuple(self.cfg.extra_rules),
            anchors_dropped=dropped,
        )
        if self.cfg.class_claims:
            c.claim = self._class_claim(design.get("claim"), [p.region for p in c.parts], cid)
        c.baseline = self._snapshot()
        self.campaign = c
        self.campaigns.append(c)
        if self.cfg.claims_on:
            for j, part in enumerate(c.parts):
                self.claims.announce(
                    Claim(
                        origin=self.index,
                        claim_id=part.claim_id,
                        region=part.region.pack(self.space),
                        rule=part.rule,
                        hypothesis=self._claim_text(c),
                        round=rnd,
                        expires=rnd + c.budget + CLAIM_SLACK,
                        ttl=self.cfg.claim_ttl,
                        note=c.note if self.cfg.share_detail else "",
                        predicts=c.claim["class"] if c.claim and c.claim["part"] == j else "",
                    )
                )
        self._opened(c, design, rnd)
        return c

    def _class_claim(self, raw: Any, regions: Sequence[Any], cid: str) -> Optional[Dict[str, Any]]:
        """A design's class claim, kept only if it names a real part and a known class."""
        if not isinstance(raw, dict):
            return None
        try:
            j = int(raw.get("part", 0))
        except (TypeError, ValueError):
            return None
        cls = str(raw.get("class", ""))
        allowed = hsignals.claim_classes(self.problem.describe().get("classes") or ())
        if not 0 <= j < len(regions) or cls not in allowed:
            return None
        return {"part": j, "class": cls, "claim_id": f"{cid}p{j}"}

    def _opened(self, c: Campaign, design: Optional[Dict[str, Any]], rnd: int) -> None:
        if not self._watching():
            return
        self._emit(
            "campaign_open",
            rnd,
            claim_id=c.claim_id,
            source=c.source,
            region=c.footprint(self.space).label(self.space),
            parts=[p.to_dict(self.space) for p in c.parts],
            claim=dict(c.claim) if c.claim else None,
            budget=c.budget,
            test=c.test.to_dict(),
            p_success=c.p_success,
            hypothesis=c.hypothesis,
            rationale=c.rationale,
            note=c.note,
            anchors_dropped=list(c.anchors_dropped),
            **({"workspace": design["workspace_log"]} if design and "workspace_log" in design else {}),
        )

    def _verdict_chars(self) -> int:
        """How much of the hypothesis a verdict carries, by channel width."""
        if self.cfg.share_detail:
            return max(DETAIL_CHARS, int(self.cfg.hypothesis_chars))
        return VERDICT_CHARS if self.cfg.share_text else 80

    def _claim_text(self, c: Campaign) -> str:
        """The hypothesis a claim carries: nothing, a truncation, or all of it."""
        if self.cfg.share_detail:
            return c.hypothesis[: max(DETAIL_CHARS, int(self.cfg.hypothesis_chars))]
        return c.hypothesis[:CLAIM_CHARS] if self.cfg.share_text else ""

    def _hit_box(self, c: Campaign):
        """The tightest box over this campaign's hits, packed for the wire, or ``None``.

        A box, not the points: region language is what a claim already travels in, so saying
        where the hits were adds no kind of information the channel did not already carry.
        """
        rows = [r for r in c.runs if r.get("F") is not None and self._is_hit(float(r["score"]))]
        if not rows:
            return None
        F = np.atleast_2d(np.asarray([r["F"] for r in rows], dtype=float))
        bounds: Dict[str, Any] = {}
        picks: Dict[str, Any] = {}
        for i, d in enumerate(self.space.dims):
            col = F[:, i]
            if d.kind == KIND_CAT:
                picks[d.name] = sorted({int(round(v)) for v in col})
            else:
                bounds[d.name] = [float(col.min()), float(col.max())]
        return self.space.region(bounds=bounds, picks=picks).pack(self.space)

    def _intersect(self, a, b):
        """The tighter of two regions, coordinate by coordinate."""
        bounds = dict(getattr(b, "bounds", {}))
        for name, (lo, hi) in getattr(a, "bounds", {}).items():
            if name in bounds:
                blo, bhi = bounds[name]
                lo, hi = max(lo, blo), min(hi, bhi)
                if hi <= lo:
                    lo, hi = blo, bhi
            bounds[name] = (lo, hi)
        picks = dict(getattr(b, "picks", {}))
        for name, opts in getattr(a, "picks", {}).items():
            keep = tuple(sorted(set(opts) & set(picks[name]))) if name in picks else tuple(opts)
            picks[name] = keep or picks.get(name, tuple(opts))
        return self.space.region(bounds=bounds, picks=picks)

    # -- one experiment --------------------------------------------------------------------

    def propose(self, rnd: int, taken: Sequence[int] = ()) -> Pick:
        """This round's experiment: the rule's argmax inside the campaign's region.

        ``taken`` are keys already claimed by another agent this round. Nothing is evaluated
        twice in a round, as the reference comparisons require, and the exclusion happens in
        the pool draw so the rule never sees a point it cannot have.

        While a follow-up ticket has draws left, the pick is its next draw instead and the
        campaign waits.
        """
        ticket = self._drawing_ticket()
        if ticket is not None:
            return self._followup_pick(ticket, rnd, taken)
        warm = rnd < self.cfg.warmup_rounds
        if not warm and self.campaign is None:
            self.set_campaign(
                None
                if self.planner is None
                else random_design(self.space, self.rng, budget_range=self.cfg.budget_range),
                rnd,
            )
        c = self.campaign
        region = self.territory_region() if warm or c is None else c.region
        exclude = sorted(self._keys | {int(k) for k in taken})
        cands = self.problem.pool(self.rng, self.cfg.pool_size, exclude, region)
        skipped = 0
        if self.cfg.skip_peer_inputs and (self._peer_X or self._relayed_X):
            cands, skipped = self._skip_reported(cands, exclude, region)
        self._cands = cands
        if len(cands) == 0:
            raise RuntimeError(f"{self.problem.problem_id}: empty candidate pool at round {rnd}")
        mask = region.mask(self.space, cands.F)

        if warm or c is None:
            idx = np.flatnonzero(mask)
            i = int(self.rng.choice(idx if idx.size else np.arange(len(cands))))
            self.diag = {"mode": "warmup", "rule": "random", "pool": len(cands)}
            rule, claim_id = "random", ""
        else:
            i = self._score_pick(cands, mask, rnd, c)
            rule, claim_id = c.rule, c.claim_id
        if self.cfg.skip_peer_inputs:
            self.diag["skipped"] = skipped
        key, X, F, meta = cands.row(i)
        return Pick(agent=self.index, key=key, X=X, F=F, meta=meta, rule=rule, claim_id=claim_id)

    def _followup_pick(self, t: Ticket, rnd: int, taken: Sequence[int]) -> Pick:
        """A ticket's draw: the unrun candidate nearest the surprise, inside the ticket's box.

        Distances are taken in the unit cube of the region coordinates, the units the box is
        drawn in. The pool comes from the problem as any other, so a box the pool cannot fill
        is topped up from anywhere and the nearest candidate still wins.
        """
        exclude = sorted(self._keys | {int(k) for k in taken})
        cands = self.problem.pool(self.rng, self.cfg.pool_size, exclude, t.region)
        skipped = 0
        if self.cfg.skip_peer_inputs and (self._peer_X or self._relayed_X):
            cands, skipped = self._skip_reported(cands, exclude, t.region)
        self._cands = cands
        if len(cands) == 0:
            raise RuntimeError(f"{self.problem.problem_id}: empty candidate pool at round {rnd}")
        mask = t.region.mask(self.space, cands.F)
        origin = self.space.normalize(np.asarray(t.F, dtype=float)[None, :])[0]
        d = np.sqrt(((self.space.normalize(cands.F) - origin) ** 2).sum(axis=1))
        i = hrules.pick(-d, mask)
        self.diag = {
            "mode": "followup",
            "rule": "followup",
            "ticket": t.id,
            "pool": len(cands),
            "in_region": int(np.count_nonzero(mask)),
            "region_empty": not bool(mask.any()),
            "distance": round(float(d[i]), 6),
        }
        if self.cfg.skip_peer_inputs:
            self.diag["skipped"] = skipped
        if self._watching():
            mu, sigma = self.surrogate.predict(cands.X)
            self._pool_event(rnd, "followup", None, cands, -d, mu, sigma, mask, i)
        key, X, F, meta = cands.row(i)
        return Pick(agent=self.index, key=key, X=X, F=F, meta=meta, rule="followup", claim_id=t.id)

    def _skip_reported(self, cands: Candidates, exclude: Sequence[int], region) -> Tuple[Candidates, int]:
        """Drop candidates a neighbour's token shows were already run; top the pool back up.

        Tokens carry inputs, not keys, so a candidate counts as reported when it lies within
        ``skip_radius`` of a reported input (``0``: the same input). Refills come from the same
        region with everything drawn so far excluded, at most three times, so a region that
        neighbours have mined out leaves a smaller pool rather than a stall -- and if nothing at
        all is left, the original draw stands.
        """
        known = np.asarray(self._peer_X + self._relayed_X, dtype=float)
        tol = max(float(self.cfg.skip_radius), 1e-9)
        ok = hrules.nearest_distance(cands.X, known) > tol
        skipped = int(np.count_nonzero(~ok))
        if not skipped:
            return cands, 0
        parts, have = [_take(cands, ok)], int(np.count_nonzero(ok))
        drawn = set(int(k) for k in exclude) | {int(k) for k in cands.key}
        for _ in range(3):
            if have >= self.cfg.pool_size:
                break
            more = self.problem.pool(self.rng, self.cfg.pool_size - have, sorted(drawn), region)
            if len(more) == 0:
                break
            drawn |= {int(k) for k in more.key}
            ok = hrules.nearest_distance(more.X, known) > tol
            skipped += int(np.count_nonzero(~ok))
            parts.append(_take(more, ok))
            have += int(np.count_nonzero(ok))
        out = _join(parts)
        return (out if len(out) else cands), skipped

    def _score_pick(self, cands: Candidates, mask: np.ndarray, rnd: int, c: Campaign) -> int:
        w = self.cfg.weights
        mu, sigma = self.surrogate.predict(cands.X)
        G0, P0 = PeerField(self.memory, self.source_weights())(cands.X)
        cal = Calibrator(sigma, G0, P0, w.eps)
        G, L = cal.apply(G0, P0)
        self.sigma_ref = float(sigma.max())
        nov = hrules.nearest_distance(cands.X, self._known_inputs())
        rule_error = None
        if c.rule in self.cfg.extra_rules:
            v, rule_error = self._problem_rule(c, cands, mask, mu, sigma, rnd, nov)
            if rule_error is not None:
                v = hrules.values("explore", mu, sigma, G, L, nov, beta=w.beta, lam=w.lam, gamma=w.gamma)
        else:
            v = hrules.values(c.rule, mu, sigma, G, L, nov, beta=w.beta, lam=w.lam, gamma=w.gamma)
        pen = self.claims.penalty(cands.F, rnd) if self.cfg.claims_on else None
        v, parts = hrules.adjust(
            v,
            sigma,
            penalty=pen,
            penalty_weight=self.cfg.claim_weight,
            novelty=nov,
            novelty_weight=self.cfg.novelty_weight,
        )
        soft = None
        if self.cfg.soft_radius > 0 and (self._peer_X or self._relayed_X):
            known = np.asarray(self._peer_X + self._relayed_X, dtype=float)
            soft = hrules.biweight(hrules.nearest_distance(cands.X, known), self.cfg.soft_radius)
            hard = hrules.pick(v, mask)
            v = hrules.discount(v, soft)
        i = hrules.pick(v, mask)
        self.diag = {
            "mode": "campaign",
            "rule": c.rule,
            "pool": len(cands),
            "in_region": int(np.count_nonzero(mask)),
            "region_empty": not bool(mask.any()),
            "claim_id": c.claim_id,
            "spent": c.spent,
            "budget": c.budget,
            "steered": float(np.mean(pen)) if pen is not None else 0.0,
            **(
                {"soft": round(float(np.mean(soft)), 6), "soft_moved": bool(i != hard)}
                if soft is not None
                else {}
            ),
            "score_baseline": bool(
                i == hrules.pick(score(mu, sigma, G, L, w.beta, w.lam, w.gamma), None)
            ),
            **{f"term_{k}": round(float(x), 6) for k, x in parts.items()},
            **{k: cal.diagnostics[k] for k in ("k", "U", "floor_active")},
        }
        if c.multipart:
            self.diag["part"] = c.part_index
        if rule_error is not None:
            self.diag["rule_error"] = rule_error
            self._emit("rule_error", rnd, rule=c.rule, claim_id=c.claim_id, error=rule_error)
        if self._watching():
            self._pool_event(rnd, c.rule, c.part_index, cands, v, mu, sigma, mask, i)
        return i

    def _problem_rule(self, c: Campaign, cands: Candidates, mask, mu, sigma, rnd: int, nov):
        """A problem-supplied rule's values, or ``(None, why)`` if it cannot be used this pick.

        Anything a rule can get wrong -- raising, the wrong shape, nothing finite -- costs this
        pick its rule and nothing more: the caller falls back to ``explore`` and records why.
        NaN is read as the worst value.
        """
        if self._rule_fns is None:
            get = getattr(self.problem, "rules", None)
            self._rule_fns = dict(get() or {}) if callable(get) else {}
        fn = self._rule_fns.get(c.rule)
        if fn is None:
            return None, f"the problem supplies no rule {c.rule!r}"
        view = RuleView(
            agent=self.index,
            round=int(rnd),
            rule=c.rule,
            part=c.part_index,
            cands=cands,
            mask=np.asarray(mask, dtype=bool),
            mu=np.asarray(mu, dtype=float),
            sigma=np.asarray(sigma, dtype=float),
            own_X=np.asarray(self._X, dtype=float) if self._X else np.zeros((0, np.shape(cands.X)[1])),
            own_F=_rows(self._F, len(self.space)),
            own_u=np.asarray(self._u, dtype=float),
            own_signals=[dict(s) for s in self._signals],
            own_families=list(self._fams),
            known_inputs=self._known_inputs(),
            seed=int(np.random.SeedSequence([self.seed, self.index, int(rnd), 4049]).generate_state(1)[0]),
            region=c.region,
            own_descriptors=_rows([z if z.size else None for z in self._desc],
                                  int(self.problem.descriptor_dim() or 0)),
            known_descriptors=self._descriptors(),
        )
        try:
            v = np.asarray(fn(view), dtype=float)
        except Exception as e:  # noqa: BLE001 -- the problem's code, not ours: any failure is one pick's
            return None, f"{type(e).__name__}: {e}"[:200]
        if v.shape != (len(cands),):
            return None, f"returned shape {v.shape}, expected ({len(cands)},)"
        v = np.where(np.isnan(v), -np.inf, v)
        if not np.isfinite(v).any():
            return None, "returned no finite value"
        return v, None

    def _pool_event(self, rnd, rule, part, cands, v, mu, sigma, mask, i) -> None:
        """The pick's top candidates by final value, and the one chosen, for the ledger."""
        v = np.asarray(v, dtype=float)
        order = np.argsort(-np.where(np.isfinite(v), v, -np.inf), kind="stable")[:N_TOP]

        def row(j):
            return {
                "key": int(cands.key[j]),
                "value": _finite(v[j]),
                "mu": _finite(mu[j]),
                "sigma": _finite(sigma[j]),
                "in_region": bool(mask[j]),
            }

        self._emit("pool", rnd, rule=rule, part=part, pool=len(cands),
                   top=[row(int(j)) for j in order], chosen=row(int(i)))

    def complete(self, pick: Pick, out: Outcome, rnd: int) -> HToken:
        """Record the result, close the campaign if this was its last experiment, emit a token.

        A follow-up draw (rule ``followup``) counts for its ticket, not for the campaign, which
        waits for it. With agency on, a surprise then opens a ticket and may close the
        campaign early, and tickets whose draws have all settled are judged.
        """
        sig = dict(getattr(out, "signals", None) or {})
        ticket = self._ticket(pick.claim_id) if pick.rule == "followup" else None
        c = self.campaign if ticket is None else None
        last = c.last if c is not None else False
        self._remember(pick.key, pick.X, out, pick.F, rnd)
        self.surrogate.update(pick.X, self._u[-1])
        insight = None
        if ticket is not None:
            ticket.spent += 1
            ticket.runs.append({"round": rnd, "key": int(pick.key), **({"signals": sig} if sig else {})})
        elif c is not None:
            c.add_run(
                {
                    "round": rnd,
                    "key": int(pick.key),
                    "point": self.problem.label(int(pick.key), pick.meta),
                    "score": round(float(out.score), 6),
                    "family": out.family,
                    # Only the detailed channel needs where a hit sat, and only in region
                    # coordinates; keeping it off by default leaves every other arm's records
                    # byte for byte what they were.
                    **(
                        {"F": [float(v) for v in np.ravel(pick.F)]}
                        if self.cfg.share_detail
                        else {}
                    ),
                    **({"signals": dict(sig)} if sig else {}),
                }
            )
            if last or c.done:
                # A campaign already closed (early, by a surprise) is not judged twice.
                if not (self._agency and c.verdict is not None):
                    insight = self._close(c, rnd)
        if self._agency:
            self._noticed(len(self._u) - 1, rnd, ticket)
            self._open_tickets(rnd)
            self._check_tickets(rnd)
            if insight is None and self._insight_out:
                insight = self._insight_out
            self._insight_out = None
        claims_out, verdicts_out = self.claims.outbox() if self.cfg.claims_on else ([], [])
        x_shared = self.problem.privatize(pick.X, self.token_rng)
        if self.cfg.token_noise > 0:
            x_shared = self._jitter(x_shared)
        relayed: List[Dict[str, Any]] = []
        if self.cfg.input_hops > 1:
            self._seen_inputs.add(_input_key(x_shared))  # my own run, if it comes back to me
            relayed, self._relay_out = self._relay_out, []
        bits: Dict[str, Any] = {}
        revised: List[Dict[str, Any]] = []
        if self.cfg.share_signals:
            bits = self._bits(sig, pick.F)
            revised, self._revised_out = self._revised_out, []
        tok = encode(
            agent_id=self.agent_id,
            source=self.index,
            round=rnd,
            x_shared=x_shared,
            u=self._u[-1],
            history=self._u,
            quantizer=self.quantizer,
            new_best=len(self._u) == 1 or self._u[-1] > max(self._u[:-1]),
            insight=insight,
            descriptor=self._shared_descriptor(out.descriptor),
            family=out.family if self.cfg.share_family else "",
            claims=claims_out,
            verdicts=verdicts_out,
            relayed=relayed,
            signals=bits,
            revised=revised,
        )
        self.records.append(
            {
                "round": rnd,
                "agent": self.index,
                "key": int(pick.key),
                "point": self.problem.label(int(pick.key), pick.meta),
                "score": float(out.score),
                "family": out.family,
                "descriptor": [float(z) for z in np.asarray(out.descriptor, dtype=float).ravel()],
                "rule": pick.rule,
                "claim_id": pick.claim_id,
                "bits": tok.payload_bits(),
                "trust": None if self.trust is None else self.trust.mean(),
                **{f"d_{k}": x for k, x in self.diag.items()},
                **({"extras": out.extras} if out.extras else {}),
                **({"signals": sig} if sig else {}),
            }
        )
        if self._watching():
            self._emit(
                "token_sent",
                rnd,
                key=int(pick.key),
                bits=tok.payload_bits(),
                family=tok.family,
                kind=sig.get("kind"),
                mismatch=sig.get("mismatch"),
                surprise=hsignals.is_surprise(sig),
                signals=dict(tok.signals),
                claims=[m.claim_id for m in claims_out],
                verdicts=[m.claim_id for m in verdicts_out],
                relayed=len(relayed),
                revised=len(revised),
            )
        return tok

    def _bits(self, sig: Dict[str, Any], F) -> Dict[str, Any]:
        """A result's signal bits for the wire; a surprise also says where it sat."""
        bits = hsignals.on_token(sig)
        if hsignals.is_surprise(sig) and F is not None:
            bits["at"] = [round(float(v), 4) for v in np.ravel(F)]
        return bits

    def _close(self, c: Campaign, rnd: int, early: Optional[str] = None) -> str:
        """Judge a campaign, send its verdict (and its class claim's status), tell the planner.

        Returns the one-line insight the token carries.
        """
        v = c.judge(self._evidence(c), early=early)
        insight = self._insight(c, v)
        if self.cfg.claims_on:
            found = self._found(c) if self.cfg.share_hits else None
            verdict = Verdict(
                origin=self.index,
                claim_id=c.claim_id,
                region=c.footprint(self.space).pack(self.space),
                metric=v["metric"],
                value=v["value"],
                op=v["op"],
                threshold=v["threshold"],
                success=v["success"],
                hypothesis=c.hypothesis[: self._verdict_chars()],
                round=rnd,
                ttl=self.cfg.verdict_ttl,
                note=c.note if self.cfg.share_detail else "",
                hit_box=self._hit_box(c) if self.cfg.share_detail else None,
                **(
                    {}
                    if found is None
                    else {
                        "spent": found["spent"],
                        "hits": found["hits"],
                        "hit_families": found["new_hit_families"],
                        "best": found["best"],
                    }
                ),
            )
            self.claims.settle(verdict)
            if c.claim is not None:
                self.claims.settle(self._claim_verdict(c, rnd))
        if self.planner is not None:
            getattr(self.planner, "record_outcome", lambda *a, **k: None)(c, v)
        if self._watching():
            self._emit("campaign_verdict", rnd, claim_id=c.claim_id, verdict=dict(v),
                       brier=c.brier(), early=c.early)
            if c.claim is not None:
                self._emit("claim_status", rnd, campaign=c.claim_id, **dict(c.claim))
        return insight

    def _claim_verdict(self, c: Campaign, rnd: int) -> Verdict:
        """A class claim's status as a verdict on the claim of its part."""
        cl = c.claim
        share = cl["shows"] / cl["settled"] if cl.get("settled") else None
        return Verdict(
            origin=self.index,
            claim_id=cl["claim_id"],
            region=c.parts[cl["part"]].region.pack(self.space),
            metric="class_claim",
            value=None if share is None else round(float(share), 6),
            op=">=",
            threshold=0.5,
            success=cl["status"] == "supported",
            hypothesis=c.hypothesis[: self._verdict_chars()],
            round=rnd,
            ttl=self.cfg.verdict_ttl,
            predicts=cl["class"],
            status=cl["status"],
        )

    # -- surprises, tickets, late labels ---------------------------------------------------

    def _noticed(self, i: int, rnd: int, ticket: Optional[Ticket]) -> None:
        """What a surprise sets off, from ``complete`` or ``revise`` alike.

        It is listed (once per result) for the planner. A ticket's own draw counts for that
        ticket and sets off nothing more, or a region full of surprises would follow itself up
        forever; any other surprise queues a ticket (``followup``) and closes the open campaign
        early (``replan_on_surprise``).
        """
        if not hsignals.is_surprise(self._signals[i]):
            return
        s, new = self._own_surprise(i)
        if not new:
            return
        if ticket is not None:
            s["ticket"], s["status"] = ticket.id, "follow-up draw"
            return
        if self.cfg.followup > 0 and s.get("F") is not None:
            self._queue.append(s["id"])
            s["status"] = "queued"
        if self.cfg.replan_on_surprise:
            self._close_early(rnd)

    def _close_early(self, rnd: int) -> None:
        """Close the open campaign on its runs so far; the planner is asked again next."""
        c = self.campaign
        if c is None or c.done or c.verdict is not None or c.source == "none":
            return
        self._insight_out = self._close(c, rnd, early="surprise")

    def _own_surprise(self, i: int) -> Tuple[Dict[str, Any], bool]:
        key = ("own", int(self._keylist[i]))
        if key in self._surprise_at:
            return self.surprises[self._surprise_at[key]], False
        sig = self._signals[i]
        F = self._F[i]
        s = {
            "id": f"s{len(self.surprises)}",
            "agent": self.index,
            "round": self._rounds[i],
            "key": int(self._keylist[i]),
            "F": None if F is None else [float(v) for v in F],
            "observed": sig.get("observed"),
            "predicted": sig.get("predicted"),
            "kind": sig.get("kind"),
            "novelty": sig.get("novelty"),
            "ticket": None,
            "status": None,
        }
        self._surprise_at[key] = len(self.surprises)
        self.surprises.append(s)
        return s, True

    def _peer_surprise(self, source: int, rnd: int, bits: Dict[str, Any]) -> None:
        """A neighbour's surprise, from its token or a revision it sent: listed, never followed up."""
        key = ("peer", int(source), int(rnd))
        if key in self._surprise_at:
            return
        at = bits.get("at")
        self._surprise_at[key] = len(self.surprises)
        self.surprises.append(
            {
                "id": f"s{len(self.surprises)}",
                "agent": int(source),
                "round": int(rnd),
                "key": None,
                "F": None if at is None else [float(v) for v in at],
                "observed": None,
                "predicted": None,
                "kind": bits.get("kind"),
                "novelty": None,
                "ticket": None,
                "status": None,
            }
        )

    def _surprise_by_id(self, sid: str) -> Optional[Dict[str, Any]]:
        for s in self.surprises:
            if s["id"] == sid:
                return s
        return None

    def _drawing_ticket(self) -> Optional[Ticket]:
        """The first ticket with draws left (first in, first out), if any."""
        for t in self.tickets:
            if t.drawing:
                return t
        return None

    def _ticket(self, tid: str) -> Optional[Ticket]:
        for t in self.tickets:
            if t.id == tid:
                return t
        return None

    def _open_tickets(self, rnd: int) -> None:
        """Give queued surprises a ticket while fewer than ``max_open_tickets`` are drawing.

        A ticket whose draws are spent but not yet settled does not hold a slot: a label that
        never settles would otherwise block every later surprise.
        """
        while self._queue and sum(1 for t in self.tickets if t.drawing) < self.cfg.max_open_tickets:
            s = self._surprise_by_id(self._queue.pop(0))
            if s is None or s.get("F") is None:
                continue
            centre = self.space.normalize(np.asarray(s["F"], dtype=float)[None, :])[0]
            t = Ticket(
                id=f"a{self.index}t{len(self.tickets)}",
                surprise=s["id"],
                origin_key=int(s["key"]),
                F=list(s["F"]),
                kind=str(s.get("kind") or ""),
                draws=int(self.cfg.followup),
                region=self.space.box_around(centre, self.cfg.followup_half_width),
                opened=rnd,
            )
            self.tickets.append(t)
            s["ticket"], s["status"] = t.id, "open"
            self._emit("ticket_open", rnd, ticket=t.id, surprise=t.surprise, origin_key=t.origin_key,
                       recipe=self._recipe(t.F), F=list(t.F), kind=t.kind, draws=t.draws,
                       region=t.region.label(self.space))

    def _check_tickets(self, rnd: int, final: bool = False) -> None:
        """Close every ticket that can be closed; its status travels as a verdict."""
        for t in self.tickets:
            if t.status is not None:
                continue
            st = t.outcome(final)
            if st is None:
                continue
            t.status, t.closed = st, int(rnd)
            s = self._surprise_by_id(t.surprise)
            if s is not None:
                s["status"] = st
            if self.cfg.claims_on:
                self.claims.settle(
                    Verdict(
                        origin=self.index,
                        claim_id=t.id,
                        region=t.region.pack(self.space),
                        metric="ticket",
                        value=float(t.reproduced()),
                        op=">=",
                        threshold=1.0,
                        success=st == "reproduced",
                        hypothesis=f"follow-up of {t.surprise} ({t.kind})"[: self._verdict_chars()],
                        round=int(rnd),
                        ttl=self.cfg.verdict_ttl,
                        predicts=t.kind,
                        status=st,
                    )
                )
            self._emit("ticket_close", rnd, **t.to_dict(self.space),
                       keys=[int(r["key"]) for r in t.runs], final=bool(final))

    def revise(self, key: int, outcome: Outcome, rnd: int) -> List[Dict[str, Any]]:
        """A pending result of this agent settled: replace its score, family, descriptor, signals.

        The GP is refit on the corrected history, the run entry of whichever campaign (or
        ticket) holds the point is updated, and a revised surprise sets off what it would have
        in :meth:`complete` (a ticket, an early close). A campaign already judged keeps its
        verdict -- it was judged on what was known then -- and tokens already sent are not
        recalled; with ``share_signals`` the next token carries the revision. Call it between
        rounds, before planning (see the module notes); the same revisions at the same rounds
        give the same picks.

        Returns this call's events (``label_revision`` first), whether or not
        ``AgentConfig.events`` is on; with it on they are also in :meth:`drain_events`.
        """
        self._collect = []
        try:
            i = self._index_of(key)
            if i is None:
                raise KeyError(f"agent {self.index} ran no point with key {key}")
            sig = dict(getattr(outcome, "signals", None) or {})
            old = {"score": self._score_at(i), "family": self._fams[i], "signals": dict(self._signals[i])}
            self._u[i] = float(outcome.score) if self.problem.maximize else -float(outcome.score)
            self._fams[i] = str(outcome.family or "")
            z = np.asarray(outcome.descriptor, dtype=float).ravel()
            if z.size:
                self._desc[i] = z
            self._signals[i] = sig
            refit = getattr(self.surrogate, "refit", None)
            if callable(refit):
                refit(self._X, self._u)
            holder, judged, ticket = None, None, None
            for c in self.campaigns:
                for r in c.runs:
                    if int(r["key"]) == int(key):
                        r["score"] = round(float(outcome.score), 6)
                        r["family"] = outcome.family
                        _set_signals(r, sig)
                        holder, judged = c.claim_id, c.verdict is not None
            for t in self.tickets:
                for r in t.runs:
                    if int(r["key"]) == int(key):
                        _set_signals(r, sig)
                        ticket = t
            self._emit(
                "label_revision",
                rnd,
                key=int(key),
                ran=self._rounds[i],
                old=old,
                new={"score": float(outcome.score), "family": self._fams[i], "signals": dict(sig)},
                campaign=holder,
                judged=judged,
                ticket=None if ticket is None else ticket.id,
                surprise=hsignals.is_surprise(sig),
                refit=callable(refit),
            )
            if self.cfg.share_signals:
                fam = self._fams[i] if self.cfg.share_family else ""
                self._revised_out.append(
                    {"r": self._rounds[i], **({"family": fam} if fam else {}), **self._bits(sig, self._F[i])}
                )
            self._noticed(i, rnd, ticket)
            self._open_tickets(rnd)
            self._check_tickets(rnd)
            return self._collect
        finally:
            self._collect = None

    def finish(self, rnd: int) -> List[Dict[str, Any]]:
        """The run is over: close every ticket still open on what is known now.

        Returns this call's events (``ticket_close``), as :meth:`revise` does.
        """
        self._collect = []
        try:
            self._check_tickets(rnd, final=True)
            return self._collect
        finally:
            self._collect = None

    def drain_events(self) -> List[Dict[str, Any]]:
        """Every event since the last call, oldest first, and clear them.

        Each is ``{"type", "agent", "round", ...}``. Collected only with
        ``AgentConfig.events``; otherwise this is always empty and nothing was built.
        """
        out, self._events = self._events, []
        return out

    def _watching(self) -> bool:
        return self.cfg.events or self._collect is not None

    def _emit(self, _type: str, _round: int, **fields) -> None:
        if not self._watching():
            return
        ev = {"type": _type, "agent": self.index, "round": int(_round), **fields}
        if self.cfg.events:
            self._events.append(ev)
        if self._collect is not None:
            self._collect.append(ev)

    def _index_of(self, key: int) -> Optional[int]:
        """Where an own result sits in the history (shared initial points are not this agent's)."""
        for i in range(len(self._keylist) - 1, self._n_loaded - 1, -1):
            if self._keylist[i] == int(key):
                return i
        return None

    def _score_at(self, i: int) -> float:
        return float(self._u[i]) if self.problem.maximize else -float(self._u[i])

    def _insight(self, c: Campaign, v: Dict[str, Any]) -> str:
        """The one line that leaves with the campaign's last result: claim, reason, outcome."""
        val = "n/a" if v["value"] is None else f"{v['value']:g}"
        return (
            f"{c.hypothesis or '(no hypothesis)'} | why: {c.rationale or '(none)'} | "
            f"{'held' if v['success'] else 'failed'} ({v['metric']}={val} vs {v['op']} {v['threshold']:g})"
        )[: self.cfg.insight_chars]

    # -- bookkeeping -----------------------------------------------------------------------

    def _remember(self, key: int, X: np.ndarray, out: Outcome, F=None, rnd: Optional[int] = None) -> None:
        self._keys.add(int(key))
        self._keylist.append(int(key))
        self._X.append(np.asarray(X, dtype=float))
        self._u.append(float(out.score) if self.problem.maximize else -float(out.score))
        self._desc.append(np.asarray(out.descriptor, dtype=float).ravel())
        self._fams.append(str(out.family or ""))
        self._signals.append(dict(getattr(out, "signals", None) or {}))
        self._F.append(None if F is None else np.asarray(F, dtype=float).ravel())
        self._rounds.append(None if rnd is None else int(rnd))

    @property
    def own_picks(self) -> List[np.ndarray]:
        """The embeddings this agent actually spent budget on."""
        return self._X[self._n_loaded :]

    def _known_inputs(self) -> np.ndarray:
        """Everything anyone has run, as far as I know: my points plus every peer input I ever
        received -- not just the ones still in the FIFO."""
        rows = list(self._X) + list(self._peer_X)
        if not rows:
            return np.zeros((0, 1))
        d = max(len(x) for x in rows)
        return np.asarray([x for x in rows if len(x) == d], dtype=float)

    def _descriptors(self, include_peers: bool = True) -> np.ndarray:
        rows = [z for z in self._desc if z.size]
        if include_peers:
            rows += [z for z in self._peer_desc if z.size]
        if not rows:
            return np.zeros((0, 0))
        d = max(len(z) for z in rows)
        return np.asarray([z for z in rows if len(z) == d], dtype=float)

    def _snapshot(self) -> Dict[str, Any]:
        Z = self._descriptors()
        return {
            "known_descriptors": Z,
            "known_families": {f for f in self._fams if f} | set(self._peer_fams),
            "hull": hmetrics.hull_volume(Z),
            "n": len(self._u),
        }

    def _evidence(self, c: Campaign) -> Evidence:
        """The campaign's own runs against the snapshot taken when it opened."""
        if self._agency:
            return self._evidence_keyed(c)
        n = len(c.runs)
        mine = self._descriptors(include_peers=False)
        Z = mine[-n:] if n and len(mine) >= n else np.zeros((0, 0))
        base = c.baseline.get("known_descriptors", np.zeros((0, 0)))
        after = np.vstack([base, Z]) if base.size and Z.size and base.shape[1] == Z.shape[1] else (
            Z if Z.size else base
        )
        return Evidence(
            scores=np.asarray([r["score"] for r in c.runs], dtype=float),
            descriptors=Z,
            families=[r["family"] for r in c.runs],
            known_descriptors=base,
            known_families=set(c.baseline.get("known_families", set())),
            hull_before=float(c.baseline.get("hull", 0.0)),
            hull_after=hmetrics.hull_volume(after),
            hit_threshold=self.cfg.hit_threshold,
            maximize=self.problem.maximize,
        )

    def _evidence_keyed(self, c: Campaign) -> Evidence:
        """The campaign's runs found by key, its pending ones kept out of every count.

        With follow-up draws between a campaign's runs and labels revised after the fact, its
        runs are no longer the last ``n`` of the history, so descriptors are looked up by key.
        (The legacy path, :meth:`_evidence`, keeps the tail it always used.)
        """
        at = {k: i for i, k in enumerate(self._keylist)}
        sigs = [dict(r.get("signals") or {}) for r in c.runs]
        keep = [j for j, sg in enumerate(sigs) if not hsignals.pending(sg)]
        rows = [self._desc[at[int(c.runs[j]["key"])]] for j in keep if int(c.runs[j]["key"]) in at]
        rows = [z for z in rows if z.size]
        if rows:
            d = max(len(z) for z in rows)
            Z = np.asarray([z for z in rows if len(z) == d], dtype=float)
        else:
            Z = np.zeros((0, 0))
        base = c.baseline.get("known_descriptors", np.zeros((0, 0)))
        after = np.vstack([base, Z]) if base.size and Z.size and base.shape[1] == Z.shape[1] else (
            Z if Z.size else base
        )
        return Evidence(
            scores=np.asarray([c.runs[j]["score"] for j in keep], dtype=float),
            descriptors=Z,
            families=[c.runs[j]["family"] for j in keep],
            known_descriptors=base,
            known_families=set(c.baseline.get("known_families", set())),
            hull_before=float(c.baseline.get("hull", 0.0)),
            hull_after=hmetrics.hull_volume(after),
            hit_threshold=self.cfg.hit_threshold,
            maximize=self.problem.maximize,
            signals=sigs,
            parts=[int(r.get("part", 0)) for r in c.runs],
        )

    def _jitter(self, x) -> np.ndarray:
        """v2's continuous input channel: Gaussian noise on the shared input, per coordinate."""
        x = np.asarray(x, dtype=float)
        scale = self.problem.input_scale()
        s = np.ones(x.shape) if scale is None else np.broadcast_to(np.asarray(scale, dtype=float), x.shape)
        return x + self.token_rng.normal(0.0, 1.0, size=x.shape) * (self.cfg.token_noise * s)

    def _shared_descriptor(self, z: np.ndarray) -> List[float]:
        if not self.cfg.share_descriptor or self.descriptor_scale is None:
            return []
        return quantize_descriptor(np.asarray(z, dtype=float).ravel(), self.descriptor_scale)

    # -- neighbours ------------------------------------------------------------------------

    def observe(self, tokens: Sequence[HToken]) -> None:
        """Ingest neighbour tokens: trust, memory, footprints, claims and verdicts."""
        peers = sorted(
            [t for t in tokens if t.source != self.index],
            key=lambda t: (t.round, t.source),
        )
        if not peers:
            return
        if self.trust is not None:
            for t in peers:
                self.trust.ensure(t.source)
            if self.surrogate.fitted:
                self._update_trust(peers)
        self.memory.add(peers)
        self.memory.prune()
        for t in peers:
            self._peer_X.append(np.asarray(t.x, dtype=float))
            if self.cfg.input_hops > 1:
                self._take_relay(t)
            z = np.asarray(getattr(t, "descriptor", []), dtype=float)
            if z.size and self.descriptor_scale is not None and z.size == len(self.descriptor_scale):
                self._peer_desc.append(z * self.descriptor_scale)
            fam = getattr(t, "family", "")
            if fam:
                self._peer_fams.add(fam)
        if self._watching():
            self._received(peers)
        elif self.cfg.claims_on:
            self.claims.ingest(
                [c for t in peers for c in getattr(t, "claims", [])],
                [v for t in peers for v in getattr(t, "verdicts", [])],
            )
        for t in peers:
            self._take_signals(t)

    def _received(self, peers: Sequence[HToken]) -> None:
        """Ingest token by token, so each ``token_received`` event can say what it brought.

        The same claims and verdicts in the same order as one batched ingest: claims and
        verdicts are kept apart, so interleaving them by token changes nothing.
        """
        for t in peers:
            tot = self.claims.totals
            n0 = (tot["claims_in"], tot["claims_new"], tot["verdicts_in"], tot["verdicts_new"])
            claims, verdicts = list(getattr(t, "claims", [])), list(getattr(t, "verdicts", []))
            if self.cfg.claims_on:
                self.claims.ingest(claims, verdicts)
            n_in = tot["claims_in"] - n0[0] + tot["verdicts_in"] - n0[2]
            n_new = tot["claims_new"] - n0[1] + tot["verdicts_new"] - n0[3]
            self._emit(
                "token_received",
                t.round,
                sender=int(t.source),
                family=getattr(t, "family", ""),
                signals=dict(getattr(t, "signals", None) or {}),
                claims=[m.claim_id for m in claims],
                verdicts=[m.claim_id for m in verdicts],
                accepted=int(n_new),
                skipped=int(n_in - n_new) if self.cfg.claims_on else len(claims) + len(verdicts),
                revised=len(getattr(t, "revised", None) or []),
            )

    def _take_signals(self, t: HToken) -> None:
        """A neighbour's signal bits: its surprises join the list, its revised families the set."""
        bits = getattr(t, "signals", None) or {}
        if bits.get("surprise") is True:
            self._peer_surprise(t.source, t.round, bits)
        for r in getattr(t, "revised", None) or []:
            if r.get("family"):
                self._peer_fams.add(str(r["family"]))
            if r.get("surprise") is True and r.get("r") is not None:
                self._peer_surprise(t.source, int(r["r"]), r)

    def _take_relay(self, t: HToken) -> None:
        """Note a neighbour's input and the inputs it relayed; queue what may travel further."""
        hops = self.cfg.input_hops
        direct = _input_key(t.x)
        if direct not in self._seen_inputs:
            self._seen_inputs.add(direct)
            self._relay_out.append({"h": 1, "x": list(t.x)})
        for r in getattr(t, "relayed", []):
            k = _input_key(r["x"])
            if k in self._seen_inputs:
                continue
            self._seen_inputs.add(k)
            self._relayed_X.append(np.asarray(r["x"], dtype=float))
            if int(r["h"]) + 1 < hops:
                self._relay_out.append({"h": int(r["h"]) + 1, "x": list(r["x"])})

    def _update_trust(self, tokens: Sequence[HToken]) -> None:
        mu, sd = self.surrogate.predict(np.asarray([t.x for t in tokens], dtype=float))
        qhat = [
            score_against(float(m), self._u, self.quantizer, maximize=True)
            for m in self.surrogate.raw_mean(mu)
        ]
        rel = relative_sigma(sd, self.sigma_ref)
        by_src: Dict[int, List[float]] = defaultdict(list)
        for t, qh, s in zip(tokens, qhat, rel):
            by_src[t.source].append(agreement(t.q, qh, s))
        for j, a in by_src.items():
            self.trust.update(j, a)

    def source_weights(self) -> Optional[Dict[int, float]]:
        return None if self.trust is None else self.trust.weights()

    # -- read-outs -------------------------------------------------------------------------

    def best_so_far(self) -> Optional[Tuple[int, float]]:
        if not self._u:
            return None
        i = int(np.argmax(self._u))
        return (self._keylist[i], float(self._u[i]))

    def trace(self) -> Dict[str, Any]:
        """Everything the run record keeps about this agent."""
        judged = [c for c in self.campaigns if c.verdict]
        return {
            "agent": self.index,
            "agent_id": self.agent_id,
            "n_evaluated": len(self._u),
            "best": max(self._u) if self._u else None,
            "families": sorted({f for f in self._fams if f}),
            "peer_families": sorted(self._peer_fams),
            "campaigns": [c.to_dict(self.space) for c in self.campaigns],
            "calibration": hmetrics.calibration(
                [c.p_success for c in judged], [bool(c.verdict["success"]) for c in judged]
            ),
            "claims": dict(self.claims.totals),
            "trust": None if self.trust is None else self.trust.snapshot(),
            "records": self.records,
            **({"relayed_inputs": len(self._relayed_X)} if self.cfg.input_hops > 1 else {}),
            **(
                {
                    "tickets": [t.to_dict(self.space) for t in self.tickets],
                    "surprises": [self._surprise_row(s) for s in self.surprises],
                }
                if self._agency
                else {}
            ),
        }


def _role_of(role: Any, index: int) -> str:
    """An agent's role from ``describe()["role"]``: one text, a sequence, or a mapping."""
    if role is None:
        return ""
    if isinstance(role, str):
        return role
    if isinstance(role, dict):
        return str(role.get(index, role.get(str(index), "")))
    seq = list(role)
    return str(seq[index % len(seq)]) if seq else ""


def _rows(rows: Sequence[Optional[np.ndarray]], width: int) -> np.ndarray:
    """Row-aligned vectors as one array; a missing row is NaN, so alignment is kept."""
    if not rows:
        return np.zeros((0, width))
    w = max([len(r) for r in rows if r is not None] or [width])
    return np.asarray([np.full(w, np.nan) if r is None else r for r in rows], dtype=float)


def _finite(x) -> Optional[float]:
    """A float for a JSON record; ``None`` for inf or NaN, which strict JSON cannot hold."""
    x = float(x)
    return round(x, 6) if np.isfinite(x) else None


def _set_signals(run: Dict[str, Any], sig: Dict[str, Any]) -> None:
    if sig:
        run["signals"] = dict(sig)
    else:
        run.pop("signals", None)


def _input_key(x) -> Tuple[float, ...]:
    """An input as a hashable value. Relays copy the reported coordinates exactly, so two copies
    of one run compare equal and two different runs (noise included) do not."""
    return tuple(float(v) for v in np.asarray(x, dtype=float).ravel())


def _take(c: Candidates, mask: np.ndarray) -> Candidates:
    """The rows of a pool where ``mask`` holds."""
    idx = np.flatnonzero(mask)
    meta = list(c.meta)
    return Candidates(
        key=np.asarray(c.key)[idx],
        X=np.asarray(c.X)[idx],
        F=np.asarray(c.F)[idx],
        meta=[meta[i] for i in idx] if meta else [],
    )


def _join(parts: Sequence[Candidates]) -> Candidates:
    """Several pools as one, in order."""
    return Candidates(
        key=np.concatenate([np.asarray(p.key) for p in parts]),
        X=np.vstack([np.asarray(p.X) for p in parts]),
        F=np.vstack([np.asarray(p.F) for p in parts]),
        meta=[m for p in parts for m in p.meta],
    )
