"""The signal contract: what the application says about a result beyond its score.

A score is one number, and the snapshot read-out showed it was the wrong one to steer by: the
states worth finding sat inside every method's hull, and the campaign that found the first of them
was written off because its test counted families. What a planner needs is what the read-out
looks at -- whether the observed class differs from the one predicted for the recipe, of which
kind, how novel the state is, and whether the label is final yet. The application computes that
per result as a plain dict (``Outcome.signals``); the agent treats it as opaque except for the
keys below, so the same machinery runs on a problem that has no classes at all (every dict
empty, every new path a no-op).

Keys (every value may be ``None`` when unknown)::

    settled           the label is final; False means pending (it may be revised later)
    stable            the coarse class held over the last snapshots
    observed          the observed class
    predicted         the class predicted for the recipe
    observed_coarse   coarse(observed)
    predicted_coarse  coarse(predicted)
    mismatch          the coarse classes differ (only when settled and stable)
    kind              one of KINDS, when there is a mismatch or the state is outside the known set
    outside_nine      the state is not one of the known classes
    novelty           distance to the nearest known class
    surprise          settled and stable and kind in SURPRISE_KINDS

The functions here are the only places that read those keys, so a rule about what counts as
pending or as a surprise is written once.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: Ways an observed state can differ from its prediction, as the read-out names them.
KINDS = ("wetting_absent", "wetting_present", "over_demixed", "under_demixed", "outside_nine")
#: The kinds worth a follow-up. ``under_demixed`` is the sequential demixer doing what it does.
SURPRISE_KINDS = ("wetting_absent", "wetting_present", "over_demixed", "outside_nine")
#: What a class claim may name besides the problem's classes and the kinds: any surprise kind.
MISMATCH = "mismatch"
#: The signal bits a token carries when ``AgentConfig.share_signals`` is on.
TOKEN_KEYS = ("kind", "mismatch", "surprise")


def pending(sig: Optional[Mapping[str, Any]]) -> bool:
    """Whether a result's label may still change. Only an explicit ``settled: False`` says so:
    a result with no signals (a problem without them, a failed run) is as final as it gets."""
    return bool(sig) and sig.get("settled") is False


def is_surprise(sig: Optional[Mapping[str, Any]]) -> bool:
    """A settled result the application flags as a surprise."""
    return bool(sig) and sig.get("surprise") is True and not pending(sig)


def shows(sig: Mapping[str, Any], cls: str) -> bool:
    """Whether a result shows what a class claim names: a class, a kind, or any surprise kind."""
    if cls == MISMATCH:
        return sig.get("kind") in SURPRISE_KINDS
    if cls in KINDS:
        return sig.get("kind") == cls
    return cls in (sig.get("observed"), sig.get("observed_coarse"))


def claim_classes(classes: Sequence[str] = ()) -> Tuple[str, ...]:
    """Everything a class claim may name, in a fixed order: the problem's classes, then
    ``mismatch``, then the kinds. Duplicates are dropped, first occurrence kept."""
    return tuple(dict.fromkeys([str(c) for c in classes] + [MISMATCH, *KINDS]))


def on_token(sig: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The few bits of a result's signals that travel: kind, mismatch, surprise, when known."""
    return {k: sig[k] for k in TOKEN_KEYS if sig and sig.get(k) is not None}
