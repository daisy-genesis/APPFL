"""ADKO-H unit tests. ``python -m unittest discover -s tests/decentralized -t .``

Offline and fast: the only language model used is the mock planner, and the molecule tests skip
themselves unless ``ADKO_PMO_POOL`` points at a pool. What is actually checked is the part a
GPU run cannot afford to get wrong -- that a campaign's verdict comes from the data, that the
claim channel deduplicates and expires, that a region built from a model's output is always
legal, and that a seed reproduces a run exactly.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from appfl.decentralized.algorithm.adko_h import metrics as hmetrics
from appfl.decentralized.algorithm.adko_h import rules as hrules
from appfl.decentralized.algorithm.adko_h.agent import AgentConfig
from appfl.decentralized.algorithm.adko_h.campaign import (
    Campaign,
    Evidence,
    Test,
    random_design,
)
from appfl.decentralized.algorithm.adko_h.claims import Claim, ClaimStore, Verdict
from appfl.decentralized.algorithm.adko_h.llm import build_planner
from appfl.decentralized.algorithm.adko_h.llm import schema as hschema
from appfl.decentralized.algorithm.adko_h.loop import RunConfig, run
from appfl.decentralized.algorithm.adko_h.problems.synthetic import Synthetic
from appfl.decentralized.algorithm.adko_h.region import Dim, Region, RegionSpace
from appfl.decentralized.algorithm.adko_h.surrogate import GP
from appfl.decentralized.algorithm.adko_h.token import HToken, encode
from appfl.decentralized.algorithm.adko_v2.knowledge_token import Quantizer
from appfl.decentralized.topology import build_topology


def space2() -> RegionSpace:
    return RegionSpace(
        [
            Dim("x", "real", 0.0, 10.0, meaning="first"),
            Dim("s", "cat", options=("a", "b", "c"), meaning="second"),
        ]
    )


class TestRegion(unittest.TestCase):
    def setUp(self):
        self.sp = space2()
        self.F = np.array([[1.0, 0.0], [5.0, 1.0], [9.0, 2.0], [5.0, 2.0]])

    def test_mask_and_full(self):
        r = self.sp.region(bounds={"x": (4.0, 6.0)}, picks={"s": [1, 2]})
        np.testing.assert_array_equal(r.mask(self.sp, self.F), [False, True, False, True])
        np.testing.assert_array_equal(self.sp.full().mask(self.sp, self.F), [True] * 4)

    def test_bad_input_is_clipped_not_raised(self):
        r = self.sp.region(
            bounds={"x": (-50.0, 500.0), "nope": (1, 2), "s": (0, 1)},
            picks={"s": [9, 1, -3], "x": [0]},
        )
        self.assertEqual(r.bounds, {"x": (0.0, 10.0)})
        self.assertEqual(r.picks, {"s": (1,)})

    def test_reversed_interval_is_sorted(self):
        self.assertEqual(self.sp.region(bounds={"x": (7.0, 2.0)}).bounds["x"], (2.0, 7.0))

    def test_empty_interval_dropped(self):
        self.assertEqual(self.sp.region(bounds={"x": (3.0, 3.0)}).bounds, {})

    def test_share_and_overlap(self):
        r = self.sp.region(bounds={"x": (0.0, 5.0)}, picks={"s": [0]})
        self.assertAlmostEqual(r.share(self.sp), 0.5 * (1 / 3))
        self.assertAlmostEqual(self.sp.full().share(self.sp), 1.0)
        self.assertAlmostEqual(r.overlap(self.sp, r), 1.0)
        far = self.sp.region(bounds={"x": (6.0, 10.0)}, picks={"s": [1]})
        self.assertAlmostEqual(r.overlap(self.sp, far), 0.0)

    def test_pack_round_trip(self):
        r = self.sp.region(bounds={"x": (2.0, 4.0)}, picks={"s": [0, 2]})
        back = self.sp.unpack(r.pack(self.sp))
        self.assertEqual(back.bounds, r.bounds)
        self.assertEqual(back.picks, r.picks)
        self.assertEqual(self.sp.unpack(self.sp.full().pack(self.sp)), Region())

    def test_label_is_human_readable(self):
        r = self.sp.region(bounds={"x": (2.0, 4.0)}, picks={"s": [0, 2]})
        self.assertEqual(r.label(self.sp), "x[2,4] s{a,c}")
        self.assertEqual(self.sp.full().label(self.sp), "whole space")

    def test_territory_boxes_cover_the_cube(self):
        centers = self.sp.centers(4, seed=0)
        self.assertEqual(centers.shape, (4, 2))
        masks = [self.sp.cell_mask(self.F, centers, i) for i in range(4)]
        np.testing.assert_array_equal(np.sum(masks, axis=0), [1, 1, 1, 1])

    def test_box_around_takes_a_half_width_per_coordinate(self):
        sp = RegionSpace([Dim("a", "real", 0.0, 10.0), Dim("b", "real", 0.0, 2.0)])
        self.assertEqual(sp.box_around([0.5, 0.5], [0.1, 0.25]).bounds, {"a": (4.0, 6.0), "b": (0.5, 1.5)})
        self.assertEqual(sp.box_around([0.5, 0.5], 0.1).bounds, sp.box_around([0.5, 0.5], [0.1, 0.1]).bounds)


class TestCampaign(unittest.TestCase):
    def test_metrics(self):
        ev = Evidence(
            scores=np.array([0.1, 0.9]),
            descriptors=np.array([[0.0, 0.0], [3.0, 4.0]]),
            families=["a", "b", ""],
            known_descriptors=np.array([[0.0, 0.0]]),
            known_families={"a"},
            hull_before=1.0,
            hull_after=2.5,
        )
        self.assertEqual(Test("new_families").value(ev), 1.0)
        self.assertEqual(Test("novel_distance").value(ev), 5.0)
        self.assertAlmostEqual(Test("hull_growth").value(ev), 1.5)
        self.assertAlmostEqual(Test("max_objective").value(ev), 0.9)
        self.assertAlmostEqual(Test("mean_objective").value(ev), 0.5)

    def test_ops_and_validation(self):
        ev = Evidence(scores=np.array([0.3]))
        self.assertTrue(Test("max_objective", ">=", 0.2).passed(ev))
        self.assertTrue(Test("max_objective", "<=", 0.5).passed(ev))
        self.assertFalse(Test("max_objective", ">=", 0.5).passed(ev))
        with self.assertRaises(ValueError):
            Test("not_a_metric")
        with self.assertRaises(ValueError):
            Test("new_families", "==", 1)

    def test_novel_distance_with_nothing_known_is_infinite(self):
        ev = Evidence(descriptors=np.array([[1.0]]), known_descriptors=np.zeros((0, 0)))
        self.assertEqual(Test("novel_distance").value(ev), float("inf"))

    def test_judge_is_idempotent_and_records_infinity_as_none(self):
        sp = space2()
        c = Campaign(region=sp.full(), test=Test("novel_distance", ">=", 1.0), p_success=0.75)
        ev = Evidence(descriptors=np.array([[1.0]]), known_descriptors=np.zeros((0, 0)))
        v = c.judge(ev)
        self.assertTrue(v["success"])
        self.assertIsNone(v["value"])
        self.assertIs(c.judge(Evidence()), v)
        self.assertAlmostEqual(c.brier(), 0.0625)

    def test_budget_is_clamped_except_for_the_no_planner_arm(self):
        sp = space2()
        self.assertEqual(Campaign(region=sp.full(), budget=99).budget, 10)
        self.assertEqual(Campaign(region=sp.full(), budget=1).budget, 2)
        self.assertEqual(Campaign(region=sp.full(), budget=10**9, source="none").budget, 10**9)

    def test_unknown_rule_falls_back(self):
        self.assertEqual(Campaign(region=space2().full(), rule="telepathy").rule, "adko")

    def test_random_design_is_legal(self):
        sp = space2()
        rng = np.random.default_rng(0)
        for _ in range(20):
            d = random_design(sp, rng)
            c = Campaign(**{k: v for k, v in d.items() if k != "source"}, source=d["source"])
            self.assertIn(c.rule, hrules.RULES)
            self.assertGreaterEqual(c.budget, 2)
            self.assertLessEqual(c.budget, 10)
            self.assertLessEqual(c.region.share(sp), 1.0)

    def test_random_design_can_be_centred_and_shaped(self):
        sp = RegionSpace([Dim("a", "real", 0.0, 10.0), Dim("b", "real", 0.0, 2.0)])
        d = random_design(sp, np.random.default_rng(0), center=[0.5, 0.5], widths=[0.1, 0.25])
        self.assertEqual(d["region"].bounds, {"a": (4.0, 6.0), "b": (0.5, 1.5)})

    def test_hit_metrics_count_only_outcomes_past_the_threshold(self):
        ev = Evidence(
            scores=np.array([0.9, 0.2, 0.7, 0.95]),
            families=["a", "b", "c", "a"],
            known_families={"c"},
            hit_threshold=0.6,
        )
        self.assertEqual(Test("hits").value(ev), 3.0)
        self.assertEqual(Test("new_hit_families").value(ev), 1.0)  # c was known; b missed
        self.assertEqual(Test("new_families").value(ev), 2.0)
        self.assertEqual(Test("hits").value(Evidence(scores=np.array([0.9]))), 0.0)  # no threshold
        low = Evidence(scores=np.array([0.1, 0.5]), families=["x", "y"], hit_threshold=0.2, maximize=False)
        self.assertEqual(Test("new_hit_families").value(low), 1.0)


class TestClaims(unittest.TestCase):
    def setUp(self):
        self.sp = space2()
        self.store = ClaimStore(self.sp, source=0)
        self.region = self.sp.region(bounds={"x": (0.0, 5.0)})

    def claim(self, origin=1, cid="a1c0", ttl=2, expires=10):
        return Claim(
            origin=origin,
            claim_id=cid,
            region=self.region.pack(self.sp),
            rule="adko",
            hypothesis="h",
            round=0,
            expires=expires,
            ttl=ttl,
        )

    def test_dedupe_and_relay_decrements_ttl(self):
        self.store.ingest([self.claim()], [])
        out_c, _ = self.store.outbox()
        self.assertEqual([c.ttl for c in out_c], [1])
        self.store.ingest([self.claim()], [])  # the same claim again
        self.assertEqual(self.store.outbox(), ([], []))
        self.assertEqual(self.store.totals["claims_new"], 1)
        self.assertEqual(self.store.totals["claims_in"], 2)

    def test_last_hop_is_not_forwarded(self):
        self.store.ingest([self.claim(ttl=1)], [])
        self.assertEqual(self.store.outbox(), ([], []))

    def test_own_claims_are_ignored_on_the_way_in(self):
        self.store.ingest([self.claim(origin=0)], [])
        self.assertEqual(self.store.totals["claims_new"], 0)

    def test_penalty_is_an_indicator_and_expires(self):
        self.store.ingest([self.claim(expires=5), self.claim(cid="a1c1", expires=5)], [])
        F = np.array([[1.0, 0.0], [9.0, 0.0]])
        np.testing.assert_array_equal(self.store.penalty(F, now=3), [1.0, 0.0])
        np.testing.assert_array_equal(self.store.penalty(F, now=6), [0.0, 0.0])

    def test_verdicts_are_kept_newest_first_and_capped(self):
        store = ClaimStore(self.sp, source=0, keep=3)
        for i in range(5):
            store.ingest([], [Verdict(origin=2, claim_id=f"c{i}", metric="new_families", value=i, success=bool(i % 2))])
        got = [v.claim_id for v in store.recent_verdicts()]
        self.assertEqual(got, ["c4", "c3", "c2"])
        self.assertIn("HELD", store.digest())

    def test_digests_never_crash_when_empty(self):
        self.assertEqual(self.store.digest(), "(nothing yet)")
        self.assertEqual(self.store.claim_digest(0), "(none)")

    def test_findings_ride_on_a_verdict_only_when_shared(self):
        bare = Verdict(origin=1, claim_id="a1c0", metric="new_families", value=3.0, success=True)
        self.assertFalse({"spent", "hits", "hit_families", "best"} & set(bare.to_dict()))
        self.assertNotIn("found", bare.line(self.sp))
        rich = Verdict(
            origin=1, claim_id="a1c0", region=self.region.pack(self.sp), metric="new_families",
            value=3.0, success=True, spent=30, hits=4, hit_families=3, best=0.47,
        )
        self.assertEqual(rich.to_dict()["hits"], 4)
        line = rich.line(self.sp)
        self.assertIn("4 hits", line)
        self.assertIn("x[0,5]", line)
        self.assertEqual(Verdict(**rich.to_dict()).hit_families, 3)  # survives a relay


class TestRules(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.0, 1.0, 2.0])
        self.sigma = np.array([1.0, 0.5, 0.25])
        self.G = np.array([0.0, 0.0, 1.0])
        self.L = np.array([1.0, 0.0, 0.0])
        self.nov = np.array([0.0, 5.0, 1.0])

    def test_each_rule_picks_what_it_says(self):
        v = lambda r: hrules.values(r, self.mu, self.sigma, self.G, self.L, self.nov)
        self.assertEqual(int(np.argmax(v("exploit"))), 2)
        self.assertEqual(int(np.argmax(v("explore"))), 0)
        self.assertEqual(int(np.argmax(v("novelty"))), 1)
        self.assertEqual(int(np.argmax(v("indep"))), 2)
        self.assertEqual(int(np.argmax(v("adko"))), 2)
        np.testing.assert_allclose(v("indep"), self.mu + 2.0 * self.sigma)

    def test_unknown_rule_is_the_settled_score(self):
        np.testing.assert_allclose(
            hrules.values("nonsense", self.mu, self.sigma, self.G, self.L),
            hrules.values("adko", self.mu, self.sigma, self.G, self.L),
        )

    def test_penalty_and_novelty_scale_with_sigma(self):
        base = hrules.values("exploit", self.mu, self.sigma, self.G, self.L)
        v, parts = hrules.adjust(base, self.sigma, penalty=np.array([1.0, 0, 0]), penalty_weight=2.0)
        u = float(np.sqrt(np.mean(self.sigma**2)))
        self.assertAlmostEqual(v[0], base[0] - 2.0 * u)
        self.assertAlmostEqual(parts["scale"], u)
        v2, _ = hrules.adjust(base, self.sigma, novelty=self.nov, novelty_weight=1.0)
        self.assertGreater(v2[1], base[1])

    def test_nearest_distance_handles_an_empty_history(self):
        d = hrules.nearest_distance(np.zeros((3, 2)), np.zeros((0, 2)))
        self.assertTrue(np.all(np.isinf(d)))
        d2 = hrules.nearest_distance(np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[0.0, 0.0]]))
        np.testing.assert_allclose(d2, [0.0, 5.0])

    def test_pick_falls_back_when_the_region_is_empty(self):
        v = np.array([0.0, 9.0, 1.0])
        self.assertEqual(hrules.pick(v, np.array([True, False, True])), 2)
        self.assertEqual(hrules.pick(v, np.zeros(3, dtype=bool)), 1)


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.sp = space2()

    def test_format_mentions_every_coordinate(self):
        fmt = hschema.campaign_format(self.sp)
        props = fmt["schema"]["properties"]["region"]["properties"]
        self.assertEqual(set(props), {"x", "s"})
        self.assertEqual(set(fmt["schema"]["properties"]["region"]["required"]), {"x", "s"})

    def test_parse_a_good_answer(self):
        d = hschema.parse(
            self.sp,
            {
                "region": {"x": [2, 4], "s": None},
                "rule": "novelty",
                "budget": 6,
                "hypothesis": "h",
                "rationale": "r",
                "test": {"metric": "new_families", "op": ">=", "threshold": 2},
                "p_success": 0.6,
            },
        )
        self.assertEqual(d["region"].bounds, {"x": (2.0, 4.0)})
        self.assertEqual(d["region"].picks, {})
        self.assertEqual((d["rule"], d["budget"], d["source"]), ("novelty", 6, "llm"))
        self.assertEqual(d["test"].metric, "new_families")

    def test_parse_refuses_an_empty_answer(self):
        self.assertIsNone(hschema.parse(self.sp, {}))
        self.assertIsNone(hschema.parse(self.sp, None))
        self.assertIsNone(hschema.parse(self.sp, {"region": None, "rule": "adko"}))

    def test_parse_survives_nonsense_fields(self):
        d = hschema.parse(
            self.sp,
            {"region": {"x": "wide"}, "test": {"metric": "vibes", "op": "~", "threshold": "lots"}},
        )
        self.assertEqual(d["test"].metric, "new_families")
        self.assertEqual(d["test"].op, ">=")


class TestToken(unittest.TestCase):
    def test_round_trip_with_campaign_messages(self):
        tok = encode(
            agent_id="Agent1",
            source=1,
            round=3,
            x_shared=[0.5, 0.25],
            u=1.0,
            history=[0.0, 1.0],
            quantizer=Quantizer(8),
            new_best=True,
            insight="held",
            descriptor=[0.5, -0.25],
            family="scaffoldA",
            claims=[Claim(origin=1, claim_id="a1c0", region=[None, None])],
            verdicts=[Verdict(origin=1, claim_id="a1c0", success=True)],
        )
        back = HToken.deserialize(tok.serialize())
        self.assertEqual(back.family, "scaffoldA")
        self.assertEqual(back.descriptor, tok.descriptor)
        self.assertEqual(back.claims[0].claim_id, "a1c0")
        self.assertTrue(back.verdicts[0].success)
        self.assertTrue(back.is_best)
        self.assertLess(back.payload_bits(), back.size_bits())


class TestMetrics(unittest.TestCase):
    def test_cumulative_distinct_counts_budget_not_families(self):
        self.assertEqual(hmetrics.cumulative_distinct(["a", "a", "", "b"]), [1, 1, 1, 2])

    def test_n_circles(self):
        Z = np.array([[0.0], [0.5], [2.0], [4.0]])
        self.assertEqual(hmetrics.n_circles(Z, 1.0), 3)
        self.assertEqual(hmetrics.n_circles(np.zeros((0, 1)), 1.0), 0)

    def test_auc_top_k_rewards_finding_good_things_early(self):
        early = hmetrics.auc_top_k([1.0, 0.0, 0.0, 0.0], k=1)
        late = hmetrics.auc_top_k([0.0, 0.0, 0.0, 1.0], k=1)
        self.assertGreater(early, late)
        self.assertAlmostEqual(hmetrics.auc_top_k([0.5, 0.5], k=1), 0.5)

    def test_diverse_hits_ignores_repeats_of_one_family(self):
        self.assertEqual(hmetrics.diverse_hits([0.9] * 5, ["a"] * 5, 0.5), 1)
        self.assertEqual(hmetrics.diverse_hits([0.9, 0.1], ["a", "b"], 0.5), 1)

    def test_calibration_skill_against_the_base_rate(self):
        good = hmetrics.calibration([0.9, 0.1], [True, False])
        self.assertGreater(good["skill"], 0.0)
        flat = hmetrics.calibration([0.5, 0.5], [True, False])
        self.assertAlmostEqual(flat["skill"], 0.0)

    def test_hull_volume_degenerate(self):
        self.assertEqual(hmetrics.hull_volume(np.zeros((2, 3))), 0.0)


class TestTopology(unittest.TestCase):
    def test_random_regular_is_regular_and_connected(self):
        ids = [f"N{i}" for i in range(24)]
        t = build_topology("random_regular", ids, k=4, seed=1)
        self.assertTrue(all(len(t.neighbors(i)) == 4 for i in ids))
        d = t.describe()
        self.assertGreater(d["fiedler_value"], 1e-9)  # connected

    def test_odd_degree_times_odd_nodes_is_reduced(self):
        t = build_topology("random_regular", [f"N{i}" for i in range(5)], k=3, seed=0)
        self.assertEqual(t.k, 2)

    def test_same_seed_same_graph(self):
        ids = [f"N{i}" for i in range(12)]
        a = build_topology("random_regular", ids, k=4, seed=3)
        b = build_topology("random_regular", ids, k=4, seed=3)
        self.assertEqual([a.neighbors(i) for i in ids], [b.neighbors(i) for i in ids])


class TestRequestBody(unittest.TestCase):
    """What will actually hit vLLM. Checked here because a GPU hour is a bad place to find a
    malformed body, and nothing else in this suite talks to a server."""

    def setUp(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient

        sp = space2()
        self.policy = CampaignPolicy(MockClient(sp), sp, model="Qwen/Qwen3.8-27B-FP8")
        self.ctx = {
            "agent": 0,
            "round": 7,
            "problem": {"task": "t", "outcome": "o", "hints": "h"},
            "dims": sp.describe(),
            "rules": list(hrules.RULES),
            "budget_range": [2, 10],
            "metrics": ["new_families"],
            "n_evaluated": 3,
            "best": [{"point": "p", "score": 0.5, "family": "f"}],
            "my_families": ["f"],
            "peer_families": [],
            "territory": "x[0,5]",
            "first_campaign": True,
            "active_claims": "(none)",
            "verdicts": "(nothing yet)",
            "history": [],
        }

    def test_chat_body_is_what_vllm_expects(self):
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        body = chat_request(self.policy._request(self.ctx))
        self.assertEqual(body["model"], "Qwen/Qwen3.8-27B-FP8")
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])
        self.assertEqual(body["response_format"]["json_schema"]["name"], "campaign")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertTrue(body["chat_template_kwargs"]["enable_thinking"])
        self.assertGreater(body["max_tokens"], 1000)
        self.assertIn("campaign", body["messages"][0]["content"].lower())
        self.assertIn("x[0,5]", body["messages"][1]["content"])

    def test_thinking_off_sends_no_thinking_keys(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        sp = space2()
        body = chat_request(
            CampaignPolicy(MockClient(sp), sp, thinking=None)._request(self.ctx)
        )
        self.assertNotIn("reasoning_effort", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_same_prompt_same_seed(self):
        from appfl.decentralized.algorithm.adko_v2.llm.client import chat_request

        a = chat_request(self.policy._request(self.ctx))
        b = chat_request(self.policy._request(self.ctx))
        self.assertEqual(a["seed"], b["seed"])
        other = dict(self.ctx, round=8)
        self.assertNotEqual(chat_request(self.policy._request(other))["seed"], a["seed"])

    def test_the_sdk_takes_the_body_and_sends_the_vllm_fields(self):
        # Through the real openai SDK to a fake server. create() raises TypeError on a keyword
        # it does not know (chat_template_kwargs), before any request leaves; the fields must
        # still reach the server at the top level of the JSON, where vLLM reads them.
        import asyncio
        import json

        try:  # openai >= 3 is built on httpx2
            import httpx2 as httpx
        except ImportError:
            import httpx
        from openai import AsyncOpenAI

        from appfl.decentralized.algorithm.adko_v2.llm.client import LocalClient

        sent = []

        def reply(req):
            sent.append(json.loads(req.content))
            return httpx.Response(200, json={
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "{}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        client = LocalClient("http://vllm.test/v1")
        client.api = AsyncOpenAI(
            base_url="http://vllm.test/v1", api_key="local", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
        )
        out = asyncio.run(client.complete(self.policy._request(self.ctx)))
        self.assertEqual(out["status"], "stop")
        self.assertTrue(sent[0]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(sent[0]["reasoning_effort"], "low")
        self.assertEqual(sent[0]["temperature"], 0.2)
        self.assertIn("seed", sent[0])
        self.assertNotIn("extra_body", sent[0])

    def test_a_fenced_or_chatty_reply_still_parses(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import _json

        self.assertEqual(_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_json('Here you go:\n{"a": 1}\nhope that helps'), {"a": 1})
        self.assertIsNone(_json("no json at all"))
        self.assertIsNone(_json(""))

    def test_the_mock_answer_satisfies_the_parser(self):
        import asyncio

        from appfl.decentralized.algorithm.adko_h.llm.policy import _json

        resp = asyncio.run(self.policy.client.complete(self.policy._request(self.ctx), self.ctx))
        design = hschema.parse(self.policy.space, _json(resp["text"]))
        self.assertIsNotNone(design)
        self.assertIn(design["rule"], hrules.RULES)


class _FakeRaw:
    def __init__(self, text, cost):
        from types import SimpleNamespace as NS

        self.headers = {"x-litellm-response-cost": str(cost)}
        msg = NS(content=text, model_extra={})
        self._resp = NS(
            id="fake",
            choices=[NS(message=msg, finish_reason="stop")],
            usage=NS(prompt_tokens=100, completion_tokens=50),
        )

    def parse(self):
        return self._resp


class _FakeAPI:
    """Stands in for the gateway: records request bodies, returns a canned design."""

    def __init__(self, text, cost=0.01):
        from types import SimpleNamespace as NS

        self.bodies = []
        outer = self

        async def create(**body):
            outer.bodies.append(body)
            return _FakeRaw(text, cost)

        self.chat = NS(completions=NS(with_raw_response=NS(create=create)))


class TestMag(unittest.TestCase):
    """The MAG path without the network: what is sent, what is charged, when it stops."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.sp = space2()
        self.design = (
            '{"region": {"x": [1, 3], "s": null}, "rule": "explore", "budget": 4, '
            '"hypothesis": "h", "rationale": "r", '
            '"test": {"metric": "new_families", "op": ">=", "threshold": 1}, "p_success": 0.4}'
        )

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, cap=None, cost=0.01):
        from appfl.decentralized.algorithm.adko_h.llm.mag import Ledger, MagClient, Slots

        ledger = Ledger(os.path.join(self.tmp.name, "ledger.txt"), cap)
        slots = Slots(n=2, gap=0.0, root=os.path.join(self.tmp.name, "slots"))
        return MagClient(slots=slots, ledger=ledger, api=_FakeAPI(self.design, cost))

    def ctx(self, agent=0):
        return {"agent": agent, "round": 3, "problem": {}, "dims": "", "budget_range": [2, 10],
                "metrics": [], "best": [], "history": []}

    def test_vllm_only_fields_are_not_sent_and_cost_is_recorded(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        c = self.client(cost=0.02)
        pol = CampaignPolicy(c, self.sp, model="claude-x@amazon", thinking="high", temperature=None)
        designs = pol.plan([self.ctx(0), self.ctx(1)])
        self.assertEqual([d["rule"] for d in designs], ["explore", "explore"])
        for body in c.api.bodies:
            for key in ("seed", "chat_template_kwargs", "reasoning_effort", "temperature"):
                self.assertNotIn(key, body)
            self.assertEqual(body["response_format"]["json_schema"]["name"], "campaign")
        self.assertAlmostEqual(pol.totals["cost_usd"], 0.04)
        self.assertAlmostEqual(c.ledger.total(), 0.04)

    def test_the_cap_stops_the_run_instead_of_falling_back(self):
        from appfl.decentralized.algorithm.adko_h.llm.mag import BudgetSpent
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        c = self.client(cap=0.015, cost=0.01)
        pol = CampaignPolicy(c, self.sp, model="m", temperature=None)
        pol.plan([self.ctx(0)])  # $0.01 spent, under the cap
        pol.plan([self.ctx(1)])  # $0.02 spent, now over
        with self.assertRaises(BudgetSpent):
            pol.plan([self.ctx(2)])

    def test_the_token_never_reaches_a_log(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        secret = "sk-not-a-real-token-but-a-canary"
        old = os.environ.get("AMSC_TOKEN")
        os.environ["AMSC_TOKEN"] = secret
        try:
            pol = CampaignPolicy(self.client(), self.sp, run_dir=os.path.join(self.tmp.name, "run"),
                                 model="m", temperature=None)
            pol.plan([self.ctx(0)])
            pol.close()
            for path in Path(self.tmp.name).rglob("*"):
                if path.is_file() and path.suffix in (".jsonl", ".txt", ".json"):
                    self.assertNotIn(secret, path.read_text())
        finally:
            if old is None:
                os.environ.pop("AMSC_TOKEN", None)
            else:
                os.environ["AMSC_TOKEN"] = old

    def test_one_event_loop_across_rounds(self):
        """An async HTTP client is bound to the loop it first ran on; a planner that opened a
        new loop per round broke every call after the first round against a real server."""
        import asyncio

        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        design = self.design

        class LoopBound:
            name = "loopbound"
            loop = None

            async def complete(self, request, ctx=None):
                here = asyncio.get_running_loop()
                if LoopBound.loop is None:
                    LoopBound.loop = here
                if here is not LoopBound.loop or LoopBound.loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                return {"text": design, "usage": {}, "status": "stop"}

        pol = CampaignPolicy(LoopBound(), self.sp, model="m")
        for rnd in range(4):
            pol.plan([self.ctx(a) for a in range(3)])
        self.assertEqual(pol.totals["errors"], 0)
        pol.close()

    def test_a_planner_that_keeps_failing_stops_the_run(self):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy

        class Broken:
            name = "broken"

            async def complete(self, request, ctx=None):
                raise ConnectionError("gateway unreachable")

        pol = CampaignPolicy(Broken(), self.sp, model="m")
        with self.assertRaises(RuntimeError):
            for i in range(4):
                pol.plan([self.ctx(a) for a in range(4)])


def short(arm: str, seed: int = 0, rounds: int = 8, **kw):
    """One tiny run of an arm, with the knobs the arms differ in."""
    problem = Synthetic(seed=seed)
    planner = build_planner(problem.region_space, kw.pop("planner_kind", "none"), seed=seed)
    cfg = RunConfig(
        rounds=rounds,
        agents=3,
        seed=seed,
        topology="random_regular",
        topology_kwargs={"k": 2, "seed": 0},
        planner=kw.pop("planner", "none"),
        label=arm,
        start_points=6,
        agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, **kw),
    )
    rec = run(problem, GP, cfg, planner=planner)
    if planner is not None:
        planner.close()
    return rec


class TestLoop(unittest.TestCase):
    def test_plain_adko_never_judges_a_campaign(self):
        rec = short("adko")
        self.assertEqual(rec["campaigns"]["n"], 0)
        self.assertEqual(rec["endpoints"]["evaluations"], 8 * 3)

    def test_arms_run_and_report_endpoints(self):
        for arm, kw in (
            ("adko", {}),
            ("adko-div", {"novelty_weight": 1.0}),
            ("random-campaign", {"planner": "random"}),
            ("adkoh", {"planner": "llm", "planner_kind": "mock"}),
        ):
            with self.subTest(arm=arm):
                rec = short(arm, **kw)
                e = rec["endpoints"]
                self.assertGreater(e["distinct_families"], 0)
                self.assertEqual(e["evaluations"], 24)
                self.assertEqual(e["redundancy"], 0)  # nothing evaluated twice

    def test_campaigns_are_judged_and_scored(self):
        rec = short("adkoh", rounds=14, planner="llm", planner_kind="mock")
        c = rec["campaigns"]
        self.assertGreater(c["n"], 0)
        self.assertIsNotNone(c["calibration"]["brier"])
        self.assertTrue(set(c["rules"]) <= set(hrules.RULES))

    def test_one_seed_reproduces_exactly(self):
        a = short("adkoh", planner="llm", planner_kind="mock")
        b = short("adkoh", planner="llm", planner_kind="mock")
        self.assertEqual(a["endpoints"], b["endpoints"])

    def test_different_seeds_differ(self):
        a = short("adko", seed=0)
        b = short("adko", seed=1)
        self.assertNotEqual(a["endpoints"]["hull_volume"], b["endpoints"]["hull_volume"])

    def test_claims_change_the_token_not_the_endpoints_shape(self):
        on = short("adkoh", planner="llm", planner_kind="mock", claims_on=True)
        off = short("adkoh", planner="llm", planner_kind="mock", claims_on=False)
        self.assertGreater(on["bits_per_token"], off["bits_per_token"])
        self.assertEqual(set(on["endpoints"]), set(off["endpoints"]))


class TestFootprint(unittest.TestCase):
    """The novelty rule has to see the whole team, not just the FIFO window."""

    def agent(self, memory_rounds=1):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        problem = Synthetic(seed=0)
        return ADKOHAgent(
            "Agent0",
            0,
            GP(),
            problem,
            AgentConfig(memory_rounds=memory_rounds),
            neighbor_indices=[1],
            degree=1,
        )

    def test_peer_inputs_outlive_the_token_memory(self):
        a = self.agent(memory_rounds=1)
        for rnd in range(6):
            a.observe([encode(
                agent_id="Agent1", source=1, round=rnd, x_shared=[float(rnd), 0.0, 0.0, 0.0, 0.0],
                u=0.5, history=[0.5], quantizer=Quantizer(8),
            )])
        self.assertLessEqual(len(a.memory), 2)          # the FIFO forgot most of them
        self.assertEqual(len(a._peer_X), 6)             # the footprint did not
        known = a._known_inputs()
        self.assertEqual(len(known), 6)
        # A candidate sitting on a long-forgotten peer point is not novel.
        d = hrules.nearest_distance(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]), known)
        self.assertAlmostEqual(float(d[0]), 0.0)


class TestPlanners(unittest.TestCase):
    """The controls a planner's regions are compared with: random boxes, and its own replayed."""

    def test_random_boxes_sit_on_anchors_with_the_given_widths(self):
        from appfl.decentralized.algorithm.adko_h.loop import RandomPlanner

        rp = RandomPlanner(space2(), seed=0, anchor=lambda rng: np.array([0.3, 0.0]), widths=[0.05, 0.0])
        for d in rp.plan([{}, {}, {}]):
            self.assertEqual(d["region"].bounds["x"], (2.5, 3.5))

    def test_replay_draws_only_the_listed_designs(self):
        from appfl.decentralized.algorithm.adko_h.loop import ReplayPlanner

        sp = space2()
        designs = [
            {"bounds": {"x": [1.0, 2.0]}, "picks": {}, "rule": "explore", "budget": 7,
             "test": {"metric": "hits", "op": ">=", "threshold": 2}},
            {"bounds": {"x": [6.0, 9.0]}, "picks": {"s": [2]}},
        ]
        got = ReplayPlanner(sp, designs, seed=0).plan([{}] * 20)
        self.assertEqual({d["region"].label(sp) for d in got}, {"x[1,2]", "x[6,9] s{c}"})
        one = next(d for d in got if d["region"].label(sp) == "x[1,2]")
        self.assertEqual((one["rule"], one["budget"], one["test"].metric), ("explore", 7, "hits"))
        again = ReplayPlanner(sp, designs, seed=0).plan([{}] * 20)
        self.assertEqual([d["region"].label(sp) for d in again], [d["region"].label(sp) for d in got])

    def test_a_fixed_rule_overrides_the_planner(self):
        rec = short("random-region", rounds=14, planner="random", force_rule="adko", claims_on=False)
        self.assertGreater(rec["campaigns"]["n"], 0)
        self.assertEqual(set(rec["campaigns"]["rules"]), {"adko"})

    def test_replay_arm_runs_from_a_file(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "designs.json"
            path.write_text(json.dumps([{"bounds": {"x1": [0.0, 0.4], "x3": [0.1, 0.5]}, "picks": {}}]))
            cfg = RunConfig(
                rounds=12, agents=3, seed=0, topology="random_regular",
                topology_kwargs={"k": 2, "seed": 0}, planner="replay", replay_from=str(path),
                start_points=6, agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, claims_on=False),
            )
            rec = run(Synthetic(seed=0), GP, cfg)
        self.assertGreater(rec["campaigns"]["n"], 0)
        self.assertGreaterEqual(rec["planner"]["designs"], rec["campaigns"]["n"])  # + unfinished


class FixedPool:
    """Sixteen fixed points on a diagonal, so a skipped point is visibly missing from a pool."""

    problem_id = "fixed16"
    maximize = True
    kernel = "euclidean"

    def __init__(self, n: int = 16):
        self.region_space = RegionSpace([Dim("x", "real", 0.0, 1.0)])
        self.X = np.repeat(np.linspace(0.0, 1.0, n)[:, None], 2, axis=1)

    def pool(self, rng, n, exclude=None, region=None):
        from appfl.decentralized.algorithm.adko_h.problem import Candidates

        idx = np.setdiff1d(np.arange(len(self.X)), np.asarray(list(exclude or []), dtype=np.int64))
        take = np.sort(rng.choice(idx, size=min(int(n), len(idx)), replace=False)) if len(idx) else idx
        return Candidates(key=take, X=self.X[take], F=self.X[take, :1], meta=[None] * len(take))

    def evaluate(self, picks):
        from appfl.decentralized.algorithm.adko_h.problem import Outcome

        return [Outcome(score=float(p.X[0]), descriptor=np.zeros(0), family="") for p in picks]

    def privatize(self, X, rng):
        return np.asarray(X, dtype=float)

    def input_scale(self):
        return None

    def label(self, key, meta=None):
        return str(key)

    def describe(self):
        return {"task": "", "outcome": "", "hints": "", "dims": self.region_space.describe()}


class TestSkipAndNoise(unittest.TestCase):
    """Skipping what neighbours ran, and what noise on the shared input does to it."""

    def agent(self, **kw):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        self.problem = FixedPool()
        return ADKOHAgent(
            "Agent0", 0, GP(), self.problem, AgentConfig(pool_size=16, warmup_rounds=0, **kw),
            neighbor_indices=[1], degree=1,
        )

    def reported(self, a, x):
        a.observe([encode(agent_id="Agent1", source=1, round=0, x_shared=list(x), u=0.5,
                          history=[0.5], quantizer=Quantizer(8))])

    def test_a_reported_input_leaves_the_pool(self):
        a = self.agent(skip_peer_inputs=True)
        self.reported(a, self.problem.X[5])
        a.propose(1)
        self.assertNotIn(5, set(a._cands.key))
        self.assertEqual(len(a._cands), 15)
        self.assertEqual(a.diag["skipped"], 1)

    def test_without_skipping_it_stays(self):
        a = self.agent()
        self.reported(a, self.problem.X[5])
        a.propose(1)
        self.assertIn(5, set(a._cands.key))

    def test_noise_defeats_an_exact_match_but_not_a_radius(self):
        exact = self.agent(skip_peer_inputs=True)
        self.reported(exact, self.problem.X[5] + 0.01)  # the input as a noisy token reports it
        exact.propose(1)
        self.assertIn(5, set(exact._cands.key))
        near = self.agent(skip_peer_inputs=True, skip_radius=0.03)
        self.reported(near, self.problem.X[5] + 0.01)
        near.propose(1)
        self.assertNotIn(5, set(near._cands.key))

    def test_token_noise_moves_the_shared_input_not_my_own(self):
        a = self.agent(token_noise=0.5)
        pick = a.propose(0)
        tok = a.complete(pick, self.problem.evaluate([pick])[0], 0)
        self.assertFalse(np.allclose(tok.x, pick.X))
        np.testing.assert_array_equal(a._X[-1], pick.X)
        b = self.agent()
        pb = b.propose(0)
        tb = b.complete(pb, self.problem.evaluate([pb])[0], 0)
        np.testing.assert_array_equal(tb.x, pb.X)


class TestInputRelay(unittest.TestCase):
    """``input_hops``: reported inputs travel past the in-neighbours, for the filter only."""

    def node(self, i, **kw):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        self.problem = FixedPool()
        cfg = AgentConfig(pool_size=16, warmup_rounds=0, skip_peer_inputs=True, **kw)
        return ADKOHAgent(f"Agent{i}", i, GP(), self.problem, cfg, neighbor_indices=[i + 1], degree=1)

    def token(self, i, x, rnd=0, relayed=()):
        return encode(agent_id=f"Agent{i}", source=i, round=rnd, x_shared=list(x), u=0.5,
                      history=[0.5], quantizer=Quantizer(8), relayed=relayed)

    def step(self, a, rnd):
        pick = a.propose(rnd)
        return a.complete(pick, self.problem.evaluate([pick])[0], rnd)

    def test_off_by_default_and_costs_nothing(self):
        a = self.node(1)
        a.observe([self.token(2, self.problem.X[5])])
        tok = self.step(a, 1)
        self.assertEqual(tok.relayed, [])
        self.assertEqual(tok.payload_bits(), self.token(1, tok.x).payload_bits())

    def test_two_hops_reach_the_filter_and_stop_there(self):
        b = self.node(1, input_hops=2)
        b.observe([self.token(2, self.problem.X[5])])  # Agent2 ran point 5; Agent0 cannot hear it
        tb = self.step(b, 1)
        self.assertEqual(tb.relayed, [{"h": 1, "x": list(self.problem.X[5])}])
        self.assertEqual(tb.payload_bits() - self.token(1, tb.x).payload_bits(), 8 * 2 + 8)
        a = self.node(0, input_hops=2)
        a.observe([tb])
        a.propose(2)
        self.assertNotIn(5, set(a._cands.key))
        self.assertEqual(len(a._relayed_X), 1)
        ta = self.step(a, 2)
        self.assertNotIn(list(self.problem.X[5]), [r["x"] for r in ta.relayed])  # 2 hops is all

    def test_three_hops_relay_once_more(self):
        a = self.node(0, input_hops=3)
        a.observe([self.token(1, self.problem.X[1], relayed=[{"h": 1, "x": list(self.problem.X[5])}])])
        ta = self.step(a, 1)
        self.assertIn({"h": 2, "x": list(self.problem.X[5])}, ta.relayed)

    def test_relayed_inputs_feed_only_the_filter(self):
        a = self.node(0, input_hops=2)
        a.observe([self.token(1, self.problem.X[1], relayed=[{"h": 1, "x": list(self.problem.X[5])}])])
        self.assertEqual(len(a._peer_X), 1)
        self.assertEqual(len(a._known_inputs()), 1)
        self.assertEqual(len(a.memory.X), 1)

    def test_one_run_heard_twice_is_kept_once_and_my_own_is_not_kept(self):
        a = self.node(0, input_hops=2)
        mine = self.step(a, 0)
        x5 = {"h": 1, "x": list(self.problem.X[5])}
        back = {"h": 1, "x": list(mine.x)}
        a.observe([self.token(1, self.problem.X[1], 1, relayed=[x5, back]),
                   self.token(2, self.problem.X[2], 1, relayed=[x5])])
        self.assertEqual(len(a._relayed_X), 1)

    def test_relaying_without_the_filter_is_refused(self):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        with self.assertRaises(ValueError):
            ADKOHAgent("Agent0", 0, GP(), FixedPool(), AgentConfig(input_hops=2),
                       neighbor_indices=[1], degree=1)

    def test_a_run_with_relay(self):
        import json
        import tempfile

        def go(hops, out=None):
            cfg = RunConfig(
                rounds=10, agents=6, seed=0, topology="random_regular",
                topology_kwargs={"k": 2, "seed": 0}, planner="none", start_points=6,
                agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, skip_peer_inputs=True,
                                      skip_radius=1e-6, input_hops=hops),
            )
            return run(Synthetic(seed=0), GP, cfg, out=out)

        with tempfile.TemporaryDirectory() as d:
            rec = go(2, d)
            agents = json.loads((Path(d) / "run.json").read_text())["agents"]
        self.assertTrue(all(a["relayed_inputs"] > 0 for a in agents))
        self.assertGreater(rec["bits_per_token"], go(1)["bits_per_token"])


class TestSoftFilter(unittest.TestCase):
    """``soft_radius``: candidates near a reported input are discounted, not dropped."""

    def test_biweight_is_one_at_the_report_and_zero_at_the_radius(self):
        from appfl.decentralized.algorithm.adko_h import rules

        k = rules.biweight(np.array([0.0, 0.4, 0.8, 1.2]), 0.8)
        np.testing.assert_allclose(k, [1.0, 0.5625, 0.0, 0.0])

    def test_discount_keeps_order_and_moves_the_pick_off_a_near_repeat(self):
        from appfl.decentralized.algorithm.adko_h import rules

        v = np.array([-1.0, 0.0, 2.0, 3.0])
        np.testing.assert_allclose(rules.discount(v, np.zeros(4)), v)
        out = rules.discount(v, np.array([0.0, 0.0, 0.0, 0.75]))
        np.testing.assert_allclose(out, [-1.0, 0.0, 2.0, 0.0])
        self.assertEqual(int(np.argmax(out)), 2)
        self.assertTrue(np.isneginf(rules.discount(np.array([-np.inf, 1.0]), np.ones(2))[0]))

    def test_off_by_default_and_refused_without_the_filter(self):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        self.assertEqual(AgentConfig().soft_radius, 0.0)
        with self.assertRaises(ValueError):
            ADKOHAgent("Agent0", 0, GP(), FixedPool(), AgentConfig(soft_radius=0.5),
                       neighbor_indices=[1], degree=1)

    def test_a_run_with_the_soft_ring(self):
        import json
        import tempfile

        def go(soft, out=None):
            cfg = RunConfig(
                rounds=12, agents=6, seed=0, topology="random_regular",
                topology_kwargs={"k": 2, "seed": 0}, planner="none", start_points=6,
                agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, skip_peer_inputs=True,
                                      skip_radius=1e-6, soft_radius=soft),
            )
            return run(Synthetic(seed=0), GP, cfg, out=out)

        with tempfile.TemporaryDirectory() as d:
            go(0.5, d)
            recs = [r for a in json.loads((Path(d) / "run.json").read_text())["agents"] for r in a["records"]]
        soft = [r for r in recs if "d_soft" in r]
        self.assertTrue(soft)
        self.assertTrue(all(0.0 <= r["d_soft"] <= 1.0 for r in soft))
        self.assertTrue(any(r["d_soft_moved"] for r in soft))
        with tempfile.TemporaryDirectory() as d:
            go(0.0, d)
            recs0 = [r for a in json.loads((Path(d) / "run.json").read_text())["agents"] for r in a["records"]]
        self.assertFalse(any("d_soft" in r for r in recs0))


class TestPromptV2(unittest.TestCase):
    def ctx(self, **kw):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        a = ADKOHAgent("Agent0", 0, GP(), Synthetic(seed=0), AgentConfig(hit_threshold=0.6, **kw),
                       neighbor_indices=[1], degree=1, planner=object())
        return a.plan_context(5)

    def test_v2_defines_hits_and_drops_the_old_steering(self):
        from appfl.decentralized.algorithm.adko_h.llm import prompts

        ctx = self.ctx()
        system, user = prompts.system_v2(ctx), prompts.user_v2(ctx)
        self.assertIn("at least 0.6", user)
        self.assertIn("base rates", user)
        self.assertNotIn("The default", system)
        self.assertNotIn("working elsewhere", system)
        self.assertNotIn("1 is a modest claim", user)
        self.assertIn("claims", system)
        self.assertIn("active claims", user)

    def test_v2_without_claims_does_not_mention_them(self):
        from appfl.decentralized.algorithm.adko_h.llm import prompts

        ctx = self.ctx(claims_on=False)
        self.assertNotIn("claims", prompts.system_v2(ctx))
        self.assertNotIn("active claims", prompts.user_v2(ctx))

    def test_v2_schema_allows_the_hit_metrics(self):
        fmt = hschema.campaign_format(space2(), metrics=hschema.METRICS_V2)
        self.assertIn("new_hit_families", fmt["schema"]["properties"]["test"]["properties"]["metric"]["enum"])

    def test_a_v2_run_judges_campaigns_and_shares_findings(self):
        problem = Synthetic(seed=0)
        planner = build_planner(problem.region_space, "mock", seed=0, prompt="v2")
        cfg = RunConfig(
            rounds=14, agents=3, seed=0, topology="random_regular", topology_kwargs={"k": 2, "seed": 0},
            planner="llm", start_points=6,
            agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, hit_threshold=0.6, share_hits=True,
                                  claim_weight=0.0),
        )
        rec = run(problem, GP, cfg, planner=planner)
        planner.close()
        self.assertGreater(rec["campaigns"]["n"], 0)


class TestDetailChannel(unittest.TestCase):
    """The wide campaign channel: full text, the planner's note, and where the hits were."""

    def agent(self, **kw):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        return ADKOHAgent("Agent0", 0, GP(), Synthetic(seed=0),
                          AgentConfig(hit_threshold=0.6, **kw),
                          neighbor_indices=[1], degree=1, planner=object())

    def test_an_empty_note_stays_off_the_wire(self):
        self.assertNotIn("note", Claim(origin=1, claim_id="a1c0").to_dict())
        d = Verdict(origin=1, claim_id="a1c0").to_dict()
        self.assertNotIn("note", d)
        self.assertNotIn("hit_box", d)

    def test_a_detailed_claim_and_verdict_carry_what_they_were_given(self):
        c = Claim(origin=1, claim_id="a1c0", note="small rings looked dead")
        self.assertEqual(c.to_dict()["note"], "small rings looked dead")
        sp = space2()
        v = Verdict(origin=2, claim_id="a2c1", region=sp.full().pack(sp), metric="hits",
                    value=3.0, threshold=2.0, success=True, hypothesis="mid x is rich",
                    spent=10, hits=3, hit_families=2, best=0.71,
                    note="x near 5 pays, s=a does not", hit_box=sp.region(bounds={"x": [4.0, 6.0]}).pack(sp))
        line = v.line(sp)
        self.assertIn("3 hits", line)
        self.assertIn("its hits sat in", line)
        self.assertIn("x near 5 pays", line)

    def test_a_verdict_without_findings_still_shows_its_note(self):
        v = Verdict(origin=3, claim_id="a3c0", metric="hits", threshold=1.0, note="nothing here")
        self.assertIn("nothing here", v.line())

    def test_the_schema_asks_for_a_note_only_when_detail_is_on(self):
        plain = hschema.campaign_format(space2(), metrics=hschema.METRICS_V2)
        rich = hschema.campaign_format(space2(), metrics=hschema.METRICS_V2, note=True)
        self.assertNotIn("note", plain["schema"]["properties"])
        self.assertIn("note", rich["schema"]["properties"])
        self.assertIn("note", rich["schema"]["required"])

    def test_the_parser_keeps_a_note(self):
        d = hschema.parse(space2(), {"region": {"x": [1.0, 2.0]}, "rule": "adko", "budget": 5,
                                     "note": "x low is empty",
                                     "test": {"metric": "hits", "op": ">=", "threshold": 1}})
        self.assertEqual(d["note"], "x low is empty")

    def test_the_detailed_prompt_asks_for_the_note_and_the_plain_one_does_not(self):
        from appfl.decentralized.algorithm.adko_h.llm import prompts

        rich = prompts.system_v2(self.agent(share_detail=True).plan_context(5))
        plain = prompts.system_v2(self.agent().plan_context(5))
        self.assertIn("one sentence to your neighbours", rich)
        self.assertNotIn("one sentence to your neighbours", plain)
        # Without claims there is nobody to write to, so the note is not asked for either.
        off = prompts.system_v2(self.agent(share_detail=True, claims_on=False).plan_context(5))
        self.assertNotIn("one sentence to your neighbours", off)

    def test_the_hit_box_covers_the_hits_and_leaves_the_misses_out(self):
        a = self.agent(share_detail=True)
        sp = a.space
        c = a.set_campaign({"region": sp.full(), "rule": "adko", "budget": 4,
                            "note": "worth a look"}, 0)
        names = [d.name for d in sp.dims]
        for score, F in ((0.9, [0.2] * len(names)), (0.1, [0.9] * len(names)),
                         (0.7, [0.4] * len(names))):
            c.runs.append({"round": 0, "key": 0, "point": "p", "score": score,
                           "family": "f", "F": F})
        box = a._hit_box(c)
        region = sp.unpack(box)
        for name in names:
            lo, hi = region.bounds[name]
            self.assertAlmostEqual(lo, 0.2, places=6)
            self.assertAlmostEqual(hi, 0.4, places=6)

    def test_no_hits_means_no_box(self):
        a = self.agent(share_detail=True)
        c = a.set_campaign({"region": a.space.full(), "rule": "adko", "budget": 2}, 0)
        c.runs.append({"round": 0, "key": 0, "point": "p", "score": 0.1, "family": "f",
                       "F": [0.5] * len(a.space.dims)})
        self.assertIsNone(a._hit_box(c))

    def test_the_note_reaches_a_neighbour_through_claim_and_verdict(self):
        a = self.agent(share_detail=True, share_hits=True, claim_weight=0.0)
        a.set_campaign({"region": a.space.full(), "rule": "adko", "budget": 3,
                        "note": "mid range pays"}, 0)
        claims, _ = a.claims.outbox()
        self.assertEqual(claims[0].note, "mid range pays")

    def test_a_detailed_run_is_still_a_run(self):
        problem = Synthetic(seed=0)
        planner = build_planner(problem.region_space, "mock", seed=0, prompt="v2", note=True)
        cfg = RunConfig(
            rounds=14, agents=3, seed=0, topology="random_regular",
            topology_kwargs={"k": 2, "seed": 0}, planner="llm", start_points=6,
            agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, hit_threshold=0.6,
                                  share_hits=True, share_detail=True, claim_weight=0.0),
        )
        rec = run(problem, GP, cfg, planner=planner)
        planner.close()
        self.assertGreater(rec["campaigns"]["n"], 0)


class TestComposedArms(unittest.TestCase):
    """The composed arm and its ablations must differ from it in exactly one switch each."""

    def test_each_ablation_takes_one_part_off(self):
        from appfl.decentralized.algorithm.adko_h.run import ARMS

        base = ARMS["adkoh3"]
        for name, key in (("adkoh3-blind", "blind"),
                          ("adkoh3-nofindings", "share_hits"),
                          ("adkoh3-penalty", "claim_weight"),
                          ("adkoh3-rich", "share_detail")):
            diff = {
                k for k in set(base) | set(ARMS[name])
                if base.get(k, False) != ARMS[name].get(k, False)
            }
            self.assertEqual(diff, {key}, f"{name} differs from adkoh3 in {diff}")

    def test_the_composed_arm_is_v2_named_with_claims_that_carry_findings_and_no_fence(self):
        from appfl.decentralized.algorithm.adko_h.run import ARMS, V2_NOISE

        a = ARMS["adkoh3"]
        self.assertEqual((a["prompt"], a["claim_weight"], a["share_hits"], a["token_noise"]),
                         ("v2", 0.0, True, V2_NOISE))
        self.assertNotIn("claims_on", a)  # claims are on; only the penalty is off
        self.assertNotIn("blind", a)  # the objective is named; blind is the ablation

    def test_the_noise_ablation_is_adkoh2_so_it_needs_no_arm(self):
        from appfl.decentralized.algorithm.adko_h.run import ARMS

        quiet = {k: v for k, v in ARMS["adkoh3"].items() if k != "token_noise"}
        self.assertEqual(quiet, ARMS["adkoh2"])

    def test_an_arms_noise_applies_without_a_flag_and_a_flag_overrides_it(self):
        from appfl.decentralized.algorithm.adko_h.run import V2_NOISE, main, parse

        for argv, want in (
            (["--arm", "adkoh3", "--planner", "mock"], V2_NOISE),
            (["--arm", "adkoh3", "--planner", "mock", "--token-noise", "0.5"], 0.5),
            (["--arm", "adko"], 0.0),
        ):
            with self.subTest(argv=argv):
                parse(argv + ["--problem", "synthetic", "--rounds", "0"])  # argv is legal
        # The level reaches AgentConfig, which is what the run actually uses.
        seen = {}
        import appfl.decentralized.algorithm.adko_h.run as runmod

        real = runmod.run
        runmod.run = lambda problem, gp, cfg, **kw: (
            seen.setdefault("noise", cfg.agent_cfg.token_noise),
            {"endpoints": {"distinct_families": 0, "diverse_hits": 0, "n_circles": 0,
                           "hull_volume": 0.0, "auc_top10": 0.0, "best": 0.0},
             "campaigns": {"n": 0, "passed": 0, "calibration": {"brier": None}},
             "bits_per_token": 0.0, "seconds": 0, "planner": {}},
        )[1]
        import contextlib
        import io

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--problem", "synthetic", "--arm", "adkoh3", "--planner", "mock",
                      "--rounds", "4", "--agents", "2", "--seeds", "0"])
            self.assertAlmostEqual(seen["noise"], V2_NOISE)
        finally:
            runmod.run = real


@unittest.skipUnless(os.environ.get("ADKO_PMO_POOL"), "set ADKO_PMO_POOL to run the PMO tests")
class TestPMO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from appfl.decentralized.algorithm.adko_h.problems.pmo import PMO

        cls.problem = PMO("qed", limit=2000)

    def test_pool_respects_a_region_and_excludes_what_is_run(self):
        sp = self.problem.region_space
        r = sp.region(bounds={"qed": (0.7, 1.0)})
        c = self.problem.pool(np.random.default_rng(0), 64, exclude=[], region=r)
        self.assertEqual(len(c), 64)
        self.assertTrue(r.mask(sp, c.F).all())
        again = self.problem.pool(np.random.default_rng(0), 64, exclude=list(c.key), region=r)
        self.assertEqual(len(set(again.key) & set(c.key)), 0)

    def test_evaluate_scores_and_names_a_scaffold(self):
        from appfl.decentralized.algorithm.adko_h.problem import Pick

        c = self.problem.pool(np.random.default_rng(1), 4)
        outs = self.problem.evaluate(
            [Pick(agent=0, key=int(k), X=c.X[i], F=c.F[i], meta=c.meta[i]) for i, k in enumerate(c.key)]
        )
        self.assertEqual(len(outs), 4)
        self.assertTrue(all(0.0 <= o.score <= 1.0 for o in outs))
        self.assertEqual(self.problem.descriptor_dim(), 3)
        self.assertTrue(any(o.family for o in outs))

    def test_oracle_calls_are_cached(self):
        from appfl.decentralized.algorithm.adko_h.problem import Pick

        c = self.problem.pool(np.random.default_rng(2), 2)
        picks = [Pick(agent=0, key=int(k), X=c.X[i], F=c.F[i], meta=c.meta[i]) for i, k in enumerate(c.key)]
        before = self.problem.calls
        self.problem.evaluate(picks)
        self.problem.evaluate(picks)
        self.assertEqual(self.problem.calls - before, 2)

    def test_a_blind_description_does_not_name_the_objective(self):
        from appfl.decentralized.algorithm.adko_h.problems.pmo import PMO

        blind = PMO("qed", limit=2000, blind=True).describe()
        self.assertNotIn("objective 'qed'", blind["task"])
        self.assertIn("hidden objective", blind["task"])
        self.assertIn("objective 'qed'", self.problem.describe()["task"])

    def test_input_scale_is_the_embedding_spread(self):
        s = self.problem.input_scale()
        self.assertEqual(s.shape, (32,))
        self.assertTrue(np.all(s > 0))



# -- agency (design 2026-11-11, items A0-A12) ----------------------------------------------------

from appfl.decentralized.algorithm.adko_h.problem import Outcome, Pick  # noqa: E402
from appfl.decentralized.algorithm.adko_h.signals import SURPRISE_KINDS  # noqa: E402

#: Signals as the harness would send them: a surprise, a settled match, a pending result.
SURPRISE = {"settled": True, "stable": True, "observed": "P1", "predicted": "W1-X", "mismatch": True,
            "kind": "wetting_absent", "outside_nine": False, "novelty": 0.8, "surprise": True}
MATCH = {"settled": True, "stable": True, "observed": "W1-X", "predicted": "W1-X", "mismatch": False,
         "kind": None, "outside_nine": False, "novelty": 0.1, "surprise": False}
PENDING = {"settled": False, "stable": None, "observed": "P1", "predicted": "W1-X", "mismatch": None,
           "kind": None, "outside_nine": None, "novelty": 5.0, "surprise": False}


class SignalSynthetic(Synthetic):
    """The synthetic problem with signals: a crude predictor (``alpha`` below x1 = 0.5, ``beta``
    above), so mismatches and surprises occur, and some results pending (x5 above 0.85). It
    also names a target, a role per agent, its classes, and any rules a test gives it."""

    CLASSES = ("alpha", "beta", "gamma", "delta", "epsilon", "background")

    def __init__(self, seed: int = 0, fns=None):
        super().__init__(seed=seed)
        self.fns = dict(fns or {})

    def evaluate(self, picks):
        outs = super().evaluate(picks)
        for p, o in zip(picks, outs):
            x = np.asarray(p.X, dtype=float)
            pred, pend = ("alpha" if x[0] < 0.5 else "beta"), bool(x[4] > 0.85)
            kind = None
            if o.family != pred:
                kind = "under_demixed" if o.family == "background" else (
                    "over_demixed" if x[2] > 0.5 else "wetting_absent")
            o.signals = {"settled": not pend, "stable": True, "observed": o.family, "predicted": pred,
                         "mismatch": None if pend else o.family != pred, "kind": kind,
                         "novelty": round(float(x[3]), 4),
                         "surprise": (not pend) and kind in SURPRISE_KINDS}
        return outs

    def describe(self):
        d = super().describe()
        d.update(target="Each laboratory scores a result 1 when its observed class differs from the "
                        "predicted one.", role=["hunt wetting changes", "hunt extra phases"],
                 classes=list(self.CLASSES), rule_notes={"lowx1": "the smallest x1 in the region"})
        return d

    def rules(self):
        return dict(self.fns)


def agency(problem=None, index=0, start=0, **kw):
    """One agent on the signal problem with a planner attached (so it plans) and no warmup;
    ``start`` shared initial points loaded first, so its GP is fitted."""
    from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

    problem = problem or SignalSynthetic(seed=0)
    kw.setdefault("budget_range", (2, 10))
    cfg = AgentConfig(warmup_rounds=0, pool_size=32, claim_weight=0.0, **kw)
    a = ADKOHAgent(f"Agent{index}", index, GP(), problem, cfg, neighbor_indices=[1 - index],
                   degree=1, planner=object())
    if start:
        c = problem.start(0, start)
        a.load(c, problem.evaluate([Pick(agent=-1, key=int(k), X=c.X[i], F=c.F[i])
                                    for i, k in enumerate(c.key)]))
    return a


def step(a, rnd, sig=None, score=0.5, family="alpha"):
    """One pick and its result, with the signals the test wants."""
    pick = a.propose(rnd)
    out = Outcome(score=score, descriptor=np.asarray(pick.X[:3], dtype=float), family=family,
                  signals=dict(sig or {}))
    return pick, a.complete(pick, out, rnd)


def design(sp, budget=5, rule="explore", **kw):
    """A legacy design: one region, one rule, one budget."""
    return {"region": sp.full(), "rule": rule, "budget": budget, "hypothesis": "h",
            "test": Test("mismatches", ">=", 1.0), "p_success": 0.5, "source": "test", **kw}


def parts_design(sp, parts, **kw):
    """A version-3 design: ``parts`` as (bounds, rule, n[, anchor])."""
    rows = [{"region": sp.region(bounds=b), "rule": r, "n": n, "anchor": (rest[0] if rest else None)}
            for b, r, n, *rest in parts]
    return {"parts": rows, "hypothesis": "h", "test": Test("mismatches", ">=", 1.0),
            "p_success": 0.5, "source": "test", **kw}


class TestAgencyDefaults(unittest.TestCase):
    """A0: with every new field at its default, nothing an earlier arm produced changes."""

    def test_new_fields_default_off(self):
        c = AgentConfig()
        self.assertEqual(
            (c.parts, c.replan_on_surprise, c.followup, c.anchors, c.class_claims, c.extra_rules,
             c.prompt, c.share_signals, c.events),
            (False, False, 0, False, False, (), "v2", False, False),
        )

    def test_v1_and_v2_requests_hash_as_before(self):
        # The cache keys the code produced before agency existed, for one fixed context: a legacy
        # run replays from its answer cache only if these do not move.
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient
        from appfl.decentralized.algorithm.adko_v2.llm.client import request_key

        problem = Synthetic(seed=0)
        a = ADKOHAgent("Agent0", 0, GP(), problem, AgentConfig(hit_threshold=0.6, share_hits=True),
                       neighbor_indices=[1], degree=1, planner=object(),
                       centers=problem.region_space.centers(2, 0))
        ctx = a.plan_context(5)
        want = {"v1": "9b9e42f3eb8fb2bc6500636d982e3e4649e630a21dfb2439df546dfc5aa054c3",
                "v2": "dbe409f5533f20c875732705e20f91e8b455565dbf04b0bf4241954f22f857af"}
        for prompt, key in want.items():
            pol = CampaignPolicy(MockClient(problem.region_space), problem.region_space, prompt=prompt,
                                 budget_range=(2, 10), note=prompt == "v2")
            self.assertEqual(request_key(pol._request(ctx), "mock"), key, prompt)

    def test_a_token_without_signals_has_exactly_the_old_fields(self):
        tok = encode(agent_id="Agent0", source=0, round=0, x_shared=[0.1], u=0.5, history=[0.5],
                     quantizer=Quantizer(8))
        self.assertEqual(set(tok.to_dict()), {"x", "q", "insight", "tag", "provenance", "descriptor",
                                              "family", "claims", "verdicts", "relayed"})
        self.assertEqual(HToken.deserialize(tok.serialize()).serialize(), tok.serialize())

    def test_events_watch_and_change_nothing(self):
        def go(events):
            import json
            import tempfile

            problem = Synthetic(seed=0)
            planner = build_planner(problem.region_space, "mock", seed=0, prompt="v2")
            cfg = RunConfig(rounds=12, agents=3, seed=0, topology="random_regular",
                            topology_kwargs={"k": 2, "seed": 0}, planner="llm", start_points=6,
                            agent_cfg=AgentConfig(warmup_rounds=2, pool_size=64, hit_threshold=0.6,
                                                  share_hits=True, events=events))
            with tempfile.TemporaryDirectory() as d:
                run(problem, GP, cfg, planner=planner, out=d)
                agents = json.loads((Path(d) / "run.json").read_text())["agents"]
                csv = (Path(d) / "records.csv").read_text()
            planner.close()
            return csv, [a["records"] for a in agents], [a["campaigns"] for a in agents]

        self.assertEqual(go(False), go(True))

    def test_a_default_agent_collects_no_events(self):
        a = agency()
        a.set_campaign(design(a.space), 0)
        step(a, 0, SURPRISE)
        self.assertEqual(a.drain_events(), [])
        self.assertEqual(a.tickets, [])


class TestSignals(unittest.TestCase):
    """A1: signals on outcomes, kept per run, on the token only when shared."""

    def test_signals_are_kept_and_recorded(self):
        a = agency()
        a.set_campaign(design(a.space), 0)
        _, tok = step(a, 0, SURPRISE)
        self.assertEqual(a._signals[-1], SURPRISE)
        self.assertEqual(a.records[-1]["signals"], SURPRISE)
        self.assertEqual(a.campaign.runs[-1]["signals"], SURPRISE)
        self.assertEqual(tok.signals, {})  # not shared unless asked
        self.assertNotIn("signals", tok.to_dict())

    def test_shared_bits_are_few_and_a_surprise_says_where_it_sat(self):
        a = agency(share_signals=True)
        a.set_campaign(design(a.space), 0)
        p, tok = step(a, 0, SURPRISE)
        self.assertEqual(tok.signals, {"mismatch": True, "kind": "wetting_absent", "surprise": True,
                                       "at": [round(float(v), 4) for v in p.F]})
        _, plain = step(a, 1, MATCH)
        self.assertEqual(plain.signals, {"mismatch": False, "surprise": False})
        back = HToken.deserialize(tok.serialize())
        self.assertEqual(back.signals, tok.signals)
        bare = encode(agent_id="Agent0", source=0, round=0, x_shared=tok.x, u=0.5, history=[0.5],
                      quantizer=Quantizer(8), family=tok.family)
        self.assertGreater(tok.payload_bits(), bare.payload_bits())

    def test_a_neighbours_surprise_is_listed_with_an_id(self):
        a = agency(prompt="v3")
        a.set_campaign(design(a.space), 0)
        step(a, 0, SURPRISE)
        at = [0.9, 0.05, 0.9, 0.9, 0.9]
        a.observe([encode(agent_id="Agent1", source=1, round=0, x_shared=at, u=0.5, history=[0.5],
                          quantizer=Quantizer(8),
                          signals={"kind": "over_demixed", "mismatch": True, "surprise": True, "at": at})])
        rows = a.plan_context(1)["surprises"]
        self.assertEqual([(s["id"], s["source"], s["kind"]) for s in rows],
                         [("s0", "mine", "wetting_absent"), ("s1", "a1", "over_demixed")])
        self.assertIn("x1=0.9", rows[1]["recipe"])


class TestRevise(unittest.TestCase):
    """A2: a pending result settles later; the agent's view follows, judged verdicts stay."""

    def test_revise_replaces_score_family_signals_and_refits(self):
        a = agency(share_signals=True, start=6)
        c = a.set_campaign(design(a.space, budget=4), 0)
        p0, _ = step(a, 0, PENDING, score=0.1, family="")
        mu0, _ = a.surrogate.predict(p0.X[None, :])
        events = a.revise(p0.key, Outcome(score=0.9, descriptor=np.ones(3), family="beta", signals=MATCH), 1)
        self.assertEqual(events[0]["type"], "label_revision")
        self.assertEqual((events[0]["old"]["score"], events[0]["new"]["score"]), (0.1, 0.9))
        self.assertEqual(events[0]["campaign"], c.claim_id)
        self.assertEqual((a._u[-1], a._fams[-1], a._signals[-1]), (0.9, "beta", MATCH))
        np.testing.assert_array_equal(a._desc[-1], np.ones(3))
        self.assertEqual((c.runs[0]["score"], c.runs[0]["family"], c.runs[0]["signals"]), (0.9, "beta", MATCH))
        mu1, _ = a.surrogate.predict(p0.X[None, :])
        self.assertNotAlmostEqual(float(mu0[0]), float(mu1[0]))
        _, tok = step(a, 1, MATCH)  # the next token carries the revision
        self.assertEqual(tok.revised, [{"r": 0, "family": "beta", "mismatch": False, "surprise": False}])
        self.assertEqual(a.drain_events(), [])  # returned, not collected: events are off

    def test_a_judged_campaign_keeps_its_verdict(self):
        a = agency(parts=True)  # any agency switch: with none, signals never reach a verdict
        c = a.set_campaign(design(a.space, budget=2), 0)
        p0, _ = step(a, 0, PENDING)
        step(a, 1, PENDING)
        before = dict(c.verdict)
        self.assertEqual((before["value"], before["n_pending"]), (0.0, 2))
        a.revise(p0.key, Outcome(score=0.5, signals=SURPRISE), 2)
        self.assertEqual(c.verdict, before)
        self.assertEqual(c.runs[0]["signals"], SURPRISE)

    def test_only_own_runs_can_be_revised(self):
        a = agency(start=3)
        with self.assertRaises(KeyError):
            a.revise(a._keylist[0], Outcome(score=0.5), 0)  # a shared initial point
        with self.assertRaises(KeyError):
            a.revise(10**9, Outcome(score=0.5), 0)

    def test_the_same_revisions_give_the_same_picks(self):
        def go():
            a = agency(followup=1, replan_on_surprise=True)
            a.set_campaign(design(a.space, budget=6), 0)
            keys = []
            for rnd in range(8):
                if rnd == 3:
                    a.revise(keys[1], Outcome(score=0.7, signals=SURPRISE), rnd)
                if a.needs_plan(rnd):
                    a.set_campaign(design(a.space, budget=3, rule="novelty"), rnd)
                p, _ = step(a, rnd, PENDING if rnd < 2 else MATCH)
                keys.append(p.key)
            return keys, [r["rule"] for r in a.records]

        self.assertEqual(go(), go())


class TestParts(unittest.TestCase):
    """A3: multi-part campaigns; a legacy design is one part."""

    def test_a_legacy_design_is_one_part(self):
        sp = space2()
        c = Campaign(region=sp.full(), rule="explore", budget=4)
        self.assertEqual((len(c.parts), c.parts[0].n, c.budget, c.multipart), (1, 4, 4, False))
        self.assertNotIn("parts", c.to_dict(sp))
        c.add_run({"key": 1})
        self.assertEqual(c.runs, [{"key": 1}])  # no part tag on a legacy run

    def test_counts_are_clamped_to_the_budget_range(self):
        from appfl.decentralized.algorithm.adko_h.campaign import clamp_counts

        self.assertEqual(clamp_counts([4, 4, 4], (2, 6)), [4, 1, 1])
        self.assertEqual(clamp_counts([0, 1], (5, 9)), [1, 4])
        self.assertEqual(clamp_counts([1, 1, 1, 1], (2, 3)), [1, 1, 1])
        self.assertEqual(clamp_counts([2, 3], (2, 6)), [2, 3])

    def test_parts_run_in_order_and_are_judged_together(self):
        a = agency(parts=True)
        sp = a.space
        c = a.set_campaign(parts_design(sp, [({"x1": (0.0, 0.2)}, "explore", 2),
                                             ({"x1": (0.6, 0.8)}, "novelty", 1)]), 0)
        self.assertEqual((c.budget, c.multipart, c.rule), (3, True, "explore"))
        claims, _ = a.claims.outbox()
        self.assertEqual([m.claim_id for m in claims], ["a0c0p0", "a0c0p1"])  # one claim per part
        got = [step(a, r, MATCH)[0] for r in range(3)]
        self.assertEqual([p.rule for p in got], ["explore", "explore", "novelty"])
        self.assertTrue(all(0.0 <= p.F[0] <= 0.2 for p in got[:2]))
        self.assertTrue(0.6 <= got[2].F[0] <= 0.8)
        self.assertEqual([r["part"] for r in c.runs], [0, 0, 1])
        self.assertIsNotNone(c.verdict)
        self.assertEqual(sp.unpack(a.claims.verdicts[(0, "a0c0")].region).bounds["x1"], (0.0, 0.8))

    def test_without_parts_a_design_keeps_its_first_part(self):
        a = agency()
        c = a.set_campaign(parts_design(a.space, [({"x1": (0.0, 0.2)}, "explore", 3),
                                                  ({"x1": (0.6, 0.8)}, "novelty", 3)]), 0)
        self.assertEqual((len(c.parts), c.budget), (1, 3))

    def test_a_part_with_no_candidates_falls_back_and_says_so(self):
        a = agency(parts=True)
        c = a.set_campaign(parts_design(a.space, [({"x1": (0.95, 1.0), "x2": (0.95, 1.0)}, "explore", 2)]), 0)
        a.propose(0)
        self.assertTrue(a.diag["region_empty"])
        self.assertEqual(a.diag["part"], 0)
        self.assertEqual(c.part_index, 0)


class TestSignalTests(unittest.TestCase):
    """A4: the signal metrics, pending runs left out, and class claims with code-set status."""

    def test_signal_metrics_leave_pending_runs_out(self):
        other = {**SURPRISE, "kind": "over_demixed", "novelty": 2.0}
        ev = Evidence(signals=[SURPRISE, MATCH, PENDING, other], parts=[0, 0, 0, 1])
        self.assertEqual(Test("mismatches").value(ev), 2.0)
        self.assertAlmostEqual(Test("mismatch_rate").value(ev), 2 / 3)
        self.assertEqual(Test("surprises").value(ev), 2.0)
        self.assertEqual(Test("max_novelty").value(ev), 2.0)  # the pending 5.0 does not count
        none = Evidence(signals=[PENDING])
        self.assertEqual((Test("mismatch_rate").value(none), Test("max_novelty").value(none)), (0.0, 0.0))

    def test_a_verdict_counts_pending_runs_only_when_there_are_signals(self):
        sp = space2()
        c = Campaign(region=sp.full(), test=Test("mismatch_rate", ">=", 0.5))
        v = c.judge(Evidence(signals=[SURPRISE, PENDING], parts=[0, 0]))
        self.assertEqual((v["value"], v["success"], v["n_pending"]), (1.0, True, 1))
        self.assertNotIn("n_pending", Campaign(region=sp.full()).judge(Evidence()))

    def test_claim_status_is_set_by_code(self):
        from appfl.decentralized.algorithm.adko_h.campaign import claim_status

        ev = Evidence(signals=[SURPRISE, MATCH, PENDING, MATCH], parts=[0, 0, 0, 1])
        self.assertEqual(claim_status(ev, 0, "mismatch")["status"], "supported")  # 1 of 2 settled
        self.assertEqual(claim_status(ev, 0, "P1")["status"], "supported")
        self.assertEqual(claim_status(ev, 1, "wetting_absent")["status"], "refuted")
        self.assertEqual(claim_status(ev, 2, "mismatch")["status"], "unsettled")
        self.assertEqual(claim_status(Evidence(signals=[PENDING], parts=[0]), 0, "P1")["status"], "unsettled")

    def test_a_class_claim_is_announced_and_its_status_travels(self):
        a = agency(parts=True, class_claims=True)
        sp = a.space
        c = a.set_campaign(parts_design(sp, [({"x1": (0.0, 0.4)}, "explore", 2), ({}, "novelty", 1)],
                                        claim={"part": 0, "class": "mismatch"}), 0)
        claims, _ = a.claims.outbox()
        self.assertEqual([m.predicts for m in claims], ["mismatch", ""])
        step(a, 0, SURPRISE)
        step(a, 1, MATCH)
        _, tok = step(a, 2, MATCH)
        self.assertEqual(c.claim["status"], "supported")
        self.assertEqual(c.verdict["claim"]["status"], "supported")
        cv = [v for v in tok.verdicts if v.metric == "class_claim"]
        self.assertEqual([(v.claim_id, v.status, v.predicts, v.value) for v in cv],
                         [("a0c0p0", "supported", "mismatch", 0.5)])
        self.assertIn('SUPPORTED: class claim "mismatch"', cv[0].line(sp))

    def test_a_claim_on_no_part_or_an_unknown_class_is_dropped(self):
        a = agency(parts=True, class_claims=True)
        for claim in ({"part": 3, "class": "alpha"}, {"part": 0, "class": "unicorn"}, "alpha"):
            c = a.set_campaign(parts_design(a.space, [({}, "explore", 2)], claim=claim), 0)
            self.assertIsNone(c.claim)
        self.assertEqual(a.set_campaign(parts_design(a.space, [({}, "explore", 2)],
                                                     claim={"part": 0, "class": "alpha"}), 0).claim["class"],
                         "alpha")  # one of the problem's classes


class TestTickets(unittest.TestCase):
    """A5: code-owned follow-up of a surprise."""

    def test_a_surprise_opens_a_ticket_and_the_campaign_waits(self):
        a = agency(followup=2)
        c = a.set_campaign(design(a.space, budget=6), 0)
        p0, tok0 = step(a, 0, SURPRISE)
        t = a.tickets[0]
        self.assertEqual((t.id, t.surprise, t.origin_key, t.kind, t.draws), ("a0t0", "s0", p0.key, "wetting_absent", 2))
        self.assertTrue(t.drawing)
        p1 = a.propose(1)
        self.assertEqual((p1.rule, p1.claim_id), ("followup", "a0t0"))
        sp = a.space
        mask = t.region.mask(sp, a._cands.F)
        d = np.sqrt(((sp.normalize(a._cands.F) - sp.normalize(p0.F[None, :])[0]) ** 2).sum(axis=1))
        self.assertAlmostEqual(float(d[list(a._cands.key).index(p1.key)]), float(d[mask].min()))
        self.assertTrue(t.region.mask(sp, p1.F[None, :])[0])
        a.complete(p1, Outcome(score=0.5, signals=MATCH), 1)
        self.assertEqual((c.spent, t.spent), (1, 1))  # the campaign waited
        _, tok2 = step(a, 2, SURPRISE)  # the second draw reproduces it
        self.assertEqual((t.status, len(a.tickets)), ("reproduced", 1))  # a draw opens no ticket
        tv = [v for v in tok2.verdicts if v.metric == "ticket"]
        self.assertEqual([(v.claim_id, v.status, v.predicts, v.value) for v in tv],
                         [("a0t0", "reproduced", "wetting_absent", 1.0)])
        p3, _ = step(a, 3, MATCH)
        self.assertEqual((p3.rule, c.spent), ("explore", 2))  # and resumed
        self.assertEqual(a.surprises[0]["status"], "reproduced")
        self.assertEqual(a.surprises[1]["status"], "follow-up draw")

    def test_ticket_status_follows_its_draws(self):
        from appfl.decentralized.algorithm.adko_h.campaign import Ticket

        def ticket(*sigs, draws=2):
            t = Ticket(id="a0t0", surprise="s0", origin_key=1, F=[0.5], kind="wetting_absent", draws=draws)
            t.runs = [{"key": i, "signals": dict(s)} for i, s in enumerate(sigs)]
            t.spent = len(sigs)
            return t

        self.assertEqual(ticket(MATCH, SURPRISE).outcome(), "reproduced")
        self.assertEqual(ticket(MATCH, MATCH).outcome(), "not_reproduced")
        self.assertEqual(ticket(MATCH, {**SURPRISE, "kind": "over_demixed"}).outcome(), "not_reproduced")
        self.assertIsNone(ticket(MATCH, PENDING).outcome())  # waits for the label
        self.assertEqual(ticket(MATCH, PENDING).outcome(final=True), "unsettled")
        self.assertIsNone(ticket(MATCH).outcome())  # a draw left
        self.assertEqual(ticket(MATCH).outcome(final=True), "unsettled")

    def test_surprises_queue_beyond_the_open_tickets(self):
        a = agency(followup=1, max_open_tickets=1)
        a.set_campaign(design(a.space, budget=8), 0)
        k0 = step(a, 0, PENDING)[0].key
        k1 = step(a, 1, PENDING)[0].key
        a.revise(k0, Outcome(score=0.5, signals=SURPRISE), 2)  # a late surprise opens a ticket too
        a.revise(k1, Outcome(score=0.5, signals=SURPRISE), 2)
        self.assertEqual((len(a.tickets), a._queue), (1, ["s1"]))
        p, _ = step(a, 2, MATCH)  # t0's only draw; t0 stops drawing and s1 gets its ticket
        self.assertEqual((p.claim_id, [t.id for t in a.tickets], a._queue), ("a0t0", ["a0t0", "a0t1"], []))
        self.assertEqual(step(a, 3, MATCH)[0].claim_id, "a0t1")

    def test_the_end_of_the_run_closes_what_is_open(self):
        a = agency(followup=2)
        a.set_campaign(design(a.space), 0)
        step(a, 0, SURPRISE)
        step(a, 1, PENDING)
        events = a.finish(2)
        self.assertEqual([(e["type"], e["status"], e["final"]) for e in events],
                         [("ticket_close", "unsettled", True)])
        self.assertEqual(a.tickets[0].status, "unsettled")


class TestReplan(unittest.TestCase):
    """A6: a surprise closes the campaign early; the planner is asked once the draws are spent."""

    def test_early_close_and_replan_after_the_draws(self):
        a = agency(followup=1, replan_on_surprise=True)
        c = a.set_campaign(design(a.space, budget=5), 0)
        step(a, 0, MATCH)
        _, tok = step(a, 1, SURPRISE)
        self.assertEqual((c.verdict["early"], c.verdict["spent"], c.done), ("surprise", 2, True))
        self.assertEqual(c.to_dict(a.space)["early"], "surprise")
        self.assertEqual([v.claim_id for v in tok.verdicts], ["a0c0"])
        self.assertIsNotNone(tok.insight)
        self.assertFalse(a.needs_plan(2))  # the follow-up draw first
        step(a, 2, MATCH)
        self.assertTrue(a.needs_plan(3))

    def test_without_followup_the_next_round_replans(self):
        a = agency(replan_on_surprise=True)
        c = a.set_campaign(design(a.space, budget=5), 0)
        step(a, 0, SURPRISE)
        self.assertTrue(c.done and a.needs_plan(1))

    def test_a_late_surprise_closes_the_open_campaign_too(self):
        a = agency(replan_on_surprise=True)
        a.set_campaign(design(a.space, budget=2), 0)
        k = step(a, 0, PENDING)[0].key
        step(a, 1, MATCH)  # the first campaign closes normally
        c2 = a.set_campaign(design(a.space, budget=5), 2)
        step(a, 2, MATCH)
        events = a.revise(k, Outcome(score=0.5, signals=SURPRISE), 3)
        self.assertEqual(c2.verdict["early"], "surprise")
        # Events are off, yet revise returns what it set off: the revision and the early verdict.
        self.assertEqual([e["type"] for e in events], ["label_revision", "campaign_verdict"])
        self.assertEqual(a.drain_events(), [])


class TestAnchors(unittest.TestCase):
    """A7: a part anchored on a surprise contains it."""

    def test_widen_and_hull(self):
        sp = space2()
        r = sp.region(bounds={"x": (2.0, 4.0)}, picks={"s": [0]})
        w = sp.widen(r, [7.0, 2.0])
        self.assertEqual((w.bounds, w.picks), ({"x": (2.0, 7.0)}, {"s": (0, 2)}))
        self.assertEqual(sp.widen(sp.full(), [7.0, 2.0]), sp.full())
        h = sp.hull([r, sp.region(bounds={"x": (6.0, 8.0)}, picks={"s": [1]})])
        self.assertEqual((h.bounds, h.picks), ({"x": (2.0, 8.0)}, {"s": (0, 1)}))
        self.assertEqual(sp.hull([r, sp.full()]), sp.full())

    def test_an_anchor_widens_its_part_and_nothing_else(self):
        a = agency(parts=True, anchors=True)
        a.set_campaign(design(a.space), 0)
        p0, _ = step(a, 0, SURPRISE)
        f = [float(v) for v in p0.F]
        c = a.set_campaign(parts_design(a.space, [({"x1": (0.0, 0.01), "x2": (0.0, 0.01)}, "explore", 2, "s0"),
                                                  ({"x1": (0.0, 0.01)}, "explore", 2)]), 1)
        b = c.parts[0].region.bounds
        self.assertEqual(c.parts[0].anchor, "s0")
        self.assertTrue(c.parts[0].region.mask(a.space, np.asarray([f]))[0])
        self.assertEqual((b["x1"][0], b["x2"][0]), (0.0, 0.0))  # only stretched up to the recipe
        self.assertEqual(c.parts[1].region.bounds, {"x1": (0.0, 0.01)})

    def test_an_unknown_anchor_is_dropped_and_logged(self):
        a = agency(parts=True, anchors=True, events=True)
        c = a.set_campaign(parts_design(a.space, [({"x1": (0.0, 0.2)}, "explore", 2, "s9")]), 0)
        self.assertEqual((c.parts[0].anchor, c.anchors_dropped), (None, ["s9"]))
        self.assertEqual(c.parts[0].region.bounds, {"x1": (0.0, 0.2)})
        self.assertEqual(a.drain_events()[0]["anchors_dropped"], ["s9"])

    def test_anchors_off_ignores_them(self):
        a = agency(parts=True)
        a.set_campaign(design(a.space), 0)
        step(a, 0, SURPRISE)
        c = a.set_campaign(parts_design(a.space, [({"x1": (0.0, 0.01)}, "explore", 2, "s0")]), 1)
        self.assertEqual((c.parts[0].anchor, c.anchors_dropped, c.parts[0].region.bounds),
                         (None, [], {"x1": (0.0, 0.01)}))


class TestProblemRules(unittest.TestCase):
    """A8: rules a problem supplies, and the fallback when one breaks."""

    def setUp(self):
        self.seen = []

        def low(view):
            self.seen.append(view)
            return -view.cands.F[:, 0]

        self.problem = SignalSynthetic(seed=0, fns={"lowx1": low, "boom": lambda v: 1 / 0,
                                                    "short": lambda v: np.zeros(3)})

    def test_a_problem_rule_ranks_the_pool(self):
        from appfl.decentralized.algorithm.adko_h.problem import RuleView

        a = agency(self.problem, extra_rules=("lowx1",))
        a.set_campaign(design(a.space, rule="lowx1"), 0)
        p = a.propose(0)
        self.assertEqual(p.rule, "lowx1")
        self.assertAlmostEqual(p.F[0], float(a._cands.F[:, 0].min()))
        v = self.seen[0]
        self.assertIsInstance(v, RuleView)
        self.assertEqual((v.agent, v.round, v.rule, v.part, v.mu.shape, v.own_X.shape[0]),
                         (0, 0, "lowx1", 0, (len(a._cands),), 0))
        again = SignalSynthetic(seed=0, fns=self.problem.fns)
        b = agency(again, extra_rules=("lowx1",))
        b.set_campaign(design(b.space, rule="lowx1"), 0)
        self.assertEqual(b.propose(0).key, p.key)
        self.assertEqual(self.seen[-1].seed, v.seed)  # deterministic per (seed, agent, round)

    def test_the_view_carries_the_region_and_the_descriptors(self):
        a = agency(self.problem, extra_rules=("lowx1",))
        a.set_campaign(parts_design(a.space, [({"x1": (0.1, 0.6)}, "lowx1", 3)]), 0)
        step(a, 0)
        a.complete(a.propose(1), Outcome(score=-1.0), 1)  # a failed run: no descriptor
        a.propose(2)
        v = self.seen[-1]
        self.assertIs(v.region, a.campaign.region)
        self.assertEqual(v.region.bounds["x1"], (0.1, 0.6))
        self.assertEqual(v.own_descriptors.shape, (len(v.own_X), 3))
        np.testing.assert_allclose(v.own_descriptors[0], a._desc[0])
        self.assertTrue(np.isnan(v.own_descriptors[1]).all())  # row-aligned: the failed run is a NaN row
        np.testing.assert_allclose(v.known_descriptors, a._descriptors())
        self.assertIsNone(v.progress)  # the agent does not know the run's budget; the driver may

    def test_a_rule_that_breaks_falls_back_to_explore(self):
        for name, why in (("boom", "ZeroDivisionError"), ("short", "shape"), ("absent", "supplies no rule")):
            with self.subTest(rule=name):
                a = agency(self.problem, extra_rules=(name,), events=True)
                a.set_campaign(design(a.space, rule=name), 0)
                p = a.propose(0)
                _, sigma = a.surrogate.predict(a._cands.X)
                self.assertEqual(p.key, int(a._cands.key[int(np.argmax(sigma))]))
                self.assertIn(why, a.diag["rule_error"])
                self.assertIn("rule_error", [e["type"] for e in a.drain_events()])

    def test_extra_rules_reach_the_planner_and_may_not_shadow_built_ins(self):
        from appfl.decentralized.algorithm.adko_h.agent import ADKOHAgent

        a = agency(self.problem, extra_rules=("lowx1",), prompt="v3")
        self.assertEqual(a.plan_context(0)["rules"], list(hrules.RULES) + ["lowx1"])
        with self.assertRaises(ValueError):
            ADKOHAgent("Agent0", 0, GP(), self.problem, AgentConfig(extra_rules=("explore",)))
        self.assertEqual(hrules.valid_rule("lowx1"), "adko")
        self.assertEqual(hrules.valid_rule("lowx1", ("lowx1",)), "lowx1")


class FakeWorkspace:
    """The harness's workspace, as APPFL sees it: three calls, and a record of each."""

    def __init__(self, block="notes: none yet"):
        self.block = block
        self.applied = []

    def context(self, agent, rnd):
        return self.block

    def schema(self, agent):
        return {"type": "object", "additionalProperties": False, "required": ["write"],
                "properties": {"write": {"type": "array", "minItems": 1,
                                         "items": {"type": "object", "additionalProperties": False,
                                                   "required": ["path", "text"],
                                                   "properties": {"path": {"type": "string"},
                                                                  "text": {"type": "string"}}}}}}

    def apply(self, agent, rnd, value):
        self.applied.append((agent, rnd, value))
        return {"agent": agent, "round": rnd, "ops": len((value or {}).get("write", []))}


class TestWorkspace(unittest.TestCase):
    """A9: the workspace hook. APPFL passes text and an object through and knows no files."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.problem = SignalSynthetic(seed=0)
        a = agency(self.problem, prompt="v3", parts=True)
        self.ctx = a.plan_context(5)

    def tearDown(self):
        self.tmp.cleanup()

    def policy(self, ws, prompt="v3"):
        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient

        sp = self.problem.region_space
        return CampaignPolicy(MockClient(sp), sp, run_dir=self.tmp.name, prompt=prompt, workspace=ws,
                              budget_range=(2, 6))

    def test_context_schema_and_apply(self):
        import json

        ws = FakeWorkspace("NOTES-42")
        pol = self.policy(ws)
        req = pol._request(self.ctx)
        self.assertIn("## My workspace\nNOTES-42", req["input"])
        sch = req["text"]["format"]["schema"]
        self.assertIn("workspace", sch["properties"])
        self.assertIn("workspace", sch["required"])
        (d,) = pol.plan([self.ctx])
        self.assertEqual(ws.applied, [(0, 5, {"write": [{"path": "mock", "text": "mock"}]})])
        self.assertEqual(d["workspace_log"], {"agent": 0, "round": 5, "ops": 1})
        pol.close()
        row = json.loads((Path(self.tmp.name) / "llm_calls.jsonl").read_text().splitlines()[-1])
        self.assertEqual(row["workspace"], {"agent": 0, "round": 5, "ops": 1})

    def test_the_block_is_in_the_cache_key_and_a_replay_applies_again(self):
        from appfl.decentralized.algorithm.adko_v2.llm.client import request_key

        a, b = self.policy(FakeWorkspace("one")), self.policy(FakeWorkspace("two"))
        self.assertNotEqual(request_key(a._request(self.ctx), "mock"), request_key(b._request(self.ctx), "mock"))
        b.close()
        a.plan([self.ctx])
        a.close()
        ws = FakeWorkspace("one")
        replay = self.policy(ws)
        replay.plan([self.ctx])
        replay.close()
        self.assertEqual(replay.totals["cached"], 1)
        self.assertEqual(len(ws.applied), 1)

    def test_a_v2_planner_takes_a_workspace_too(self):
        ws = FakeWorkspace("v2 notes")
        pol = self.policy(ws, prompt="v2")
        ctx = agency(self.problem).plan_context(5)
        self.assertTrue(pol._request(ctx)["input"].endswith("## My workspace\nv2 notes"))
        (d,) = pol.plan([ctx])
        pol.close()
        self.assertEqual(d["workspace"], {"write": [{"path": "mock", "text": "mock"}]})


class TestPromptV3(unittest.TestCase):
    """A10: the version-3 prompt, schema, parser and mock."""

    def agent(self, **kw):
        kw.setdefault("prompt", "v3")
        return agency(SignalSynthetic(seed=0, fns={"lowx1": lambda v: -v.cands.F[:, 0]}), **kw)

    def test_the_prompt_says_target_role_signals_surprises_and_follow_ups(self):
        from appfl.decentralized.algorithm.adko_h.llm import prompts

        a = self.agent(parts=True, anchors=True, class_claims=True, followup=1, replan_on_surprise=True,
                       extra_rules=("lowx1",))
        a.set_campaign(design(a.space, budget=3), 0)
        step(a, 0, SURPRISE)
        step(a, 1, PENDING)
        ctx = a.plan_context(2)
        system, user = prompts.system_v3(ctx), prompts.user_v3(ctx)
        for text in ("observed class differs from the predicted one", "hunt wetting changes", "s0 (mine",
                     "a0t0 for s0 (wetting_absent)", "Use 1 to 3 parts", "lowx1: the smallest x1",
                     "mismatch_rate:", "Class claim options: alpha", "closed early by a surprise",
                     "1 settled, 0 pending"):
            self.assertIn(text, user)
        for text in ("anchor", "ends your current campaign early", "Code sets the claim's status"):
            self.assertIn(text, system)
        plain = self.agent()
        self.assertNotIn("anchor", prompts.system_v3(plain.plan_context(0)))
        self.assertIn("Use one part", prompts.user_v3(plain.plan_context(0)))

    def test_text_limits_are_stated_stored_and_carried(self):
        from appfl.decentralized.algorithm.adko_h.llm import prompts

        long = {"hypothesis": "h" * 700, "rationale": "r" * 1000, "note": "n" * 600}
        # The defaults: the limits every earlier arm ran with, stated in the v3 prompt.
        a = self.agent(share_detail=True)
        c = a.set_campaign({**design(a.space, budget=3), **long}, 0)
        self.assertEqual((len(c.hypothesis), len(c.rationale), len(c.note)), (300, 600, 240))
        system = prompts.system_v3(a.plan_context(1))
        self.assertIn("at most 300 characters", system)
        self.assertIn("in at most 240 characters", system)
        self.assertNotIn("one sentence", system)
        # Raised limits: stored, stated in the prompt and the schema, and carried whole on the wire.
        b = self.agent(share_detail=True, note_chars=500, hypothesis_chars=600, rationale_chars=900,
                       insight_chars=1700)
        c = b.set_campaign({**design(b.space, budget=3), **long}, 0)
        self.assertEqual((len(c.hypothesis), len(c.rationale), len(c.note)), (600, 900, 500))
        self.assertEqual(len(b._claim_text(c)), 600)
        self.assertEqual(b._verdict_chars(), 600)
        ctx = b.plan_context(1)
        system = prompts.system_v3(ctx)
        for text in ("in at most 600 characters", "in at most 900 characters", "in at most 500 characters"):
            self.assertIn(text, system)
        fmt = hschema.campaign_format_v3(b.space, note=True, text_limits=ctx["text_limits"])["schema"]
        self.assertIn("at most 600 characters", fmt["properties"]["hypothesis"]["description"])
        self.assertIn("at most 500 characters", fmt["properties"]["note"]["description"])

    def test_the_schema_has_exactly_what_the_arm_uses(self):
        sp = SignalSynthetic().region_space
        fmt = hschema.campaign_format_v3(sp, budget_range=(2, 6), rules=hrules.RULES + ("lowx1",),
                                         max_parts=3, classes=("alpha", "mismatch"), anchors=True)
        s = fmt["schema"]
        self.assertEqual((fmt["name"], s["properties"]["parts"]["maxItems"]), ("campaign_v3", 3))
        part = s["properties"]["parts"]["items"]["properties"]
        self.assertEqual(part["rule"]["enum"][-1], "lowx1")
        self.assertIn("anchor", part)
        self.assertEqual(s["properties"]["claim"]["properties"]["class"]["enum"], ["alpha", "mismatch"])
        self.assertEqual(s["properties"]["test"]["properties"]["metric"]["enum"][:4],
                         ["mismatches", "mismatch_rate", "surprises", "max_novelty"])
        self.assertEqual(set(s["required"]), set(s["properties"]))
        bare = hschema.campaign_format_v3(sp)["schema"]
        self.assertNotIn("claim", bare["properties"])
        self.assertNotIn("anchor", bare["properties"]["parts"]["items"]["properties"])
        self.assertNotIn("note", bare["properties"])

    def test_the_parser_clips_and_refuses(self):
        sp = SignalSynthetic().region_space
        parse = lambda data, **kw: hschema.parse_v3(sp, data, budget_range=(2, 6), max_parts=2,  # noqa: E731
                                                     claim_classes=("alpha", "mismatch"), **kw)
        d = parse({"parts": [{"region": {"x1": [0.3, 0.1]}, "rule": "telepathy", "n": 9, "anchor": "s2"},
                             {"region": {}, "rule": "novelty", "n": 0, "anchor": "nonsense"},
                             {"region": {}, "rule": "adko", "n": 1}],
                   "test": {"metric": "mismatch_rate", "op": ">=", "threshold": 0.3},
                   "claim": {"part": 1, "class": "mismatch"}, "p_success": 7, "workspace": {"w": 1}})
        self.assertEqual([(p["rule"], p["n"], p["anchor"]) for p in d["parts"]],
                         [("explore", 5, "s2"), ("novelty", 1, None)])
        self.assertEqual(d["parts"][0]["region"].bounds, {"x1": (0.1, 0.3)})
        self.assertEqual((d["budget"], d["rule"], d["p_success"], d["claim"], d["workspace"]),
                         (6, "explore", 1.0, {"part": 1, "class": "mismatch"}, {"w": 1}))
        self.assertEqual(d["test"].label(), "mismatch_rate >= 0.3")
        self.assertIsNone(parse({"parts": [{"region": {}, "rule": "x", "n": 1}],
                                 "claim": {"part": 5, "class": "alpha"}}))  # nothing said at all
        self.assertIsNone(parse({"region": {"x1": [0, 1]}}))  # a legacy answer is not a v3 one
        self.assertIsNone(parse("nope"))
        kept = parse({"parts": [{"region": {"x1": [0, 0.5]}, "rule": "adko", "n": 2}],
                      "claim": {"part": 1, "class": "alpha"}})
        self.assertIsNone(kept["claim"])  # names a part that does not exist

    def test_the_mock_answers_what_the_schema_asks(self):
        import asyncio

        from appfl.decentralized.algorithm.adko_h.llm.policy import CampaignPolicy, MockClient, _json

        a = self.agent(parts=True, anchors=True, class_claims=True, extra_rules=("lowx1",))
        a.set_campaign(design(a.space), 0)
        step(a, 0, SURPRISE)
        sp = a.space
        pol = CampaignPolicy(MockClient(sp), sp, prompt="v3", budget_range=(2, 6))
        for rnd in range(1, 9):
            ctx = a.plan_context(rnd)
            resp = asyncio.run(pol.client.complete(pol._request(ctx), ctx))
            raw = _json(resp["text"])
            d = pol._parse_v3(raw, ctx)
            self.assertIsNotNone(d)
            self.assertTrue(set(raw) >= {"parts", "claim", "test"})
            self.assertEqual(raw["parts"][0]["anchor"], "s0")
            self.assertTrue(2 <= d["budget"] <= 6)
            self.assertTrue(all(p["rule"] in hrules.RULES + ("lowx1",) for p in d["parts"]))
        pol.close()

    def test_a_v3_run_with_everything_on_is_a_run_and_reproduces(self):
        import json
        import tempfile

        def go():
            problem = SignalSynthetic(seed=0, fns={"lowx1": lambda v: -v.cands.F[:, 0]})
            planner = build_planner(problem.region_space, "mock", seed=0, prompt="v3", budget_range=(2, 6))
            cfg = RunConfig(
                rounds=24, agents=3, seed=0, topology="random_regular", topology_kwargs={"k": 2, "seed": 0},
                planner="llm", start_points=6,
                agent_cfg=AgentConfig(warmup_rounds=2, pool_size=48, budget_range=(2, 6), parts=True,
                                      replan_on_surprise=True, followup=2, anchors=True, class_claims=True,
                                      extra_rules=("lowx1",), prompt="v3", share_signals=True,
                                      claim_weight=0.0),
            )
            with tempfile.TemporaryDirectory() as d:
                rec = run(problem, GP, cfg, planner=planner, out=d)
                agents = json.loads((Path(d) / "run.json").read_text())["agents"]
            planner.close()
            return rec, agents

        rec, agents = go()
        self.assertGreater(rec["campaigns"]["n"], 0)
        self.assertEqual(rec["endpoints"]["redundancy"], 0)
        self.assertTrue(any(a["tickets"] for a in agents))
        self.assertTrue(any(c.get("parts") for a in agents for c in a["campaigns"]))
        self.assertTrue(any(s["source"] != "mine" for a in agents for s in a["surprises"]))
        rules = {r["rule"] for a in agents for r in a["records"]}
        self.assertIn("followup", rules)
        again, agents2 = go()
        self.assertEqual(rec["endpoints"], again["endpoints"])
        self.assertEqual([a["records"] for a in agents], [a["records"] for a in agents2])


class TestEvents(unittest.TestCase):
    """A11: what happened, drained per agent for the ledger."""

    def test_every_event_type_with_its_keys(self):
        a = agency(followup=1, class_claims=True, parts=True, share_signals=True, events=True,
                   problem=SignalSynthetic(seed=0, fns={"boom": lambda v: 1 / 0}), extra_rules=("boom",))
        a.set_campaign(parts_design(a.space, [({}, "boom", 2)], claim={"part": 0, "class": "mismatch"}), 0)
        k0 = step(a, 0, PENDING)[0].key
        step(a, 1, SURPRISE)  # closes the campaign (2 picks) and opens a ticket
        a.observe([encode(agent_id="Agent1", source=1, round=1, x_shared=[0.1] * 5, u=0.5, history=[0.5],
                          quantizer=Quantizer(8))])
        step(a, 2, MATCH)  # the ticket's draw: not reproduced
        a.revise(k0, Outcome(score=0.4, signals=MATCH), 3)
        events = a.drain_events()
        self.assertEqual(a.drain_events(), [])
        self.assertTrue(all({"type", "agent", "round"} <= set(e) for e in events))
        by = {}
        for e in events:
            by.setdefault(e["type"], e)
        self.assertEqual(set(by), {"campaign_open", "pool", "rule_error", "token_sent", "campaign_verdict",
                                   "claim_status", "ticket_open", "token_received", "ticket_close",
                                   "label_revision"})
        self.assertTrue({"parts", "claim", "test", "budget"} <= set(by["campaign_open"]))
        self.assertTrue({"verdict", "brier", "early"} <= set(by["campaign_verdict"]))
        self.assertEqual((by["claim_status"]["status"], by["claim_status"]["class"]), ("supported", "mismatch"))
        self.assertTrue({"top", "chosen", "rule", "part"} <= set(by["pool"]))
        self.assertTrue({"key", "value", "mu", "sigma", "in_region"} <= set(by["pool"]["chosen"]))
        self.assertLessEqual(len(by["pool"]["top"]), 8)
        self.assertTrue({"bits", "family", "kind", "surprise", "claims", "verdicts", "relayed"}
                        <= set(by["token_sent"]))
        self.assertTrue({"sender", "accepted", "skipped"} <= set(by["token_received"]))
        self.assertEqual(by["ticket_close"]["status"], "not_reproduced")


if __name__ == "__main__":
    unittest.main()
