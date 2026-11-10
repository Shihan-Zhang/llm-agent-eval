"""E2 阶段 4:限速、重试、调用包装、Outcome Monitor 补丁与编排逻辑。假时钟、假客户端、假 agent,不联网。"""

import json
import os
import random
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import contracts as C  # noqa: E402
import ratelimit as RL  # noqa: E402
import run_e2 as E  # noqa: E402
import toolmaze_run as T  # noqa: E402


# ---------------------------------------------------------------- 令牌桶
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_bucket_waits_until_enough_tokens():
    clk = FakeClock()
    b = RL.TokenBucket(RL.Limits(tpm=60_000, rpm=1_000, headroom=1.0), clock=clk, sleep=clk.sleep)
    assert b.acquire(50_000) == 0                       # 桶满时不等
    waited = b.acquire(20_000)                          # 剩 10,000,每秒回 1,000 → 约等 10 秒
    assert waited == pytest.approx(10, abs=0.05)


def test_bucket_respects_request_limit():
    clk = FakeClock()
    b = RL.TokenBucket(RL.Limits(tpm=10**9, rpm=60, headroom=1.0), clock=clk, sleep=clk.sleep)
    for _ in range(60):
        assert b.acquire(1) == 0
    assert b.acquire(1) == pytest.approx(1, abs=0.05)   # 每秒回 1 次请求


def test_bucket_settle_refunds_and_charges():
    clk = FakeClock()
    b = RL.TokenBucket(RL.Limits(tpm=60_000, rpm=1_000, headroom=1.0), clock=clk, sleep=clk.sleep)
    b.acquire(30_000)
    b.settle(30_000, 10_000)                            # 估多了,退回 20,000
    assert b.tok == pytest.approx(50_000)
    b.settle(0, 70_000)                                 # 估少了,补扣,可以为负
    assert b.tok == pytest.approx(-20_000)
    assert b.acquire(1_000) == pytest.approx(21, abs=0.05)


def test_headroom_shrinks_capacity():
    b = RL.TokenBucket(RL.Limits(tpm=40_000, rpm=1_000, headroom=0.9))
    assert b.cap_tok == pytest.approx(36_000) and b.cap_req == pytest.approx(900)


def test_estimator_uses_conservative_prior_then_calibrates():
    e = RL.TokenEstimator(chars_per_token=2.5, output_ratio=0.05)
    assert e.estimate(2_500, 16_000) == 1_000 + 800
    e.observe(12_000, 1_000)                            # 样本太少,仍用初值
    assert e.cpt == 2.5
    e.observe(12_000, 2_000)                            # 累计 24,000 字符 / 3,000 token = 8,封顶 2 × 2.5
    assert e.cpt == 5.0


# ---------------------------------------------------------------- 重试
class Boom(Exception):
    def __init__(self, status=None):
        super().__init__("boom")
        self.status_code = status


class RateLimitError(Exception):
    pass


def test_transient_errors_are_retried_with_capped_exponential_backoff():
    sleeps, n = [], {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] < 4:
            raise RateLimitError()
        return "ok"

    assert T.call_with_retry(fn, 8, 5, 12, sleep=sleeps.append, rng=random.Random(0)) == "ok"
    assert len(sleeps) == 3
    for s, full in zip(sleeps, [5, 10, 12]):
        assert full * 0.5 <= s <= full


def test_non_transient_errors_are_not_retried():
    sleeps = []
    with pytest.raises(Boom):
        T.call_with_retry(lambda: (_ for _ in ()).throw(Boom(400)), 8, 1, 1, sleep=sleeps.append)
    assert sleeps == []
    assert T.is_transient(Boom(429)) and T.is_transient(Boom(503)) and not T.is_transient(Boom(400))


def test_retry_gives_up_after_max_attempts():
    sleeps = []
    with pytest.raises(Boom):
        T.call_with_retry(lambda: (_ for _ in ()).throw(Boom(429)), 3, 1, 1, sleep=sleeps.append)
    assert len(sleeps) == 2


# ---------------------------------------------------------------- 调用包装
def _resp(pt=100, ct=10, finish="stop"):
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, completion_tokens_details=None,
                            prompt_tokens_details=None)
    msg = SimpleNamespace(reasoning_content=None)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg, finish_reason=finish)])


class FakeLimiter:
    def __init__(self):
        self.calls = []

    def acquire(self, n):
        self.calls.append(("acquire", n))
        return 0.0

    def settle(self, est, actual):
        self.calls.append(("settle", est, actual))


def test_wrapper_sets_timeout_extra_body_logs_and_counts(tmp_path):
    seen = {}

    def create(**kw):
        seen.update(kw)
        return _resp(120, 8, "length")

    log = tmp_path / "calls.jsonl"
    lim, est = FakeLimiter(), RL.TokenEstimator(2.0, 0.1)
    w = T.make_create_wrapper(create, timeout=300, extra_body={"enable_thinking": False}, log_path=str(log),
                              limiter=lim, estimator=est, retry={"max_attempts": 2, "base_s": 0, "cap_s": 0})
    T.EPISODE.labels, T.EPISODE.stats = {"pair": "C3_task_001_P3", "arm": "full"}, T.new_stats()
    w(model="m", messages=[{"role": "user", "content": "x" * 10}], max_tokens=100)
    assert seen["timeout"] == 300 and seen["extra_body"] == {"enable_thinking": False}
    st = T.EPISODE.stats
    assert (st["calls"], st["prompt_tokens"], st["completion_tokens"], st["length_finishes"]) == (1, 120, 8, 1)
    assert lim.calls[0][0] == "acquire" and lim.calls[1] == ("settle", lim.calls[0][1], 128)
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["pair"] == "C3_task_001_P3" and rec["arm"] == "full" and rec["ok"]


def test_wrapper_counts_api_failure_after_retries_exhausted():
    def create(**kw):
        raise Boom(429)

    w = T.make_create_wrapper(create, timeout=1, extra_body=None, log_path=None, limiter=None, estimator=None,
                              retry={"max_attempts": 2, "base_s": 0, "cap_s": 0})
    T.EPISODE.labels, T.EPISODE.stats = {}, T.new_stats()
    with pytest.raises(Boom):
        w(model="m", messages=[])
    assert (T.EPISODE.stats["attempts"], T.EPISODE.stats["retries"], T.EPISODE.stats["api_failures"]) == (2, 1, 1)


# ---------------------------------------------------------------- Outcome Monitor 补丁
def test_call_arguments_parses_string_dict_and_bad_json():
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": '{"id": "F1"}'}},
                                                 {"function": {"name": "b", "arguments": {"x": 1}}}]}]
    assert T.call_arguments(msgs, 0) == {"id": "F1"}
    assert T.call_arguments(msgs, 1) == {"x": 1}
    assert T.call_arguments(msgs, 9) == {"id": "F1"}
    assert T.call_arguments([{"role": "assistant", "tool_calls": [{"function": {"arguments": "{bad"}}]}]) == {}


def _agent_cls():
    class FakeAgent:
        def __init__(self):
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: _resp())))
            self.messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "quote",
                                                                                 "arguments": '{"id": "S2"}'}}]}]
            self.tools = [{"type": "function", "function": {"name": n}} for n in ("quote", "quote_alt", "other")]
            self.received = []

        def receive_tool_result(self, tool_name, result, tool_call_index=0):
            self.received.append(result)
    return FakeAgent


def _contract():
    rows = [{"price": 10.0 * i, "id": f"S{i}"} for i in (1, 2, 3, 4)]
    return C.learn([C.Sample(f"w{i}", "quote", {"id": r["id"]}, r) for i, r in enumerate(rows)])["quote"]


def test_monitor_off_passes_the_same_object_on_wraps_with_visible_tools(tmp_path):
    cls = _agent_cls()
    T.apply_patches({"patch": {}}, str(tmp_path / "cfg.yaml"), agent_cls=cls)
    T.apply_patches({"patch": {}}, str(tmp_path / "cfg.yaml"), agent_cls=cls)      # 只打一次
    agent = cls()
    bad = {"price": -1.0, "id": "S2"}
    T.EPISODE.monitor, T.EPISODE.stats, T.EPISODE.labels = None, T.new_stats(), {"pair": "p", "arm": "baseline"}
    agent.receive_tool_result("quote", bad)
    assert agent.received[-1] is bad
    log = tmp_path / "receipts.jsonl"
    T.EPISODE.monitor = {"registry": {"quote": _contract()}, "variant": "full", "log": str(log),
                         "subs": {"quote": ("quote_alt", "quote_hidden")}}
    T.EPISODE.labels = {"pair": "p", "arm": "full"}
    agent.receive_tool_result("quote", bad)
    view = agent.received[-1]
    assert view["tool_result"] is bad
    assert view["outcome_contract"]["admissible_recovery_tools"] == ["quote", "quote_alt"]
    assert T.EPISODE.stats["receipts"] == 1
    assert json.loads(log.read_text(encoding="utf-8"))["violations"][0]["property"] == "nonpositive_value"
    T.EPISODE.monitor = None


def test_monitor_setting_is_thread_local(tmp_path):
    cls = _agent_cls()
    T.apply_patches({"patch": {}}, str(tmp_path / "cfg.yaml"), agent_cls=cls)
    T.EPISODE.monitor = {"registry": {"quote": _contract()}, "variant": "full", "log": None, "subs": {}}
    out = {}

    def other():
        a = cls()
        r = {"price": -1.0, "id": "S2"}
        a.receive_tool_result("quote", r)
        out["same"] = a.received[-1] is r

    th = threading.Thread(target=other)
    th.start()
    th.join()
    T.EPISODE.monitor = None
    assert out["same"]                                   # 另一个线程没有设置 monitor,不受影响


def test_patched_init_wraps_client_create(tmp_path):
    cls = _agent_cls()
    T.apply_patches({"patch": {"request_timeout_s": 7}}, str(tmp_path / "c.yaml"), agent_cls=cls)
    a = cls()
    T.EPISODE.stats, T.EPISODE.labels = T.new_stats(), {}
    a.client.chat.completions.create(model="m", messages=[])
    assert T.EPISODE.stats["calls"] == 1


BARE_PARAMS = {"type": "object", "properties": {"alert_card": {"type": "object", "description": "card"},
                                                "title": {"type": "string"}}, "required": ["alert_card", "title"]}


def _initializing_agent_cls():
    class InitAgent(_agent_cls()):
        def initialize(self, task_description, tool_definitions):
            self.tools = [{"type": "function", "function": {"name": d, "parameters": BARE_PARAMS}} for d in tool_definitions]
    return InitAgent


@pytest.mark.parametrize("flag,expected", [(True, True), (False, None)])
def test_bare_object_patch_adds_additional_properties_after_initialize(tmp_path, flag, expected):
    # 硅基流动上无属性 object 参数被压成 {}(schema_probe/results/summary.md);补丁只在配置打开时生效
    cls = _initializing_agent_cls()
    T.apply_patches({"patch": {"bare_object_additional_properties": flag}}, str(tmp_path / "c.yaml"), agent_cls=cls)
    a = cls()
    a.initialize("task", ["archive"])
    card = a.tools[0]["function"]["parameters"]["properties"]["alert_card"]
    assert card.get("additionalProperties") is expected
    assert a.tools[0]["function"]["parameters"]["properties"]["title"] == {"type": "string"}
    assert BARE_PARAMS["properties"]["alert_card"] == {"type": "object", "description": "card"}   # 不改上游的定义对象


# ---------------------------------------------------------------- 编排
MANIFEST = {"main": [{"tier": "c3", "task_id": "C3_task_001", "mode": "P3"},
                     {"tier": "c4", "task_id": "C4_task_002", "mode": "P4"}],
            "c1_supplement": [{"tier": "c1", "task_id": "C1_task_001", "mode": "P3"}]}


def test_plan_order_is_seeded_and_keeps_both_arms():
    pairs = [E.Pair("c3", f"C3_task_{i:03d}", "P3") for i in range(20)]
    p1, p2 = E.plan_order(pairs, ["baseline", "full"], 7), E.plan_order(pairs, ["baseline", "full"], 7)
    assert p1 == p2 and sorted(p.key for p, _ in p1) == sorted(p.key for p in pairs)
    assert all(sorted(a) == ["baseline", "full"] for _, a in p1)
    assert len({tuple(a) for _, a in p1}) == 2                       # 组内先后确实被打乱
    assert [p for p, _ in p1] != pairs


def test_pairs_for_groups_and_pilot_excludes_manifest():
    assert [p.mode for p in E.pairs_for("p0", MANIFEST)] == ["P0", "P0"]
    assert [p.key for p in E.pairs_for("main", MANIFEST)] == ["C3_task_001_P3", "C4_task_002_P4"]
    avail = {"c3": [f"C3_task_{i:03d}" for i in range(1, 10)], "c4": [f"C4_task_{i:03d}" for i in range(1, 10)]}
    pilot = E.pairs_for("pilot", MANIFEST, avail, n=5, seed=3)
    assert len(pilot) == 5 and len({p.task_id for p in pilot}) == 5
    assert not {p.task_id for p in pilot} & {"C3_task_001", "C4_task_002"}
    assert [(p.tier, p.mode) for p in pilot][:4] == [("c3", "P3"), ("c3", "P4"), ("c4", "P3"), ("c4", "P4")]


def _rec(pair, arm, attempt, fail=0):
    return {"pair": pair, "arm": arm, "attempt": attempt, "api_failures": fail}


def test_pair_state_and_rerun_counting():
    arms = ["baseline", "full"]
    assert E.pair_state([], "x", arms) == ("todo", 0)
    recs = [_rec("x", "baseline", 1), _rec("x", "full", 1, fail=1)]
    assert E.pair_state(recs, "x", arms) == ("retry", 1)              # 有 API 错误 → 整对重跑
    recs += [_rec("x", "full", 2)]
    assert E.pair_state(recs, "x", arms) == ("retry", 2)              # 只有一组完成(中断)→ 重跑
    recs += [_rec("x", "baseline", 2)]
    assert E.pair_state(recs, "x", arms) == ("done", 2)
    assert E.pair_reruns(recs + [_rec("y", "baseline", 1), _rec("y", "full", 1)]) == 1


def test_estimate_extrapolates_tokens_time_cost_and_context():
    def r(arm, pt, ct, calls, lat, mx):
        return {"pair": "p", "arm": arm, "attempt": 1, "prompt_tokens": pt, "completion_tokens": ct, "calls": calls,
                "latency_s": lat, "max_prompt_tokens": mx, "receipts": 0}
    recs = [r("baseline", 10_000, 500, 10, 60, 3_000), r("full", 12_000, 500, 12, 72, 7_000)]
    est = E.estimate(recs, 80, ["baseline", "full"], 4, {"tpm": 40_000, "rpm": 1_000, "headroom": 0.9},
                     {"input": 0.24, "output": 1.80}, num_ctx=8192, max_tokens=2048)
    assert est["total_prompt_tokens"] == 80 * 22_000 and est["total_completion_tokens"] == 80 * 1_000
    assert est["rate_limit_bound_minutes"] == pytest.approx(80 * 23_000 / 36_000, abs=0.05)
    assert est["latency_bound_minutes"] == pytest.approx(80 * 132 / 4 / 60, abs=0.05)
    assert est["estimated_minutes"] == max(est["rate_limit_bound_minutes"], est["latency_bound_minutes"])
    assert est["estimated_cost_usd"] == pytest.approx(80 * 22_000 / 1e6 * 0.24 + 80_000 / 1e6 * 1.80, abs=0.01)
    assert est["prompt_budget"] == 6144 and est["fits_context"] is False


@pytest.mark.parametrize("name", ["e2_api.yaml", "e2_api_v2.yaml", "e2_local.yaml"])
def test_real_configs_have_valid_arms_and_groups(name):
    cfg = E.load_cfg(os.path.join(ROOT, "eval", "agent_error", "configs", name))
    for run, spec in cfg["e2"]["runs"].items():
        variants = {a: E.arm_variant(v) for a, v in spec["arms"].items()}
        assert set(variants.values()) <= {None, "full", "stripped"}, run
        assert spec["episodes"] in {"main", "main_ext", "ext", "p0", "c1", "pilot"}, run
    assert "main" in cfg["e2"]["runs"]
    assert cfg["agent"]["temperature"] == 0


def test_v2_config_differs_from_old_only_in_schema_patch_and_output():
    old, new = (E.load_cfg(os.path.join(ROOT, "eval", "agent_error", "configs", n)) for n in ("e2_api.yaml", "e2_api_v2.yaml"))
    assert new["patch"].pop("bare_object_additional_properties") is True and "bare_object_additional_properties" not in old["patch"]
    assert new["e2"]["results_dir"] != old["e2"]["results_dir"]
    assert new["e2"].pop("pilot_results_dir") == old["e2"]["results_dir"] and new["e2"].pop("tag") == "api_v2"
    for d in (old, new):
        d["e2"].pop("results_dir")
    old["e2"]["runs"].pop("pilot")
    # ① v2 补充:补充清单 120 对,与 main 同样的两组,结果写在单独的 main_supp 目录
    assert new["e2"]["runs"].pop("main_supp") == {"episodes": "ext", "arms": new["e2"]["runs"]["main"]["arms"]}
    assert new == old


def test_arm_variant_accepts_yaml_false_and_rejects_unknown():
    assert E.arm_variant(False) is None and E.arm_variant("off") is None and E.arm_variant("full") == "full"
    with pytest.raises(ValueError):
        E.arm_variant("always_warn")


def test_latest_records_keeps_last_attempt():
    recs = [_rec("x", "full", 1, fail=1), _rec("x", "full", 2)]
    assert E.latest_records(recs) == [recs[1]]
