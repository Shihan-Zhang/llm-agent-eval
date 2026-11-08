"""E2 契约挖掘与检查(contracts.py)和回合清单(e2_manifest.py):不读真实数据、不调用模型。"""

import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import contracts as C  # noqa: E402
import e2_manifest as M  # noqa: E402

README = os.path.join(ROOT, "eval", "agent_error", "README.md")


def _readme_hash(name):
    with open(README, encoding="utf-8") as fh:
        return re.search(rf"{name}: ([0-9a-f]{{64}})", fh.read()).group(1)


def S(result, args=None, tool="t", wf="w1"):
    return C.Sample(wf, tool, args or {}, result)


def learn_one(results, args=None):
    args = args or [{}] * len(results)
    return C.learn([S(r, a, wf=f"w{i}") for i, (r, a) in enumerate(zip(results, args))])["t"]


def props(violations):
    return [v.property for v in violations]


# ---------------------------------------------------------------- 冻结规格
def test_spec_hash_matches_readme():
    assert C.SPEC_SHA256 == _readme_hash("SPEC_SHA256")


def test_manifest_file_matches_readme_hash():
    with open(M.OUT, encoding="utf-8") as fh:
        assert M.digest(json.load(fh)) == _readme_hash("MANIFEST_SHA256")


# ---------------------------------------------------------------- 挖掘:各类门槛
def test_echo_needs_two_equal_pairs():
    c = learn_one([{"id": "A"}, {"id": "B"}], [{"id": "A"}, {"id": "B"}])
    assert c.echo == ("id",)
    one_pair = C.learn([S({"id": "A"}, {"id": "A"}), S({"id": "B"})])["t"]
    assert one_pair.echo == ()
    mismatch = learn_one([{"id": "A"}, {"id": "X"}], [{"id": "A"}, {"id": "B"}])
    assert mismatch.echo == ()


def test_positivity_needs_three_positive_numbers():
    assert learn_one([{"p": 1.5}, {"p": 2}, {"p": 3}]).positive == ("p",)
    assert learn_one([{"p": 1}, {"p": 2}]).positive == ()
    assert learn_one([{"p": 1}, {"p": 0}, {"p": 3}]).positive == ()
    assert learn_one([{"p": True}, {"p": True}, {"p": True}]).positive == ()   # 布尔不算数值


def test_domain_needs_lexicon_name_four_samples_and_at_most_eight_values():
    c = learn_one([{"status": s} for s in ["OK", "ok ", "Delayed", "ok"]])
    assert c.domains == {"status": ("delayed", "ok")}
    assert learn_one([{"note": "a"}] * 4).domains == {}                       # 名字不在词表
    assert learn_one([{"seat_class": "eco"}] * 3).domains == {}               # 样本不足 4 个
    many = learn_one([{"status": f"s{i}"} for i in range(9)])
    assert many.domains == {}                                                 # 9 个不同值


def test_affine_fit_and_rejections():
    c = learn_one([{"n": n, "total": 2.5 * n + 1} for n in (1, 2, 3, 4)])
    (fx, fy, a, b), = c.affine
    assert (fx, fy) == ("n", "total") and a == pytest.approx(2.5) and b == pytest.approx(1)
    assert learn_one([{"n": n, "sq": n * n} for n in (1, 2, 3, 4)]).affine == ()      # 非线性
    assert learn_one([{"n": n, "k": 7} for n in (1, 2, 3, 4)]).affine == ()          # 常数列不足 3 个不同值
    assert learn_one([{"n": n, "t": 2 * n} for n in (1, 2, 3)]).affine == ()         # 样本不足 4 个


def test_order_on_fixed_date_pairs_only():
    ok = [{"check_in_date": "2025-07-01", "check_out_date": "2025-07-05"},
          {"check_in_date": "2025-08-01", "check_out_date": "2025-08-02"}]
    assert learn_one(ok).order == (("check_in_date", "check_out_date"),)
    bad = ok + [{"check_in_date": "2025-09-03", "check_out_date": "2025-09-01"}]
    assert learn_one(bad).order == ()
    other = [{"from_date": "2025-07-01", "to_date": "2025-07-05"}] * 2
    assert learn_one(other).order == ()                                       # 不在三组日期对里
    nested = learn_one([{"slot": {"start_time": "09:00", "end_time": "10:00"}}] * 2)
    assert nested.order == (("slot.start_time", "slot.end_time"),)


def test_positivity_and_domain_only_on_required_fields():
    c = learn_one([{"p": 1, "status": "a"}, {"p": 2, "status": "a"}, {"status": "a"}, {"status": "a"}])
    assert "p" not in c.required and c.positive == ()
    assert c.domains == {"status": ("a",)}


def test_tool_with_single_sample_gets_no_contract():
    assert C.learn([S({"a": 1})]) == {}


# ---------------------------------------------------------------- 运行时检查:8 类 + order
@pytest.fixture
def contract():
    rows = [{"id": f"F{i}", "status": "ok", "price": 10.0 * i, "qty": i, "total": 3 * i,
             "check_in_date": "2025-07-01", "check_out_date": "2025-07-05"} for i in (1, 2, 3, 4)]
    args = [{"id": f"F{i}"} for i in (1, 2, 3, 4)]
    return learn_one(rows, args)


def clean(i=2):
    return {"id": f"F{i}", "status": "ok", "price": 10.0 * i, "qty": i, "total": 3 * i,
            "check_in_date": "2025-07-01", "check_out_date": "2025-07-05"}


def test_clean_result_has_no_violation(contract):
    assert C.check(contract, {"id": "F2"}, clean()) == []


def test_no_contract_or_non_dict_result_is_unchecked(contract):
    assert C.check(None, {}, {"anything": 1}) == []
    assert C.check(contract, {}, "plain text") == []


@pytest.mark.parametrize("mutate, prop", [
    (lambda r: r.pop("status"), "missing_learned_field"),
    (lambda r: r.update(extra=1), "unexpected_field"),
    (lambda r: r.update(qty="2"), "learned_type_mismatch"),
    (lambda r: r.update(id="F9"), "learned_echo_mismatch"),
    (lambda r: r.update(price=-5.0), "nonpositive_value"),
    (lambda r: r.update(status="cancelled"), "unseen_category"),
    (lambda r: r.update(total=999), "affine_relation_broken"),
    (lambda r: r.update(check_in_date="2025-07-09"), "temporal_order_violated"),
])
def test_each_violation_class(contract, mutate, prop):
    r = clean()
    mutate(r)
    assert prop in props(C.check(contract, {"id": "F2"}, r))


def test_explicit_error_field_and_witness_values(contract):
    r = clean()
    r["error"] = "Service Temporarily Unavailable"
    v = C.check(contract, {"id": "F2"}, r)
    assert props(v)[:2] == ["unexpected_field", "explicit_error"]
    r2 = clean()
    r2["price"] = -1.0
    w, = [x for x in C.check(contract, {"id": "F2"}, r2) if x.property == "nonpositive_value"]
    assert (w.path, w.expected, w.observed) == ("price", "> 0", -1.0)


def test_integer_passes_number_type_but_not_vice_versa():
    c = learn_one([{"x": 1.5}, {"x": 2.5}])
    assert C.check(c, {}, {"x": 3}) == []
    c2 = learn_one([{"x": 1}, {"x": 2}])
    assert props(C.check(c2, {}, {"x": 2.5})) == ["learned_type_mismatch"]


def test_affine_runtime_tolerance_is_looser_than_fit():
    c = learn_one([{"n": n, "t": 2.0 * n} for n in (1, 2, 3, 4)])
    assert C.check(c, {}, {"n": 5, "t": 10 + 1e-7}) == []            # 运行时容差 1e-5
    assert props(C.check(c, {}, {"n": 5, "t": 10.1})) == ["affine_relation_broken"]


def test_contract_json_round_trip(contract):
    assert C.ToolContract.from_json(json.loads(json.dumps(contract.to_json()))) == contract


# ---------------------------------------------------------------- 分折与交叉拟合
def test_folds_partition_tasks_deterministically():
    ids = [f"C1_task_{i:03d}" for i in range(1, 101)]
    f1, f2 = C.assign_folds(ids), C.assign_folds(list(reversed(ids)))
    assert f1 == f2 and set(f1) == set(ids)
    assert sorted(set(f1.values())) == [0, 1, 2, 3, 4]
    assert all(list(f1.values()).count(k) == 20 for k in range(5))


def test_templates_must_not_span_folds():
    C.check_templates_disjoint({"a": 0, "b": 1}, {"a": "T1", "b": "T2"})
    with pytest.raises(ValueError):
        C.check_templates_disjoint({"a": 0, "b": 1}, {"a": "T1", "b": "T1"})


def test_crossfit_registry_never_sees_its_own_fold():
    folds = {f"w{i}": i % 5 for i in range(10)}
    samples = [S({"v": 1}, tool="common", wf=w) for w in folds] + [S({"v": 2}, tool="common", wf=w) for w in folds]
    samples += [S({"s": 1}, tool="sentinel", wf="w3"), S({"s": 2}, tool="sentinel", wf="w3")]   # 只在第 3 折
    regs = C.crossfit(samples, folds)
    assert "sentinel" not in regs[3]
    assert all("sentinel" in regs[k] for k in (0, 1, 2, 4))
    assert C.registry_for(regs, folds, "w3") is regs[3]
    assert C.registry_for(regs, folds, "w8") is regs[3]


def test_crossfit_rejects_samples_without_fold():
    with pytest.raises(ValueError):
        C.crossfit([S({"a": 1}, wf="unknown")], {"w1": 0})


# ---------------------------------------------------------------- 读入:只用 P0、去掉错误与重复
def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def test_load_gold_reads_only_p0_and_dedupes_shared_steps(tmp_path):
    step = {"tool_name": "get", "arguments": {"id": 1}, "output": {"id": 1}, "status": "success"}
    err = {"tool_name": "get", "arguments": {"id": 2}, "output": {"error": "boom"}, "status": "success"}
    _write(tmp_path / "data/perturbed_tasks/c1/C1_task_001_P0.json",
           {"task_id": "C1_task_001", "template_id": "T1", "perturbation_mode": "P0", "execution_trace": [step, err]})
    _write(tmp_path / "data/perturbed_tasks/c1/C1_task_001_P3.json",
           {"task_id": "C1_task_001", "perturbation_mode": "P3",
            "execution_trace": [{**step, "output": {"id": 999}, "is_perturbed": True}]})
    _write(tmp_path / "data/perturbed_tasks/c3/C3_task_001_P0.json",
           {"task_id": "C3_task_001", "template_id": "T3", "perturbation_mode": "P0",
            "valid_paths": [{"execution_trace": [step]}, {"execution_trace": [step]}]})
    samples, templates = C.load_gold(str(tmp_path), tiers=("c1", "c3"))
    assert [(s.workflow, s.result) for s in samples] == [("C1_task_001", {"id": 1}), ("C3_task_001", {"id": 1})]
    assert templates == {"C1_task_001": "T1", "C3_task_001": "T3"}


def test_load_e1_keeps_only_clean_results_paired_by_call_id(tmp_path):
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "type": "tool_call", "tool_call": {"id": "c1", "name": "get", "arguments": {"id": 1}}},
            {"role": "tool", "name": "get", "call_id": "c1", "content": {"id": 1},
             "metadata": {"perturbation_status": "clean"}},
            {"role": "assistant", "type": "tool_call", "tool_call": {"id": "c2", "name": "get", "arguments": {"id": 2}}},
            {"role": "tool", "name": "get", "call_id": "c2", "content": {"error": "x"},
             "metadata": {"perturbation_status": "clean"}},
            {"role": "assistant", "type": "tool_call", "tool_call": {"id": "c3", "name": "get", "arguments": {"id": 3}}},
            {"role": "tool", "name": "get", "call_id": "c3", "content": {"id": 3},
             "metadata": {"perturbation_status": "perturbed"}}]
    _write(tmp_path / "C1_task_001_P0_inference.json", {"task_id": "C1_task_001", "mode": "P0", "messages": msgs})
    _write(tmp_path / "C1_task_001_P3_inference.json", {"task_id": "C1_task_001", "mode": "P3", "messages": msgs})
    samples = C.load_e1(str(tmp_path))
    assert [(s.args, s.result) for s in samples] == [({"id": 1}, {"id": 1})]


# ---------------------------------------------------------------- 回合清单
def _fake_tiers(root, n=45):
    for tier in ("c1", "c3", "c4"):
        for i in range(1, n + 1):
            for m in ("P0", "P3", "P4"):
                _write(root / f"data/perturbed_tasks/{tier}/{tier.upper()}_task_{i:03d}_{m}.json", {})


def test_manifest_strata_unique_and_deterministic(tmp_path):
    _fake_tiers(tmp_path)
    m1, m2 = M.build(str(tmp_path)), M.build(str(tmp_path))
    assert m1 == m2
    main = m1["main"]
    assert len(main) == 80
    for tier in ("c3", "c4"):
        for mode in ("P3", "P4"):
            assert sum(1 for e in main if e["tier"] == tier and e["mode"] == mode) == 20
    assert len({e["task_id"] for e in main}) == 80                  # 80 个互不相同的任务
    assert len(m1["c1_supplement"]) == 90


def test_manifest_refuses_too_few_tasks(tmp_path):
    _fake_tiers(tmp_path, n=30)
    with pytest.raises(ValueError):
        M.build(str(tmp_path))
