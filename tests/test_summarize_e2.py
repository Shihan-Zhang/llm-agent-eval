"""E2 汇总(summarize_e2.py):回执三分类、反应分类、回执与轨迹对齐、配对统计、清单过滤与输入超长标记。"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval", "agent_error"))
import contracts as C  # noqa: E402
import run_e2 as E  # noqa: E402
import summarize_e2 as S  # noqa: E402


def call(i, name, args=None):
    return {"role": "assistant", "type": "tool_call", "tool_call": {"id": f"c{i}", "name": name, "arguments": args or {}}}


def res(i, name, content, perturbed=False):
    return {"role": "tool", "name": name, "call_id": f"c{i}", "content": content,
            "metadata": {"perturbation_status": "perturbed" if perturbed else "clean"}}


def test_result_class_three_way():
    assert S.result_class(res(1, "t", {"price": -1}, perturbed=True)) == "injected"
    assert S.result_class(res(1, "t", {"error": "price_data.price_usd is required"})) == "tool_error"
    assert S.result_class(res(1, "t", {"price": 3})) == "normal"
    assert S.result_class(res(1, "t", {"price": 3, "error": ""})) == "normal"          # 空 error 不算


def test_response_retry_switch_other():
    ev = [{"tool": "quote"}, {"tool": "quote"}, {"tool": "quote_alt"}, {"tool": "book"}]
    assert S.response(ev, 0, ["quote", "quote_alt"]) == "retry"
    assert S.response(ev, 1, ["quote", "quote_alt"]) == "switch"
    assert S.response(ev, 2, ["quote_alt"]) == "other"                                 # 下一个工具不在回执列表里
    assert S.response(ev, 3, ["book"]) == "other"                                      # 没有后续调用


def _contract():
    rows = [{"price": float(i)} for i in (1, 2, 3)]
    return C.learn([C.Sample(f"w{i}", "quote", {}, r) for i, r in enumerate(rows)])["quote"]


def test_align_receipts_matches_in_order_and_rejects_mismatch():
    reg = {"quote": _contract()}
    ev = [{"tool": "quote", "args": {}, "result": {"price": 2.0}, "cls": "normal"},
          {"tool": "quote", "args": {}, "result": {"price": -1.0}, "cls": "injected"}]
    rec = {"tool": "quote", "receipt": {}}
    assert S.align_receipts(ev, reg, [rec]) == [(1, rec)]
    with pytest.raises(ValueError):
        S.align_receipts(ev, reg, [])
    with pytest.raises(ValueError):
        S.align_receipts(ev, reg, [{"tool": "other", "receipt": {}}])


def _record(pair, arm, passed, attempt=1, fail=0, mx=1000, tier="c3"):
    return {"pair": pair, "arm": arm, "attempt": attempt, "passed": passed, "api_failures": fail, "tier": tier,
            "max_prompt_tokens": mx, "error_in_answer": False, "harness_error": None}


def test_build_end_to_end_on_synthetic_run():
    pairs = [E.Pair("c3", "C3_task_001", "P3"), E.Pair("c3", "C3_task_002", "P3"), E.Pair("c4", "C4_task_003", "P4")]
    records = [
        _record("C3_task_001_P3", "baseline", False), _record("C3_task_001_P3", "full", True),           # 胜
        _record("C3_task_002_P3", "baseline", True), _record("C3_task_002_P3", "full", True, mx=7000),   # 都过,超长
        _record("C4_task_003_P4", "baseline", False), _record("C4_task_003_P4", "full", False, fail=1),  # API 错误
        _record("C3_task_099_P3", "baseline", False), _record("C3_task_099_P3", "full", True),           # 试跑,不在清单
    ]
    win_traj = {"messages": [call(1, "quote"), res(1, "quote", {"price": -1.0}, perturbed=True),
                             call(2, "quote"), res(2, "quote", {"price": 2.0})]}
    tie_traj = {"messages": [call(1, "quote"), res(1, "quote", {"error": "bad args"}),
                             call(2, "book"), res(2, "book", {"ok": 1})]}
    inferences = {("C3_task_001_P3", "full"): win_traj, ("C3_task_002_P3", "full"): tie_traj}
    receipts = {("C3_task_001_P3", 1): [{"tool": "quote", "receipt": {"admissible_recovery_tools": ["quote", "q2"]}}],
                ("C3_task_002_P3", 1): [{"tool": "quote", "receipt": {"admissible_recovery_tools": ["quote"]}}]}
    folds = {"C3_task_001": 0, "C3_task_002": 1, "C4_task_003": 2}
    regs = [{"quote": _contract()} for _ in range(5)]
    s = S.build(records, pairs, inferences, receipts, (regs, folds), prompt_budget=6144)

    assert s["n_manifest"] == 3 and s["done"] == 2 and s["unresolved"] == ["C4_task_003_P4"]
    assert (s["primary"]["n"], s["primary"]["wins"], s["primary"]["losses"]) == (2, 1, 0)
    assert s["cls_n"] == {"injected": 1, "normal": 2, "tool_error": 1}
    assert s["cls_r"] == {"injected": 1, "tool_error": 1}
    assert s["cls_ge2"]["injected"] == 1 and s["cls_ge2"]["tool_error"] == 0
    assert s["resp_win"]["injected"]["retry"] == 1 and s["resp_all"]["tool_error"]["other"] == 1
    assert s["win_label"] == {"retry": 1}
    assert s["overflow"] == [("C3_task_002_P3", "full", 7000)]
    text = S.render(s, "test")
    assert "C3_task_099" not in text and "胜)1 对" in text


def test_paired_difference_intervals_when_no_losses():
    rows = [(False, True)] * 11 + [(False, False)] * 69
    p = S.paired(rows)
    assert p["diff"] == pytest.approx(11 / 80)
    lo, hi = p["diff_wilson"]
    assert (round(lo, 4), round(hi, 4)) == (0.0785, 0.2297)          # 报告里的 [7.9%, 23.0%]
    # 某边缘为 0 → φ = 0:L = δ − √((p₁−l₁)² + (u₂−p₂)²),U = u₁(基线 0/80 时 p₂ = l₂ = 0)
    l1, u1 = S.wilson(11, 80)
    _, u2 = S.wilson(0, 80)
    nlo, nhi = p["diff_newcombe"]
    assert nlo == pytest.approx(11 / 80 - ((11 / 80 - l1) ** 2 + u2 ** 2) ** 0.5)
    assert nhi == pytest.approx(u1)
    assert nlo < lo                                                   # Newcombe 计入了负的不确定性,更宽


def test_paired_difference_wilson_only_without_losses_and_newcombe_symmetric():
    p = S.paired([(False, True)] * 5 + [(True, False)] * 5 + [(True, True)] * 10 + [(False, False)] * 30)
    assert p["diff"] == 0 and p["diff_wilson"] is None
    lo, hi = p["diff_newcombe"]
    assert lo < 0 < hi and lo == pytest.approx(-hi)


def test_render_shows_paired_difference():
    rows = [(False, True)] * 11 + [(False, False)] * 69
    text = "\n".join(S.diff_lines(S.paired(rows)))
    assert "13.8%" in text and "Wilson [7.9%, 23.0%]" in text and "Newcombe 方法 10 [6.3%, 23.0%]" in text


def test_pass_hat_k_unbiased_estimator():
    from fractions import Fraction
    assert S.pass_hat_k(6, 10, 4) == Fraction(15, 210)              # C(6,4)/C(10,4)
    assert S.pass_hat_k(0, 10, 3) == 0                               # c = 0
    assert S.pass_hat_k(10, 10, 3) == 1 and S.pass_hat_k(5, 5, 5) == 1   # c = n
    assert S.pass_hat_k(3, 10, 4) == 0                               # c < k:不可能 k 次全过
    assert S.pass_hat_k(7, 10, 1) == Fraction(7, 10)                 # k = 1 就是完成率
    for c, n, k in ((3, 3, 4), (0, 2, 3)):                           # k > n
        with pytest.raises(ValueError):
            S.pass_hat_k(c, n, k)
    for c, n, k in ((1, 3, 0), (4, 3, 2), (-1, 3, 1)):               # k < 1、c 越界
        with pytest.raises(ValueError):
            S.pass_hat_k(c, n, k)


def test_flip_rate_is_mean_pairwise_disagreement():
    from fractions import Fraction
    # 每题:两两比较中结果不同的比例 2c(n−c)/(n(n−1)):0、1、2·1·2/(3·2) = 2/3
    items = [[True, True], [True, False], [False, False, True]]
    assert S.flip_rate(items) == (0 + 1 + Fraction(2, 3)) / 3
    assert S.flip_rate([[True], [False, False]]) == 0                # n < 2 的题不计入
    assert S.flip_rate([[True], [False]]) is None                    # 没有任何一题跑过两次


def test_repeat_summary_k1_withholds_pass_k_and_flip():
    r = S.repeat_summary([[True], [False], [True]])
    assert r == {"k": 1, "k_max": 1, "pass_k": None, "flip": None}
    r2 = S.repeat_summary([[True, True], [True, False], [False, False, False]])
    assert r2["k"] == 2 and r2["k_max"] == 3
    assert r2["pass_k"] == (1 + 0 + 0) / S.Fraction(3)               # 公共 k = 2:C(2,2)/C(2,2)、C(1,2)、C(0,2)/C(3,2)


def _two_pair_records(second_attempt_valid):
    recs = [_record("C3_task_001_P3", "baseline", False), _record("C3_task_001_P3", "full", True),
            _record("C3_task_002_P3", "baseline", True), _record("C3_task_002_P3", "full", True)]
    if second_attempt_valid:                                         # 同一配对两组都再跑一次、都无 API 错误
        recs += [_record("C3_task_001_P3", "baseline", True, attempt=2), _record("C3_task_001_P3", "full", True, attempt=2),
                 _record("C3_task_002_P3", "baseline", True, attempt=2), _record("C3_task_002_P3", "full", False, attempt=2)]
    return recs


def _build_two_pairs(records):
    pairs = [E.Pair("c3", "C3_task_001", "P3"), E.Pair("c3", "C3_task_002", "P3")]
    traj = {"messages": [call(1, "book"), res(1, "book", {"ok": 1})]}
    inferences = {(p.key, "full"): traj for p in pairs}
    return S.build(records, pairs, inferences, {}, ([{} for _ in range(5)], {"C3_task_001": 0, "C3_task_002": 1}),
                   prompt_budget=None)


def test_report_k1_shows_k_column_and_reason_without_pass_k():
    s = _build_two_pairs(_two_pair_records(False))
    assert s["repeats"]["baseline"]["k"] == 1 and s["repeats"]["full"]["pass_k"] is None
    text = S.render(s, "test")
    assert "| 组 | 完成率 [95% Wilson] | 每题重复次数 k | pass^k |" in text
    assert "| 基线 | 1/2 = 50.0%" in text and "| 1 | — |" in text
    assert "k = 1" in text and "不输出 pass^k" in text and "翻转率" in text


def test_report_k2_shows_pass_k_and_flip_rate():
    s = _build_two_pairs(_two_pair_records(True))
    rb, rf = s["repeats"]["baseline"], s["repeats"]["full"]
    assert (rb["k"], rb["pass_k"], rb["flip"]) == (2, S.Fraction(1, 2), S.Fraction(1, 2))   # 001: F,T;002: T,T
    assert (rf["k"], rf["pass_k"], rf["flip"]) == (2, S.Fraction(1, 2), S.Fraction(1, 2))   # 001: T,T;002: T,F
    text = S.render(s, "test")
    assert "| 2 | 50.0% |" in text and "重跑翻转率" in text and "不输出 pass^k" not in text


def test_attempt_with_api_failure_is_not_a_repeat():
    recs = _two_pair_records(False) + [_record("C3_task_001_P3", "baseline", True, attempt=2),
                                       _record("C3_task_001_P3", "full", True, attempt=2, fail=1)]
    s = _build_two_pairs(recs)
    assert s["repeats"]["baseline"]["k"] == 1                       # 第 2 次尝试回执组有 API 错误,整次不算


def test_stripped_run_compares_full_against_stripped():
    pairs = [E.Pair("c3", "C3_task_001", "P3"), E.Pair("c3", "C3_task_002", "P3")]
    records = [
        _record("C3_task_001_P3", "stripped", False), _record("C3_task_001_P3", "full", True),     # 胜:只有完整回执完成
        _record("C3_task_002_P3", "stripped", True), _record("C3_task_002_P3", "full", False),     # 负
    ]
    traj = {"messages": [call(1, "book"), res(1, "book", {"ok": 1})]}
    inferences = {("C3_task_001_P3", "full"): traj, ("C3_task_002_P3", "full"): traj}
    folds = {"C3_task_001": 0, "C3_task_002": 1}
    regs = [{} for _ in range(5)]
    s = S.build(records, pairs, inferences, {}, (regs, folds), prompt_budget=None, control="stripped")
    assert s["done"] == 2 and (s["primary"]["wins"], s["primary"]["losses"]) == (1, 1)
    text = S.render(s, "test")
    assert "去掉备用工具列表 vs 完整回执" in text and "基线" not in text


def test_compared_arms_from_real_configs():
    api = E.load_cfg(os.path.join(os.path.dirname(S.HERE), "agent_error", "configs", "e2_api.yaml"))["e2"]["runs"]
    assert S.compared_arms(api["main"]) == ("baseline", "full")
    assert S.compared_arms(api["p0"]) == ("baseline", "full")
    assert S.compared_arms(api["stripped"]) == ("stripped", "full")
    with pytest.raises(ValueError):
        S.compared_arms({"arms": {"baseline": "off", "stripped": "stripped"}})



def _run_main_on_stripped(tmp_path, monkeypatch, outcomes, pilot=(), reference=None):
    """在 tmp_path 下造一个 ② 运行目录(full / stripped 两组),经 main() 汇总,返回汇总文本。
    outcomes = [(stripped 通过, full 通过)],依次对应冻结清单 main 的前几对;
    pilot = 试跑用过的任务编号;reference = ① 完整回执组每对的通过与否(列表,与 outcomes 对齐)。"""
    import json
    pairs = E.pairs_for("main", S.M.load())[:len(outcomes)]
    res_dir = tmp_path / "res"
    out = res_dir / "stripped"
    traj = {"messages": [call(1, "book"), res(1, "book", {"ok": 1})]}
    recs = []
    for p, (s_pass, f_pass) in zip(pairs, outcomes):
        for arm, passed in (("stripped", s_pass), ("full", f_pass)):
            recs.append(_record(p.key, arm, passed, tier=p.tier))
            d = out / arm / "m" / "fc" / p.tier / "inferences"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{p.key}_inference.json").write_text(json.dumps(traj), encoding="utf-8")
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (out / "receipts.jsonl").write_text("", encoding="utf-8")
    if pilot:
        (res_dir / "pilot").mkdir()
        (res_dir / "pilot" / "order.json").write_text(
            json.dumps({"order": [{"tier": "c3", "task_id": t, "mode": "P3", "arms": []} for t in pilot]}), encoding="utf-8")
    if reference is not None:
        (res_dir / "main").mkdir()
        ref = [_record(p.key, "full", passed, tier=p.tier) for p, passed in zip(pairs, reference)]
        (res_dir / "main" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(json.dumps({"agent": {"model": "m", "max_tokens": 100}, "e2": {
        "results_dir": str(res_dir),
        "runs": {"stripped": {"episodes": "main", "arms": {"full": "full", "stripped": "stripped"}}}}}),
        encoding="utf-8")
    (tmp_path / "eval" / "results").mkdir(parents=True)
    monkeypatch.setattr(S, "ROOT", str(tmp_path))
    monkeypatch.setattr(S.C, "load", lambda path: ([{} for _ in range(5)], {p.task_id: 0 for p in pairs}))
    monkeypatch.setattr(S.TS, "load_bare_params", lambda d: frozenset({("book", "card")}))
    monkeypatch.setattr(sys, "argv", ["summarize_e2.py", "--config", str(cfg), "--run", "stripped"])
    S.main()
    return (tmp_path / "eval" / "results" / "e2_api_stripped_summary.md").read_text(encoding="utf-8"), pairs


def test_main_summarizes_stripped_run_against_stripped_arm(tmp_path, monkeypatch):
    """回归:main() 原先不把对照组传给 build,默认找 baseline,② 的 80 对全被判为未完成。"""
    text, _ = _run_main_on_stripped(tmp_path, monkeypatch, [(False, True), (False, False)])
    assert "两组都完成且无 API 错误 2 对" in text
    assert "只有完整回执组完成(胜)1 对" in text
    assert "敏感性分析" not in text and "复现性" not in text      # 没有试跑重叠、没有 ① 记录时不出这两节


def test_cohen_kappa_known_values():
    rows = [(True, True)] * 20 + [(True, False)] * 5 + [(False, True)] * 10 + [(False, False)] * 15
    assert S.cohen_kappa(rows) == pytest.approx(0.4)        # p_o 0.7,p_e 0.5·0.6 + 0.5·0.4 = 0.5
    assert S.cohen_kappa([(True, True), (False, False)]) == pytest.approx(1.0)
    assert S.cohen_kappa([(False, False)] * 3) is None


def test_sensitivity_and_replication_sections(tmp_path, monkeypatch):
    outcomes = [(False, True), (False, True), (False, False), (True, True)]
    pilot_task = E.pairs_for("main", S.M.load())[0].task_id
    text, pairs = _run_main_on_stripped(tmp_path, monkeypatch, outcomes, pilot=[pilot_task, "C3_task_999"],
                                        reference=[True, False, False, True])
    # (a) 去掉第 1 对(试跑重叠)后剩 3 对:胜 1(第 2 对)、负 0
    assert "## 敏感性分析:去掉与试跑重叠的任务" in text and f"去掉 1 对({pairs[0].key}" in text
    assert "剩 3 对" in text and "胜 : 负 = 1 : 0" in text
    # (b) 本次完整回执组 [T, T, F, T] vs ① [T, F, F, T]:一致 3/4,κ = (0.75 − 0.5)/(1 − 0.5) = 0.5
    assert "共同的 4 对" in text and "一致率 3/4 = 75.0%" in text and "Cohen κ = 0.50" in text
    assert "| ① 通过 | 2 | 0 |" in text and "| ① 失败 | 1 | 1 |" in text
    assert "PA = 2a/(2a+b+c) = 2×2/(2×2 + 0 + 1) = 0.80" in text


def test_high_agreement_but_low_kappa_and_pa():
    # 100 对:都过 1、只有 ① 过 4、只有本次过 4、都失败 91。总一致率 92%,但通过率只有 5%:
    # p_e = 0.05² + 0.95² = 0.905,κ = (0.92 − 0.905)/(1 − 0.905) ≈ 0.16;PA = 2/(2 + 4 + 4) = 0.2
    ref = {f"k{i}": i < 5 for i in range(100)}
    new = {f"k{i}": (i == 0 or 5 <= i < 9) for i in range(100)}
    rep = S.replication({k: (False, v) for k, v in new.items()}, ref)
    assert (rep["both"], rep["ref_only"], rep["new_only"], rep["neither"]) == (1, 4, 4, 91)
    assert rep["agree"] / rep["n"] == pytest.approx(0.92)
    assert rep["kappa"] == pytest.approx(0.015 / 0.095)
    assert rep["pa"] == pytest.approx(0.2)


def test_positive_agreement_undefined_without_any_pass():
    assert S.positive_agreement(0, 0, 0) is None
    rep = S.replication({"k1": (False, False), "k2": (True, False)}, {"k1": False, "k2": False})
    assert rep["pa"] is None and rep["kappa"] is None and rep["agree"] == 2


# ---------------------------------------------------------------- 管线标注(无属性 object 参数)
BARE = frozenset({("archive", "card")})


def _inf(*values):
    return {"messages": [call(i, "archive", {"card": v}) for i, v in enumerate(values)]}


def test_pipeline_counts_bare_arguments_per_arm_and_reads_patch_flag():
    inferences = {("p1", "baseline"): _inf({}, {"id": "A1"}), ("p1", "full"): _inf({}, "", None, "A1")}
    pl = S.pipeline({}, inferences, BARE)
    assert pl["patched"] is False
    assert pl["counts"] == {"baseline": {"empty": 1, "nonempty": 1}, "full": {"empty": 3, "other": 1}}
    assert S.pipeline({"bare_object_additional_properties": True}, {}, BARE)["patched"] is True


def test_pipeline_lines_mark_old_pipeline_and_show_counts():
    old = "\n".join(S.pipeline_lines(S.pipeline({}, {("p1", "full"): _inf({}, {})}, BARE)))
    assert S.OLD_PIPELINE in old and "完整回执组:无属性 object 参数 2 次,传空 2/2" in old
    new = "\n".join(S.pipeline_lines(S.pipeline({"bare_object_additional_properties": True},
                                               {("p1", "full"): _inf({"id": "A1"})}, BARE)))
    assert S.OLD_PIPELINE not in new and "additionalProperties: true" in new and "非空对象 1" in new
    local = "\n".join(S.pipeline_lines(S.pipeline({}, {("p1", "full"): _inf({"id": "A1"}, {})}, BARE)))
    assert S.OLD_PIPELINE not in local and "未打补丁" in local          # 本地运行没有被压成 {},不贴旧管线标签
    assert S.pipeline_lines(None) == []


def test_main_reports_pipeline_from_order_and_inferences(tmp_path, monkeypatch):
    text, _ = _run_main_on_stripped(tmp_path, monkeypatch, [(True, True)])   # 没有 order.json,轨迹里没有 card 参数
    assert "工具 schema 未打补丁;本运行没有调用带无属性 object 参数的工具" in text
