"""
阶段 C 的调查与规则层试算,写 eval/results/e2_judge_survey.md(生成文件,勿手改)。不调用模型。

1. ToolMaze 任务 expected_result 的格式(层级 × 模式),P1–P4 带 is_perturbed 的条数;
   P0 叶子值类型(启发式分类器 survey_leaf_type,与调查时相同)。
2. ① API 被判失败的回合,只看最终回答能否判对错(A 没有最终回答 / B 含损坏值 / C 含某条路径全部参考 ID / D 其余)。
3. 脱敏要处理的痕迹:推理文件字段、注入标签、助手文字里的回执词汇、token 与调用数。
4. 规则层(rules.py)在 ① 全部回合上的 L1 / L2 / L3 落点。
5. 人工核对清单:L2 判失败的回合随机抽 AUDIT_N 个(种子 AUDIT_SEED)+ 点名的回合;L3 判"正确"的回合待 LLM 裁判后补。
6. 管线检查:schema 只写 type: object 的参数,API 与本地运行里传空 / 非空的次数。

用法(仓库根目录):python eval/agent_error/judge/survey.py [--toolmaze DIR] [--run-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import re
import statistics
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))
from judge import reference as REF  # noqa: E402
from judge import rules as RU  # noqa: E402
from judge import trajectory as TR  # noqa: E402
from judge.normalize import leaves, norm  # noqa: E402

TOOLMAZE = os.path.join(ROOT, "external", "agent_error", "toolmaze")
RUN_DIR = os.path.join(ROOT, "external", "agent_error", "results", "e2", "api", "main")
LOCAL_RUN_DIR = os.path.join(ROOT, "external", "agent_error", "results", "e2", "local", "main")
OUT = os.path.join(ROOT, "eval", "results", "e2_judge_survey.md")
AUDIT_SEED, AUDIT_N = 20261109, 20
AUDIT_NAMED = (("C4_task_015_P4", "full"), ("C4_task_052_P4", "baseline"), ("C4_task_052_P4", "full"),
               ("C3_task_039_P3", "full"), ("C4_task_014_P3", "full"))   # 上一轮规则判通过 / 与 ToolMaze 相反的回合
RECEIPT_VOCAB = re.compile(r"outcome[_ ]contract|admissible|recovery[_ ]tools?|inconsistent|violat\w*|contract|invariant|receipt", re.I)
FAIL_HINT = re.compile(r"(unable|could not|couldn't|cannot|can't|failed|unsuccessful|not able|error|无法|失败)", re.I)

# ---------------------------------------------------------------- 调查用的启发式叶子分类(与调查时相同,只用于第 1、2 节)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?([+-]\d{2}:\d{2}|Z)?)?$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]+$", re.I)
_IDLIKE = re.compile(r"^(?=.*\d)[A-Za-z0-9_\-]{4,}$")
_ID_KEY = re.compile(r"(^|_)(id|code|ref|number|no|sku|uuid|token)$", re.I)
LEAF_TYPES = ("短标签(≤3 词)", "ID", "自由文本(>3 词)", "数字", "日期时间", "布尔", "空值")


def survey_leaf_type(key: str | None, v: object) -> str:
    if v is None:
        return "空值"
    if isinstance(v, bool):
        return "布尔"
    if isinstance(v, (int, float)):
        return "ID" if _ID_KEY.search(key or "") else "数字"
    if not isinstance(v, str):
        return "其他"
    s = v.strip()
    if _DATE.match(s) or _TIME.match(s):
        return "日期时间"
    if _EMAIL.match(s) or _ID_KEY.search(key or "") or _IDLIKE.match(s):
        return "ID"
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return "数字"
    return "短标签(≤3 词)" if len(s.split()) <= 3 else "自由文本(>3 词)"


def er_steps(er: dict) -> list:
    return [c.get("output") for c in er["tool_calls"]] if "tool_calls" in er else list(er["results"].values())


def pct(k: int, n: int) -> str:
    return f"{k}/{n} = {k / n:.1%}" if n else "—"


# ---------------------------------------------------------------- 1. expected_result
def survey_expected(tm: str) -> list[str]:
    fmt, pert = collections.Counter(), collections.Counter()
    last_t, all_t = collections.Counter(), collections.Counter()
    ex = collections.defaultdict(list)
    total = 0
    for f in sorted(glob.glob(os.path.join(tm, "data", "perturbed_tasks", "*", "*.json"))):
        tier, mode = os.path.basename(os.path.dirname(f)), f.rsplit("_", 1)[1][:-5]
        with open(f, encoding="utf-8") as fh:
            t = json.load(fh)
        ers = [t["expected_result"]] if "expected_result" in t else [vp["expected_result"] for vp in t["valid_paths"]]
        for er in ers:
            total += 1
            form = "tool_calls 列表" if "tool_calls" in er else "{status, message, results}"
            fmt[(tier, mode, form)] += 1
            if "tool_calls" in er and any(c.get("is_perturbed") for c in er["tool_calls"]):
                pert[mode] += 1
            if mode != "P0":
                continue
            st = er_steps(er)
            for k, v in leaves(st):
                all_t[survey_leaf_type(k, v)] += 1
            for k, v in leaves(st[-1]):
                typ = survey_leaf_type(k, v)
                last_t[typ] += 1
                if len(ex[typ]) < 2 and all(k != e[0] for e in ex[typ]):
                    ex[typ].append((k, v))
    lines = ["## 1. expected_result 的格式", "",
             f"共 {total} 条(C1 每个文件 1 条;C2/C3 每题 2 条有效路径、C4 6 条);全部是结构化对象,没有单个值。", "",
             "| 层级 | 模式 | 格式 | 条数 |", "|---|---|---|---|"]
    lines += [f"| {t.upper()} | {m} | {f} | {n} |" for (t, m, f), n in sorted(fmt.items())]
    lines += ["", "含 is_perturbed 调用的条数(按模式):" + ",".join(f"{m} {n}" for m, n in sorted(pert.items()))
              + "。P1–P4 是带故障的参考轨迹,正确值取同一题的 P0。", "",
              "P0 叶子值类型(启发式:按字段名与正则):", "",
              "| 类型 | 最后一步 | 全部步骤 | 最后一步的例子 |", "|---|---|---|---|"]
    nl, na = sum(last_t.values()), sum(all_t.values())
    for typ in LEAF_TYPES:
        lines.append(f"| {typ} | {pct(last_t[typ], nl)} | {pct(all_t[typ], na)} | "
                     + "; ".join(f"`{k}: {json.dumps(v, ensure_ascii=False)}`" for k, v in ex[typ]) + " |")
    return lines


# ---------------------------------------------------------------- ① 运行数据
def load_run(run_dir: str) -> dict:
    recs = {}
    with open(os.path.join(run_dir, "episodes.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["api_failures"] == 0:
                recs[(r["pair"], r["arm"])] = r          # 文件顺序,后写的覆盖先写的 = 最近一次
    return recs


def episode_file(run_dir: str, r: dict, kind: str) -> str:
    sub, suffix = ("inferences", "inference") if kind == "inference" else ("evaluations", "eval")
    return glob.glob(os.path.join(run_dir, r["arm"], "*", "fc", r["tier"], sub, f"{r['pair']}_{suffix}.json"))[0]


def task_of(pair: str) -> str:
    return pair.rsplit("_", 1)[0]


# ---------------------------------------------------------------- 2. 只看最终回答
def survey_answers(tm: str, run_dir: str, recs: dict) -> list[str]:
    cls, by_arm, sub = collections.Counter(), collections.Counter(), collections.Counter()
    failed = sorted(k for k, r in recs.items() if not r["passed"])
    for pair, arm in failed:
        r = recs[(pair, arm)]
        traj = TR.load(episode_file(run_dir, r, "inference"))
        ref = REF.load_task_ref(tm, r["tier"], task_of(pair))
        ids = [{str(v) for k, v in leaves(p[-1].output)
                if survey_leaf_type(k, v) == "ID" and v not in (None, "N/A", "") and len(norm(v)) >= 4} for p in ref.paths]
        corrupted = TR.corrupted_values(traj, ref)
        ans = traj.final_answer
        if not ans or not ans.strip():
            c = "A"
        else:
            t = norm(ans)
            hit = lambda v: len(norm(v)) >= 4 and norm(v) in t  # noqa: E731
            if any(x in t for x in corrupted):
                c = "B"
            elif any(R and all(hit(v) for v in R) for R in ids):
                c = "C"
            else:
                c = "D"
                part = any(R and any(hit(v) for v in R) for R in ids)
                sub["有参考 ID,只含部分" if part else ("所有路径最后一步都没有 ID 类字段" if not any(ids) else "有参考 ID,一个都不含")] += 1
                sub["带失败措辞(启发式)" if FAIL_HINT.search(ans) else "不带失败措辞(启发式)"] += 1
        cls[c] += 1
        by_arm[(c, arm)] += 1
    names = {"A": "A 没有最终回答", "B": "B 最终回答含损坏值", "C": "C 含某条路径全部最后一步参考 ID", "D": "D 其余(要 LLM 裁判)"}
    lines = ["", "## 2. ① API 失败回合:只看最终回答", "",
             f"失败回合 {len(failed)} 个(基线 {sum(1 for _, a in failed if a == 'baseline')}、"
             f"完整回执 {sum(1 for _, a in failed if a == 'full')})。C 类匹配上参考 ID 也不说明结果正确:回放沙盒下最终 ID 基本固定。", "",
             "| 类别 | 回合 | 基线 | 完整回执 |", "|---|---|---|---|"]
    lines += [f"| {names[c]} | {cls[c]} | {by_arm[(c, 'baseline')]} | {by_arm[(c, 'full')]} |" for c in "ABCD"]
    lines += ["", "D 细分:" + ";".join(f"{k} {v}" for k, v in sorted(sub.items()))]
    return lines


# ---------------------------------------------------------------- 3. 脱敏痕迹
def survey_traces(run_dir: str, recs: dict) -> list[str]:
    fields, status = collections.Counter(), collections.Counter()
    vocab, stats = collections.Counter(), collections.defaultdict(lambda: collections.defaultdict(list))
    prompts = collections.defaultdict(set)
    for (pair, arm), r in sorted(recs.items()):
        with open(episode_file(run_dir, r, "inference"), encoding="utf-8") as fh:
            d = json.load(fh)
        fields.update(("顶层", k) for k in d)
        msgs = d["messages"]
        for m in msgs:
            fields.update((m.get("role"), k) for k in m)
            fields.update((m.get("role"), "metadata." + k) for k in (m.get("metadata") or {}))
            if m.get("role") == "tool":
                status[(arm, (m.get("metadata") or {}).get("perturbation_status"))] += 1
        vocab[(arm, any(m.get("role") == "assistant" and RECEIPT_VOCAB.search(m.get("content") or "") for m in msgs))] += 1
        stats[arm]["输入 token"].append(d["tokens"]["input_tokens"])
        stats[arm]["工具调用数"].append(sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_call")))
        stats[arm]["消息数"].append(len(msgs))
        prompts[pair].add(msgs[0].get("content"))
    lines = ["", "## 3. 脱敏要处理的痕迹(① API 推理文件)", "",
             "字段:" + ",".join(f"{r}.{k} {n}" for (r, k), n in sorted(fields.items(), key=lambda x: (str(x[0][0]), x[0][1]))), "",
             "工具消息 perturbation_status:" + ",".join(f"{a}/{s} {n}" for (a, s), n in sorted(status.items())), "",
             f"助手文字含回执词汇的回合:基线 {vocab[('baseline', True)]}/{vocab[('baseline', True)] + vocab[('baseline', False)]},"
             f"完整回执 {vocab[('full', True)]}/{vocab[('full', True)] + vocab[('full', False)]}", "",
             "| 中位数 | 基线 | 完整回执 |", "|---|---|---|"]
    for name in ("输入 token", "工具调用数", "消息数"):
        lines.append(f"| {name} | {statistics.median(stats['baseline'][name]):g} | {statistics.median(stats['full'][name]):g} |")
    lines += ["", f"用户提示两组完全相同的配对:{sum(len(v) == 1 for v in prompts.values())}/{len(prompts)}"]
    return lines


# ---------------------------------------------------------------- 4. 规则层落点 + 5. 核对清单
def judge_all(tm: str, run_dir: str, recs: dict) -> dict:
    catalog = REF.load_catalog(os.path.join(tm, "tools", "definitions"))
    out = {}
    for (pair, arm), r in sorted(recs.items()):
        traj = TR.load(episode_file(run_dir, r, "inference"))
        out[(pair, arm)] = RU.judge(traj, REF.load_task_ref(tm, r["tier"], task_of(pair)), catalog)
    return out


LEVEL_NAMES = {"answer_corrupted": "L1 最终回答含损坏值 → 失败", "corrupted_argument": "L2 损坏值进入参数 → 失败",
               "missing_action": "L2 缺必需动作 → 失败", "empty_argument": "L2 必需参数为空 → 失败",
               "pass": "L2 全部满足 → 通过", "l3": "L3 交给 LLM 裁判"}


def survey_levels(recs: dict, verdicts: dict) -> list[str]:
    tab = collections.Counter()
    issues = collections.Counter()
    no_answer = collections.Counter()
    for (pair, arm), v in verdicts.items():
        tm_pass = "通过" if recs[(pair, arm)]["passed"] else "失败"
        tab[(v.reason, tm_pass, arm)] += 1
        for i in v.issues:
            issues[(tm_pass, i)] += 1
    lines = ["", "## 4. 规则层在 ① 全部回合上的落点", "",
             "| 判定 | ToolMaze 失败:基线 | ToolMaze 失败:回执 | ToolMaze 通过:基线 | ToolMaze 通过:回执 | 合计 |",
             "|---|---|---|---|---|---|"]
    for reason, name in LEVEL_NAMES.items():
        row = [tab[(reason, p, a)] for p in ("失败", "通过") for a in ("baseline", "full")]
        lines.append(f"| {name} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    totals = [sum(tab[(r, p, a)] for r in LEVEL_NAMES) for p in ("失败", "通过") for a in ("baseline", "full")]
    lines.append("| 合计 | " + " | ".join(map(str, totals)) + f" | {sum(totals)} |")
    lines += ["", "L3 的问题(一个回合可有多个,取问题最少的那条有效路径):"
              + ";".join(f"ToolMaze {p} · {RU.ISSUE_NAMES[i]} {n}" for (p, i), n in sorted(issues.items()))]
    return lines


def audit_list(recs: dict, verdicts: dict) -> list[str]:
    l2_fail = sorted(k for k, v in verdicts.items() if v.level in ("L1", "L2") and v.outcome == "fail" and k not in AUDIT_NAMED)
    sample = sorted(random.Random(AUDIT_SEED).sample(l2_fail, min(AUDIT_N, len(l2_fail))))
    lines = ["", "## 5. 人工核对清单", "",
             f"L1/L2 判失败的 {len(l2_fail)} 个回合(不含点名的)里随机抽 {len(sample)} 个(种子 {AUDIT_SEED}),"
             f"加点名的 {len(AUDIT_NAMED)} 个;L3 判正确的回合待 LLM 裁判后全部补进来。", "",
             "| 回合 | 组 | 来源 | ToolMaze | 规则判定 |", "|---|---|---|---|---|"]
    for (pair, arm), src in [(k, "点名") for k in AUDIT_NAMED] + [(k, "随机") for k in sample]:
        v = verdicts[(pair, arm)]
        detail = LEVEL_NAMES[v.reason] + (f"({'、'.join(RU.ISSUE_NAMES[i] for i in v.issues)})" if v.issues else "")
        lines.append(f"| {pair} | {arm} | {src} | {'通过' if recs[(pair, arm)]['passed'] else '失败'} | {detail} |")
    return lines


def survey_object_params(tm: str, runs: dict[str, str]) -> list[str]:
    """schema 里只写 type: object、没有 properties 的参数,各次运行里模型传空 {} / 非空对象的次数。"""
    bare = set()
    for fn in REF.DEFINITION_FILES:
        with open(os.path.join(tm, "tools", "definitions", fn), encoding="utf-8") as fh:
            for t in yaml.safe_load(fh).get("tools") or []:
                spec = ((t.get("paradigms") or {}).get("function_call") or {}).get("spec") or {}
                for k, p in ((spec.get("parameters") or {}).get("properties") or {}).items():
                    if p.get("type") == "object" and not p.get("properties"):
                        bare.add((t["name"], k))
    lines = ["", "## 6. 管线检查:schema 只写 type: object 的参数", "",
             f"ToolMaze 工具定义里这类参数 {len(bare)} 个(都没有 properties)。各次运行里模型传的值:", "",
             "| 运行 | 空 {} | 非空对象 | 其他 |", "|---|---|---|---|"]
    for name, rd in runs.items():
        c = collections.Counter()
        for f in glob.glob(os.path.join(rd, "*", "*", "fc", "*", "inferences", "*_inference.json")):
            with open(f, encoding="utf-8") as fh:
                msgs = json.load(fh)["messages"]
            for m in msgs:
                tc = m.get("tool_call") or {}
                for k, v in (tc.get("arguments") or {}).items():
                    if (tc.get("name"), k) in bare:
                        c["empty" if v in ({}, None) else ("object" if isinstance(v, dict) else "other")] += 1
        lines.append(f"| {name} | {c['empty']} | {c['object']} | {c['other']} |")
    lines += ["", "API 一律为空、本地一律非空:空对象来自 API 端(推测是服务端对无 properties 的 object 做受约束解码或解析),"
              "不是模型行为。API 运行里这类动作都带着空数据执行,第 4 节的必需参数为空多数源于此。"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段 C 调查与规则层试算")
    ap.add_argument("--toolmaze", default=TOOLMAZE)
    ap.add_argument("--run-dir", default=RUN_DIR, help="① API 的运行目录")
    ap.add_argument("--local-run-dir", default=LOCAL_RUN_DIR, help="① 本地的运行目录(第 6 节对照)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    recs = load_run(a.run_dir)
    verdicts = judge_all(a.toolmaze, a.run_dir, recs)
    lines = ["# E2 阶段 C:调查与规则层试算(由 eval/agent_error/judge/survey.py 生成,勿手改)", ""]
    lines += survey_expected(a.toolmaze)
    lines += survey_answers(a.toolmaze, a.run_dir, recs)
    lines += survey_traces(a.run_dir, recs)
    lines += survey_levels(recs, verdicts)
    lines += audit_list(recs, verdicts)
    lines += survey_object_params(a.toolmaze, {"① API Qwen3.5-35B-A3B(硅基流动)": a.run_dir,
                                              "① 本地 qwen3i-8k(Ollama)": a.local_run_dir})
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
