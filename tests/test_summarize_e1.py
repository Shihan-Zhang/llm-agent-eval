"""E1 汇总脚本:配对计数和"声称重试过但没有"的判定(不读真实结果、不调用模型)。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
from summarize_e1 import paired_counts, unbacked_retry_claim  # noqa: E402


def _call(name):
    return {"role": "assistant", "type": "tool_call", "tool_call": {"name": name, "arguments": {}}}


def _result(name, perturbed):
    return {"role": "tool", "name": name, "metadata": {"perturbation_status": "perturbed" if perturbed else "clean"}}


def _final(text):
    return {"role": "assistant", "type": "final_answer", "content": text}


def test_paired_counts_only_discordant_pairs_and_skips_incomplete_tasks():
    passed = {
        ("t1", "P1"): True, ("t1", "P3"): False, ("t1", "P2"): True, ("t1", "P4"): True,   # 1 对显式独过
        ("t2", "P1"): False, ("t2", "P3"): True, ("t2", "P2"): False, ("t2", "P4"): False,  # 1 对隐式独过
        ("t3", "P1"): True,                                                                  # 缺 P3,不成对
    }
    assert paired_counts(passed) == (4, 1, 1)


def test_retry_claim_with_single_call_is_flagged():
    msgs = [_call("book"), _result("book", True), _final("Task aborted: booking failed after retry.")]
    assert unbacked_retry_claim(msgs) == ["book"]


def test_retry_claim_backed_by_second_call_is_not_flagged():
    msgs = [_call("book"), _result("book", True), _call("book"), _result("book", False),
            _final("The first attempt failed; I retried and it succeeded.")]
    assert unbacked_retry_claim(msgs) == []


def test_no_claim_or_no_fault_is_not_flagged():
    assert unbacked_retry_claim([_call("book"), _result("book", True), _final("Task aborted.")]) == []
    assert unbacked_retry_claim([_call("book"), _result("book", False), _final("Done after retry.")]) == []
