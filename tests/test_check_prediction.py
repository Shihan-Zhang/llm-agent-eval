"""eval/agent_error/check_prediction.py:启动前确认 predictions.md 里有对应运行的预测行。"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "agent_error"))
import check_prediction as CP  # noqa: E402

TEXT = ("运行\t要预测的量\n"
        "① v2\t基线和回执两组的路径层完成率10和25\n"
        "① 新旧对比\t会升\n"
        "③ v2\t\n"
        "② v2 \n")


def test_has_prediction_requires_label_followed_by_content():
    assert CP.has_prediction(TEXT, "① v2")
    assert not CP.has_prediction(TEXT, "③ v2")       # 只有标签,内容为空
    assert not CP.has_prediction(TEXT, "② v2")
    assert not CP.has_prediction(TEXT, "① v3")


def test_label_must_start_the_line_and_be_a_whole_word():
    assert not CP.has_prediction("说明:① v2\t这里只是提到\n", "① v2")
    assert not CP.has_prediction("① v20\t别的运行\n", "① v2")


def test_cli_exit_codes(tmp_path):
    p = tmp_path / "predictions.md"
    p.write_text(TEXT, encoding="utf-8")
    script = os.path.join(os.path.dirname(CP.__file__), "check_prediction.py")
    ok = subprocess.run([sys.executable, script, str(p), "① v2"], capture_output=True)
    bad = subprocess.run([sys.executable, script, str(p), "③ v2"], capture_output=True)
    missing = subprocess.run([sys.executable, script, str(tmp_path / "none.md"), "① v2"], capture_output=True)
    assert (ok.returncode, bad.returncode, missing.returncode) == (0, 1, 1)
