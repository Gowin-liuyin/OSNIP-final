import os
import subprocess
import sys

from osnip_repro.process_inspection import command_tokens, process_command


def test_command_tokens_keeps_windows_separators():
    cmd = r"python F:\OSNIP\scripts\train_streaming.py --output-dir F:\OSNIP\reports\run1"
    tokens = command_tokens(cmd)
    assert tokens[1] == r"F:\OSNIP\scripts\train_streaming.py"
    assert tokens[tokens.index("--output-dir") + 1] == r"F:\OSNIP\reports\run1"


def test_command_tokens_strips_one_layer_of_quotes():
    tokens = command_tokens('python "C:/a b/train_streaming.py" --x 1')
    assert tokens[1] == "C:/a b/train_streaming.py"
    assert tokens[2:] == ["--x", "1"]


def test_process_command_sees_live_and_exited_processes():
    assert process_command(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = child.pid
    assert child.wait() == 0
    assert process_command(pid) is None