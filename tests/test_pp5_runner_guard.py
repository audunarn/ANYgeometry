import importlib.util
from pathlib import Path
import pytest

def test_outer_runner_console_scope():
    spec=importlib.util.spec_from_file_location('guard',Path(__file__).resolve().parents[1]/'tools/pp5_runner_guard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    system=Path('C:/Windows/System32')
    assert module.console_command([str(system/'conhost.exe'),'0x4'],system)
    for command in ([str(system/'conhost.exe')],['C:/elsewhere/conhost.exe','0x4'],[str(system/'cmd.exe'),'0x4'],[str(system/'conhost.exe'),'bad']):
        assert not module.console_command(command,system)
