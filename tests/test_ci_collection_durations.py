"""Collection errors must survive CI reporting without another crash."""
import importlib.util
from pathlib import Path

def test_collection_failure_without_classname_keeps_its_node_and_duration(tmp_path):
    p=Path(__file__).resolve().parents[1]/'tools/ci_durations.py'
    spec=importlib.util.spec_from_file_location('durations',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=tmp_path/'failed.xml'
    report.write_text('<testsuite><testcase name="tests/test_missing.py" time="1.5"><error message="missing dependency"/></testcase></testsuite>')
    assert module.read_junit(report)=={'tests/test_missing.py':1.5}
