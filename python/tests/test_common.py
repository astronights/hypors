import json

import pytest

from hypors import TailType, TestResult
from hypors.common import TailType as CommonTailType
from hypors.common import TestResult as CommonTestResult


def make_result():
    return TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True,
    )


def test_types_are_importable_from_both_paths():
    assert TailType is CommonTailType
    assert TestResult is CommonTestResult


def test_tail_type_equality():
    assert TailType.Left == TailType.Left
    assert TailType.Right != TailType.Left
    assert TailType.Two == TailType.Two


def test_tail_type_name_and_repr():
    assert TailType.Left.name == "Left"
    assert repr(TailType.Two) == "TailType.Two"


def test_test_result_attributes():
    result = make_result()
    assert result.test_statistic == 2.5
    assert result.p_value == 0.05
    assert result.confidence_interval == (1.0, 3.0)
    assert result.null_hypothesis == "No effect"
    assert result.alt_hypothesis == "There is an effect"
    assert result.reject_null is True


def test_test_result_is_read_only():
    result = make_result()
    with pytest.raises(AttributeError):
        result.p_value = 0.5


def test_test_result_repr_is_python_spelled():
    assert "reject_null=True" in repr(make_result())


def test_test_result_serialisation():
    result_dict = json.loads(json.dumps(make_result().to_dict()))
    assert result_dict["test_statistic"] == 2.5
    assert result_dict["p_value"] == 0.05
    assert result_dict["confidence_interval"] == [1.0, 3.0]
    assert result_dict["null_hypothesis"] == "No effect"
    assert result_dict["reject_null"] is True


def test_test_result_equality():
    assert make_result() == make_result()


def test_test_result_inequality():
    other = TestResult(
        test_statistic=1.5,
        p_value=0.10,
        confidence_interval=(0.5, 2.5),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=False,
    )
    assert make_result() != other


def test_submodules_are_fully_qualified():
    # A bare __name__ of "t" names a module that cannot be imported back,
    # and shows up in every repr and traceback.
    import hypors.anova
    import hypors.chi_square
    import hypors.mann_whitney
    import hypors.proportion
    import hypors.t
    import hypors.z

    for module, expected in [
        (hypors.anova, "hypors.anova"),
        (hypors.chi_square, "hypors.chi_square"),
        (hypors.mann_whitney, "hypors.mann_whitney"),
        (hypors.proportion, "hypors.proportion"),
        (hypors.t, "hypors.t"),
        (hypors.z, "hypors.z"),
    ]:
        assert module.__name__ == expected


def test_test_result_survives_pickle_and_copy():
    # Results get cached to disk and sent across process pools.
    import copy
    import pickle

    result = make_result()
    assert pickle.loads(pickle.dumps(result)) == result
    assert copy.copy(result) == result
    assert copy.deepcopy(result) == result


def test_tail_type_survives_pickle_and_copy():
    import copy
    import pickle

    for tail in (TailType.Left, TailType.Right, TailType.Two):
        assert pickle.loads(pickle.dumps(tail)) == tail
        assert copy.deepcopy(tail) == tail


def test_tail_type_from_name_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown TailType"):
        TailType._from_name("Sideways")
