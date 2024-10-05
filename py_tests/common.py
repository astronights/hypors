import pytest

from hypors.common import TailType, TestResult
import json

def test_tail_type_equality():
    # Test equality of TailType enum variants
    assert TailType.Left == TailType.Left
    assert TailType.Right != TailType.Left
    assert TailType.Two == TailType.Two

def test_test_result_creation():
    # Test the creation of TestResult
    result = TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True
    )
    
    assert isinstance(result, TestResult)
    assert result.test_statistic == 2.5
    assert result.p_value == 0.05
    assert result.confidence_interval == (1.0, 3.0)
    assert result.null_hypothesis == "No effect"
    assert result.alt_hypothesis == "There is an effect"
    assert result.reject_null is True

def test_test_result_serialization():
    # Test serialization of TestResult
    result = TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True
    )
    
    result_json = json.dumps(result.to_dict())
    result_dict = json.loads(result_json)
    
    assert result_dict['test_statistic'] == 2.5
    assert result_dict['p_value'] == 0.05
    assert result_dict['confidence_interval'] == [1.0, 3.0]
    assert result_dict['null_hypothesis'] == "No effect"
    assert result_dict['alt_hypothesis'] == "There is an effect"
    assert result_dict['reject_null'] is True

def test_test_result_equality():
    # Test equality of TestResult instances
    result1 = TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True
    )
    
    result2 = TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True
    )
    
    assert result1 == result2

def test_test_result_inequality():
    # Test inequality of TestResult instances
    result1 = TestResult(
        test_statistic=2.5,
        p_value=0.05,
        confidence_interval=(1.0, 3.0),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=True
    )
    
    result2 = TestResult(
        test_statistic=1.5,
        p_value=0.10,
        confidence_interval=(0.5, 2.5),
        null_hypothesis="No effect",
        alt_hypothesis="There is an effect",
        reject_null=False
    )
    
    assert result1 != result2
