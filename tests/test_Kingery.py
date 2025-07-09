import pytest

from src.utils.Kingery import KingeryBulmashModel


def test_metric_properties_are_positive():
    # 测试公制默认情况下各属性均为正数且类型正确
    model = KingeryBulmashModel(neq=1000, distance=100, unit_system='metric')
    assert isinstance(model.time_of_arrival, float)
    assert model.time_of_arrival > 0
    assert isinstance(model.incident_pressure, float)
    assert model.incident_pressure > 0
    assert isinstance(model.reflected_pressure, float)
    assert model.reflected_pressure >= model.incident_pressure
    assert isinstance(model.positive_phase_duration, float)
    assert model.positive_phase_duration > 0
    assert isinstance(model.incident_impulse, float)
    assert model.incident_impulse > 0
    assert isinstance(model.reflected_impulse, float)
    assert model.reflected_impulse >= model.incident_impulse
    assert isinstance(model.shock_front_velocity, float)
    assert model.shock_front_velocity > 0


def test_to_dict_contains_all_keys():
    # 测试 to_dict() 返回字典包含预期的字段
    model = KingeryBulmashModel(neq=500, distance=50, unit_system='metric')
    data = model.to_dict()
    expected_keys = {
        'time_of_arrival',
        'incident_pressure',
        'reflected_pressure',
        'positive_phase_duration',
        'incident_impulse',
        'reflected_impulse',
        'shock_front_velocity'
    }
    assert expected_keys.issubset(set(data.keys()))


def test_imperial_unit_system_switch():
    # 测试英制单位下属性仍然可用
    model = KingeryBulmashModel(neq=2204.62, distance=328.08, unit_system='imperial')
    # 仅需验证属性存在且为 float
    for attr in [
        'time_of_arrival',
        'incident_pressure',
        'reflected_pressure',
        'positive_phase_duration',
        'incident_impulse',
        'reflected_impulse',
        'shock_front_velocity'
    ]:
        value = getattr(model, attr)
        assert isinstance(value, float)


def test_invalid_parameters_raise_error():
    # 测试无效参数会抛出错误
    with pytest.raises(Exception):
        KingeryBulmashModel(neq=-1, distance=100)
    with pytest.raises(Exception):
        KingeryBulmashModel(neq=1000, distance=-10)


