from src.physics.calculate_blast_parameters import calculate_td
import pytest
# ✅ 测试函数主体
# ✅ 自由场测试
def test_calculate_td():
    td = calculate_td(
        charge_weight=10000,
        prev_distance=0,
        distance=500,
        is_free_field=True
    )
    assert 120 <= td <= 140, f"自由场 td 应在合理范围，实际: {td}"

# ✅ 反射传播测试（参数化多组距离）
@pytest.mark.parametrize("prev_distance, distance, prev_td", [
    (500, 600, 130),
    (500, 750, 130),
    (500, 1000, 130)
])
def test_reflected_td(prev_distance, distance, prev_td):
    td = calculate_td(
        charge_weight=None,
        prev_distance=prev_distance,
        distance=distance,
        prev_td=prev_td,
        is_free_field=False
    )
    expected = prev_td * (distance / prev_distance)
    assert abs(td - expected) < 1e-6

# ✅ 异常测试：自由场缺炸药质量
def test_free_field_missing_charge_weight():
    with pytest.raises(ValueError, match="自由场计算必须提供 charge_weight"):
        calculate_td(
            charge_weight=None, prev_distance=0, distance=500, is_free_field=True
        )

# ✅ 异常测试：反射传播缺 prev_td
def test_reflected_missing_prev_td():
    with pytest.raises(ValueError, match="反射传播计算 td 需要提供"):
        calculate_td(
            charge_weight=None, prev_distance=500, distance=750, is_free_field=False
        )

# ✅ 异常测试：distance 为负
def test_negative_distance():
    with pytest.raises(ValueError, match="distance 必须为正数"):
        calculate_td(
            charge_weight=10000, prev_distance=0, distance=-100, is_free_field=True
        )

if __name__ == "__main__":
    test_calculate_td()