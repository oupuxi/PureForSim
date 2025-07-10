"""
sim/kingery.py

封装 kingery-bulmash 包，提供便捷的爆炸冲击波参数计算接口。

安装依赖：
    pip install kingery-bulmash
"""
from typing import Literal, Dict, Any, Optional
import kingery_bulmash as kb

UnitSystem = Literal['metric', 'imperial']


class KingeryBulmashModel:
    """
    Kingery-Bulmash 封装类，支持 fallback 容错输出。

    参数:
      - neq: TNT 等效量 (kg 或 lb)
      - distance: 探测点距离 (m 或 ft)
      - unit_system: 'metric' (kg/m) 或 'imperial' (lb/ft)
      - safe: 是否检查缩比距离范围
      - fallback: 若查表失败，是否启用默认值而不中断程序
    """
    TD_MAX_MS = 200.0
    I_MAX_KPAMS = 50.0
    PS_MIN_KPA = 1.0

    def __init__(self,
                 neq: float,
                 distance: float,
                 unit_system: UnitSystem = 'metric',
                 safe: bool = True,
                 fallback: bool = True):
        self.neq = neq
        self.distance = distance
        self.unit = (kb.Units.METRIC if unit_system == 'metric' else kb.Units.IMPERIAL)
        self.safe = safe
        self.fallback = fallback
        self._res: Optional[kb.Blast_Parameters] = None
        self._compute()

    def _compute(self) -> None:
        try:
            self._res = kb.Blast_Parameters(
                unit_system=self.unit,
                neq=self.neq,
                distance=self.distance,
                safe=self.safe
            )
        except Exception:
            if self.fallback:
                self._res = None
            else:
                raise

    def _guard(self, attr: str, default: float) -> float:
        if self._res is not None:
            return getattr(self._res, attr)
        return default

    def is_using_fallback(self) -> bool:
        return self._res is None

    @property
    def time_of_arrival(self) -> float:
        return self._guard("time_of_arrival", 999.0)

    @property
    def incident_pressure(self) -> float:
        return self._guard("incident_pressure", self.PS_MIN_KPA)

    @property
    def reflected_pressure(self) -> float:
        return self._guard("reflected_pressure", self.PS_MIN_KPA)

    @property
    def positive_phase_duration(self) -> float:
        return self._guard("positive_phase_duration", self.TD_MAX_MS)

    @property
    def incident_impulse(self) -> float:
        return self._guard("incident_impulse", self.I_MAX_KPAMS)

    @property
    def reflected_impulse(self) -> float:
        return self._guard("reflected_impulse", self.I_MAX_KPAMS)

    @property
    def shock_front_velocity(self) -> float:
        return self._guard("shock_front_velocity", 340.0)

    def to_dict(self) -> Dict[str, Any]:
        if self._res is not None:
            return self._res.to_dict()
        else:
            return {
                "neq": self.neq,
                "distance": self.distance,
                "time_of_arrival_ms": 999.0,
                "incident_pressure_kpa": self.PS_MIN_KPA,
                "reflected_pressure_kpa": self.PS_MIN_KPA,
                "positive_phase_duration_ms": self.TD_MAX_MS,
                "incident_impulse_kpams": self.I_MAX_KPAMS,
                "reflected_impulse_kpams": self.I_MAX_KPAMS,
                "shock_front_velocity_mps": 340.0,
                "note": "fallback defaults used due to out-of-range Z"
            }

    def __repr__(self) -> str:
        if self._res is not None:
            return (f"<KingeryBulmashModel: neq={self.neq}, distance={self.distance}, "
                    f"td={self.positive_phase_duration:.2f} ms, Ps={self.incident_pressure:.2f} kPa>")
        else:
            return (f"<KingeryBulmashModel [fallback]: neq={self.neq}, distance={self.distance} "
                    f"(超出KB表范围，使用默认值)>")


# 示例用法
if __name__ == "__main__":
    model = KingeryBulmashModel(neq=10000, distance=9999, unit_system='metric')
    print(model)
    if model.is_using_fallback():
        print("⚠ 当前为 fallback 模式：结果为兜底值，不可用于精确分析")
    else:
        print("✅ 成功查表：结果为真实 KB 模型值")

    print("所有参数及其单位：")
    print(model.to_dict())


# 示例用法
if __name__ == "__main__":
    model = KingeryBulmashModel(neq=10000, distance=50, unit_system='metric')
    print("到达时间：", model.time_of_arrival)
    print("入射峰压：", model.incident_pressure)
    print("反射峰压：", model.reflected_pressure)
    print("正相位持续时间：", model.positive_phase_duration)
    print("入射冲量：", model.incident_impulse)
    print("反射冲量：", model.reflected_impulse)
    print("冲击波前缘速度：", model.shock_front_velocity)
    print("所有参数及其单位：")
    print(model.to_dict())
    data = model.to_dict()
    print("计算结果：", data)
