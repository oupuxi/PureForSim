"""
sim/kingery.py

封装 kingery-bulmash 包，提供便捷的爆炸冲击波参数计算接口。

安装依赖：
    pip install kingery-bulmash
"""
from typing import Literal, Dict, Any

import kingery_bulmash as kb  # :contentReference[oaicite:0]{index=0}


UnitSystem = Literal['metric', 'imperial']


class KingeryBulmashModel:
    """
    Kingery-Bulmash 封装类

    参数:
      - neq: TNT 等效量 (kg 或 lb)
      - distance: 探测点距离 (m 或 ft)
      - unit_system: 'metric' (kg/m) 或 'imperial' (lb/ft)
      - safe: 是否开启越界检查，默认 True
    越界会报错
    """
    def __init__(self,
                 neq: float,
                 distance: float,
                 unit_system: UnitSystem = 'metric',
                 safe: bool = True):
        self.neq = neq
        self.distance = distance
        self.safe = safe
        self.unit = (kb.Units.METRIC if unit_system == 'metric'
                     else kb.Units.IMPERIAL)
        self._compute()

    def _compute(self) -> None:
        try:
            self._res = kb.Blast_Parameters(
                unit_system=self.unit,
                neq=self.neq,
                distance=self.distance,
                safe=self.safe
            )
        except Exception as e:
            # 计算失败时可根据需求捕获或重抛
            raise RuntimeError(f"Kingery-Bulmash 计算出错：{e}")

    @property
    def time_of_arrival(self) -> float:
        """到达时间 (ms)"""
        return self._res.time_of_arrival

    @property
    def incident_pressure(self) -> float:
        """入射峰压 (kPa 或 psi)"""
        return self._res.incident_pressure

    @property
    def reflected_pressure(self) -> float:
        """反射峰压 (kPa 或 psi)"""
        return self._res.reflected_pressure

    @property
    def positive_phase_duration(self) -> float:
        """正相位持续时间 (ms)"""
        return self._res.positive_phase_duration

    @property
    def incident_impulse(self) -> float:
        """入射冲量 (kPa·ms 或 psi·ms)"""
        return self._res.incident_impulse

    @property
    def reflected_impulse(self) -> float:
        """反射冲量 (kPa·ms 或 psi·ms)"""
        return self._res.reflected_impulse

    @property
    def shock_front_velocity(self) -> float:
        """冲击波前缘速度 (m/s 或 ft/s)"""
        return self._res.shock_front_velocity

    def to_dict(self) -> Dict[str, Any]:
        """
        返回所有参数及其单位：
        {
          'neq': …,
          'distance': …,
          'time_of_arrival_ms': …,
          'incident_pressure_kpa': …,
          …,
          'all_units': { 'incident_pressure': 'kPa', … }
        }
        """
        return self._res.to_dict()


# 示例用法
if __name__ == "__main__":
    model = KingeryBulmashModel(neq=10000, distance=500, unit_system='metric')
    print("到达时间：", model.time_of_arrival)
    data = model.to_dict()
    print("计算结果：", data)
