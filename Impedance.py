import numpy as np
from scipy.optimize import brentq
from enum import Enum
from typing import overload, Optional
from dataclasses import dataclass
import cmath
import pandas as pd
from datetime import datetime

t1 = lambda u, c=0.5: -c*u**3 + 2*c*u**2 - c*u
t2 = lambda u, c=0.5: (2-c)*u**3 + (c-3)*u**2 + 1
t3 = lambda u, c=0.5: (c-2)*u**3 + (3-2*c)*u**2 + c*u
t4 = lambda u, c=0.5: c*u**3 - c*u**2

splineFunctionX = lambda px, u, c=0.5: t1(u, c)*px[0] + t2(u,c)*px[1] + t3(u,c)*px[2] + t4(u,c)*px[3]
splineFunctionY = lambda py, u, c=0.5: t1(u, c)*py[0] + t2(u,c)*py[1] + t3(u,c)*py[2] + t4(u,c)*py[3]
splineFunctionZ = lambda pz, u, c=0.5: t1(u, c)*pz[0] + t2(u,c)*pz[1] + t3(u,c)*pz[2] + t4(u,c)*pz[3]

def findUForTargetTime(z, targetTime, c=0.5):
    def z_of_u(u):
        return t1(u, c)*z[0] + t2(u,c)*z[1] + t3(u,c)*z[2] + t4(u,c)*z[3]
    
    f = lambda u: z_of_u(u) - targetTime

    try:
        u_target = brentq(f, 0, 1)
        return u_target
    except Exception:
        u_candidates = [0, 1]
        u_target = min(u_candidates, key=lambda u: abs(z_of_u(u) - targetTime))
        return u_target

def splineTargetTime(px, py, pz, target_time, c=0.5):
    px = np.array(px, dtype=float)
    py = np.array(py, dtype=float)
    pz = np.array(pz, dtype=float)

    # 端点よりも一つ外に端点の値をコピーし、両端で補間可能にする
    px = np.insert(px, [0,-1], [px[0], px[-1]])
    py = np.insert(py, [0,-1], [py[0], py[-1]])
    pz = np.insert(pz, [0,-1], [pz[0], pz[-1]])

    target_index = None

    # pzを走査し、target_timeがある区間の最初のインデックスを探す
    if target_time < pz[0] or pz[-1] < target_time:
        print(f"{pz[0]} ~ {pz[-1]}")
        raise ValueError(f"target_time: {target_time} は範囲外です")
    for i in range(1, len(pz)-2):
        if pz[i] <= target_time and target_time < pz[i+1]:
            target_index = i
            break
    
    if target_index is not None:
        z1 = pz[target_index-1]; z2 = pz[target_index]; z3 = pz[target_index+1]; z4 = pz[target_index+2]
        u = findUForTargetTime([z1, z2, z3, z4], target_time, c)
        z = splineFunctionZ([pz[target_index-1], pz[target_index], pz[target_index+1], pz[target_index+2]], u)
        x = splineFunctionX([px[target_index-1], px[target_index], px[target_index+1], px[target_index+2]], u)
        y = splineFunctionY([py[target_index-1], py[target_index], py[target_index+1], py[target_index+2]], u)
        return [float(x), float(y), float(z)]
    else:
        raise ValueError("indexが見つかりませんでした")

class AngleMode(Enum):
    DEGREE = 1
    RADIAN = 2

@dataclass(frozen=True)
class Impedance:
    frequency: float
    z_re: float
    z_im: float
    
    @property
    def z_complex(self) -> complex:
        return complex(self.z_re, self.z_im)
    
    @property
    def magnitude(self) -> float:
        return abs(self.z_complex)
    
    @property
    def phase_radian(self) -> float:
        return cmath.phase(self.z_complex)
    
    @property
    def phase_degree(self) -> float:
        return np.degrees(self.phase_radian)


class ImpedanceSeries:
    def __init__(self, measurements: list[Impedance]):
        self._measurements: list[Impedance] = measurements
        self._df: pd.DataFrame | None = None

    @property
    def measurements(self) -> list[Impedance]:
        return self._measurements
    
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            data = [
                {
                    "frequency": m.frequency,
                    "z_re": m.z_re,
                    "z_im": m.z_im,
                    "magnitude": m.magnitude,
                    "phase_degree": m.phase_degree
                }
                for m in self._measurements
            ]
            self._df = pd.DataFrame(data).set_index("frequency")
        return self._df
    
    def __len__(self) -> int:
        return len(self._measurements)
    
    def __getitem__(self, index: int) -> Impedance:
        return self._measurements[index]
    
    def getRealImagImpedance(self, min_f:float=0, max_f:float=np.inf) -> tuple[np.ndarray, np.ndarray]:
        df_filtered = self._getDataFrameFilteredByFrequency(min_f, max_f)
        return df_filtered["z_re"].to_numpy(), df_filtered["z_im"].to_numpy()

    def getImpedance(self, min_f:float=0, max_f:float=np.inf) -> np.ndarray:
        z_re, z_im = self.getRealImagImpedance()
        return np.concatenate((z_re, z_im))
    
    def getPhase(self, min_f:float=0, max_f:float=np.inf) -> np.ndarray:
        df_filtered = self._getDataFrameFilteredByFrequency(min_f, max_f)
        return df_filtered["phase_degree"].to_numpy()
    
    def getMagnitude(self, min_f:float=0, max_f:float=np.inf) -> np.ndarray:
        df_filtered = self._getDataFrameFilteredByFrequency(min_f, max_f)
        return df_filtered["magnitude"].to_numpy()
    
    def getFrequency(self, min_f:float=0, max_f:float=np.inf) -> np.ndarray:
        df_filtered = self._getDataFrameFilteredByFrequency(min_f, max_f)
        return df_filtered.index.to_numpy()
    
    def getDataFrame(self, min_f:float=0, max_f:float=np.inf) -> pd.DataFrame:
        return self._getDataFrameFilteredByFrequency(min_f, max_f)

    def _getDataFrameFilteredByFrequency(self, min_f:float, max_f:float) -> pd.DataFrame:
        df = self.df
        return df[(df.index >= min_f) and (df.index <= max_f)]

class TimeImpedance(Impedance):
    time: datetime

class ThreeDimentionImpedance:
    def __init__(self, measurements: list[TimeImpedance]):
        self._measurements = measurements
        self._df:pd.DataFrame | None = None

    @property
    def measurements(self):
        return self._measurements
    
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            data = [
                {
                    "time": m.time,
                    "frequency": m.frequency,
                    "z_re": m.z_re,
                    "z_im": m.z_im,
                    "magnitude": m.magnitude,
                    "phase_degree": m.phase_degree
                }
                for m in self._measurements
            ]
            self._df = pd.DataFrame(data).set_index(["time", "frequency"])
        return self._df
    
    def __len__(self) -> int:
        return len(self._measurements)
    
    def __getitem__(self, index: int) -> TimeImpedance:
        return self._measurements[index]
    
    def getImpedanceAtFrequency(self, f:float) ->np.ndarray:
        pass

    def getRealImagImpedanceAtFrequency(self, f:float) -> tuple[np.ndarray, np.ndarray]:
        pass

    def getPhaseAtFrequency(self, f:float) -> np.ndarray:
        pass

    def getMagnitudeAtFrequency(self, f:float) -> np.ndarray:
        pass

    def getTimeAtFrequency(self, f:float) -> np.ndarray:
        pass

    def getDataFrameAtFrequency(self, f:float) -> pd.DataFrame:
        pass

    # --- 補間値を出力 ---
    def getImpedanceAtTime(self, t) -> np.ndarray:
        pass

    def getRealImagImpedanceAtTime(self, t:float) -> tuple[np.ndarray, np.ndarray]:
        pass

    def getPhaseAtTime(self, t:float) -> np.ndarray:
        pass

    def getFrequencyAtTime(self, t:float) -> np.ndarray:
        pass

    def getDataFrameAtTime(self, t:float) -> pd.DataFrame:
        pass



"""
Impedance:
    property:
        frequency: float
        z_re: float
        z_im: float
        z_complex: complex
        magnitude: float
        phase_degree: float
        phase_radian

TimeImpedance(Impedance):
    property:
        time: datetime

ImpedanceSeries:
    property:
        measurements: list[Impedance]
        df: pandas.DataFrame

    operation:
        getImpedance(min_f, max_f): numpy.ndarray   [z_re1, z_re2, ..., z_im1, z_im2, ...]
        getRealImagImpedance(): numpy.ndarray, numpy.ndarray    [z_re1, z_re2, ...], [z_im1, z_im2, ...]
        getPhase(): numpy.ndarray
        getMagnitude(): numpy.ndarray
        getFrequency(): numpy.ndarray
        getDataFrame(): pandas.DataFrame

        __len__(): int  データ点数を返す
        __getitem__(index): Impedance 

3DImpedance:
    property:
        measurements: list[TimeImpedance]
        df: pandas. DataFrame

    operation:
        getImpedanceAtFrequency(f): numpy.ndarray   [z_re1, z_re2, ..., z_im1, z_im2, ...]
        getRealImagImpedanceAtFrequency(f): numpy.ndarray, numpy.ndarray    [z_re1, z_re2, ...], [z_im1, z_im2, ...]
        getPhaseAtFrequency(f): numpy.ndarray
        getMagnitudeAtFrequency(f): numpy.ndarray
        getTimeAtFrequency(f): numpy.ndarray
        getDataFrameAtFrequency(f): pandas.DataFrame

        --- 補間値を出力 ---
        getImpedanceAtTime(t): numpy.ndarray   [z_re1, z_re2, ..., z_im1, z_im2, ...]
        getRealImagImpedanceAtTime(t): numpy.ndarray, numpy.ndarray    [z_re1, z_re2, ...], [z_im1, z_im2, ...]
        getPhaseAtTime(t): numpy.ndarray
        getFrequencyAtTime(t): numpy.ndarray
        getDataFrameAtTime(t): pandas.DataFrame

        __len__(): int  データ点数を返す
        __getitem__(index): Impedance 
"""