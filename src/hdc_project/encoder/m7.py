"""Module M7: Segment manager and reset policies."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .m0 import M0_NewKey

__all__ = [
    "M7_SegMgr",
    "M7_SegMgr_new",
    "M7_curKey",
    "M7_onBoundary",
]


class M7_SegMgr:
    def __init__(self, seed: int, D: int):
        self.D = int(D)
        self._rng = np.random.default_rng(seed)
        self.seg_idx = -1
        self._K: Optional[np.ndarray] = None
        self.onBoundary()

    def curKey(self) -> np.ndarray:
        assert self._K is not None
        self._K.setflags(write=False)
        return self._K

    def onBoundary(self) -> np.ndarray:
        self.seg_idx += 1
        sub = int(self._rng.integers(0, 2**31 - 1))
        K = M0_NewKey(sub, self.D)
        K.setflags(write=False)
        self._K = K
        return self._K

    def nextKey_correlated(self, rho: float) -> np.ndarray:
        assert self._K is not None
        f = (1.0 - float(rho)) / 2.0
        flips = self._rng.random(self.D) < f
        Kp = self._K.copy()
        Kp[flips] = -Kp[flips]
        Kp.setflags(write=False)
        self.seg_idx += 1
        self._K = Kp
        return self._K


def M7_SegMgr_new(seed: int, D: int) -> M7_SegMgr:
    return M7_SegMgr(seed, D)


def M7_curKey(segmgr: M7_SegMgr) -> np.ndarray:
    return segmgr.curKey()


def M7_onBoundary(segmgr: M7_SegMgr) -> np.ndarray:
    return segmgr.onBoundary()
