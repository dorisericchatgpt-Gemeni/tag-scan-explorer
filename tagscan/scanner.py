"""
Two-galvo scan geometry with exact vector reflection.

Layout ("best arrangement"): X galvo -> 1:1 relay -> Y galvo -> scan/tube lens (M2) ->
objective pupil. Each mirror sits on a plane conjugate to the pupil.

Each mirror's rotation axis should lie in the mirror surface AND be perpendicular to the
plane of incidence. If the axis is tilted by beta inside the mirror surface (no longer
perpendicular to the plane of incidence), the reflected beam sweeps a cone instead of a
plane and the scan line bows. With beta = 0 the two scans are exactly separable and the
field is a square grid.

Angles: optical scan angle = 2 x mechanical angle. An ideal afocal relay maps
tan(angle) -> tan(angle)/M2, and an ideal (distortion-free) objective maps the pupil
angle to the field position x = F tan(alpha_p).
"""
from __future__ import annotations

import numpy as np


def _rot(axis: np.ndarray, ang: float, v: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of vector(s) v (..., 3) about unit axis by ang."""
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(a, v) * s + np.outer(v @ a, a).reshape(v.shape) * (1 - c)


def _reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    return d - 2 * (d @ n)[..., None] * n


class _Mirror:
    """Mirror in its own frame: nominal incoming beam along +z, plane of incidence = xz,
    nominal reflection toward the output frame (e_scan, e_cross, e_axis)."""

    def __init__(self, incidence_deg: float, beta_deg: float):
        phi = np.radians(incidence_deg)
        self.n0 = np.array([np.sin(phi), 0.0, -np.cos(phi)])
        a0 = np.array([0.0, 1.0, 0.0])                    # perpendicular to plane of incidence
        t = np.cross(self.n0, a0)                         # in the mirror surface, in the plane
        b = np.radians(beta_deg)
        self.axis = np.cos(b) * a0 + np.sin(b) * t
        d0 = np.array([0.0, 0.0, 1.0])
        self.e_axis = _reflect(d0, self.n0)
        # scan direction defined by the ideal (beta = 0) mirror
        d_small = _reflect(d0, _rot(a0, 1e-6, self.n0))
        es = d_small - self.e_axis
        es -= (es @ self.e_axis) * self.e_axis
        self.e_scan = es / np.linalg.norm(es)
        self.e_cross = np.cross(self.e_axis, self.e_scan)

    def out_tan(self, d_in: np.ndarray, mech: float) -> tuple[np.ndarray, np.ndarray]:
        n = _rot(self.axis, mech, self.n0)
        d = _reflect(d_in, n)
        za = d @ self.e_axis
        return (d @ self.e_scan) / za, (d @ self.e_cross) / za


def fov_grid(n: int, optical_deg: float, beta_x: float, beta_y: float,
             incidence_deg: float, M2: float, F_mm: float):
    """Return field positions (fx, fy) in um on an n x n grid of galvo angles, and the
    pupil-angle tangents (tan_px, tan_py). Index [i, j]: i = X galvo, j = Y galvo."""
    th = np.radians(np.linspace(-optical_deg, optical_deg, n)) / 2.0     # mechanical
    mx, my = _Mirror(incidence_deg, beta_x), _Mirror(incidence_deg, beta_y)
    d0 = np.array([0.0, 0.0, 1.0])
    TX = np.empty((n, n))
    TY = np.empty((n, n))
    for i, a in enumerate(th):
        sx, cx = mx.out_tan(d0, a)               # X galvo: scan along its e_scan
        # 1:1 relay inverts angles; the X deviation must arrive along mirror Y's
        # cross axis (y of its frame), the bow component along its scan axis (x).
        d_in = np.array([-cx, -sx, 1.0])
        d_in /= np.linalg.norm(d_in)
        for j, b in enumerate(th):
            sy, cy = my.out_tan(d_in, b)
            TX[i, j] = cy                         # carried X scan
            TY[i, j] = sy                         # Y scan
    # sign convention: field x grows with the X galvo angle, y with the Y galvo angle
    if TX[-1, n // 2] < TX[0, n // 2]:
        TX = -TX
    if TY[n // 2, -1] < TY[n // 2, 0]:
        TY = -TY
    tpx, tpy = TX / M2, TY / M2
    return F_mm * 1e3 * tpx, F_mm * 1e3 * tpy, tpx, tpy
