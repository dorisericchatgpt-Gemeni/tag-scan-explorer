"""
TAG + galvo two-photon scan volume -- computation core.

Scalar Debye wave model (same physics as tag_unified_psf: aplanatic sqrt(cos) weight,
air->tissue Fresnel transmission, propagation with k_z3 in tissue, true J0 TAG phase),
extended with the knobs of the explorer:

  1  TAG beam 1/e^2 radius w_TAG      -> relay magnification M = beam_pupil / w_TAG
  2  TAG offset d from the plane conjugate to the pupil (TAG space; d > 0 = TAG on the
     laser side of that plane, so the TAG field propagates +d before being relayed)
  3  peak TAG power V_T (dioptres)
  4  galvo rotation-axis tilt (scan bow)            -> scanner.py
  5  galvo offset dg from the pupil-conjugate plane -> beam walks s = M2^2 dg tan(a_p) on
     the pupil: off-centre TAG phase (shear) + clipping (dimming)
  6  entrance-pupil offset e from the objective's front focal plane (0 = telecentric):
     sample-side chief-ray angle u = atan(r e / F^2) -> scan lines lean, magnification
     changes with depth, the tilted cone crosses the flat interface obliquely
  7  depth D0 below the tissue surface of the D = 0 focus
  +  objective field curvature (corner sag S, quadratic in field radius)
  +  off-axis vignetting (pupil clipped by an offset circle; corner retained area eta)

Two modes:
  fast    : exact 1-D radial wave integral on axis; off-axis effects as corrections
            (lean geometric, shear by first-order perturbation, dimming by pupil integrals)
  precise : 2-D pupil integral at several field radii (tilted cone, decentred TAG phase,
            clipped pupil), all from the same wave model

All off-axis effects are radially symmetric (pupil walk, chief-ray tilt, vignetting and
field curvature all point along the field radius), so the 2-D work is done on a set of
field radii and interpolated; galvo bow only moves where each field point lands.

Depth z is in micrometres below the tissue surface; optical lengths are in mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import numpy as np
from scipy.special import j0, j1, jn_zeros

from . import tag_unified_psf as tup
from .scanner import fov_grid, scan_metrics

THR = float(np.exp(-2))          # usable threshold: e^-2 of the global brightest point


# =============================================================================
# parameters
# =============================================================================
@dataclass(frozen=True)
class Fixed:
    wavelength_um: float = 0.94
    F_mm: float = 16.8               # objective focal length
    R_mm: float = 10.0               # objective pupil radius (air objective, NA 0.595)
    beam_pupil_mm: float = 10.0      # 1/e^2 beam radius at the objective pupil
    n_tissue: float = 1.36
    tag_freq_hz: float = 190e3
    sound_speed_m_s: float = 980.0
    galvo_optical_deg: float = 14.0  # +- optical scan angle of each galvo
    M2: float = 4.0                  # galvo -> pupil (scan lens / tube lens)

    @property
    def k0(self) -> float:           # rad / mm (vacuum = air)
        return 2 * np.pi / (self.wavelength_um * 1e-3)

    @property
    def k_r(self) -> float:          # TAG radial wave number, rad / mm
        return 2 * np.pi * self.tag_freq_hz / self.sound_speed_m_s * 1e-3

    @property
    def r_zero_mm(self) -> float:    # first zero of J0(k_r rho) in the TAG plane
        return 2.404825557695773 / self.k_r


@dataclass
class Params:
    w_tag_mm: float = 10.0 / (20.0 / 3.34)   # 1.670 mm  (M = 5.99)
    tag_offset_mm: float = 0.0
    vt_peak_D: float = 10.0
    beta_x_deg: float = 0.0
    beta_y_deg: float = 0.0
    galvo_incidence_deg: float = 45.0
    galvo_offset_mm: float = 0.0
    pupil_offset_mm: float = 0.0             # e; 0 = telecentric, F = stop at the lens
    focus_depth_um: float = 279.6            # D0 (current lab geometry, L2 = 9.8 mm)
    field_curv_um: float = 0.0               # corner sag, + = deeper
    vignette_corner: float = 1.0             # retained pupil area fraction at the corner
    n_field: int = 21                        # field points per axis
    n_time: int = 25                         # TAG phases over half a period (c = +1 .. -1)
    piezo_range_um: float = 100.0            # piezo page: objective travel +- (um)


@dataclass
class Result:
    mode: str
    params: Params
    fx: np.ndarray            # (n, n) field-point x at D0, um (after bow)
    fy: np.ndarray
    z: np.ndarray             # (nz,) depth grid, um (before field curvature)
    V: np.ndarray             # (n, n, nz) normalised max_c |E|^4
    X: np.ndarray             # (n, n, nz) true lateral position of each sample, um
    Y: np.ndarray
    Z: np.ndarray             # (n, n, nz) true depth (incl. field curvature), um
    off: np.ndarray           # (n, n, nz) true radial lateral offset of the focus, um
    zd0: np.ndarray           # (n, n) depth of the c = 0 focus (D = 0 surface), um
    summary: dict
    warnings: list = field(default_factory=list)


# =============================================================================
# TAG field (radial, complex) at the TAG plane after the offset d
# =============================================================================
class _QDHT:
    """Quasi-discrete Hankel transform of order 0 (Guizar-Sicairos & Gutierrez-Vega 2004).
    T is symmetric and its own inverse, so propagation is T diag(H) T on scaled vectors."""

    def __init__(self, n: int, r_max: float):
        a = jn_zeros(0, n + 1)
        S = a[-1]
        a = a[:-1]
        self.rho = a * r_max / S
        self.k = a / r_max
        J1 = np.abs(j1(a))
        self.T = 2 * j0(np.outer(a, a) / S) / (np.outer(J1, J1) * S)
        self.sc = J1 / r_max

    def propagate(self, f: np.ndarray, H: np.ndarray) -> np.ndarray:
        """f: (..., n) samples at self.rho; H: (n,) transfer function at self.k."""
        F1 = f / self.sc
        F2 = (F1 @ self.T) * H
        return (F2 @ self.T) * self.sc


_QDHT_CACHE: dict = {}


def tag_field(p: Params, fx: Fixed, cs: np.ndarray, n_hankel: int = 2048):
    """Return (rho_grid_mm, E[c, rho]) of the TAG output field after propagating d.
    The Gaussian amplitude is included (beam 1/e^2 radius w_TAG)."""
    Phi = 2 * fx.k0 * (p.vt_peak_D * 1e-3) / fx.k_r ** 2
    if abs(p.tag_offset_mm) < 1e-9:
        rho = np.linspace(0, 1.6 * p.w_tag_mm, 1500)
        E = np.exp(-(rho / p.w_tag_mm) ** 2)[None, :] * np.exp(1j * Phi * cs[:, None] * j0(fx.k_r * rho)[None, :])
        return rho, E
    r_max = max(6.0, 4.0 * p.w_tag_mm)
    key = (n_hankel, round(r_max, 6))   # 1024 pts ~1 %, 2048 pts ~0.4 % vs 2-D FFT reference
    if key not in _QDHT_CACHE:
        _QDHT_CACHE[key] = _QDHT(n_hankel, r_max)
    q = _QDHT_CACHE[key]
    E0 = np.exp(-(q.rho / p.w_tag_mm) ** 2)[None, :] * np.exp(1j * Phi * cs[:, None] * j0(fx.k_r * q.rho)[None, :])
    kz = np.sqrt(np.maximum(fx.k0 ** 2 - q.k ** 2, 0.0))
    H = np.exp(1j * (kz - fx.k0) * p.tag_offset_mm)
    return q.rho, q.propagate(E0, H)


def _interp_c(rho_grid, E, rho):
    """Complex interpolation of E[c, rho_grid] at points rho (any shape) -> (nc, *rho.shape)."""
    out = np.empty((E.shape[0],) + np.shape(rho), complex)
    rr = np.ravel(rho)
    for i in range(E.shape[0]):
        out[i] = (np.interp(rr, rho_grid, E[i].real, right=0.0)
                  + 1j * np.interp(rr, rho_grid, E[i].imag, right=0.0)).reshape(np.shape(rho))
    return out


# =============================================================================
# 1-D on-axis integral
# =============================================================================
class _Axial1D:
    def __init__(self, fx: Fixed, M: float, nr: int = 1400):
        self.fx = fx
        r = np.linspace(0, fx.R_mm, nr + 1)[1:] - 0.5 * fx.R_mm / nr
        self.r = r
        k0, n3 = fx.k0, fx.n_tissue
        kt = k0 * r / fx.F_mm
        self.kz1 = np.sqrt(k0 ** 2 - kt ** 2)
        self.kz3 = np.sqrt((n3 * k0) ** 2 - kt ** 2)
        cos1 = self.kz1 / k0
        fres = tup._fresnel_avg(self.kz1, self.kz3, 1.0, n3)
        self.w = np.sqrt(cos1) / cos1 * fres * kt * (k0 / fx.F_mm) * (r[1] - r[0])
        self.M = M

    def I2(self, Ep: np.ndarray, h_mm: float, z_um: np.ndarray) -> np.ndarray:
        """Ep: (nc, nr) pupil field; returns |E|^4 (nc, nz)."""
        vec = Ep * (self.w * np.exp(-1j * self.kz1 * h_mm))[None, :]
        Mz = np.exp(1j * np.outer(self.kz3, z_um * 1e-3))
        return np.abs(vec @ Mz) ** 4


def calibrate_h(p: Params, fx: Fixed) -> float:
    """Air focus distance beyond the interface so that the D = 0 (c = 0) 2P peak is at D0."""
    M = fx.beam_pupil_mm / p.w_tag_mm
    ax = _Axial1D(fx, M)
    Ep = np.exp(-(ax.r / fx.beam_pupil_mm) ** 2)[None, :]          # c = 0, d = 0
    h = p.focus_depth_um * 1e-3 / fx.n_tissue
    for _ in range(5):
        z = p.focus_depth_um + np.linspace(-80, 80, 641)
        zp = tup.parabolic_peak(z, ax.I2(Ep, h, z)[0])
        h += (p.focus_depth_um - zp) * 1e-3 / fx.n_tissue
    return h


# =============================================================================
# 2-D pupil integral at one field radius
# =============================================================================
def _profile_2d(fx: Fixed, M: float, h: float, z: np.ndarray, rho_g, Etag,
                u1: float, s, delta: float, npx: int, dh=None):
    """Return (I2[c, z], zpk[c], xoff[c]) for a field point whose chief ray leaves the
    objective at angle u1 (air), with the beam walked by s (mm) and the pupil clipped by a
    circle offset delta (mm), all along +x (the field radius). xoff is the lateral offset of
    the focus from the refracted chief ray through the c = 0 focus depth (um).
    Piezo page: s may be an array (one beam walk per row) and dh (mm, per row) is how far
    the objective moved toward the sample for that row (its air focus goes dh deeper)."""
    k0, n3, F, R = fx.k0, fx.n_tissue, fx.F_mm, fx.R_mm
    g = (np.arange(-npx, npx + 1) + 0.0) * (R / npx)
    PX, PY = np.meshgrid(g, g, indexing="ij")
    m = (PX ** 2 + PY ** 2 <= R ** 2) & ((PX - delta) ** 2 + PY ** 2 <= R ** 2)
    px, py = PX[m], PY[m]
    sth2 = (px ** 2 + py ** 2) / F ** 2
    cth = np.sqrt(1 - sth2)
    ux, uy, uz = px / F, py / F, cth
    cu, su = np.cos(u1), np.sin(u1)
    uxt, uzt = ux * cu + uz * su, -ux * su + uz * cu          # cone tilted by u1 about y
    kx, ky, kz1 = k0 * uxt, k0 * uy, k0 * uzt
    kz3 = np.sqrt(np.maximum((n3 * k0) ** 2 - kx ** 2 - ky ** 2, 0.0))
    fres = tup._fresnel_avg(np.maximum(kz1, 1e-9), kz3, 1.0, n3)
    w = np.sqrt(cth) / cth * fres * (R / npx) ** 2 / F ** 2
    if np.ndim(s) == 0:
        Ep = _interp_c(rho_g, Etag, np.hypot(px - s, py) / M)   # (nc, N), Gaussian included
    else:
        Ep = np.stack([_interp_c(rho_g, Etag[i:i + 1], np.hypot(px - si, py) / M)[0]
                       for i, si in enumerate(np.asarray(s))])
    tan3 = np.tan(np.arcsin(su / n3))
    # the tilted cone aims at the air point (0, h): its chief ray meets the interface at
    # x = -h tan(u1) and continues in tissue as x(z) = -h tan(u1) + z tan(u3)
    x_int = -h * np.tan(u1)
    vec = Ep * (w * np.exp(-1j * (kz1 * h - kx * x_int)))[None, :]
    if dh is not None:
        # objective moved toward the sample by dh: the whole converging cone translates
        # rigidly, so its air aim point moves straight down (h -> h + dh) with no lateral
        # change. Paraxially the tissue focus therefore does not lean even when the
        # objective is not telecentric (unlike TAG, which changes the cone's convergence).
        vec = vec * np.exp(-1j * np.outer(dh, kz1))
        # each piezo focus sits at a fixed lateral position, so search along a VERTICAL
        # line through the D0 point of the refracted chief ray instead of along that ray
        zc = z[z.size // 2] * 1e-3
        vec = vec * np.exp(1j * kx * zc * tan3)[None, :]
        tan3 = 0.0
    vec = vec.astype(np.complex64)
    kxl = (kx * tan3 + kz3).astype(np.float32)                 # phase rate along the search line

    kxl = kxl.astype(np.float64)
    dzm = (z[1] - z[0]) * 1e-3
    step = np.exp(1j * kxl * dzm)                              # z grid is uniform

    def axial(V):
        out = np.empty((V.shape[0], z.size), np.float64)
        for a in range(0, z.size, 128):
            nb = min(128, z.size - a)
            blk = np.empty((kxl.size, nb), np.complex128)
            blk[:, 0] = np.exp(1j * kxl * z[a] * 1e-3)
            blk[:, 1:] = step[:, None]
            Mz = np.cumprod(blk, axis=1).astype(np.complex64)   # exp(i k z) by recurrence
            out[:, a:a + nb] = np.abs(V @ Mz) ** 4
        return out

    I = axial(vec)
    zpk = np.array([tup.parabolic_peak(z, I[i]) for i in range(I.shape[0])])
    # lateral search at each phase's peak depth, relative to the chief ray
    xs = np.linspace(-25, 25, 251)
    rows = vec * np.exp(1j * (np.outer(zpk * 1e-3, kx * tan3 + kz3))).astype(np.complex64)
    Lx = np.abs(rows @ np.exp(1j * np.outer(kx, xs * 1e-3)).astype(np.complex64)) ** 4
    xoff = np.array([tup.parabolic_peak(xs, Lx[i]) for i in range(Lx.shape[0])])
    I = axial(vec * np.exp(1j * np.outer(xoff * 1e-3, kx)).astype(np.complex64))
    zpk = np.array([tup.parabolic_peak(z, I[i]) for i in range(I.shape[0])])
    return I, zpk, xoff


def _pupil_factor(fx: Fixed, s: float, delta: float, n: int = 161) -> float:
    """2P peak factor at c = 0 from beam walk s and vignetting offset delta: |sum P|^4 ratio."""
    R = fx.R_mm
    g = np.linspace(-R, R, n)
    PX, PY = np.meshgrid(g, g, indexing="ij")
    ap = PX ** 2 + PY ** 2 <= R ** 2
    cth = np.sqrt(np.clip(1 - (PX ** 2 + PY ** 2) / fx.F_mm ** 2, 1e-6, 1))
    base = np.where(ap, np.sqrt(cth) / cth, 0.0)
    ref = (base * np.exp(-(PX ** 2 + PY ** 2) / fx.beam_pupil_mm ** 2)).sum()
    m = ap & ((PX - delta) ** 2 + PY ** 2 <= R ** 2)
    val = (np.where(m, base, 0) * np.exp(-((PX - s) ** 2 + PY ** 2) / fx.beam_pupil_mm ** 2)).sum()
    return float((val / ref) ** 4)


def _vig_offset(fx: Fixed, eta: float) -> float:
    """Offset of the clipping circle that leaves area fraction eta of the pupil."""
    if eta >= 0.9999:
        return 0.0
    R = fx.R_mm
    lo, hi = 0.0, 2 * R
    for _ in range(60):
        d = 0.5 * (lo + hi)
        A = 2 * R ** 2 * np.arccos(d / (2 * R)) - 0.5 * d * np.sqrt(4 * R ** 2 - d ** 2)
        if A / (np.pi * R ** 2) > eta:
            lo = d
        else:
            hi = d
    return 0.5 * (lo + hi)


def _shear_perturbation(fx: Fixed, M: float, Phi: float, s: float) -> float:
    """First-order lateral focus shift at c = +1 from a TAG phase decentred by s (mm).
    Weighted least-squares tilt of -s.grad(phi) over the pupil -> dx = -g F / k0 (um)."""
    r = np.linspace(0, fx.R_mm, 2001)[1:]
    cth = np.sqrt(1 - (r / fx.F_mm) ** 2)
    a = np.exp(-(r / fx.beam_pupil_mm) ** 2) * np.sqrt(cth) / cth
    w = a ** 2
    g = s * Phi * (fx.k_r / M) * np.sum(w * j1(fx.k_r * r / M) * r ** 2) / np.sum(w * r ** 3)
    return -g * fx.F_mm / fx.k0 * 1e3


def _envelope(I: np.ndarray, zpk: np.ndarray, z: np.ndarray, nsub: int = 8) -> np.ndarray:
    """max over a CONTINUOUS TAG sweep. The sampled phases are ~10 um apart near the sweep
    centre, more than the axial PSF, so max over samples alone leaves dark stripes. Between
    neighbouring phases the profile is moved along z by the peak displacement and blended."""
    V = I.max(axis=0).copy()
    for i in range(len(zpk) - 1):
        dz = zpk[i + 1] - zpk[i]
        for t in np.linspace(0, 1, nsub + 2)[1:-1]:
            a = np.interp(z - t * dz, z, I[i], left=0, right=0)
            b = np.interp(z + (1 - t) * dz, z, I[i + 1], left=0, right=0)
            V = np.maximum(V, (1 - t) * a + t * b)
    return V


# =============================================================================
# driver
# =============================================================================
def compute(p: Params, mode: str = "fast", fx: Fixed = Fixed(), cancel=lambda: False,
            progress=lambda f: None) -> Result | None:
    warn = []
    piezo = mode.startswith("piezo")
    precise = mode in ("precise", "piezo_precise")
    M = fx.beam_pupil_mm / p.w_tag_mm
    Phi = 0.0 if piezo else 2 * fx.k0 * (p.vt_peak_D * 1e-3) / fx.k_r ** 2
    cs = np.cos(2 * np.pi * np.linspace(0, 0.5, p.n_time))
    ic0 = int(np.argmin(np.abs(cs)))
    # piezo: objective displacement per sample (mm, + = toward the sample); index 0 is the
    # shallow end, like c = +1 on the TAG page
    dh = (-p.piezo_range_um * cs * 1e-3) if piezo else None
    if not piezo and p.w_tag_mm > fx.r_zero_mm:
        warn.append(f"TAG beam radius {p.w_tag_mm:.2f} mm exceeds the J0 first zero "
                    f"({fx.r_zero_mm:.2f} mm): the outer beam sees reversed TAG phase.")

    # ---- field geometry (galvo bow) ----
    fxg, fyg, tanpx, tanpy = fov_grid(p.n_field, fx.galvo_optical_deg, p.beta_x_deg, p.beta_y_deg,
                                      p.galvo_incidence_deg, fx.M2, fx.F_mm)
    rf = np.hypot(fxg, fyg)                                   # um
    c_ = p.n_field // 2
    r_edge = float(np.hypot(fxg[:, c_], fyg[:, c_]).max())      # end of the centre fast line
    r_corner = float(max(np.hypot(fxg[i, j], fyg[i, j]) for i in (0, -1) for j in (0, -1)))
    radii = np.unique(np.round(np.r_[np.linspace(0, 1, 5) * r_corner, r_edge, r_corner, rf.max()], 3))

    # ---- depth grid ----
    dz_per_D = fx.n_tissue * fx.F_mm ** 2 / M ** 2              # um per dioptre (paraxial)
    if piezo:
        amp = fx.n_tissue * p.piezo_range_um
    else:
        amp = dz_per_D * p.vt_peak_D / max(0.1, 1 - abs(p.tag_offset_mm) * p.vt_peak_D * 1e-3)
    W = 1.3 * amp + 60
    step = 1.0 if precise else 0.5
    z = p.focus_depth_um + np.arange(-W, W + step, step)

    h = calibrate_h(p, fx)
    if piezo:     # no TAG: flat Gaussian beam at the pupil, one identical row per sample
        q = replace(p, vt_peak_D=0.0, tag_offset_mm=0.0)
        rho_g, Etag = tag_field(q, fx, cs)
    else:
        rho_g, Etag = tag_field(p, fx, cs, 2048 if precise else 1024)
    delta_c = _vig_offset(fx, p.vignette_corner)

    nR = radii.size
    Vr = np.empty((nR, z.size))
    zpk = np.empty((nR, cs.size))
    lat = np.empty((nR, cs.size))         # radial lateral offset of each phase's focus, um
    pkr = np.empty((nR, cs.size))         # peak |E|^4 of each phase

    if not precise:
        ax = _Axial1D(fx, M)
        Ep = _interp_c(rho_g, Etag, ax.r / M)
        if piezo:   # h -> h + dh (objective closer = focus deeper)
            Ep = Ep * np.exp(-1j * np.outer(dh, ax.kz1))
        I = ax.I2(Ep, h, z)
        zp0 = np.array([tup.parabolic_peak(z, I[i]) for i in range(cs.size)])
        env0 = _envelope(I, zp0, z)
        for k, r in enumerate(radii):
            r_mm = r * 1e-3
            s = fx.M2 ** 2 * p.galvo_offset_mm * r_mm / fx.F_mm
            delta = delta_c * (r / r_corner if r_corner > 0 else 0)
            fac = _pupil_factor(fx, s, delta)      # piezo pupil walk (<~10 um) neglected here
            u1 = np.arctan(r_mm * p.pupil_offset_mm / fx.F_mm ** 2)
            tan3 = np.tan(np.arcsin(np.sin(u1) / fx.n_tissue))
            dx1 = _shear_perturbation(fx, M, Phi, s)
            Vr[k] = env0 * fac
            zpk[k] = zp0
            if piezo:   # rigid translation of the cone: no lean or shear to first order
                lat[k] = 0.0
            else:
                lat[k] = (zp0 - zp0[ic0]) * tan3 + dx1 * cs
            pkr[k] = I.max(axis=1) * fac
    else:
        npx = int(np.clip(np.ceil(Phi * fx.k_r / M * 0.6 * fx.R_mm / 1.5), 64, 150))
        for k, r in enumerate(radii):
            if cancel():
                return None
            r_mm = r * 1e-3
            s = fx.M2 ** 2 * p.galvo_offset_mm * r_mm / fx.F_mm
            if piezo:   # the pupil moves with the objective: extra walk -dh * tan(alpha_p)
                s = s - dh * (r_mm / fx.F_mm)
            delta = delta_c * (r / r_corner if r_corner > 0 else 0)
            u1 = np.arctan(r_mm * p.pupil_offset_mm / fx.F_mm ** 2)
            tan3 = np.tan(np.arcsin(np.sin(u1) / fx.n_tissue))
            I, zp, xo = _profile_2d(fx, M, h, z, rho_g, Etag, u1, s, delta, npx, dh)
            Vr[k] = _envelope(I, zp, z)
            zpk[k] = zp
            lat[k] = (xo - xo[ic0]) if piezo else (zp - zp[ic0]) * tan3 + (xo - xo[ic0])
            pkr[k] = I.max(axis=1)
            progress((k + 1) / nR)

    gmax = Vr.max()
    Vr /= gmax
    pkr /= gmax

    # lateral offset as a function of depth, per radius (from the focus at that depth)
    offr = np.empty_like(Vr)
    for k in range(nR):
        o = np.argsort(zpk[k])
        offr[k] = np.interp(z, zpk[k][o], lat[k][o])

    # ---- assemble the volume on the field grid ----
    n = p.n_field
    wgt = np.clip(np.interp(rf.ravel(), radii, np.arange(nR)), 0, nR - 1)
    i0 = np.floor(wgt).astype(int)
    i1 = np.minimum(i0 + 1, nR - 1)
    t = (wgt - i0)[:, None]
    V = ((1 - t) * Vr[i0] + t * Vr[i1]).reshape(n, n, -1)
    off = ((1 - t) * offr[i0] + t * offr[i1]).reshape(n, n, -1)
    zd0 = ((1 - t[:, 0]) * zpk[i0, ic0] + t[:, 0] * zpk[i1, ic0]).reshape(n, n)
    fc = p.field_curv_um * (rf / r_corner) ** 2 if r_corner > 0 else np.zeros_like(rf)
    zd0 = zd0 + fc
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(rf > 0, fxg / rf, 0.0)
        uy = np.where(rf > 0, fyg / rf, 0.0)
    X = fxg[..., None] + off * ux[..., None]
    Y = fyg[..., None] + off * uy[..., None]
    Z = z[None, None, :] + fc[..., None]

    # ---- summary ----
    def stats(k):
        ok = pkr[k] >= THR
        fck = p.field_curv_um * (radii[k] / r_corner) ** 2 if r_corner > 0 else 0.0
        d = {"sweep": (zpk[k].min() + fck, zpk[k].max() + fck),
             "usable": ((zpk[k][ok].min() + fck, zpk[k][ok].max() + fck) if ok.any() else (np.nan, np.nan)),
             "brightness": float(pkr[k].max()),
             "shear_um": float(lat[k][0] - lat[k][-1]),       # c = +1 (shallow) minus c = -1 (deep)
             "d0_depth": float(zpk[k][ic0] + fck)}
        d["centre"] = 0.5 * (d["sweep"][0] + d["sweep"][1])
        return d

    k_edge = int(np.argmin(np.abs(radii - r_edge)))
    k_corner = int(np.argmin(np.abs(radii - r_corner)))
    summ = {"axis": stats(0), "edge": stats(k_edge), "corner": stats(k_corner),
            "M_net": M, "dz_per_D_um": dz_per_D, "Phi_rad": Phi, "h_air_mm": h,
            "fov_x_mm": 2 * r_edge * 1e-3, "r_corner_um": r_corner, "scan": scan_metrics(fxg, fyg),
            "radii_um": radii, "zpk": zpk, "lat": lat, "peak": pkr, "cs": cs,
            "drive": "piezo" if piezo else "tag",
            "dz_per_um_piezo": (float((zpk[0].max() - zpk[0].min()) / (2 * p.piezo_range_um))
                                if piezo and p.piezo_range_um > 0 else np.nan)}
    if (zpk.min() + min(0.0, p.field_curv_um)) < 0:
        warn.append("Part of the sweep lies above the tissue surface (in air); "
                    "the model treats it as tissue, so those depths are not valid.")
    if precise and npx >= 150:
        warn.append("Pupil sampling hit its cap (150 px radius); very large V_T may alias.")
    return Result(mode, p, fxg, fyg, z, V, X, Y, Z, off, zd0, summ, warn)


def centroid_shift(p: Params, fx: Fixed = Fixed()) -> tuple[float, float]:
    """On-axis shift (um) of the |E|^4 centroid at c = +1 (top) and c = -1 (bottom) caused
    by the TAG offset, relative to the same phases at d = 0. Centroids move smoothly with
    d; the brightest point can hop between Bessel side lobes at the sweep ends."""
    M = fx.beam_pupil_mm / p.w_tag_mm
    cs = np.array([1.0, -1.0])
    h = calibrate_h(p, fx)
    ax = _Axial1D(fx, M)
    z = p.focus_depth_um + np.arange(-250, 250.25, 0.25)

    def cen(q):
        I = ax.I2(_interp_c(*tag_field(q, fx, cs), ax.r / M), h, z)
        return (I * z).sum(1) / I.sum(1)

    a = cen(p)
    b = cen(Params(**{**asdict(p), "tag_offset_mm": 0.0}))
    return float(a[0] - b[0]), float(a[1] - b[1])
