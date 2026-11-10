"""Regression checks: the explorer's model against tag_unified_psf and independent references.
Run: python -m pytest -q tests"""
import dataclasses

import numpy as np
import pytest
from scipy.special import j0

from tagscan import tag_unified_psf as tup
from tagscan.model import Params, Fixed, compute, centroid_shift, tag_field, _QDHT
from tagscan.scanner import fov_grid, scan_metrics


@pytest.fixture(scope="module")
def ref_axis():
    cfg = tup.apply_sampling_guard(tup.PRESETS["mesoscale"], verbose=False)
    tr = tup.scan_trajectory(cfg, 0.0)
    zp = tr["z_peak"] - cfg.L2_mm * 1e3
    pk = (tr["I1"] ** 2).max(axis=1)
    ok = pk >= np.exp(-2) * pk.max()
    return zp, ok


def test_fast_matches_tag_unified_on_axis(ref_axis):
    zp, ok = ref_axis
    s = compute(Params(), "fast").summary["axis"]
    assert abs(s["sweep"][0] - zp.min()) < 3.0
    assert abs(s["sweep"][1] - zp.max()) < 1.5
    # the usable end is interpolated between phases: it must fall between the reference's
    # last sample below threshold and its first sample above
    first_ok = zp[ok].min()
    last_dim = zp[~ok & (zp < first_ok)].max()
    assert last_dim - 1.5 < s["usable"][0] < first_ok + 1.5
    assert abs(s["d0_depth"] - Params().focus_depth_um) < 0.5


def test_precise_equals_fast_on_axis():
    a = compute(Params(), "fast").summary["axis"]
    b = compute(Params(), "precise").summary
    for k in ("axis", "edge", "corner"):          # telecentric, aligned: every field point the same
        assert abs(b[k]["sweep"][0] - a["sweep"][0]) < 1.0
        assert abs(b[k]["sweep"][1] - a["sweep"][1]) < 1.0
        assert abs(b[k]["brightness"] - 1.0) < 0.02


def test_galvo_shear():
    # independent 2-D wave calculation (FFT pupil, homogeneous tissue): -0.80 um per 0.1 mm at the edge
    pr = compute(Params(galvo_offset_mm=0.3, galvo_offset_y_mm=0.3), "precise").summary
    fa = compute(Params(galvo_offset_mm=0.3, galvo_offset_y_mm=0.3), "fast").summary
    assert pr["edge"]["shear_um"] == pytest.approx(-2.4, abs=0.3)
    assert fa["edge"]["shear_um"] == pytest.approx(pr["edge"]["shear_um"], rel=0.2)
    assert pr["corner"]["shear_um"] < pr["edge"]["shear_um"] < 0     # grows with field radius


def test_hankel_matches_fft_propagation():
    fx, p = Fixed(), Params(tag_offset_mm=40.0)
    cs = np.array([1.0])
    rho, E = tag_field(p, fx, cs)
    # 2-D FFT angular-spectrum reference
    N, L = 1024, 12.0
    x = (np.arange(N) - N // 2) * L / N
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    Phi = 2 * fx.k0 * (p.vt_peak_D * 1e-3) / fx.k_r ** 2
    E0 = np.exp(-(R / p.w_tag_mm) ** 2) * np.exp(1j * Phi * j0(fx.k_r * R))
    k = np.fft.fftfreq(N, L / N) * 2 * np.pi
    KX, KY = np.meshgrid(k, k)
    kz = np.sqrt(np.maximum(fx.k0 ** 2 - KX ** 2 - KY ** 2, 0))
    Ef = np.fft.ifft2(np.fft.fft2(E0) * np.exp(1j * (kz - fx.k0) * p.tag_offset_mm))
    line = Ef[N // 2, N // 2:]
    r_line = x[N // 2:]
    sel = r_line < 1.6
    ref = line[sel]
    got = np.interp(r_line[sel], rho, E[0].real) + 1j * np.interp(r_line[sel], rho, E[0].imag)
    assert np.abs(got - ref).max() < 0.03      # |E| ~ 1.6; FFT reference itself ~1 % off at N 1024


def test_tag_offset_direction():
    top, bot = centroid_shift(Params(tag_offset_mm=40.0))
    assert top < -30 and bot < -5          # whole sweep moves shallower, top end more
    t2, b2 = centroid_shift(Params(tag_offset_mm=-40.0))
    assert t2 > 5 and b2 > 30              # mirrored for d < 0


def test_tilted_cone_against_tag_unified():
    p = Params(pupil_offset_mm=16.8)
    s = compute(p, "precise").summary
    u1 = np.arctan(s["r_corner_um"] * 1e-3 * 16.8 / 16.8 ** 2)
    cfg = tup.apply_sampling_guard(dataclasses.replace(tup.PRESETS["mesoscale"], clip_vignetting=False),
                                   verbose=False)
    zp = tup.scan_trajectory(cfg, u1)["z_peak"]
    th_ref = zp.max() - zp.min()
    th = s["corner"]["sweep"][1] - s["corner"]["sweep"][0]
    # both trackers follow the brightest point, which hops between Bessel side lobes at the
    # dim sweep ends (tag_unified gives 187.3 um at 4.95 deg but 184.6 um at 5.11 deg)
    assert th == pytest.approx(th_ref, abs=5.0)


def test_scanner_square_without_bow():
    fx_, fy_, _, _ = fov_grid(9, 14.0, 0.0, 0.0, 45.0, 4.0, 16.8)
    m = 4
    assert np.allclose(fy_[:, m], 0, atol=1e-9) and np.allclose(fx_[m, :], 0, atol=1e-9)
    assert np.allclose(fx_, -fx_[::-1, :]) and np.allclose(fy_, -fy_[:, ::-1])
    assert fx_[-1, m] == pytest.approx(16.8e3 * np.tan(np.radians(14)) / 4, rel=1e-6)
    bx, by, _, _ = fov_grid(9, 14.0, 5.0, 0.0, 45.0, 4.0, 16.8)
    assert np.abs(by[:, m]).max() > 10         # tilted axis -> the X scan line bows in y


def test_shaft_in_plane_of_incidence_matches_poster_model():
    # independent vector model (LFOV4D poster, shafts tilted into the plane of incidence):
    # field 1.69 x 1.52 mm with angle/M2 mapping, every fast line bows 63 um
    fx_, fy_, _, _ = fov_grid(41, 14.0, 90.0, 90.0, 45.0, 4.0, 16.8)
    m = scan_metrics(fx_, fy_)
    assert m["width_mm"] == pytest.approx(1.69, abs=0.05)
    assert m["height_mm"] == pytest.approx(1.52, abs=0.05)
    assert m["bow_um"] == pytest.approx(63, abs=5)
    assert abs(abs(m["rotation_deg"]) - 90) < 1          # that mirror now sweeps the other axis


def test_mis_mounted_galvo_runs_through_model():
    s = compute(Params(beta_x_deg=90.0, beta_y_deg=90.0), "fast").summary
    assert s["scan"]["bow_um"] > 50
    assert s["edge"]["sweep"][1] - s["edge"]["sweep"][0] > 150


def test_piezo_focus_moves_about_n_per_um():
    s = compute(Params(piezo_range_um=100.0), "piezo_fast").summary
    k = s["dz_per_um_piezo"]
    assert 1.30 < k < 1.45              # ~n_tissue, nudged by depth-dependent spherical aberration
    assert abs(s["axis"]["d0_depth"] - Params().focus_depth_um) < 0.5


def test_piezo_small_range_matches_tag_without_power():
    a = compute(Params(piezo_range_um=0.5), "piezo_fast").summary["axis"]
    b = compute(Params(vt_peak_D=0.05), "fast").summary["axis"]
    assert abs(a["d0_depth"] - b["d0_depth"]) < 0.5
    assert abs(a["sweep"][0] - b["sweep"][0]) < 1.0


def test_piezo_precise_equals_fast_and_is_uniform_when_aligned():
    a = compute(Params(), "piezo_fast").summary
    b = compute(Params(), "piezo_precise").summary
    for k in ("axis", "edge", "corner"):
        assert abs(b[k]["sweep"][0] - a["axis"]["sweep"][0]) < 1.0
        assert abs(b[k]["sweep"][1] - a["axis"]["sweep"][1]) < 1.0
        assert abs(b[k]["brightness"] - 1.0) < 0.02


def test_piezo_has_no_shear_from_telecentricity_or_galvo_offset():
    # the objective translates the whole converging cone: unlike TAG, no lean or shear
    s = compute(Params(pupil_offset_mm=16.8, galvo_offset_mm=1.0), "piezo_precise").summary
    assert abs(s["corner"]["shear_um"]) < 0.5
    t = compute(Params(pupil_offset_mm=16.8), "precise").summary
    assert abs(t["corner"]["shear_um"]) > 5


def test_only_x_galvo_offset_shears_along_x():
    # with only the X mirror misplaced the beam walks along x everywhere, so the shear at
    # the corner is the edge shear along x and nothing along y; the Y edge is unaffected
    r = compute(Params(galvo_offset_mm=0.3), "precise")
    s = r.summary
    c = r.fx.shape[0] // 2
    it, ib = (int(np.argmin(np.abs(r.z - q))) for q in s["axis"]["sweep"])

    def shear(i, j):
        dx, dy = r.X[i, j] - r.fx[i, j], r.Y[i, j] - r.fy[i, j]
        return dx[it] - dx[ib], dy[it] - dy[ib]

    ex, ey = shear(-1, c)
    assert ex == pytest.approx(s["edge"]["shear_um"], abs=0.15) and abs(ey) < 0.05
    assert max(abs(v) for v in shear(c, -1)) < 0.05
    cx, cy = shear(-1, -1)
    assert cx == pytest.approx(ex, rel=0.1) and abs(cy) < 0.2
    assert abs(s["corner"]["shear_t_um"]) > 1.0


def test_usable_range_does_not_depend_on_phase_sampling():
    a = compute(Params(n_time=25), "fast").summary["axis"]["usable"]
    b = compute(Params(n_time=201), "fast").summary["axis"]["usable"]
    assert abs(a[0] - b[0]) < 1.5 and abs(a[1] - b[1]) < 1.5


def test_time_average_weights_the_sweep_ends():
    r = compute(Params(focus_depth_um=50.0), "fast")      # little aberration: clean comparison
    c = r.fx.shape[0] // 2
    V, A = r.V[c, c], r.Vavg[c, c]
    assert A.max() == pytest.approx(1.0)
    mid = int(np.argmin(np.abs(r.z - 50.0)))
    end = int(np.argmin(np.abs(r.z - r.summary["axis"]["usable"][1])))
    assert A[end] / A[mid] > 1.5 * V[end] / V[mid]       # longer dwell at the turning point


def test_spherical_aberration_ratio_and_warning():
    deep = compute(Params(), "fast")
    shallow = compute(Params(focus_depth_um=50.0), "fast")
    assert deep.summary["sa_2p_ratio"] == pytest.approx(0.18, abs=0.03)
    assert shallow.summary["sa_2p_ratio"] > 0.8
    assert any("spherical aberration" in w for w in deep.warnings)
    assert not any("spherical aberration" in w for w in shallow.warnings)
