"""
TAG scan-volume explorer -- desktop GUI (PySide6 + pyvistaqt + matplotlib).

Run:  python -m tagscan      (or the `tagscan` console script)
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import asdict, replace

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore, QtWidgets
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

from .model import Params, Fixed, compute, THR

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Microsoft JhengHei", "Microsoft YaHei", "PingFang TC",
                                          "Noto Sans CJK TC"]
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Microsoft JhengHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

DIM_LEVEL = 0.02     # faint envelope: the focus passes here but 2P < e^-2

# (attribute, label, min, max, step, decimals, tooltip)
SPECS = [
    ("w_tag_mm", "1. TAG 入射光 1/e² 半徑 (mm)", 0.5, 3.0, 0.01, 2,
     "relay 放大率 M = 10 mm / w_TAG 跟著變（物鏡光寬固定 10 mm）。\n"
     "光寬小 → 每 D 掃的深度 ∝ 1/M² 變小，但只用到 J0 中央較平的部分。"),
    ("tag_offset_mm", "2. TAG 離共軛面 d (mm)", -50.0, 50.0, 0.5, 1,
     "TAG 空間的錯位。d > 0：TAG 在共軛面的雷射側（光先傳播 d 才到共軛面）。"),
    ("vt_peak_D", "3. TAG 屈光度峰值 ±V_T (D)", 1.0, 20.0, 0.5, 1, "TAG 光學屈光度振幅。"),
    ("beta_x_deg", "4a. X galvo 轉軸傾斜 β_x (°)", -10.0, 10.0, 0.1, 1,
     "轉軸在鏡面內、偏離「垂直入射平面」的角度 → 掃描線弓形。"),
    ("beta_y_deg", "4b. Y galvo 轉軸傾斜 β_y (°)", -10.0, 10.0, 0.1, 1, "同上，Y 鏡。"),
    ("galvo_incidence_deg", "4c. galvo 入射角 (°)（假設）", 20.0, 70.0, 1.0, 0,
     "鏡面法線與入射光夾角，弓形大小取決於它。預設 45°。"),
    ("galvo_offset_mm", "5. galvo 離共軛面 Δg (mm)", -2.0, 2.0, 0.01, 2,
     "galvo 空間的錯位（X、Y 共用）。光束在光瞳上平移 s = M₂²·Δg·tanα。"),
    ("pupil_offset_mm", "6. 入瞳離前焦面 e (mm)", -16.8, 16.8, 0.1, 1,
     "0 = 遠心（主光線平行光軸）；16.8 = 光闌在鏡頭上。\n"
     "非遠心造成掃描線傾斜與放大率隨深度變，不造成焦面彎曲（那是場曲）。"),
    ("focus_depth_um", "7. D=0 焦點在組織表面下 (µm)", 50.0, 800.0, 1.0, 0,
     "決定空氣→組織的深度球差。279.6 µm = 現行 L2 9.8 mm。"),
    ("field_curv_um", "+ 物鏡場曲：角落下垂 (µm)", -100.0, 100.0, 1.0, 0,
     "整欄深度加上 S·(r/r_角落)²；+ = 角落較深。理想平場 = 0。"),
    ("vignette_corner", "+ 離軸暈影：角落光瞳保留比例", 0.3, 1.0, 0.01, 2,
     "光瞳被偏移圓裁切（貓眼形），偏移量 ∝ 視野半徑。1 = 無暈影。"),
]

FIXED_TEXT = (
    "<b>固定參數</b><br>"
    "λ 0.92 µm；物鏡 F 16.8 mm、空氣、光瞳 R 10 mm、NA 0.595<br>"
    "入物鏡光束 1/e² 半徑 10 mm；組織 n 1.36<br>"
    "TAG 190 kHz、聲速 980 m/s（J0 第一零點 1.97 mm）<br>"
    "galvo ±14° 光學角，M₂ 4，X→1:1 relay→Y（各自與光瞳共軛）<br>"
    "可用 = 2P 峰值 ≥ e⁻² (13.5%) × 全體積最亮點")

NOTE_TEXT = (
    "<b>讀圖</b>：深色 = 可用（2P ≥ e⁻²）；淡色 = 焦點掃過但 2P < e⁻²（≥ 2% 才畫）。<br>"
    "模型：純量 Debye 波動積分（J0 TAG、aplanatic、Fresnel、組織深度球差）。<br>"
    "快速模式：軸上精確，離軸效應為近似修正；精確模式：每個視野半徑做 2D 光瞳積分。<br>"
    "未含：散射、組織不均勻、表面傾斜、偏振、resonant 停留時間。")


class _Worker(QtCore.QObject):
    done = QtCore.Signal(object, int)
    prog = QtCore.Signal(float, int)

    def __init__(self):
        super().__init__()
        self.token = 0

    @QtCore.Slot(object, int)
    def run(self, p, token):
        try:
            res = compute(p, "precise", cancel=lambda: token != self.token,
                          progress=lambda f: self.prog.emit(f, token))
        except Exception:
            traceback.print_exc()
            res = None
        self.done.emit(res, token)


class MainWindow(QtWidgets.QMainWindow):
    request_precise = QtCore.Signal(object, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAG 掃描體積 Explorer")
        self.resize(1700, 980)
        self.p = Params()
        self.res = None
        self.token = 0
        self._first = True

        # ---------------- left: parameters
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        form = QtWidgets.QGridLayout()
        self.spins = {}
        for row, (attr, label, lo, hi, step, dec, tip) in enumerate(SPECS):
            lab = QtWidgets.QLabel(label)
            lab.setToolTip(tip)
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setSingleStep(step); sp.setDecimals(dec)
            sp.setValue(getattr(self.p, attr)); sp.setToolTip(tip); sp.setKeyboardTracking(False)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            nst = int(round((hi - lo) / step))
            sl.setRange(0, nst); sl.setValue(int(round((getattr(self.p, attr) - lo) / step)))
            sl.valueChanged.connect(lambda v, s=sp, lo=lo, st=step: s.setValue(lo + v * st))
            sp.valueChanged.connect(lambda v, s=sl, lo=lo, st=step: (s.blockSignals(True),
                                    s.setValue(int(round((v - lo) / st))), s.blockSignals(False)))
            sp.valueChanged.connect(lambda v, a=attr: self._changed(a, v))
            form.addWidget(lab, 2 * row, 0, 1, 2)
            form.addWidget(sl, 2 * row + 1, 0)
            form.addWidget(sp, 2 * row + 1, 1)
            self.spins[attr] = sp
        lv.addLayout(form)
        reset = QtWidgets.QPushButton("全部回到預設")
        reset.clicked.connect(self._reset)
        lv.addWidget(reset)

        disp = QtWidgets.QGroupBox("顯示（不影響計算）")
        dg = QtWidgets.QFormLayout(disp)
        self.exag = QtWidgets.QDoubleSpinBox(); self.exag.setRange(1, 500); self.exag.setValue(1)
        self.exag.setToolTip("把焦點的橫向偏移（剪切、傾斜）放大顯示；1 = 真實比例")
        self.zsc = QtWidgets.QDoubleSpinBox(); self.zsc.setRange(1, 20); self.zsc.setValue(5)
        self.zsc.setToolTip("3D 圖深度方向放大倍率（體積 ~2 mm 寬、~0.2 mm 厚）")
        self.exag.valueChanged.connect(self._redraw)
        self.zsc.valueChanged.connect(self._redraw)
        dg.addRow("橫向偏移顯示放大 ×", self.exag)
        dg.addRow("3D 深度顯示放大 ×", self.zsc)
        lv.addWidget(disp)
        fixed = QtWidgets.QLabel(FIXED_TEXT); fixed.setWordWrap(True)
        lv.addWidget(fixed)
        lv.addStretch(1)
        scroll = QtWidgets.QScrollArea(); scroll.setWidget(left); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)

        # ---------------- centre: 3D
        centre = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(centre)
        bar = QtWidgets.QHBoxLayout()
        for name, fn in (("俯視 XY", "xy"), ("側視 XZ", "xz"), ("側視 YZ", "yz"), ("等角", "iso")):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda _=False, f=fn: self._view(f))
            bar.addWidget(b)
        shot = QtWidgets.QPushButton("匯出 3D 截圖")
        shot.clicked.connect(self._shot)
        bar.addWidget(shot)
        self.status = QtWidgets.QLabel("")
        bar.addWidget(self.status, 1)
        cv.addLayout(bar)
        self.plotter = QtInteractor(centre)
        self.plotter.set_background("white")
        cv.addWidget(self.plotter.interactor, 1)
        note = QtWidgets.QLabel(NOTE_TEXT); note.setWordWrap(True)
        cv.addWidget(note)

        # ---------------- right: slice + numbers
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        sbar = QtWidgets.QHBoxLayout()
        self.sl_kind = QtWidgets.QComboBox()
        self.sl_kind.addItems(["XY（指定深度）", "XY：D=0 焦面", "XZ（指定 y）", "YZ（指定 x）"])
        self.sl_pos = QtWidgets.QDoubleSpinBox(); self.sl_pos.setRange(-2000, 2000); self.sl_pos.setDecimals(0)
        self.sl_pos.setValue(self.p.focus_depth_um); self.sl_pos.setSuffix(" µm")
        self.sl_kind.currentIndexChanged.connect(self._slice_kind)
        self.sl_pos.valueChanged.connect(self._draw_slice)
        sbar.addWidget(self.sl_kind); sbar.addWidget(self.sl_pos)
        rv.addLayout(sbar)
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        rv.addWidget(self.canvas, 1)
        ebar = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("匯出切面 PNG"); b1.clicked.connect(self._export_png)
        b2 = QtWidgets.QPushButton("匯出切面 CSV"); b2.clicked.connect(self._export_csv)
        ebar.addWidget(b1); ebar.addWidget(b2)
        rv.addLayout(ebar)
        self.nums = QtWidgets.QLabel(""); self.nums.setWordWrap(True)
        self.nums.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        rv.addWidget(self.nums)

        split = QtWidgets.QSplitter()
        split.addWidget(scroll); split.addWidget(centre); split.addWidget(right)
        split.setSizes([380, 820, 520])
        self.setCentralWidget(split)

        # ---------------- timers and worker thread
        self.t_fast = QtCore.QTimer(self, singleShot=True, interval=60)
        self.t_fast.timeout.connect(self._run_fast)
        self.t_prec = QtCore.QTimer(self, singleShot=True, interval=700)
        self.t_prec.timeout.connect(self._run_precise)
        self.thread = QtCore.QThread(self)
        self.worker = _Worker()
        self.worker.moveToThread(self.thread)
        self.request_precise.connect(self.worker.run)
        self.worker.done.connect(self._precise_done)
        self.worker.prog.connect(self._precise_prog)
        self.thread.start()
        self._run_fast()
        self.t_prec.start()

    # ------------------------------------------------------------------ params
    def _changed(self, attr, v):
        self.p = replace(self.p, **{attr: float(v)})
        self.token += 1
        self.worker.token = self.token
        self.t_fast.start()
        self.t_prec.start()

    def _reset(self):
        d = Params()
        for attr, sp in self.spins.items():
            sp.setValue(getattr(d, attr))

    def _run_fast(self):
        try:
            self.res = compute(self.p, "fast")
        except Exception as e:
            self.status.setText(f"<span style='color:red'>計算錯誤：{e}</span>")
            traceback.print_exc()
            return
        self.status.setText("快速模式（離軸效應為近似）— 精確計算待命…")
        self._redraw()

    def _run_precise(self):
        self.status.setText("精確計算中 0%")
        self.request_precise.emit(self.p, self.token)

    @QtCore.Slot(float, int)
    def _precise_prog(self, f, token):
        if token == self.token:
            self.status.setText(f"精確計算中 {f:.0%}")

    @QtCore.Slot(object, int)
    def _precise_done(self, res, token):
        if res is None or token != self.token:
            return
        self.res = res
        self.status.setText("<span style='color:#1a7f37'>精確模式（2D 光瞳波動積分）</span>")
        self._redraw()

    # ------------------------------------------------------------------ drawing
    def _geom(self):
        r = self.res
        k = self.exag.value()
        rf = np.hypot(r.fx, r.fy)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(rf > 0, r.fx / rf, 0.0)[..., None]
            uy = np.where(rf > 0, r.fy / rf, 0.0)[..., None]
        X = r.fx[..., None] + k * r.off * ux
        Y = r.fy[..., None] + k * r.off * uy
        return X, Y, r.Z

    def _redraw(self, *_):
        if self.res is None:
            return
        X, Y, Z = self._geom()
        zs = self.zsc.value()
        g = pv.StructuredGrid(X, Y, Z * zs)   # +z = deeper; camera 'up' is -z
        g["V"] = self.res.V.ravel(order="F")
        cam = None if self._first else self.plotter.camera_position
        self.plotter.clear()
        # solid regions (cells whose V passes the level), drawn by their outer surface
        gc = g.point_data_to_cell_data(pass_point_data=True)
        dim = gc.threshold(DIM_LEVEL, scalars="V", preference="cell").extract_surface()
        use = gc.threshold(THR, scalars="V", preference="cell").extract_surface()
        if dim.n_points:
            self.plotter.add_mesh(dim, color="#9ec5ff", opacity=0.18)
        if use.n_points:
            self.plotter.add_mesh(use, color="#1f3f8f", opacity=0.95)
        b = g.bounds   # depth is drawn stretched; label the axis with true depths
        self.plotter.show_grid(xtitle="x (um)", ytitle="y (um)", ztitle="depth below surface (um)",
                               font_size=10, color="black", fmt="%.0f",
                               axes_ranges=[b[0], b[1], b[2], b[3], b[4] / zs, b[5] / zs])
        k = self.exag.value()
        self.plotter.add_text("dark blue: usable (2P >= e^-2)   light blue: swept, 2P < e^-2\n"
                              f"depth drawn x{zs:g};  lateral offsets drawn x{k:g}"
                              + ("  (NOT true scale)" if k != 1 else ""),
                              position="upper_left", font_size=9, color="black")
        if cam is None:
            self._view("iso")
            self._first = False
        else:
            self.plotter.camera_position = cam
        self.plotter.render()
        self._numbers()
        self._draw_slice()

    def _view(self, f):
        """Camera presets with depth pointing DOWN on screen (data z = +depth)."""
        pl = self.plotter
        b = pl.bounds
        c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        d, up = {"iso": ((1.0, -1.0, -0.8), (0, 0, -1)), "xz": ((0, 1.0, 0), (0, 0, -1)),
                 "yz": ((-1.0, 0, 0), (0, 0, -1)), "xy": ((0, 0, -1.0), (0, 1, 0))}[f]
        pl.camera_position = [tuple(c + 5000 * np.array(d)), tuple(c), up]
        pl.reset_camera()
        pl.render()

    def _numbers(self):
        s = self.res.summary

        def row(name, d):
            sw, us = d["sweep"], d["usable"]
            uu = (f"{us[0]:.0f}–{us[1]:.0f}（{us[1]-us[0]:.0f}）" if np.isfinite(us[0]) else "無")
            return (f"<tr><td>{name}</td><td>{sw[0]:.0f}–{sw[1]:.0f}（{sw[1]-sw[0]:.0f}）</td>"
                    f"<td>{uu}</td><td>{d['d0_depth']:.0f}</td><td>{d['brightness']:.2f}</td>"
                    f"<td>{d['shear_um']:+.2f}</td></tr>")

        html = ("<table cellspacing=4><tr><th></th><th>主峰掃描 µm</th><th>可用 µm</th>"
                "<th>D=0 µm</th><th>相對亮度</th><th>頂−底橫移 µm</th></tr>"
                + row("軸上", s["axis"]) + row("邊緣", s["edge"]) + row("角落", s["corner"]) + "</table>"
                f"M_net {s['M_net']:.2f}；近軸每 D {s['dz_per_D_um']:.2f} µm；"
                f"TAG 相位振幅 {s['Phi_rad']:.0f} rad；FOV {s['fov_x_mm']:.2f} mm（中線寬）<br>"
                "深度 = 組織表面下；頂−底橫移 = c=+1（淺端）減 c=−1（深端）的徑向位置，"
                "負值 = 淺端偏向中心。")
        for w in self.res.warnings:
            html += f"<br><span style='color:#c00'>⚠ {w}</span>"
        self.nums.setText(html)

    # ------------------------------------------------------------------ slices
    def _slice_kind(self, i):
        r = self.res
        if i == 0:
            self.sl_pos.setValue(self.p.focus_depth_um)
        elif i in (2, 3):
            self.sl_pos.setValue(0)
        self.sl_pos.setEnabled(i != 1)
        self._draw_slice()

    def _slice_data(self):
        r = self.res
        X, Y, Z = self._geom()
        i = self.sl_kind.currentIndex()
        q = self.sl_pos.value()
        n = r.fx.shape[0]
        if i in (0, 1):
            fc = r.Z[..., 0] - r.z[0]
            tgt = (q - fc) if i == 0 else (r.zd0 - fc)
            Vs = np.empty((n, n)); Xs = np.empty((n, n)); Ys = np.empty((n, n))
            for a in range(n):
                for b in range(n):
                    Vs[a, b] = np.interp(tgt[a, b], r.z, r.V[a, b], left=0, right=0)
                    Xs[a, b] = np.interp(tgt[a, b], r.z, X[a, b])
                    Ys[a, b] = np.interp(tgt[a, b], r.z, Y[a, b])
            title = (f"XY 切面：深度 {q:.0f} µm" if i == 0 else
                     f"XY：D=0 焦面（深度 {r.zd0.min():.1f}–{r.zd0.max():.1f} µm）")
            return "xy", Xs, Ys, Vs, title, ("x (µm)", "y (µm)"), (None if i == 0 else r.zd0)
        if i == 2:
            j = int(np.argmin(np.abs(r.fy[n // 2, :] - q)))
            return ("xz", X[:, j, :], Z[:, j, :], r.V[:, j, :],
                    f"XZ 切面：y = {r.fy[n // 2, j]:.0f} µm", ("x (µm)", "深度 (µm)"), None)
        a = int(np.argmin(np.abs(r.fx[:, n // 2] - q)))
        return ("yz", Y[a, :, :], Z[a, :, :], r.V[a, :, :],
                f"YZ 切面：x = {r.fx[a, n // 2]:.0f} µm", ("y (µm)", "深度 (µm)"), None)

    def _draw_slice(self, *_):
        if self.res is None:
            return
        kind, A, B, V, title, labs, depth = self._slice_data()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        Vc = np.clip(V, 1e-4, 1)
        pm = ax.pcolormesh(A, B, Vc, shading="gouraud", cmap="magma", norm=LogNorm(1e-4, 1))
        try:
            cs = ax.contour(A, B, V, levels=[THR], colors="#00e0ff", linewidths=1.2)
        except Exception:
            cs = None
        if depth is not None:
            cl = ax.contour(A, B, depth, colors="white", linewidths=0.6, levels=6)
            ax.clabel(cl, fmt="%.1f", fontsize=7)
        if kind != "xy":
            ax.invert_yaxis()
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal")
        cb = self.fig.colorbar(pm, ax=ax)
        cb.set_label("2P = |E|⁴（全體積正規化，log）")
        k = self.exag.value()
        ax.set_title(title + "\n青線 = 可用門檻 e⁻²" + (f"；橫向偏移 ×{k:g}" if k != 1 else ""), fontsize=10)
        ax.set_xlabel(labs[0]); ax.set_ylabel(labs[1])
        self.canvas.draw_idle()
        self._slice_cache = (kind, A, B, V, title)

    def _export_png(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "匯出切面 PNG", "slice.png", "PNG (*.png)")
        if fn:
            self.fig.savefig(fn, dpi=200)

    def _export_csv(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "匯出切面 CSV", "slice.csv", "CSV (*.csv)")
        if not fn:
            return
        kind, A, B, V, title = self._slice_cache
        cols = {"xy": "x_um,y_um", "xz": "x_um,depth_um", "yz": "y_um,depth_um"}[kind]
        data = np.column_stack([A.ravel(), B.ravel(), V.ravel()])
        hdr = f"{title}\nparams: {asdict(self.p)}\nmode: {self.res.mode}\n{cols},I2P_norm"
        np.savetxt(fn, data, delimiter=",", header=hdr, fmt="%.6g", encoding="utf-8")

    def _shot(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "匯出 3D 截圖", "volume.png", "PNG (*.png)")
        if fn:
            self.plotter.screenshot(fn, scale=2)

    def closeEvent(self, ev):
        self.worker.token = -1
        self.thread.quit(); self.thread.wait(10000)
        self.plotter.close()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
