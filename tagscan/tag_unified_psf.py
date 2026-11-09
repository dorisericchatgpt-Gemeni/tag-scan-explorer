r"""
===============================================================================
 tag_unified_psf.py  --  TAG lens 2P 顯微鏡的「統一光瞳積分」模擬器
===============================================================================

一句話：把三份既有推導縫進**同一個光瞳函數**，用一次 Debye 積分同時產生
        真實掃描軌跡、PSF_XY、PSF_Z、離軸散光、以及 2P 激發強度分佈。

-------------------------------------------------------------------------------
0. 這支程式取代了什麼
-------------------------------------------------------------------------------
舊的 `tag_bessel_exact_hybrid.py` 是「貼」的：幾何模組算兩條焦線深度、Bessel
模組另外算軸上 scalar PSF，兩者不互相回饋。所以

    * PSF 裡看不到散光
    * 焦線深度裡看不到 Bessel needle 縮短
    * 而且用的是 tangent condition (tan θ10 = R/f)，不是理想物鏡

本檔是「一起算」的：散光、Bessel sidelobe、球差、渦捲、暗化全部從同一個
複數場 E(x,y,z,t) 長出來。

-------------------------------------------------------------------------------
1. 物理定式
-------------------------------------------------------------------------------
座標：z 沿光軸。孔徑（半徑 R）在 z = -L1；n1/n2 界面在 z = 0；
      n2/n3 界面（組織表面）在 z = L2；焦點在 z > L2 的組織 n3 內。

Debye 型光瞳積分（scalar，焦區近似）：

    E(r) = ∬ P(kx,ky) · exp[i Φ(kx,ky; r)] dkx dky

    P = A_gauss(r_p) / sqrt(cosθ1) · T_fresnel · exp[i φ_TAG(r_p,t)] · Vignette
    Φ = Φ_conv + k_z1·L1 + k_z2·L2 + k_z3·(z-L2) + kx·x + ky·y

其中 k_zj = sqrt((n_j k0)^2 - kt^2) （橫向 kt 跨層守恆＝Snell 的波動版）。

--- (a) 理想物鏡 = Abbe sine condition ---------------------------------------
    光瞳半徑 r_p 與角度的映射用**正弦條件**（不是 tangent）：

        r_p = F · n1 · sinθ1        ⟹   kt = n1 k0 sinθ1 = k0 · r_p / F

    也就是「光瞳半徑 ↔ 橫向波數」是線性的。這正是
    `Exactly TAG DOV - ideal objective sine condition.md` 要求的替換：

        tan θ10 = R/f      →      n1 sin θ10 = R/F
        fold R/(R²+f²)     →      fold R/(F·sqrt(W1)),   W_j ≡ n_j²F² - R²

    aplanatic apodization = sqrt(cosθ1)；在 (kx,ky) 網格上做積分時，
    Jacobian sinθdθdφ = dkx dky /((n1 k0)² cosθ) 把它翻成權重 1/sqrt(cosθ1)。

--- (b) TAG 的真實 J0 相位（不是拋物近似）-----------------------------------
    TAG 平面上的相位（r_TAG = r_p / M_net）：

        φ_TAG(r,t) = φ_amp · J0(k_r · r) · cos(2π f_TAG t)
        k_r = 2π f_TAG / v_sound
        φ_amp = 2 k0 V_T,peak / k_r²

    φ_amp 的來源：把 J0 ≈ 1 - x²/4 的 r² 係數 (-φ_amp k_r²/4) 對上薄透鏡相位
    (-k0 V/2) 得到。所以「拋物區內」本模型嚴格等於一個 vergence = V_T 的透鏡，
    「拋物區外」自動產生 Bessel needle + sidelobe + 正球差 —— 也就是
    `MIC TAG/BesselReductionFactor_說明.md` 講的三個縮短機制。

    另外注意：此相位對光線給出 δ(n1 sinθ) = -V·r_p，恰好就是 sine-condition
    檔的 direction-cosine 形式 δq = R·V_T/M²。波動模型天生站在對的那一邊。

--- (c) 離軸 = 剛體右錐傾斜 ---------------------------------------------------
    galvo 給 chief ray 傾角 Ω（cosΩ = cosωx·cosωy）。實作是把方向向量反轉回去：

        û = R_x(-Ω) · û'      (û' 由網格上的 (kx,ky) 定義)

    圓形光瞳在 k 空間被映成非圓區域，而 k_z 是 |kt| 的非線性函數 ⟹ 二次項在
    x/y 方向不相等 ⟹ **散光自動出現**，不需要另外寫 Coddington 公式。
    這與 `平界面斜聚焦散光.md` 是同一個物理，只是這裡是波動版。

--- (d-0) 有限 NA 的 fold：本模擬與 md 閉式的實質分歧 ★重要★ ----------------
    md（含 sine-condition 版）把 TAG 寫成「在光瞳高度 h 加上光線斜率 V·h」，於是

        md :   delta(n1 sin theta_a) = + R V_T / M^2

    那是**近軸薄透鏡關係**。對 aplanatic 物鏡，光瞳相位其實是出射參考球面上的
    OPD，光線傾角要沿**球面弧長** (n1 F d theta) 取梯度，不是沿平的光瞳座標 dr_p：

        delta_theta = (1/(n1 k0)) * dphi / (n1 F d theta)
        =>  wave :  delta(n1 sin theta_a) = - (R V_T / M^2) * cos^2(theta_a)

    多一個 cos^2。等價地，stationary phase 給出「光瞳環帶 k_t 的焦點位移
    = -2 beta k_z」，beta = V F^2/(2 M^2 k0)，也就是位移 ∝ cos(theta)。

    後果：折疊係數 A_marginal = A_md * cos^2(theta_10)。**md 的有限 NA 修正方向是
    反的** —— md 說 |A| 隨 NA 變大，波動說變小。實測（V2b，均勻介質）：

        NA     md 閉式    md*cos^2   近軸      波動模擬
        0.10   -7.327     -7.254     -7.290    -7.273
        0.35   -7.782     -6.829     -7.290    -7.076
        0.556  -8.771     -6.059     -7.290    -6.724      <- md 高估 30%

    NA->0 三者一致到 0.07%（V2）。波動值永遠落在「近軸」與「md*cos^2」之間，
    因為它是整個光瞳能量加權的結果；把光瞳從 Gaussian 換成均勻會往 md*cos^2 靠。

    Config.fold_mode 可切換：預設 "wave"（物理正確），"md" 用來重現舊結果。

--- (d) 參考焦點 -------------------------------------------------------------
    Φ_conv = -[ky·sqrt(W1)·sinΩ + k_z1·sqrt(W1)·cosΩ]
    對應「若全空間都是 n1」時的名目焦點（距孔徑 sqrt(W1)）。
    均勻介質 + Ω=0 + 無 TAG 時，所有 k 分量在 z = -L1 + sqrt(W1) 同相 ⟹ 完美聚焦。

-------------------------------------------------------------------------------
2. 模型邊界（老實話）
-------------------------------------------------------------------------------
  ✔ 含：有限 NA、三層平界面、離軸散光、場曲、球差、TAG 完整 J0 相位、
        Bessel sidelobe、Gaussian 光瞳截斷、孔徑暗化(vignetting)、
        Fresnel 透射振幅、2P 非線性、螢光飽和、TAG 週期 dwell 加權。
  ✘ 不含：向量（偏振）效應 —— 這是 scalar Debye。NA<=0.7 影響小但非零。
  ✘ 不含：組織散射與吸收（要接 cerebellum_scattering 那條線）。
  ✘ 不含：物鏡自身的殘餘像差（假設理想 aplanatic）。
  ✘ 不含：TAG 腔內的行波/駐波暫態（假設純駐波 J0 × cos ωt）。

-------------------------------------------------------------------------------
3. 參數說明書（Config，全部可在 CLI 或程式內改）
-------------------------------------------------------------------------------
  ── 幾何與介質 ──────────────────────────────────────────────────────────
  n1, n2, n3        : 浸潤 / 工作介質 / 組織 折射率
  R_mm              : 光瞳（孔徑）半徑 [mm]
  F_mm              : 物鏡的 **sine-condition 焦距** [mm]，NA = n1·sin θ10 = R/F
  L1_mm             : 孔徑到 n1/n2 界面的距離（n1 段厚度）[mm]
  L2_mm             : n2 工作介質層厚度 [mm]
  M1, M2            : relay 放大率；M_net = M1·M2（TAG→光瞳），M2 用於 galvo 角
  ── TAG ────────────────────────────────────────────────────────────────
  VT_peak           : TAG 峰值 vergence（TAG 平面上）[1/m]
  tag_frequency_hz  : TAG 驅動頻率 [Hz]（預設 190 kHz）
  sound_speed_m_s   : TAG 腔內流體聲速 [m/s]（矽油 ~980）
  ── 光束與波長 ──────────────────────────────────────────────────────────
  wavelength_um     : 真空波長 [µm]
  beam_radius_pupil_mm : 光瞳上的 1/e² **強度**半徑 [mm]；None → 用 R_mm
  ── 掃描 ────────────────────────────────────────────────────────────────
  galvo_x_deg, galvo_y_deg : galvo 機械掃描半角 [deg]（會除以 M2 變物空間角）
  field_list_deg    : 要分析的視場角清單（物空間 Ω，deg），None → 自動取
                      [0, 半角, 全角(對角)]
  ── 2P 螢光 ────────────────────────────────────────────────────────────
  fluor_threshold   : 螢光偵測閾值，以「該圖最大 2P 訊號」的比例表示 (0~1)
  saturation        : 飽和強度；2P 激發機率 p = 1 - exp(-s·Î²)，Î = I/I_max。
                      s → 0 為未飽和（p ∝ I²）；s = 3 已相當飽和。
  ── 數值 ────────────────────────────────────────────────────────────────
  n_grid            : k 空間網格邊長；None → 自動取 2.6 × pupil_radius_px
  pupil_radius_px   : 光瞳半徑要佔幾個網格點（決定 FOV 與 TAG 相位取樣，預設 96）
  auto_sampling     : True → 開跑前先體檢，取樣不足就自動把 pupil_radius_px 調高
                      （見下方「取樣守門員」）。--fast 會關掉它。
  max_phase_step_rad: 守門員門檻：光瞳上相鄰格點的**相位跳變**上限（預設 4.0 rad）
  n_z               : 軸向取樣點數
  z_span_factor     : 軸向掃描範圍 = z_span_factor × DOV
  n_time            : 半個 TAG 週期的時間取樣點數（cos 從 +1 到 -1）
  lateral_view_um   : XY / XZ / YZ 圖的橫向顯示半寬 [µm]
  use_fresnel       : 是否乘上 Fresnel 透射振幅
  clip_vignetting   : 是否模擬離軸時被孔徑擋掉的光線

-------------------------------------------------------------------------------
4. 產出的圖（FIGURES 字典控制開關，跑之前可以只留你要的）
-------------------------------------------------------------------------------
  01_scan_trajectory.png     真實掃描軌跡 z(t)：波動模擬的最亮深度、幾何兩條
                             焦線 z_rad/z_az、理想 paraxial 餘弦三者對照；
                             標出 Bessel 縮短比 ratio 與軌跡非線性。多視場疊圖。
  02_psf_xy.png              PSF_XY 橫截面（1P |E|² 與 2P |E|⁴），在指定 z/t/Ω；
                             含 1/e² 與 FWHM 橢圓、長軸方向（散光的指紋）。
  03_psf_z.png               PSF_Z 軸向剖面（線性 + 對數），標出 Bessel sidelobe、
                             sidelobe 能量佔比、軸向 FWHM、離焦背景比。
  04_field_sweep.png         視場掃描：DOV_rad / DOV_az / 位置散光 / 峰值強度 /
                             橫向與軸向 FWHM 隨 Ω 變化，標出 2×2 mm FOV 邊角。
  05_2p_instant_slices.png   **瞬時** 2P 激發強度：XY / XZ / YZ 三切面，含螢光
                             閾值等高線與飽和後的訊號，以及超閾值體積。
  06_2p_period_slices.png    **整個 TAG 週期積分**的 2P 激發：同樣三切面 + 閾值 +
                             飽和；旁附 arcsine dwell 密度（cos 在端點走得慢）。
  07_pupil_diagnostics.png   光瞳診斷：振幅、J0 相位、J0 vs 拋物近似的偏差、
                             拋物區半徑 vs 光束半徑（接 Gaussian preservation 文件）、
                             離軸 vignetting 圖、有效 NA。
  08_psf_metrics.png         可量測度量地圖：FWHM_x / FWHM_y / FWHM_z、峰值
                             (Strehl-like)、2P 有效體積 V_eff，隨 TAG 相位與視場變化。
  09_focal_surface.png       焦面 dome：z_rad 與 z_az 兩張曲面 vs (ωx, ωy)，
                             = 場曲 + 散光的 3D 呈現（實驗上量平面樣品即可驗）。
  10_validation.png          四道驗證（見下）。

-------------------------------------------------------------------------------
5. 內建驗證（跑完會印表，也畫成 10_validation.png）
-------------------------------------------------------------------------------
  V6  橫向取樣座標：主射線 vs 光束自己的軌跡。
      **這是修掉的最大一個 bug。** 幾何主射線以 tanΩ₃ 橫走，但焦點附近的場**不走**
      —— 它被光瞳的線性相位釘在原地。'current' preset 在 Ω=5.3° 實測：峰值整段
      ±63 µm 都在 y = 553.1 µm，主射線卻從 549.5 走到 558.1。只有在遠離焦點的幾何
      區，強度質心才恢復以 ⟨k_y⟩/⟨k_z3⟩ ≈ tanΩ₃ 前進。
      沿主射線取樣 = 掃到 PSF 核心外面：邊角處主射線橫移 8.6 µm，而 2P 光斑只有
      1.8 µm 寬。後果是軸向 PSF 被讀成**偏窄**，看起來像「離軸軸向解析度變好」：
          FWHM_z(2P)  主射線 32.96 / 27.42 / 21.02 µm
                      真實軌跡 32.96 / 34.03 / 33.85 µm  ← 其實是平的
      修法是先粗掃出 y_peak(z) 再線性擬合，餵給 axial_line 的 (y0_um, slope)。
      擬合殘差也一併回報：axial_line 只能沿**直線**走，殘差大就代表沒有任何一條
      直線能穿過核心（實測邊角殘差 0.195 µm，遠小於光斑，安全）。

  V0  數值：快速軸向變換 vs 直接硬加總（同一個真實組態，軸上 + 邊角各一組）。
      axial_line() 把光瞳投影到單一軸向座標 g = kz3 + ky·tanΩ3 之後做一次 FFT，
      成本從 O(n_z · n_k) 掉到 O(n_k + nb log nb)（在守門後的取樣下快 20–190 倍，
      整份跑完從數小時變成幾分鐘）。V0 就是拿它跟被取代的硬加總逐點比。
      實測：相對場誤差 < 1e-6，焦點位置差 < 1e-5 µm。

  ── 取樣守門員（這不是驗證，是開跑前的體檢）─────────────────────────────
      Debye 積分是 stationary-phase 求和：光瞳上相鄰格點的相位若跳超過幾個 rad，
      求和就會 alias，長出**看起來很像 Bessel sidelobe 的假振盪**。大折射率失配
      是最常見元兇（平界面球差把數百 rad 鋪在光瞳上）。
      門檻量的是**扣掉線性項之後**的最大相位跳變 —— 線性相位斜坡只是把焦點平移，
      本程式的輸出網格本來就跟著 chief ray 走，所以不花取樣。
      實測（各 preset 在 pupil_radius_px=96 下）：
          water 0.62 rad ✓通過 / mesoscale 6.4 rad → 需 208 / current 12.9 rad → 需 312
      沒有守門員時 current 在 96 px 下：FWHMz 47.6 µm、10 個假峰、sidelobe 75%；
      312 px 下：49.2 µm、2 個峰、sidelobe 24%（192 px 起就收斂了）。

  V1  均勻介質 + Ω=0 + 無 TAG + 均勻光瞳振幅
      → 焦點應精確落在 z = -L1 + sqrt(W1)，橫向 FWHM ≈ 0.5145 λ/NA（Airy）。
      實測：焦點誤差 0.000 µm，FWHM -0.02%。

  V2  近軸極限（均勻介質、NA→0.03）+ 拋物 TAG
      → 這是**唯一**該和 md 閉式吻合的地方，兩者都必須收斂到 -n3 F²V/M²。
      實測：+0.066%。

  V2b 有限 NA（均勻介質）→ 量化上面 (d-0) 的 fold 分歧。
      實測：NA 0.556 時 md 高估 30.4%。這不是 bug，是模型分歧。

  V3  完整 J0 vs 拋物近似（**均勻介質**，否則平界面 caustic 會主導、最亮點會在
      不同 lobe 之間跳，比值就量不到東西）
      → 比值應落在 BesselReductionFactor_說明.md 的預測帶。
      實測：k_r·w=1.83 時 ratio 0.820（vs 拋物）/ 0.756（vs 近軸），
      文件對 k_r·w≈2 的預測是 0.70–0.85。✓ 獨立確證。

  V4  離軸散光：真實介質、降到 NA 0.25（高 NA 下平界面球差會把焦點糊成 caustic，
      兩條焦線就分不出來）→ 波動 PSF 在兩個深度分別於 x/y 最窄。
      實測：波動 -25.41 µm vs 光追 -24.57 µm（3.4%）。

  另一個已修的隱藏 bug（值得記住）：dkx dky -> 立體角的 Jacobian 必須用**旋轉後**的
  極角 cos(theta') = kz1/(n1 k0)，不是物鏡自身的 cos(theta_1)。用錯的話，傾斜光瞳
  上的權重會左右不對稱，把光束的平均方向拉回光軸 —— 症狀是焦點不沿 chief ray 走。
  修正後以強度質心量測：NA 0.30 誤差 +0.17%、NA 0.556 誤差 -2.5%（殘差是真實的
  傾斜光瞳 vignetting）。

  V5  報告這個組態到底多像差：邊際光焦深 vs 近軸焦深 vs 波動最亮點。
      預設 'current' preset 的平界面球差是 -514 µm —— 比 DOV 大一個量級，
      解讀任何深度數字之前先看這個數。

  ── 另外兩個已修的 bug（都會靜默給錯數字）─────────────────────────────
  (i)  psf_metrics 的軸向視窗原本用**繞射極限**估寬度（2n₃λ/NA² = 8.1 µm），但真實
       caustic 是 33 µm。裝不下 → fwhm() 回 NaN → 圖 04 少兩個點、圖 08 整條
       Ω=0 曲線消失，而且是**靜默**掉的。改成先在全掃描視窗量一次實際寬度，再用
       它開視窗；焦線分裂另外用自己的窄視窗找。
  (ii) depths_at_theta 的 vignetting clip 只砍子午那一對光線，弧矢分支仍用滿孔徑，
       兩個 Coddington 分支不一致 → 邊角散光從單調的 -48.3 µm 變成非單調的 +1.6 µm，
       **大小、形狀、正負號全錯**。現在改成只**偵測**不套用（回傳 vignetted 旗標），
       射線模型就是乾淨的滿孔徑參考；波動模型自己的光瞳遮罩仍照 clip_vignetting 走。

  ── 幾何體檢（check_geometry，開跑前）───────────────────────────────
       場引擎對所有 z > L2 都用 k_z3 傳播，所以「焦點落在 n2/n3 界面前面」的組態會
       被用錯的介質算完全程且**不會報錯**。內附的 'mesoscale' preset 原本就是這樣：
       sqrt(W1) = 13.500 mm 但 L1+L2 = 14.000 mm，近軸焦點在樣品前 680 µm
       （現已改成 L1=3.5、L2=9.8）。
       這只警告不自動修 —— 要focus 到組織內多深是參數決策，不是程式該替你猜的。

-------------------------------------------------------------------------------
6. 用法
-------------------------------------------------------------------------------
    .\.venv\Scripts\python.exe tag_unified_psf.py                    # 預設全跑
    .\.venv\Scripts\python.exe tag_unified_psf.py --preset mesoscale
    .\.venv\Scripts\python.exe tag_unified_psf.py --figures 01,03,05 # 只出這幾張
    .\.venv\Scripts\python.exe tag_unified_psf.py --fast             # 降網格快看
    .\.venv\Scripts\python.exe tag_unified_psf.py --validate-only    # 只跑驗證

  輸出到  ./unified_out_<preset>/
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.special import j0, jn_zeros

SCRIPT_DIR = Path(__file__).resolve().parent
MM = 1e3  # mm -> um


# =============================================================================
# Config
# =============================================================================
@dataclass
class Config:
    name: str = "current"

    # --- geometry / media (mm) ---
    n1: float = 1.00
    n2: float = 1.334
    n3: float = 1.36
    R_mm: float = 4.0
    F_mm: float = 7.2          # sine-condition focal length: NA = n1 sin(th10) = R/F
    L1_mm: float = 2.0
    L2_mm: float = 4.0
    M1: float = 1.0
    M2: float = 8.0 / 3.0

    # --- TAG ---
    VT_peak: float = 2.5             # 1/m, at the TAG plane
    tag_frequency_hz: float = 190e3
    sound_speed_m_s: float = 980.0

    # --- beam / wavelength ---
    wavelength_um: float = 0.920
    beam_radius_pupil_mm: float | None = None   # 1/e^2 INTENSITY radius at pupil

    # --- scan ---
    galvo_x_deg: float = 10.0
    galvo_y_deg: float = 10.0
    field_list_deg: tuple[float, ...] | None = None

    # --- 2P fluorescence ---
    fluor_threshold: float = 0.5     # fraction of that panel's max 2P signal
    saturation: float = 1.5          # p = 1 - exp(-s * Ihat^2)

    # --- how the TAG perturbation folds into the marginal cone angle ---
    #   "wave" : delta(n1 sin th_a) = -(R V/M^2) cos^2(th_a)      <- physical optics
    #   "md"   : delta(n1 sin th_a) = +(R V/M^2)                  <- the md's rewrite
    # See the "finite-NA fold" note in the module docstring; they differ by cos^2.
    fold_mode: str = "wave"

    # --- numerics ---
    n_grid: int | None = None        # None -> auto (2.6 x pupil_radius_px)
    pupil_radius_px: int = 96
    auto_sampling: bool = True       # raise pupil_radius_px until the guard passes
    # Sampling guard threshold, empirically calibrated on the 'current' preset:
    # the axial profile is already converged at 6.5 rad/sample and unchanged at
    # 4.3, so 4.0 carries ~1.6x margin. See sampling_report().
    max_phase_step_rad: float = 4.0
    n_z: int = 141
    z_span_factor: float = 1.6
    n_time: int = 25                 # over HALF a TAG period (cos: +1 -> -1)
    lateral_view_um: float = 6.0
    use_fresnel: bool = True
    clip_vignetting: bool = True

    # ---------- derived ----------
    @property
    def grid_size(self) -> int:
        if self.n_grid is not None:
            return int(self.n_grid)
        return int(np.ceil(2.6 * self.pupil_radius_px / 2.0) * 2)

    @property
    def M_net(self) -> float:
        return self.M1 * self.M2

    @property
    def k0(self) -> float:
        """vacuum wavenumber, rad/um"""
        return 2.0 * np.pi / self.wavelength_um

    @property
    def NA(self) -> float:
        return self.R_mm / self.F_mm            # = n1 sin(theta10)

    @property
    def sin_th10(self) -> float:
        return self.R_mm / (self.n1 * self.F_mm)

    @property
    def theta10(self) -> float:
        return np.arcsin(np.clip(self.sin_th10, -1, 1))

    def W(self, j: int) -> float:
        """W_j = n_j^2 F^2 - R^2, in mm^2 (sine-condition radicals)."""
        n = {1: self.n1, 2: self.n2, 3: self.n3}[j]
        return n * n * self.F_mm ** 2 - self.R_mm ** 2

    @property
    def beam_radius_mm(self) -> float:
        return self.R_mm if self.beam_radius_pupil_mm is None else self.beam_radius_pupil_mm

    @property
    def k_r(self) -> float:
        """TAG acoustic radial wavenumber, rad/mm"""
        return 2.0 * np.pi * self.tag_frequency_hz / self.sound_speed_m_s * 1e-3

    @property
    def phi_amp(self) -> float:
        """TAG peak phase amplitude [rad] consistent with VT_peak."""
        # phi_amp = 2 k0 V / k_r^2 ; k0 in rad/um -> rad/mm ; V in 1/m -> 1/mm
        k0_mm = self.k0 * MM          # rad/mm
        V_mm = self.VT_peak * 1e-3    # 1/mm
        return 2.0 * k0_mm * V_mm / self.k_r ** 2

    @property
    def r_first_zero_mm(self) -> float:
        return float(jn_zeros(0, 1)[0]) / self.k_r

    def parabolic_radius_mm(self, eps: float = 0.05) -> float:
        """TAG-plane radius where the J0 parabolic approx has relative error eps."""
        return 4.0 * np.sqrt(eps) / self.k_r

    @property
    def omega_pair(self) -> tuple[float, float]:
        """object-space chief tilt (rad) at the FOV corner"""
        return (np.radians(self.galvo_x_deg) / self.M2,
                np.radians(self.galvo_y_deg) / self.M2)

    @property
    def Omega_corner(self) -> float:
        wx, wy = self.omega_pair
        return float(np.arccos(np.clip(np.cos(wx) * np.cos(wy), -1, 1)))

    def fields(self) -> np.ndarray:
        """object-space Omega values (rad) to analyse"""
        if self.field_list_deg is not None:
            return np.radians(np.asarray(self.field_list_deg, dtype=float))
        c = self.Omega_corner
        return np.array([0.0, 0.5 * c, c])


PRESETS: dict[str, Config] = {
    "current": Config(name="current"),
    "mesoscale": Config(
        name="mesoscale", n1=1.00, n2=1.00, n3=1.36,
        # L1=3.5 使空氣焦點在 WD=10 mm；L2=9.8 → 名目焦深約 280 µm。
        # M1=5/3.34（TAG 1/e^2 直徑 3.34 → 5 mm）、galvo ±14° 光學角。
        R_mm=10.0, F_mm=16.8, L1_mm=3.5, L2_mm=9.8,
        M1=5.0 / 3.34, M2=4.0, VT_peak=10.0,
        galvo_x_deg=14.0, galvo_y_deg=14.0, beam_radius_pupil_mm=10.0,
    ),
    "water": Config(
        name="water", n1=1.33, n2=1.33, n3=1.36,
        R_mm=4.2, F_mm=7.2, L1_mm=0.0, L2_mm=2.0,
        M1=1.0, M2=8.0 / 3.0, VT_peak=3.75,
        galvo_x_deg=8.0, galvo_y_deg=8.0,
    ),
}


# =============================================================================
# SECTION 1 -- geometry, sine condition (ray model, for cross-checking)
# =============================================================================
def theta_a_of_VT(cfg: Config, VT: float, mode: str | None = None) -> float:
    """
    Marginal cone half-angle at TAG vergence VT [1/m].

    mode="md"   : n1 sin(th_a) = R/F + R*VT/M^2        (the md's sine-condition form)
    mode="wave" : d(n1 sin th_a)/dV = (R/M^2) cos^2(th_a), integrated exactly:
                      sin th_a = tanh( artanh(sin th_10) + R*V/(n1 M^2) )
                  This is what a pupil-plane phase actually does to an aplanatic
                  objective -- the extra cos^2 comes from taking the phase gradient
                  along the exit reference sphere (arc length n1 F dtheta) instead
                  of along the flat pupil coordinate dr_p.
    """
    mode = mode or cfg.fold_mode
    v = VT * 1e-3                      # 1/m -> 1/mm
    if mode == "md":
        q = cfg.R_mm / cfg.F_mm + cfg.R_mm * v / cfg.M_net ** 2
        return float(np.arcsin(np.clip(q / cfg.n1, -1.0, 1.0)))
    s0 = np.clip(cfg.sin_th10, -0.999999, 0.999999)
    s = np.tanh(np.arctanh(s0) + cfg.R_mm * v / (cfg.n1 * cfg.M_net ** 2))
    return float(np.arcsin(np.clip(s, -1.0, 1.0)))


def depths_at_theta(cfg: Config, Omega: float, theta_a: float) -> dict:
    """
    Radial (tangential) and azimuthal (sagittal) focus depths below the tissue
    surface z=L2, for one cone half angle theta_a and field tilt Omega.

    Ray model, sine condition, rigid right-cone tilt -- the sine-condition
    rewrite of `Exactly TAG DOV.md` sec.12 / `... ideal objective sine condition.md` sec.12.
    Lengths in mm; returned depths in um.
    """
    n1, n2, n3 = cfg.n1, cfg.n2, cfg.n3
    R, L1, L2 = cfg.R_mm, cfg.L1_mm, cfg.L2_mm
    cosO, sinO = np.cos(Omega), np.sin(Omega)
    sa = np.sin(theta_a)

    # ---- azimuthal / sagittal: closed form ----
    P = sinO ** 2 + sa ** 2 * cosO ** 2
    bz2 = 1.0 - (n1 / n2) ** 2 * P
    cz2 = 1.0 - (n1 / n3) ** 2 * P
    if bz2 <= 0 or cz2 <= 0:
        d_az = np.nan
    else:
        b_z, c_z = np.sqrt(bz2), np.sqrt(cz2)
        x0 = R - L1 * np.tan(theta_a) / max(cosO, 1e-9)
        d_az = (n3 / n1) * (x0 * c_z / sa) - (n3 / n2) * (L2 * c_z / b_z)

    # ---- radial / tangential: two meridional marginal rays ----
    ell = R / np.tan(theta_a)                    # apex distance from the stop
    Ay = ell * sinO
    Az = ell * cosO - L1                         # apex z relative to n1/n2 face
    u1, l1 = Omega - theta_a, Omega + theta_a
    au, al = u1, l1
    # Vignetting is DETECTED but deliberately not applied here.
    #
    # Clipping only the meridional pair while the sagittal branch below keeps the
    # full aperture makes the two Coddington branches inconsistent, and the
    # inconsistency is not small: on the default preset it drove astig from a
    # monotonic -48.3 um at the FOV corner to a non-monotonic +1.6 um -- wrong
    # magnitude, wrong shape, and wrong SIGN. The unclipped numbers are monotonic
    # in Omega^2 and carry the same sign the wave model measures (V4). The wave
    # model has its own honest pupil mask (Pupil.mask, still governed by
    # cfg.clip_vignetting), so nothing is lost: this function is the clean
    # full-aperture ray reference, and `vignetted` says when to distrust it.
    yu_ap = Ay - (Az + L1) * np.tan(u1)
    yl_ap = Ay - (Az + L1) * np.tan(l1)
    vig_u, vig_l = bool(abs(yu_ap) > R), bool(abs(yl_ap) > R)

    def trace(y0, a1):
        s2, s3 = (n1 / n2) * np.sin(a1), (n1 / n3) * np.sin(a1)
        if abs(s2) > 1 or abs(s3) > 1:
            return None
        return y0 + L2 * np.tan(np.arcsin(s2)), np.tan(np.arcsin(s3))

    yu0 = Ay - Az * np.tan(au)
    yl0 = Ay - Az * np.tan(al)
    ru, rl = trace(yu0, au), trace(yl0, al)
    if ru is None or rl is None or abs(al) >= np.pi / 2:
        d_rad = np.nan
    else:
        (yuL, tu), (ylL, tl) = ru, rl
        d_rad = (ylL - yuL) / (tu - tl)

    return {
        "depth_rad_um": d_rad * MM,
        "depth_az_um": d_az * MM,
        "astig_um": (d_az - d_rad) * MM,
        "vignetted": bool(vig_u or vig_l),
    }


def dov_sine(cfg: Config, Omega: float, mode: str | None = None) -> dict:
    """
    Ray-model DOV at field angle Omega.

    Computed as the honest finite-amplitude peak-to-peak of the marginal-ray focus
    depth over the TAG swing V in [-VT_peak, +VT_peak], using theta_a_of_VT so the
    chosen fold (`wave` or `md`) is applied consistently -- no separate fold factor
    to get out of sync. Also returns the linearised md-style number for comparison.
    """
    mode = mode or cfg.fold_mode
    th0 = theta_a_of_VT(cfg, 0.0, mode)
    thp = theta_a_of_VT(cfg, +cfg.VT_peak, mode)
    thm = theta_a_of_VT(cfg, -cfg.VT_peak, mode)
    nom = depths_at_theta(cfg, Omega, th0)
    dp = depths_at_theta(cfg, Omega, thp)
    dm = depths_at_theta(cfg, Omega, thm)

    out = dict(nom)
    out["Omega_deg"] = np.degrees(Omega)
    out["fold_mode"] = mode
    for key, tag in (("depth_rad_um", "DOV_rad_um"), ("depth_az_um", "DOV_az_um")):
        out[tag] = abs(dp[key] - dm[key])
        out[tag.replace("DOV", "depth") + "_plus"] = dp[key]
        out[tag.replace("DOV", "depth") + "_minus"] = dm[key]
    out["DOV_aniso_um"] = out["DOV_az_um"] - out["DOV_rad_um"]
    return out


def closed_form_AB(cfg: Config) -> dict:
    """On-axis closed-form coefficients, sine-condition version (md sec.12.4)."""
    W1, W2, W3 = cfg.W(1), cfg.W(2), cfg.W(3)
    F, R, L1, L2 = cfg.F_mm, cfg.R_mm, cfg.L1_mm, cfg.L2_mm
    n1, n2, n3 = cfg.n1, cfg.n2, cfg.n3
    A = -F * n3 ** 2 / np.sqrt(W3)
    B2 = -L2 * R ** 2 * F * (n3 ** 2 - n2 ** 2) / (np.sqrt(W3) * W2 ** 1.5)
    B1 = -L1 * R ** 2 * F * (n3 ** 2 - n1 ** 2) / (np.sqrt(W3) * W1 ** 1.5)
    # marginal-ray nominal depth, L1-aware:
    #   d0 = sqrt(W3) * (1 - L1/sqrt(W1) - L2/sqrt(W2))
    # (md sec.4 quotes the L1 = 0 special case sqrt(W3) - L2 sqrt(W3)/sqrt(W2))
    d0 = np.sqrt(W3) * (1.0 - L1 / np.sqrt(W1) - L2 / np.sqrt(W2))
    # paraxial depth of the SAME converging cone (reference sphere radius n1 F,
    # nominal focus a distance sqrt(W1) in front of the stop):
    #   (sqrt(W1) - L1)/n1 = L2/n2 + d_par/n3
    d_par = n3 * ((np.sqrt(W1) - L1) / n1 - L2 / n2)
    cos2 = W1 / (n1 ** 2 * F ** 2)                     # cos^2(theta_10)
    dz = (A + B1 + B2) * F ** 2 / cfg.M_net ** 2 * 1e-3 * MM
    return {"A": A, "B1": B1, "B2": B2,
            "d0_um": d0 * MM, "d_paraxial_um": d_par * MM,
            "spherical_um": (d_par - d0) * MM,
            "cos2_th10": cos2,
            "dz_per_VT_um": dz,              # md sine-condition closed form
            "dz_per_VT_wave_um": dz * cos2,  # same, with the cos^2 fold correction
            "dz_per_VT_paraxial_um": -n3 * F ** 2 / cfg.M_net ** 2 * 1e-3 * MM}


def closed_form_AB_tangent(cfg: Config) -> dict:
    """Old TANGENT-condition coefficients, for the sine-vs-tangent comparison."""
    n1, n2, n3 = cfg.n1, cfg.n2, cfg.n3
    R, f, L1, L2 = cfg.R_mm, cfg.F_mm, cfg.L1_mm, cfg.L2_mm
    D2 = R ** 2 + f ** 2
    W = n3 ** 2 * D2 - n1 ** 2 * R ** 2
    U2 = n2 ** 2 * D2 - n1 ** 2 * R ** 2
    A = -f * n3 ** 2 / (n1 * np.sqrt(W))
    B2 = -L2 * n1 ** 2 * R ** 2 * f * (n3 ** 2 - n2 ** 2) / (np.sqrt(W) * U2 ** 1.5)
    B1 = -L1 * (n3 ** 2 - n1 ** 2) * R ** 2 / (n1 * f ** 2 * np.sqrt(W))
    return {"A": A, "B1": B1, "B2": B2,
            "theta10_deg": np.degrees(np.arctan2(R, f)),
            "NA": n1 * R / np.sqrt(D2),
            "dz_per_VT_um": (A + B1 + B2) * f ** 2 / cfg.M_net ** 2 * 1e-3 * MM}


# =============================================================================
# SECTION 2 -- the unified pupil
# =============================================================================
class Pupil:
    """
    Builds the k-space pupil function on an FFT grid.

    All k in rad/um, all lengths in um inside this class.
    """

    def __init__(self, cfg: Config, Omega: float, parabolic_tag: bool = False,
                 tag_on: bool = True, uniform_amp: bool = False):
        self.cfg = cfg
        self.Omega = float(Omega)
        self.parabolic_tag = parabolic_tag
        self.tag_on = tag_on
        self.uniform_amp = uniform_amp

        k0 = cfg.k0
        n1, n2, n3 = cfg.n1, cfg.n2, cfg.n3
        N = cfg.grid_size

        # --- k grid sized so the pupil radius spans `pupil_radius_px` samples ---
        # support must also cover the tilted cone: sin(th10 + Omega)
        k_pup = n1 * k0 * cfg.sin_th10                      # untilted pupil radius in k
        k_sup = n1 * k0 * min(0.999, np.sin(cfg.theta10 + abs(self.Omega)) + 0.02)
        dk = k_sup / cfg.pupil_radius_px
        self.dk = dk
        self.fov_um = 2.0 * np.pi / dk
        self.dx_um = self.fov_um / N

        kax = (np.arange(N) - N // 2) * dk
        self.kx, self.ky = np.meshgrid(kax, kax, indexing="ij")
        self.kax = kax

        # --- un-rotate to recover the un-tilted direction cosines ---
        # u' = (kx, ky, kz1)/(n1 k0)  ;  u = R_x(-Omega) u'
        n1k0 = n1 * k0
        kt2 = self.kx ** 2 + self.ky ** 2
        prop1 = kt2 < (n1k0) ** 2
        kz1 = np.sqrt(np.maximum((n1k0) ** 2 - kt2, 0.0))
        ux_p = self.kx / n1k0
        uy_p = self.ky / n1k0
        uz_p = kz1 / n1k0
        cO, sO = np.cos(self.Omega), np.sin(self.Omega)
        # R_x(-Omega): (vx, vy cos - vz sin, vy sin + vz cos)
        ux = ux_p
        uy = uy_p * cO - uz_p * sO
        uz = uy_p * sO + uz_p * cO

        sin_t1 = np.sqrt(np.clip(ux ** 2 + uy ** 2, 0, 1))
        cos_t1 = uz
        r_pup_mm = cfg.n1 * cfg.F_mm * sin_t1                # sine condition
        self.r_pup_mm = r_pup_mm

        mask = prop1 & (r_pup_mm <= cfg.R_mm) & (cos_t1 > 1e-6)

        # --- geometric vignetting at the physical stop (z = -L1) ---
        if cfg.clip_vignetting and cfg.L1_mm > 0 and abs(self.Omega) > 0:
            # ray through the nominal focus with direction u'; where does it cross z=-L1?
            W1 = cfg.W(1)
            zf = -cfg.L1_mm + np.sqrt(W1) * cO
            yf = np.sqrt(W1) * sO
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (-cfg.L1_mm - zf) / np.where(uz_p > 1e-9, uz_p, np.nan)
                x_ap = 0.0 + t * ux_p
                y_ap = yf + t * uy_p
            r_ap = np.sqrt(x_ap ** 2 + y_ap ** 2)
            mask &= np.nan_to_num(r_ap, nan=1e9) <= cfg.R_mm
        self.mask = mask
        self.vignette_frac = 1.0 - mask.sum() / max(
            1, (prop1 & (r_pup_mm <= cfg.R_mm) & (cos_t1 > 1e-6)).sum())

        # --- amplitude: Gaussian x aplanatic apodization x 1/sqrt(cos) Jacobian ---
        w_mm = cfg.beam_radius_mm
        amp = (np.ones_like(r_pup_mm) if uniform_amp
               else np.exp(-(r_pup_mm / w_mm) ** 2))         # field amplitude
        # Aplanatic apodisation is a property of the OBJECTIVE, so it uses the
        # un-tilted polar angle theta_1 (equivalently the pupil radius).
        apod = np.sqrt(np.clip(cos_t1, 1e-9, None))
        # The dkx dky -> solid-angle Jacobian uses the ACTUAL (tilted) polar angle,
        # cos(theta') = kz1/(n1 k0): solid angle is rotation invariant, the k-grid
        # is not. Using cos(theta_1) here biases the weighting across a tilted pupil
        # and drags the bundle's mean direction back toward the axis (the focus then
        # fails to walk along the chief ray).
        cos_tp = np.clip(uz_p, 1e-9, None)
        base = amp * apod / cos_tp

        # --- Fresnel amplitude transmission through the two flat interfaces ---
        kz2 = np.sqrt(np.maximum((n2 * k0) ** 2 - kt2, 0.0) + 0j)
        kz3 = np.sqrt(np.maximum((n3 * k0) ** 2 - kt2, 0.0) + 0j)
        self.kz1, self.kz2, self.kz3 = kz1, kz2.real, kz3.real
        prop = prop1 & (kt2 < (n2 * k0) ** 2) & (kt2 < (n3 * k0) ** 2)
        self.mask = self.mask & prop
        if cfg.use_fresnel:
            base = base * (_fresnel_avg(kz1, kz2.real, n1, n2)
                           * _fresnel_avg(kz2.real, kz3.real, n2, n3))

        self.amp = np.where(self.mask, base, 0.0)

        # --- TAG phase profile (time-independent part) ---
        r_tag_mm = r_pup_mm / cfg.M_net
        x = cfg.k_r * r_tag_mm
        if parabolic_tag:
            prof = 1.0 - x ** 2 / 4.0
        else:
            prof = j0(x)
        self.tag_profile = np.where(self.mask, prof, 0.0)
        self.x_tag = x

        # --- converging + layer phase (time-independent) ---
        L1u, L2u = cfg.L1_mm * MM, cfg.L2_mm * MM
        sqW1 = np.sqrt(cfg.W(1)) * MM
        phi_conv = -(self.ky * sqW1 * sO + kz1 * sqW1 * cO)
        self.phi_static = phi_conv + kz1 * L1u + self.kz2 * L2u

        # --- reference focus position (chief ray, geometric) ---
        s2 = np.clip(n1 * np.sin(self.Omega) / n2, -1, 1)
        s3 = np.clip(n1 * np.sin(self.Omega) / n3, -1, 1)
        self.tanO1 = np.tan(self.Omega)
        self.tanO2 = np.tan(np.arcsin(s2))
        self.tanO3 = np.tan(np.arcsin(s3))
        self.y_at_L2_um = (cfg.L1_mm * self.tanO1 + cfg.L2_mm * self.tanO2) * MM

        # cached compact arrays over the mask (big speed win)
        self.idx = np.nonzero(self.mask)
        self.m_amp = self.amp[self.idx]
        self.m_kx = self.kx[self.idx]
        self.m_ky = self.ky[self.idx]
        self.m_kz3 = self.kz3[self.idx]
        self.m_phi = self.phi_static[self.idx]
        self.m_tag = self.tag_profile[self.idx]

    # ---- chief-ray lateral centre at absolute depth z (um, z measured from z=0)
    def y_center_um(self, z_um: float) -> float:
        L2u = self.cfg.L2_mm * MM
        return self.y_at_L2_um + (z_um - L2u) * self.tanO3

    # ---- masked pupil field at TAG phase factor c in [-1, 1]
    def field_k(self, c: float) -> np.ndarray:
        ph = self.m_phi
        if self.tag_on:
            ph = ph + self.cfg.phi_amp * c * self.m_tag
        return self.m_amp * np.exp(1j * ph)

    # ---- same thing as a dense (N, N) array, for the separable transforms
    def field_k2d(self, c: float) -> np.ndarray:
        out = np.zeros(self.mask.shape, dtype=np.complex128)
        out[self.idx] = self.field_k(c)
        return out


def sampling_report(cfg: Config, Omega: float | None = None,
                    probe_px: int = 96) -> dict:
    """
    Is the pupil sampled finely enough for THIS configuration?

    The Debye integral is a stationary-phase sum. If the total pupil phase steps by
    more than ~a radian between adjacent grid points, the sum aliases and grows
    spurious axial structure -- ripples that look exactly like Bessel sidelobes but
    are pure numerics. A large index mismatch is the usual culprit: it puts hundreds
    of um of spherical aberration, i.e. hundreds of radians, across the pupil.

    Measured on a cheap probe grid, then scaled (the step goes as 1/pupil_radius_px).
    """
    Omega = cfg.Omega_corner if Omega is None else Omega
    probe = Config(**{**asdict(cfg), "pupil_radius_px": probe_px, "n_grid": None})
    p = Pupil(probe, Omega)
    L2u = probe.L2_mm * MM
    z_probe = L2u + closed_form_AB(probe)["d_paraxial_um"]
    ph = p.phi_static + p.kz3 * (z_probe - L2u) + probe.phi_amp * p.tag_profile
    m = p.mask
    # Strip the linear (tilt) part: a phase ramp across the pupil is a pure lateral
    # translation of the focus, which is evaluated exactly here (the output grid is
    # re-centred on the chief ray), so it costs no sampling. Only the curvature and
    # higher-order aberration content has to be resolved.
    A = np.stack([p.kx[m], p.ky[m], np.ones(m.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, ph[m], rcond=None)
    res = np.full(ph.shape, np.nan)
    res[m] = ph[m] - A @ coef
    d0 = np.where(m[:-1, :] & m[1:, :], np.abs(np.diff(res, axis=0)), np.nan)
    d1 = np.where(m[:, :-1] & m[:, 1:], np.abs(np.diff(res, axis=1)), np.nan)
    step = float(max(np.nanmax(d0), np.nanmax(d1)))
    step_now = step * probe_px / cfg.pupil_radius_px
    need = int(np.ceil(cfg.pupil_radius_px * step_now / cfg.max_phase_step_rad / 8) * 8)
    return {"step_rad_at_current": step_now,
            "limit_rad": cfg.max_phase_step_rad,
            "recommended_px": max(cfg.pupil_radius_px, need),
            "ok": step_now <= cfg.max_phase_step_rad}


def check_geometry(cfg: Config, verbose: bool = True) -> bool:
    """
    Does the nominal focus actually land inside medium 3?

    The field engine propagates with k_z3 for every z > L2, so a configuration
    whose focus sits BEFORE the n2/n3 surface is evaluated in the wrong medium and
    every number it produces is meaningless -- silently. That is not hypothetical:
    the shipped 'mesoscale' preset had sqrt(W1) = 13.500 mm but L1 + L2 = 14.000 mm,
    putting the paraxial focus 680 um in front of the sample (fixed 2026-10-17 to
    L1 = 3.5, L2 = 9.8 mm).

    The fix is a parameter decision (how deep into the tissue you mean to focus),
    so this only warns -- it will not silently move your sample for you.
    """
    cf = closed_form_AB(cfg)
    d_par, d_mar = cf["d_paraxial_um"], cf["d0_um"]
    ok = bool(d_par > 0 and np.isfinite(d_mar) and d_mar > 0)
    if not ok and verbose:
        print(f"[geometry] *** preset '{cfg.name}': focus is NOT inside n3 ***")
        print(f"[geometry]   paraxial depth {d_par:+.1f} um, marginal {d_mar:+.1f} um "
              f"below the n2/n3 face (negative = in front of the sample)")
        print(f"[geometry]   aperture->focus sqrt(W1) = {np.sqrt(cfg.W(1)):.3f} mm "
              f"but L1+L2 = {cfg.L1_mm + cfg.L2_mm:.3f} mm")
        print(f"[geometry]   reduce L1+L2 by >~{-min(d_par, d_mar)/MM/cfg.n3:.3f} mm "
              f"to place the focus in the tissue. Results below are NOT valid.")
    return ok


def apply_sampling_guard(cfg: Config, verbose: bool = True) -> Config:
    """Raise pupil_radius_px until the sampling guard passes (if auto_sampling)."""
    rep = sampling_report(cfg)
    if rep["ok"]:
        if verbose:
            print(f"[sampling] pupil_radius_px={cfg.pupil_radius_px} ok "
                  f"(max phase step {rep['step_rad_at_current']:.2f} rad "
                  f"<= {rep['limit_rad']:.1f})")
        return cfg
    msg = (f"[sampling] pupil_radius_px={cfg.pupil_radius_px} is TOO COARSE: max pupil "
           f"phase step {rep['step_rad_at_current']:.2f} rad > {rep['limit_rad']:.1f}. "
           f"Axial profiles will show spurious ripple.")
    if not cfg.auto_sampling:
        if verbose:
            print(msg + f" Set pupil_radius_px >= {rep['recommended_px']}.")
        return cfg
    new = Config(**{**asdict(cfg), "pupil_radius_px": rep["recommended_px"],
                    "n_grid": None})
    if verbose:
        print(msg)
        print(f"[sampling] auto-raising pupil_radius_px "
              f"{cfg.pupil_radius_px} -> {new.pupil_radius_px} "
              f"(grid {new.grid_size}); this costs runtime.")
    return new


def _fresnel_avg(kza, kzb, na, nb):
    """Amplitude transmission, average of s and p (scalar approximation)."""
    kza = np.maximum(kza, 1e-9)
    kzb = np.maximum(kzb, 1e-9)
    ts = 2 * kza / (kza + kzb)
    tp = 2 * na * nb * kza / (nb ** 2 * kza + na ** 2 * kzb)
    return 0.5 * (ts + tp)


# =============================================================================
# SECTION 3 -- field engine
# =============================================================================
_AXIAL_SPREAD_TOL = 0.01      # max |d|*dg (rad); residual amplitude error ~ tol^2/12
_AXIAL_SPREAD_TOL_MAX = 0.05  # relax to here before giving up on the FFT path
_AXIAL_MAX_BINS = 1 << 23     # 8.4 M bins ~ 134 MB; ifft ~2 s worst case


def _axial_line_exact(pup: Pupil, c: float, z_um: np.ndarray,
                      y0_um: float, slope: float) -> np.ndarray:
    """Reference implementation: direct masked sum, one pass per depth."""
    Pk = pup.field_k(c)
    L2u = pup.cfg.L2_mm * MM
    out = np.empty(z_um.size, dtype=np.complex128)
    for i, z in enumerate(z_um):
        yc = y0_um + (z - L2u) * slope
        ph = pup.m_kz3 * (z - L2u) + pup.m_ky * yc
        out[i] = np.sum(Pk * np.exp(1j * ph))
    return out


def axial_line(pup: Pupil, c: float, z_um: np.ndarray,
               exact: bool = False,
               y0_um: float | None = None,
               slope: float | None = None) -> np.ndarray:
    """
    E along a straight line x = 0, y = y0 + (z-L2)*slope, at absolute depths z_um.

    Defaults to the geometric chief ray (y0 = y_L2, slope = tan(Om3)). Off axis you
    almost always want focus_track() instead -- see its docstring for why the chief
    ray is the wrong line to sample near focus.

    Whatever the line, the phase collapses to a single axial coordinate:

        phi(z) = kz3*(z-L2) + ky*y(z) = (z-L2)*g + ky*y0,   g = kz3 + ky*slope

    Everything about the pupil therefore enters through the 1-D "axial spectrum"
    A(g) -- the pupil amplitude projected onto g -- which does not depend on z at
    all. Binning A onto a uniform g grid (linear spreading) turns the whole depth
    scan into one FFT: cost O(n_k + nb log nb) instead of O(n_z * n_k). At the
    sampling the guard demands (n_k ~ 3e5) that is a ~1000x saving, and it is what
    makes a converged run finish in minutes rather than hours. Note this only works
    because the line is straight in (y, z) -- a curved track would not collapse.

    Set exact=True (or pass a non-uniform z) to get the direct sum instead; V0 in
    run_validation checks the two against each other.
    """
    z_um = np.atleast_1d(np.asarray(z_um, dtype=float))
    L2u = pup.cfg.L2_mm * MM
    y0 = pup.y_at_L2_um if y0_um is None else float(y0_um)
    m = pup.tanO3 if slope is None else float(slope)
    if exact or z_um.size < 8:
        return _axial_line_exact(pup, c, z_um, y0, m)

    dz = np.diff(z_um)
    if dz[0] <= 0 or not np.allclose(dz, dz[0], rtol=1e-9, atol=1e-12):
        return _axial_line_exact(pup, c, z_um, y0, m)  # non-uniform: no FFT path

    g = pup.m_kz3 + pup.m_ky * m
    A = pup.field_k(c) * np.exp(1j * pup.m_ky * y0)

    u = z_um - L2u
    u0 = 0.5 * (float(u[0]) + float(u[-1]))            # recentre so |d| stays small
    d = u - u0
    A = A * np.exp(1j * u0 * g)
    dmax = float(np.abs(d).max())
    g_lo, g_hi = float(g.min()), float(g.max())
    span = g_hi - g_lo
    step = float(dz[0])
    if span <= 0.0 or dmax == 0.0:
        return _axial_line_exact(pup, c, z_um, y0, m)
    if span * step >= 0.98 * 2 * np.pi:                # g range would wrap: alias
        return _axial_line_exact(pup, c, z_um, y0, m)

    # nb has to resolve the phase ramp: |d|*dg <= tol with dg = 2pi/(nb*step).
    # Since dmax ~ (n_z-1)*step/2 this is nb >~ pi*n_z/tol, i.e. it is set by how
    # many depths were asked for, not by the optics. Relax tol before falling back
    # to the direct sum -- the fallback is ~100x slower, so a 5x coarser spreading
    # (error still ~1e-4 relative) is by far the better trade.
    nb = 0
    for tol in (_AXIAL_SPREAD_TOL, _AXIAL_SPREAD_TOL_MAX):
        nb = 1 << int(np.ceil(np.log2(max(2 * np.pi * dmax / (tol * step),
                                          2.0 * z_um.size, 1024.0))))
        if nb <= _AXIAL_MAX_BINS:
            break
    if nb > _AXIAL_MAX_BINS:
        return _axial_line_exact(pup, c, z_um, y0, m)
    dg = 2 * np.pi / (nb * step)                       # makes step*dg = 2pi/nb exactly

    t = (g - g_lo) / dg
    i0 = np.floor(t).astype(np.intp)
    np.clip(i0, 0, nb - 2, out=i0)
    w = t - i0
    lo, hi = A * (1.0 - w), A * w
    acc = (np.bincount(i0, lo.real, nb) + np.bincount(i0 + 1, hi.real, nb)
           + 1j * (np.bincount(i0, lo.imag, nb) + np.bincount(i0 + 1, hi.imag, nb)))

    b = np.arange(nb)
    acc *= np.exp(1j * float(d[0]) * b * dg)
    out = nb * np.fft.ifft(acc)[:z_um.size]
    out *= np.exp(1j * d * g_lo)
    a = d * dg                                         # undo mean triangular kernel
    return out / (1.0 - a * a / 12.0)


def _pupil_field_at_z(pup: Pupil, c: float, z_um: float,
                      y_ref_um: float | None = None) -> np.ndarray:
    """
    Dense (N, N) pupil field propagated to depth z_um, lateral origin folded in.

    y_ref_um sets where the transverse coordinate s = 0 sits. Default is the
    geometric chief ray y_c(z), which SLIDES with depth; pass a fixed y_ref_um to
    get a frame that does not move (what you want for an XZ/YZ slice, otherwise the
    frame's own motion shears the picture -- see focus_track).
    """
    L2u = pup.cfg.L2_mm * MM
    yc = pup.y_center_um(z_um) if y_ref_um is None else float(y_ref_um)
    P = pup.field_k2d(c)
    return P * np.exp(1j * (pup.kz3 * (z_um - L2u) + pup.ky * yc))


def transverse_line(pup: Pupil, c: float, z_um: float, axis: str,
                    span_um: float, n_pts: int,
                    y_ref_um: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    E along a line through the lateral origin at depth z_um, along 'x' or 'y'.

    The other transverse coordinate is pinned, so that k axis can be summed out
    first (a plain reduction over the dense grid), leaving one 1-D DFT.
    Cost N^2 + n_pts*N instead of n_pts*n_k.
    """
    W = _pupil_field_at_z(pup, c, z_um, y_ref_um)
    s = np.linspace(-span_um, span_um, n_pts)
    if axis == "x":
        row = W.sum(axis=1)                                # sum over ky (y = origin)
        E = np.exp(1j * np.outer(s, pup.kax)) @ row
    else:
        col = W.sum(axis=0)                                # sum over kx (x = 0)
        E = np.exp(1j * np.outer(s, pup.kax)) @ col
    return s, E


def transverse_plane(pup: Pupil, c: float, z_um: float,
                     span_um: float, n_pts: int,
                     y_ref_um: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    E on an (x, y) patch centred on the lateral origin at depth z_um.

    Separable DFT on the dense grid: E = Wx . W . Wy^T, cost n_pts*N^2 + n_pts^2*N.
    Doing it on the masked point list instead costs n_pts^2 * n_k, which is ~60x
    worse once the sampling guard pushes the pupil past ~300 px radius.
    """
    W = _pupil_field_at_z(pup, c, z_um, y_ref_um)
    s = np.linspace(-span_um, span_um, n_pts)
    Wx = np.exp(1j * np.outer(s, pup.kax))                 # (n_pts, N)
    Wy = np.exp(1j * np.outer(pup.kax, s))                 # (N, n_pts)
    return s, Wx @ W @ Wy


def focus_track(pup: Pupil, c: float, z_um: np.ndarray, span_um: float,
                n_probe: int = 17, n_lat: int = 241) -> dict:
    """
    Fit the lateral track of the ACTUAL intensity maximum: y_peak(z) ~ y0 + m(z-L2).

    Why this exists. The geometric chief ray walks sideways at tan(Omega3). The
    field near focus does not: it stays pinned at the lateral position the pupil's
    linear phase puts it (measured on the 'current' preset at Omega=5.3 deg, the
    peak sits at y = 553.1 um across the whole +-63 um scan window while the chief
    ray moves 549.5 -> 558.1). Only far outside focus, where geometry takes over,
    does the intensity centroid resume walking at <ky>/<kz3> ~ tan(Omega3).

    So sampling the axial profile on the chief ray leaves the PSF core: at the FOV
    corner the chief ray moves 8.6 um across the window while the 2P spot is 1.8 um
    wide. Every axial number taken that way is wrong, and wrong in a seductive
    direction -- it reads FWHM_z as 21 um instead of 34 um and looks like the axial
    resolution IMPROVES off axis.

    x = 0 is exact by symmetry (the tilt is about x, and the pupil depends on kx
    only through kx^2), so only y has to be searched.

    Returns y0_um, slope, the residual of the linear fit, and the peak track. The
    residual matters: axial_line() can only follow a STRAIGHT line, so a large
    residual means no single line samples the core and the caller should fall back
    to a plane-by-plane maximum.
    """
    zp = np.linspace(float(z_um[0]), float(z_um[-1]), n_probe)
    L2u = pup.cfg.L2_mm * MM
    y_anchor = pup.y_center_um(0.5 * (zp[0] + zp[-1]))
    yk = np.empty(n_probe)
    Ik = np.empty(n_probe)
    edge = np.empty(n_probe)
    for i, zz in enumerate(zp):
        s, E = transverse_line(pup, c, zz, "y", span_um, n_lat, y_ref_um=y_anchor)
        I = np.abs(E) ** 4                                 # 2P-weighted: sharper peak
        j = int(np.argmax(I))
        edge[i] = abs(s[j]) / span_um
        Ik[i] = I[j]
        yk[i] = y_anchor + parabolic_peak(s, I)

    # Weight the fit by peak intensity and drop probes whose maximum ran to the
    # edge of the search span. Far out of focus the transverse maximum is broad and
    # wanders; unweighted, two such endpoints are enough to tilt the fitted line
    # right out of the core (measured: residual 2.2 um on a 1.5 um spot). The line
    # only has to pass through the bright part, so let the bright part set it.
    w = Ik / max(Ik.max(), 1e-300)
    keep = (w >= 0.1) & (edge <= 0.9)
    if keep.sum() < 3:
        keep = np.ones(n_probe, dtype=bool)
    m, y_at_L2 = np.polyfit((zp - L2u)[keep], yk[keep], 1, w=np.sqrt(w[keep]))
    resid = float(np.abs(yk[keep] - (y_at_L2 + m * (zp - L2u)[keep])).max())
    return {"y0_um": float(y_at_L2), "slope": float(m), "resid_um": resid,
            "z_probe": zp, "y_probe": yk, "weight": w, "kept": keep,
            "edge_frac": float(edge[keep].max()),
            "chief_slope": pup.tanO3,
            "walk_um": float((z_um[-1] - z_um[0]) * (pup.tanO3 - m))}


# =============================================================================
# SECTION 4 -- observables
# =============================================================================
def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """FWHM of a single-peaked profile by linear interpolation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError(f"fwhm: length mismatch x={x.size} y={y.size}")
    if y.max() <= 0:
        return np.nan
    yn = y / y.max()
    i0 = int(np.argmax(yn))
    half = 0.5

    def cross(idx_range):
        for i in idx_range:
            a, b = yn[i], yn[i + 1]
            if (a - half) * (b - half) <= 0 and a != b:
                t = (half - a) / (b - a)
                return x[i] + t * (x[i + 1] - x[i])
        return np.nan

    left = cross(range(i0 - 1, -1, -1))
    right = cross(range(i0, len(x) - 1))
    return abs(right - left) if np.isfinite(left) and np.isfinite(right) else np.nan


def parabolic_peak(x: np.ndarray, y: np.ndarray) -> float:
    """Sub-sample peak location by parabolic interpolation."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-300:
        return float(x[i])
    d = 0.5 * (y0 - y2) / den
    return float(x[i] + d * (x[1] - x[0]))


def saturate(I2p: np.ndarray, s: float) -> np.ndarray:
    """2P excitation probability with fluorophore saturation. I2p is |E|^4."""
    if s <= 0:
        return I2p / max(I2p.max(), 1e-300)
    Ihat = I2p / max(I2p.max(), 1e-300)
    return (1.0 - np.exp(-s * Ihat)) / (1.0 - np.exp(-s))


def sidelobe_fraction(z: np.ndarray, I: np.ndarray) -> float:
    """Energy outside the main axial lobe (first minima either side of the peak)."""
    I = np.asarray(I, dtype=float)
    i0 = int(np.argmax(I))
    lo = i0
    while lo > 0 and I[lo - 1] <= I[lo]:
        lo -= 1
    hi = i0
    while hi < len(I) - 1 and I[hi + 1] <= I[hi]:
        hi += 1
    tot = np.trapezoid(I, z)
    if tot <= 0:
        return np.nan
    main = np.trapezoid(I[lo:hi + 1], z[lo:hi + 1])
    return float(1.0 - main / tot)


def tag_time_samples(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Uniform time samples over HALF a TAG period; returns (t/T, cos factor)."""
    tau = np.linspace(0.0, 0.5, cfg.n_time)     # t / T
    return tau, np.cos(2 * np.pi * tau)


# =============================================================================
# SECTION 5 -- analysis drivers
# =============================================================================
_ZWIN_CACHE: dict = {}


def _cfg_key(cfg: Config) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in asdict(cfg).items()))


def wave_scan_window(cfg: Config, Omega: float) -> tuple[float, float]:
    """
    Absolute z window (um) that actually contains the wave-optical focus over the
    whole TAG swing.

    Why not just centre on the ray depth: with a large index mismatch the plane
    interface adds hundreds of um of spherical aberration, so the marginal-ray
    crossing and the brightest point are far apart. We therefore probe the wave
    field at c = +1 and c = -1 over a generous window (paraxial depth to marginal
    depth, plus several DOV) and bracket what we find.
    """
    key = (_cfg_key(cfg), round(float(Omega), 9))
    if key in _ZWIN_CACHE:
        return _ZWIN_CACHE[key]

    cf = closed_form_AB(cfg)
    g = dov_sine(cfg, Omega)
    L2u = cfg.L2_mm * MM
    d_mar = g["depth_rad_um"] if np.isfinite(g["depth_rad_um"]) else cf["d0_um"]
    d_par = cf["d_paraxial_um"]
    dov = np.nanmax([g["DOV_rad_um"], g["DOV_az_um"]])
    if not np.isfinite(dov) or dov <= 0:
        dov = 50.0

    lo = min(d_mar, d_par) - 2.0 * dov - 20.0
    hi = max(d_mar, d_par) + 2.0 * dov + 20.0
    zp = np.linspace(L2u + lo, L2u + hi, 1501)
    pup = Pupil(cfg, Omega)
    peaks = []
    for c in (1.0, 0.0, -1.0):
        I = np.abs(axial_line(pup, c, zp)) ** 2
        peaks.append(parabolic_peak(zp, I))
    zlo, zhi = min(peaks), max(peaks)
    pad = max(0.35 * (zhi - zlo), 0.25 * dov, 5.0)
    win = (zlo - pad, zhi + pad)
    _ZWIN_CACHE[key] = win
    return win


def z_grid(cfg: Config, Omega: float) -> np.ndarray:
    """Absolute z (um) grid spanning the real TAG scan at this field angle."""
    lo, hi = wave_scan_window(cfg, Omega)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * cfg.z_span_factor
    return np.linspace(mid - half, mid + half, cfg.n_z)


_TRACK_CACHE: dict = {}


def scan_frame(cfg: Config, Omega: float, c: float,
               pup: Pupil | None = None) -> dict:
    """
    Cached focus_track() for one (field angle, TAG phase): the line the axial
    observables should be sampled along instead of the chief ray.

    Note wave_scan_window() upstream still brackets on the chief ray. That is fine
    -- it only has to find a generously padded window to search in -- but it means
    the window, not the physics, is the thing the chief ray is allowed to set.
    """
    key = (_cfg_key(cfg), round(float(Omega), 9), round(float(c), 9))
    if key in _TRACK_CACHE:
        return _TRACK_CACHE[key]
    p = Pupil(cfg, Omega) if pup is None else pup
    tr = focus_track(p, c, z_grid(cfg, Omega), cfg.lateral_view_um)
    _TRACK_CACHE[key] = tr
    return tr


def scan_trajectory(cfg: Config, Omega: float, parabolic: bool = False) -> dict:
    """
    The real axial trajectory over one TAG half period:
    wave-optical brightest depth, plus 1P/2P axial profiles at each time.
    """
    pup = Pupil(cfg, Omega, parabolic_tag=parabolic)
    z = z_grid(cfg, Omega)
    tau, cs = tag_time_samples(cfg)
    I1 = np.empty((cs.size, z.size))
    zpk = np.empty(cs.size)
    for i, c in enumerate(cs):
        tr = focus_track(pup, c, z, cfg.lateral_view_um)   # not the chief ray
        E = axial_line(pup, c, z, y0_um=tr["y0_um"], slope=tr["slope"])
        I1[i] = np.abs(E) ** 2
        zpk[i] = parabolic_peak(z, I1[i] ** 2)      # 2P-weighted brightest depth
    # geometric reference at the same TAG settings
    zr = np.empty(cs.size)
    za = np.empty(cs.size)
    L2u = cfg.L2_mm * MM
    for i, c in enumerate(cs):
        th = theta_a_of_VT(cfg, cfg.VT_peak * c)
        d = depths_at_theta(cfg, Omega, th)
        zr[i] = L2u + d["depth_rad_um"]
        za[i] = L2u + d["depth_az_um"]
    return {"z": z, "tau": tau, "cos": cs, "I1": I1,
            "z_peak": zpk, "z_rad": zr, "z_az": za, "pupil": pup}


def slices_at(cfg: Config, Omega: float, c: float, z_focus: float,
              n_lat: int = 161) -> dict:
    """
    XY / XZ / YZ 2P intensity slices around the focus at TAG factor c.

    The lateral origin is FIXED at the beam's own position in the focal plane, not
    slid along the chief ray. A sliding frame shears the XZ/YZ panels by its own
    motion (8.6 um across the window at the FOV corner, on a 1.8 um spot), which
    reads as a dramatic diagonal streak that is the frame moving, not the beam. A
    fixed frame still shows genuine tilt if there is any.
    """
    pup = Pupil(cfg, Omega)
    span = cfg.lateral_view_um
    z = z_grid(cfg, Omega)
    tr = focus_track(pup, c, z, span)
    y_ref = tr["y0_um"] + tr["slope"] * (z_focus - cfg.L2_mm * MM)

    s, Exy = transverse_plane(pup, c, z_focus, span, n_lat, y_ref_um=y_ref)
    Ixy = np.abs(Exy) ** 2

    Ixz = np.empty((z.size, n_lat))
    Iyz = np.empty((z.size, n_lat))
    for i, zz in enumerate(z):
        _, Ex = transverse_line(pup, c, zz, "x", span, n_lat, y_ref_um=y_ref)
        _, Ey = transverse_line(pup, c, zz, "y", span, n_lat, y_ref_um=y_ref)
        Ixz[i] = np.abs(Ex) ** 2
        Iyz[i] = np.abs(Ey) ** 2
    return {"s": s, "z": z, "Ixy": Ixy, "Ixz": Ixz, "Iyz": Iyz,
            "z_focus": z_focus, "pupil": pup, "track": tr, "y_ref_um": y_ref}


def slices_period(cfg: Config, Omega: float, n_lat: int = 161) -> dict:
    """
    TAG-period-integrated 2P excitation. Uniform sampling in t automatically
    carries the arcsine dwell weighting (cos moves slowly at the turning points).
    """
    pup = Pupil(cfg, Omega)
    span = cfg.lateral_view_um
    z = z_grid(cfg, Omega)
    tau, cs = tag_time_samples(cfg)
    Ixz = np.zeros((z.size, n_lat))
    Iyz = np.zeros((z.size, n_lat))
    Ixy_stack = np.zeros((n_lat, n_lat))
    zc = 0.5 * (z[0] + z[-1])
    L2u = cfg.L2_mm * MM

    # The TAG scan does NOT go straight down at finite field angle. Driving the
    # focus in depth also slides it along the chief-ray direction: measured
    # dy/dz = 0.0673 against tan(Om3) = 0.0681, so over one sweep the focus walks
    # 3.1 um sideways -- more than two 2P spot widths. (This is a different motion
    # from defocus at FIXED TAG phase, where the peak does not move laterally at
    # all; conflating the two is easy and wrong.) So the dose lies along a TILTED
    # line, and the slices are taken in that line's frame. A frame that ignored the
    # tilt would cut across the dose and destroy the dwell-time double hump.
    zf_l, yf_l = [], []
    for c in cs:
        trc = focus_track(pup, c, z, span)
        Ic = np.abs(axial_line(pup, c, z,
                               y0_um=trc["y0_um"], slope=trc["slope"])) ** 4
        zfc = parabolic_peak(z, Ic)
        zf_l.append(zfc)
        yf_l.append(trc["y0_um"] + trc["slope"] * (zfc - L2u))
    zf_a, yf_a = np.array(zf_l), np.array(yf_l)
    scan_slope, scan_y0 = np.polyfit(zf_a - L2u, yf_a, 1)

    for c in cs:
        for i, zz in enumerate(z):
            yref = scan_y0 + scan_slope * (zz - L2u)
            _, Ex = transverse_line(pup, c, zz, "x", span, n_lat, y_ref_um=yref)
            _, Ey = transverse_line(pup, c, zz, "y", span, n_lat, y_ref_um=yref)
            Ixz[i] += np.abs(Ex) ** 4
            Iyz[i] += np.abs(Ey) ** 4
        _, Exy = transverse_plane(pup, c, zc, span, n_lat,
                                  y_ref_um=scan_y0 + scan_slope * (zc - L2u))
        Ixy_stack += np.abs(Exy) ** 4
    s = np.linspace(-span, span, n_lat)

    # The DOSE ridge is not the FOCUS locus. The axial PSF (34 um FWHM here) is
    # comparable to the whole scan (46 um), so every depth receives light from TAG
    # phases whose foci sit at other lateral positions, and the superposition pulls
    # the ridge back toward the mean: measured 0.042 against a focus walk of 0.067.
    # Report both so the residual lean left in the YZ panel reads as physics.
    wcol = Iyz.max(axis=1)
    wcol = wcol / max(wcol.max(), 1e-300)
    sel = wcol > 0.25
    dose_slope = np.nan
    if sel.sum() > 2:
        ridge = np.array([parabolic_peak(s, Iyz[i]) for i in range(z.size)])
        dose_slope = float(np.polyfit(z[sel], ridge[sel], 1)[0] + scan_slope)

    return {"s": s, "z": z, "Ixz": Ixz / cs.size, "Iyz": Iyz / cs.size,
            "Ixy": Ixy_stack / cs.size, "tau": tau, "cos": cs, "pupil": pup,
            "dose_slope": dose_slope,
            "scan_slope": float(scan_slope), "scan_y0_um": float(scan_y0),
            # per-TAG-phase focus positions; NOT "z_focus", which _2p_slice_figure
            # reads as the single scalar depth of the XY panel
            "z_focus_of_c": zf_a, "y_focus_of_c": yf_a,
            "walk_um": float(yf_a.max() - yf_a.min())}


def parabolic_min(x: np.ndarray, y: np.ndarray) -> float:
    """Sub-sample minimum location, ignoring NaNs."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        return np.nan
    i = int(np.nanargmin(y))
    if i == 0 or i == len(y) - 1 or not np.isfinite(y[i - 1: i + 2]).all():
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if abs(den) < 1e-300:
        return float(x[i])
    return float(x[i] + 0.5 * (y0 - y2) / den * (x[1] - x[0]))


def psf_metrics(cfg: Config, Omega: float, c: float, n_lat: int = 161,
                n_zloc: int = 61) -> dict:
    """
    Experimentally measurable PSF metrics at one (field angle, TAG phase).

    The two astigmatic focal lines are located on a LOCAL z window around the
    brightest point -- sized from the axial FWHM and the ray-model astigmatism --
    not on the full TAG scan window, which is orders of magnitude too coarse to
    resolve a few-um focal-line split.
    """
    pup = Pupil(cfg, Omega)
    z = z_grid(cfg, Omega)
    span = cfg.lateral_view_um
    L2u = cfg.L2_mm * MM

    # --- sample along the beam, not along the chief ray (see focus_track) ---
    tr = scan_frame(cfg, Omega, c, pup)
    y0, m = tr["y0_um"], tr["slope"]
    Iz = np.abs(axial_line(pup, c, z, y0_um=y0, slope=m)) ** 2
    zf = parabolic_peak(z, Iz)

    # --- axial FWHM on a window sized by the MEASURED width ---------------------
    # The diffraction-limited estimate 2*n3*lam/NA^2 is 8.1 um for the default
    # preset while the real caustic is 33 um, so sizing the window from it made
    # fwhm() return NaN and silently drop points from figures 04 and 08. Measure
    # first on the full scan window, then refine on a window that actually fits.
    fz_est = 2.0 * cfg.n3 * cfg.wavelength_um / max(cfg.NA ** 2, 1e-6)
    fz_rough = fwhm(z, Iz ** 2)
    fz_use = fz_rough if np.isfinite(fz_rough) else 2.0 * fz_est
    zax = np.linspace(zf - 1.6 * fz_use, zf + 1.6 * fz_use, max(n_zloc, 201))
    Izax = np.abs(axial_line(pup, c, zax, y0_um=y0, slope=m)) ** 2
    fz_1p, fz_2p = fwhm(zax, Izax), fwhm(zax, Izax ** 2)

    # --- focal-line split on its own, much narrower, window --------------------
    th = theta_a_of_VT(cfg, cfg.VT_peak * c)
    g = depths_at_theta(cfg, Omega, th)
    astig_ray = g["astig_um"] if np.isfinite(g["astig_um"]) else 0.0
    # Expand until BOTH width minima are interior. A fixed guess silently pins a
    # minimum to the window edge and reports that edge as a focal line -- which is
    # how the split can come out confidently wrong.
    half = max(2.0 * fz_est, 1.6 * abs(astig_ray) + 3.0)
    for _try in range(4):
        zloc = np.linspace(zf - half, zf + half, n_zloc)
        wx_l, wy_l = [], []
        for zz in zloc:
            yref = y0 + m * (zz - L2u)
            sxx, Exx = transverse_line(pup, c, zz, "x", span, 81, y_ref_um=yref)
            syy, Eyy = transverse_line(pup, c, zz, "y", span, 81, y_ref_um=yref)
            wx_l.append(fwhm(sxx, np.abs(Exx) ** 2))
            wy_l.append(fwhm(syy, np.abs(Eyy) ** 2))
        wx, wy = np.array(wx_l), np.array(wy_l)
        if not (np.isfinite(wx).any() and np.isfinite(wy).any()):
            break
        jx, jy = int(np.nanargmin(wx)), int(np.nanargmin(wy))
        pad = max(2, n_zloc // 12)
        if pad <= jx < n_zloc - pad and pad <= jy < n_zloc - pad:
            break
        half *= 1.9
    # sagittal rays converge where the spot is narrow in x  -> z_best_x = d_az
    # tangential rays converge where the spot is narrow in y -> z_best_y = d_rad
    zx = parabolic_min(zloc, wx)
    zy = parabolic_min(zloc, wy)
    # Are the two focal lines actually distinguishable, or has a big spherical
    # caustic pulled both minima onto its own waist? Asking "is there a clear
    # minimum" is not enough -- the caustic gives each width a perfectly clear
    # minimum, at the SAME place, and the split then reads as a confident zero.
    # The real question is whether moving from one focal line to the other
    # measurably broadens the other axis.
    def _at(zq, arr):
        good = np.isfinite(arr)
        return float(np.interp(zq, zloc[good], arr[good])) if good.any() else np.nan
    with np.errstate(invalid="ignore"):
        cx = _at(zy, wx) / max(np.nanmin(wx), 1e-9) - 1.0
        cy = _at(zx, wy) / max(np.nanmin(wy), 1e-9) - 1.0
    astig_contrast = float(np.nanmin([cx, cy]))
    astig_resolved = bool(np.isfinite(astig_contrast) and astig_contrast > 0.05)

    yref_f = y0 + m * (zf - L2u)
    sx, Ex = transverse_line(pup, c, zf, "x", span, n_lat, y_ref_um=yref_f)
    sy, Ey = transverse_line(pup, c, zf, "y", span, n_lat, y_ref_um=yref_f)
    Ix, Iy = np.abs(Ex) ** 2, np.abs(Ey) ** 2

    return {
        "z_focus_um": zf,
        "peak_1p": float(Iz.max()),
        "fwhm_x_um": fwhm(sx, Ix), "fwhm_y_um": fwhm(sy, Iy),
        "fwhm_z_um": fz_1p,
        "fwhm_x_2p_um": fwhm(sx, Ix ** 2), "fwhm_y_2p_um": fwhm(sy, Iy ** 2),
        "fwhm_z_2p_um": fz_2p,
        "sidelobe_frac": sidelobe_fraction(z, Iz ** 2),
        "sidelobe_window_um": float(z[-1] - z[0]),
        "z_best_x_um": zx, "z_best_y_um": zy,
        "astig_wave_um": zx - zy,          # = d_az - d_rad, same sign as the ray model
        "astig_resolved": astig_resolved,
        "astig_contrast": astig_contrast,
        "astig_ray_um": astig_ray,
        "wx_of_z": wx, "wy_of_z": wy, "z": zloc,
        "sx": sx, "Ix": Ix, "Iy": Iy, "Iz": Iz,
        "track": tr,
        "vignette_frac": pup.vignette_frac,
    }


def effective_volume(cfg: Config, Omega: float, c: float,
                     n_lat: int = 61, n_zv: int = 81) -> float:
    """
    2P effective excitation volume  V_eff = (∫I² dV)² / ∫I⁴ dV  [um^3].
    This is the quantity FCS measures, so it is directly checkable on the bench.
    """
    pup = Pupil(cfg, Omega)
    zg = z_grid(cfg, Omega)
    z = np.linspace(zg[0], zg[-1], n_zv)
    span = cfg.lateral_view_um
    s = np.linspace(-span, span, n_lat)
    dv = (s[1] - s[0]) ** 2 * (z[1] - z[0])
    # Integration box follows the beam. On the chief ray the box slides out from
    # under the PSF off axis and truncates it, which made V_eff shrink with field
    # even though the spot was getting wider.
    tr = scan_frame(cfg, Omega, c, pup)
    L2u = cfg.L2_mm * MM
    num = 0.0
    den = 0.0
    for zz in z:
        _, E = transverse_plane(pup, c, zz, span, n_lat,
                                y_ref_um=tr["y0_um"] + tr["slope"] * (zz - L2u))
        I2 = np.abs(E) ** 4          # |E|^4  == I^2 with I = |E|^2
        num += I2.sum()
        den += (I2 ** 2).sum()
    if den <= 0:
        return np.nan
    return float((num * dv) ** 2 / (den * dv))


# =============================================================================
# SECTION 6 -- validation
# =============================================================================
def _variant(cfg: Config, **kw) -> Config:
    return Config(**{**asdict(cfg), **kw})


def _wave_shift_per_VT(cfg: Config, parabolic: bool, uniform_amp: bool = False,
                       n_z: int = 4001) -> tuple[float, float]:
    """(shift per unit V_T [um], focus depth at c=0 [um]) on axis, uniform-safe."""
    pup = Pupil(cfg, 0.0, parabolic_tag=parabolic, uniform_amp=uniform_amp)
    lo, hi = wave_scan_window(cfg, 0.0)
    pad = 0.6 * (hi - lo) + 10.0
    z = np.linspace(lo - pad, hi + pad, n_z)
    pk = [parabolic_peak(z, np.abs(axial_line(pup, c, z)) ** 2) for c in (0.0, 1.0)]
    return (pk[1] - pk[0]) / cfg.VT_peak, pk[0]


def run_validation(cfg: Config, verbose: bool = True) -> dict:
    res: dict = {}

    # ---------- V0 : numerics -- the fast axial transform vs the direct sum
    # axial_line() bins the pupil onto the axial coordinate g and FFTs. Everything
    # downstream (focus depth, DOV, FWHMz, sidelobes) rides on it, so check it
    # against the brute-force sum it replaces, on the real config, on and off axis.
    _v0 = []
    for _Om in (0.0, cfg.Omega_corner):
        _pup = Pupil(cfg, _Om)
        _lo, _hi = wave_scan_window(cfg, _Om)
        _z = np.linspace(_lo, _hi, 201)
        for _c in (0.0, 1.0):
            _f = axial_line(_pup, _c, _z)
            _e = axial_line(_pup, _c, _z, exact=True)
            _v0.append((float(np.abs(_f - _e).max() / np.abs(_e).max()),
                        parabolic_peak(_z, np.abs(_f) ** 4)
                        - parabolic_peak(_z, np.abs(_e) ** 4)))
    res["V0"] = {"max_rel_field_err": max(a for a, _ in _v0),
                 "max_peak_shift_um": max(abs(b) for _, b in _v0)}

    # ---------- V6 : lateral frame -- chief ray vs the beam's own track
    # On axis the two must agree exactly (nothing to walk). Off axis the chief ray
    # walks out of the core, which shows up as a spuriously NARROW axial PSF. This
    # test exists so that regression is loud: if someone puts the chief ray back,
    # fwhm_z at the corner drops ~40% and this line says so.
    _v6 = []
    for _Om in cfg.fields():
        _pup = Pupil(cfg, _Om)
        _z = z_grid(cfg, _Om)
        _tr = scan_frame(cfg, _Om, 0.0, _pup)
        _I_chief = np.abs(axial_line(_pup, 0.0, _z)) ** 4
        _I_track = np.abs(axial_line(_pup, 0.0, _z, y0_um=_tr["y0_um"],
                                     slope=_tr["slope"])) ** 4
        _v6.append({"Omega_deg": np.degrees(_Om),
                    "fwhm_z_chief_um": fwhm(_z, _I_chief),
                    "fwhm_z_track_um": fwhm(_z, _I_track),
                    "walk_um": _tr["walk_um"], "resid_um": _tr["resid_um"],
                    "slope": _tr["slope"], "chief_slope": _tr["chief_slope"]})
    res["V6"] = _v6

    # ---------- V1 : homogeneous, no TAG -> ideal Airy at z = -L1 + sqrt(W1)
    # uniform pupil amplitude, otherwise the Gaussian apodisation broadens the
    # spot and the Airy comparison is meaningless.
    c1 = _variant(cfg, n2=cfg.n1, n3=cfg.n1, field_list_deg=None)
    pup = Pupil(c1, 0.0, tag_on=False, uniform_amp=True)
    z_pred = -c1.L1_mm * MM + np.sqrt(c1.W(1)) * MM
    zz = np.linspace(z_pred - 30, z_pred + 30, 601)
    I = np.abs(axial_line(pup, 0.0, zz)) ** 2
    z_found = parabolic_peak(zz, I)
    s, Ex = transverse_line(pup, 0.0, z_found, "x", 4.0, 801)
    fw = fwhm(s, np.abs(Ex) ** 2)
    fw_theory = 0.5145 * c1.wavelength_um / c1.NA          # Airy FWHM
    res["V1"] = {"z_pred_um": z_pred, "z_found_um": z_found,
                 "dz_um": z_found - z_pred,
                 "fwhm_um": fw, "fwhm_airy_um": fw_theory,
                 "fwhm_err_pct": 100 * (fw - fw_theory) / fw_theory}

    # ---------- V2 : paraxial limit -- the ONE place the ray closed form must hold
    # Uniform medium (kills B1,B2) and NA -> 0 (kills every finite-NA correction),
    # so both models must collapse onto  dz = -n3 F^2 V / M^2.
    cpx = _variant(cfg, n2=cfg.n1, n3=cfg.n1, L1_mm=0.0, L2_mm=0.0,
                   R_mm=0.03 * cfg.F_mm, beam_radius_pupil_mm=None)
    dz_px, _ = _wave_shift_per_VT(cpx, parabolic=True)
    cf_px = closed_form_AB(cpx)
    res["V2"] = {"NA": cpx.NA,
                 "dz_wave_um": dz_px,
                 "dz_closed_um": cf_px["dz_per_VT_um"],
                 "dz_paraxial_um": cf_px["dz_per_VT_paraxial_um"],
                 "err_pct": 100 * (dz_px - cf_px["dz_per_VT_um"]) / abs(cf_px["dz_per_VT_um"])}

    # ---------- V2b : finite NA, uniform medium -- the fold disagreement
    cfin = _variant(cfg, n2=cfg.n1, n3=cfg.n1, L1_mm=0.0, L2_mm=0.0)
    dz_g, _ = _wave_shift_per_VT(cfin, parabolic=True)
    dz_u, _ = _wave_shift_per_VT(cfin, parabolic=True, uniform_amp=True)
    cf_fin = closed_form_AB(cfin)
    res["V2b"] = {"NA": cfin.NA,
                  "dz_wave_gauss_um": dz_g, "dz_wave_uniform_um": dz_u,
                  "dz_md_um": cf_fin["dz_per_VT_um"],
                  "dz_md_cos2_um": cf_fin["dz_per_VT_wave_um"],
                  "dz_paraxial_um": cf_fin["dz_per_VT_paraxial_um"],
                  "md_over_pct": 100 * (cf_fin["dz_per_VT_um"] - dz_g) / abs(dz_g)}

    # ---------- V3 : full J0 vs parabolic -> Bessel shortening ratio
    # Run in a UNIFORM medium so the only shortening mechanism left is the J0 shape
    # itself. With the interfaces in, the plane-interface caustic dominates and the
    # "brightest point" hops between lobes -- the ratio then measures nothing.
    c3 = _variant(cfg, n2=cfg.n1, n3=cfg.n1, L1_mm=0.0, L2_mm=0.0)
    dz_par, _ = _wave_shift_per_VT(c3, parabolic=True)
    dz_j0, _ = _wave_shift_per_VT(c3, parabolic=False)
    dz_px3 = closed_form_AB(c3)["dz_per_VT_paraxial_um"]
    res["V3"] = {"dz_J0_um": dz_j0, "dz_parabolic_um": dz_par,
                 "dz_paraxial_um": dz_px3,
                 "ratio_vs_parabolic": dz_j0 / dz_par if dz_par else np.nan,
                 "ratio_vs_paraxial": dz_j0 / dz_px3 if dz_px3 else np.nan,
                 "phi_amp_rad": cfg.phi_amp,
                 "kr_w": cfg.k_r * cfg.beam_radius_mm / cfg.M_net}

    # ---------- V4 : off-axis astigmatism, wave vs ray
    # Same media and layer thicknesses as the real system, but at a lower NA so the
    # plane-interface spherical aberration does not smear the focus into a caustic
    # and the two focal lines are actually resolvable.
    c4 = _variant(cfg, R_mm=0.25 * cfg.F_mm, beam_radius_pupil_mm=None)
    Om = c4.Omega_corner
    m = psf_metrics(c4, Om, 0.0, n_lat=121, n_zloc=81)
    res["V4"] = {"Omega_deg": np.degrees(Om), "NA": c4.NA,
                 "astig_wave_um": m["astig_wave_um"],
                 "astig_ray_um": m["astig_ray_um"],
                 "z_best_x_um": m["z_best_x_um"], "z_best_y_um": m["z_best_y_um"],
                 "cfg": c4}

    # ---------- V5 : how aberrated is the ACTUAL configuration?
    cf = closed_form_AB(cfg)
    _, z_wave = _wave_shift_per_VT(cfg, parabolic=False)
    L2u = cfg.L2_mm * MM
    res["V5"] = {"d_marginal_um": cf["d0_um"], "d_paraxial_um": cf["d_paraxial_um"],
                 "spherical_um": cf["spherical_um"],
                 "d_wave_peak_um": z_wave - L2u}

    # ---------- sine vs tangent, for the record
    ct = closed_form_AB_tangent(cfg)
    res["sine_vs_tangent"] = {
        "theta10_sine_deg": np.degrees(cfg.theta10), "NA_sine": cfg.NA,
        "theta10_tan_deg": ct["theta10_deg"], "NA_tan": ct["NA"],
        "dz_per_VT_sine_um": cf["dz_per_VT_um"],
        "dz_per_VT_tan_um": ct["dz_per_VT_um"],
        "diff_pct": 100 * (cf["dz_per_VT_um"] - ct["dz_per_VT_um"]) / abs(ct["dz_per_VT_um"]),
    }

    if verbose:
        _print_validation(cfg, res)
    return res


def _print_validation(cfg: Config, r: dict) -> None:
    W = 78
    print("=" * W)
    print(f" VALIDATION  --  preset '{cfg.name}'   NA={cfg.NA:.3f}  "
          f"lam={cfg.wavelength_um} um  phi_amp={cfg.phi_amp:.1f} rad")
    print("=" * W)
    v = r["V0"]
    print(f"V0 numerics: fast axial transform vs direct sum (real config, on+off axis)")
    print(f"   max relative field error {v['max_rel_field_err']:.2e}"
          f"   max focus shift {v['max_peak_shift_um']:.2e} um")
    print(f"V6 lateral frame: chief ray vs the beam's own track")
    print(f"   {'Omega':>7} {'FWHMz chief':>12} {'FWHMz track':>12} "
          f"{'chief walk':>11} {'fit resid':>10}")
    for v in r["V6"]:
        print(f"   {v['Omega_deg']:7.2f} {v['fwhm_z_chief_um']:12.2f} "
              f"{v['fwhm_z_track_um']:12.2f} {v['walk_um']:10.2f}u "
              f"{v['resid_um']:9.3f}u")
    v = r["V1"]
    print(f"V1 homogeneous, no TAG")
    print(f"   focus  predicted {v['z_pred_um']:10.3f} um   found {v['z_found_um']:10.3f} um"
          f"   (dz {v['dz_um']:+.3f})")
    print(f"   lateral FWHM  sim {v['fwhm_um']:.4f} um  vs Airy {v['fwhm_airy_um']:.4f} um"
          f"   ({v['fwhm_err_pct']:+.2f}%)")
    v = r["V2"]
    print(f"V2 paraxial limit (uniform medium, NA={v['NA']:.3f}) -- must match")
    print(f"   wave {v['dz_wave_um']:+9.4f}  closed {v['dz_closed_um']:+9.4f}"
          f"  paraxial {v['dz_paraxial_um']:+9.4f} um/(1/m)   ({v['err_pct']:+.3f}%)")
    v = r["V2b"]
    print(f"V2b finite NA={v['NA']:.3f}, uniform medium -- the fold disagreement")
    print(f"   wave: gauss {v['dz_wave_gauss_um']:+8.4f}  uniform {v['dz_wave_uniform_um']:+8.4f}")
    print(f"   ray : md    {v['dz_md_um']:+8.4f}  md*cos^2 {v['dz_md_cos2_um']:+8.4f}"
          f"  paraxial {v['dz_paraxial_um']:+8.4f}")
    print(f"   -> md OVER-predicts the scan range by {v['md_over_pct']:+.1f}%"
          f"  (its finite-NA correction has the wrong sign)")
    v = r["V3"]
    print(f"V3 full J0 vs parabolic, uniform medium (Bessel shortening)")
    print(f"   dz(J0) {v['dz_J0_um']:+.4f}   dz(parab) {v['dz_parabolic_um']:+.4f}"
          f"   dz(paraxial) {v['dz_paraxial_um']:+.4f}")
    print(f"   ratio vs parabolic {v['ratio_vs_parabolic']:.3f}"
          f"   vs paraxial {v['ratio_vs_paraxial']:.3f}   [doc expects 0.70-0.95]")
    print(f"   phi_amp {v['phi_amp_rad']:.1f} rad   k_r*w_TAG {v['kr_w']:.2f}"
          f"   (J0 first zero at 2.405)")
    v = r["V4"]
    print(f"V4 off-axis astigmatism, real media at reduced NA={v['NA']:.3f}, "
          f"Omega={v['Omega_deg']:.2f} deg")
    print(f"   wave {v['astig_wave_um']:+.3f} um   ray {v['astig_ray_um']:+.3f} um")
    v = r["V5"]
    print(f"V5 how aberrated is THIS configuration")
    print(f"   marginal-ray depth {v['d_marginal_um']:9.2f} um")
    print(f"   paraxial     depth {v['d_paraxial_um']:9.2f} um   "
          f"-> plane-interface SA = {v['spherical_um']:+.1f} um")
    print(f"   wave peak    depth {v['d_wave_peak_um']:9.2f} um")
    v = r["sine_vs_tangent"]
    print("-" * W)
    print(f"sine vs tangent condition (why the old code's geometry was wrong)")
    print(f"   theta10  sine {v['theta10_sine_deg']:6.2f} deg (NA {v['NA_sine']:.3f})"
          f"   tangent {v['theta10_tan_deg']:6.2f} deg (NA {v['NA_tan']:.3f})")
    print(f"   dz per unit V_T   sine {v['dz_per_VT_sine_um']:+.4f} um"
          f"   tangent {v['dz_per_VT_tan_um']:+.4f} um   ({v['diff_pct']:+.2f}%)")
    print("=" * W)


# =============================================================================
# SECTION 7 -- figures
# =============================================================================
FIGURES = {
    "01": "scan_trajectory",
    "02": "psf_xy",
    "03": "psf_z",
    "04": "field_sweep",
    "05": "2p_instant_slices",
    "06": "2p_period_slices",
    "07": "pupil_diagnostics",
    "08": "psf_metrics",
    "09": "focal_surface",
    "10": "validation",
}


def _stamp(fig, cfg: Config) -> None:
    fig.text(0.005, 0.005,
             f"{cfg.name} | NA={cfg.NA:.3f} (sine) | n={cfg.n1}/{cfg.n2}/{cfg.n3} | "
             f"R={cfg.R_mm} F={cfg.F_mm} L1={cfg.L1_mm} L2={cfg.L2_mm} mm | "
             f"VT={cfg.VT_peak}/m phi={cfg.phi_amp:.1f}rad | lam={cfg.wavelength_um}um",
             fontsize=6, color="0.35")


def fig01_scan_trajectory(cfg: Config, out: Path) -> None:
    fields = cfg.fields()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(fields)))

    trs = [scan_trajectory(cfg, Om) for Om in fields]      # expensive: compute once
    ratios, offsets = [], []
    for k, Om in enumerate(fields):
        tr = trs[k]
        tau, zp = tr["tau"], tr["z_peak"]
        lab = f"$\\Omega$={np.degrees(Om):.2f}$^\\circ$"

        # ideal cosine with the SAME endpoints (cos runs +1 -> -1 over t/T 0 -> 0.5)
        amp = 0.5 * (zp[0] - zp[-1])
        ideal = zp.mean() + amp * tr["cos"]

        axes[0].plot(tau, zp, "-o", ms=3, color=colors[k], label=f"wave, {lab}")
        if k == 0:
            axes[0].plot(tau, ideal, "-", lw=1.2, color="crimson", alpha=0.7,
                         label="ideal cosine (same endpoints)")
        ratios.append((np.degrees(Om), zp.max() - zp.min()))
        offsets.append((np.degrees(Om), np.nanmean(tr["z_rad"]) - zp.mean()))

        # de-meaned comparison: shapes and amplitudes, offset removed
        axes[1].plot(tau, zp - zp.mean(), "-o", ms=3, color=colors[k],
                     label=f"wave, {lab}")
        axes[1].plot(tau, tr["z_rad"] - np.nanmean(tr["z_rad"]), "--", lw=1.2,
                     color=colors[k], alpha=0.75)
        axes[1].plot(tau, tr["z_az"] - np.nanmean(tr["z_az"]), ":", lw=1.6,
                     color=colors[k], alpha=0.75)
        if k == 0:
            axes[1].plot(tau, ideal - ideal.mean(), "-", lw=1.2, color="crimson",
                         alpha=0.7, label="ideal cosine")

    axes[0].set_xlabel("t / T  (half TAG period)")
    axes[0].set_ylabel("focus depth  z  [$\\mu$m]  (from $n_1/n_2$ face)")
    off_txt = "/".join(f"{b:+.0f}" for _, b in offsets)
    axes[0].set_title("Real axial scan trajectory\n(wave-optical brightest point; "
                      f"ray $z_{{rad}}$ offset by {off_txt} $\\mu$m = plane-interface SA)",
                      fontsize=9)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("t / T")
    axes[1].set_ylabel("$z - \\langle z\\rangle$  [$\\mu$m]")
    axes[1].set_title("Same curves, offsets removed\n"
                      "solid: wave   dashed: $z_{rad}$   dotted: $z_{az}$   red: cosine",
                      fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    # dwell density from the SIMULATED trajectory: |dz/dt|^-1 on z(t) as computed,
    # not the textbook arcsine of an ideal cosine. Where the two differ is exactly
    # where the TAG scan is not a cosine in depth.
    zn = np.linspace(-0.995, 0.995, 400)
    axes[2].plot(zn, 1.0 / np.sqrt(1 - zn ** 2), color="darkorange",
                 label="ideal cosine (arcsine law)")
    for k, Om in enumerate(fields):
        tr = trs[k]
        zp, tau = tr["z_peak"], tr["tau"]
        zc = 0.5 * (zp.max() + zp.min())
        amp = 0.5 * (zp.max() - zp.min())
        zm = 0.5 * (zp[1:] + zp[:-1])
        dens = np.abs(np.gradient(tau)[0] / np.maximum(np.abs(np.diff(zp)), 1e-9))
        dens *= (zp.max() - zp.min()) / 0.5          # normalise to mean 1
        axes[2].plot((zm - zc) / amp, dens, "-", lw=1.0, color=colors[k], alpha=0.85,
                     label=f"sim, $\\Omega$={np.degrees(Om):.2f}$^\\circ$")
    axes[2].set_xlabel("normalised depth  $(z-z_0)/\\Delta z$")
    axes[2].set_ylabel("dwell density  $\\propto |dz/dt|^{-1}$")
    axes[2].set_title("Dwell density: simulated vs ideal\n"
                      "cosine scan lingers at the turning points", fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=6.5)
    axes[2].set_ylim(0, 8)

    txt = "  |  ".join(f"$\\Omega$={a:.1f}$^\\circ$: DOV={b:.1f}$\\mu$m" for a, b in ratios)
    fig.suptitle(f"01  Scan trajectory      {txt}", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / "01_scan_trajectory.png", dpi=150)
    plt.close(fig)


def fig02_psf_xy(cfg: Config, out: Path) -> None:
    fields = cfg.fields()
    cs = [1.0, 0.0, -1.0]
    fig, axes = plt.subplots(len(fields), len(cs) * 2,
                             figsize=(3.0 * len(cs) * 2, 3.1 * len(fields)),
                             squeeze=False)
    span = cfg.lateral_view_um
    for i, Om in enumerate(fields):
        pup = Pupil(cfg, Om)
        for j, c in enumerate(cs):
            z = z_grid(cfg, Om)
            tr = scan_frame(cfg, Om, c, pup)
            Iz = np.abs(axial_line(pup, c, z,
                                   y0_um=tr["y0_um"], slope=tr["slope"])) ** 2
            zf = parabolic_peak(z, Iz)
            s, E = transverse_plane(pup, c, zf, span, 161,
                                    y_ref_um=tr["y0_um"]
                                    + tr["slope"] * (zf - cfg.L2_mm * MM))
            I = np.abs(E) ** 2
            I /= I.max()
            for kk, (dat, lbl) in enumerate(((I, "1P  $|E|^2$"), (I ** 2, "2P  $|E|^4$"))):
                ax = axes[i][2 * j + kk]
                im = ax.imshow(dat.T, origin="lower", cmap="inferno",
                               extent=[-span, span, -span, span], vmin=0, vmax=1)
                ax.contour(s, s, dat.T, levels=[0.5], colors="c", linewidths=0.8)
                fx = fwhm(s, dat[:, dat.shape[1] // 2])
                fy = fwhm(s, dat[dat.shape[0] // 2, :])
                ax.set_title(f"$\\Omega$={np.degrees(Om):.2f}$^\\circ$  c={c:+.0f}\n"
                             f"{lbl}  FWHM {fx:.2f}/{fy:.2f} $\\mu$m", fontsize=8)
                if i == len(fields) - 1:
                    ax.set_xlabel("x [$\\mu$m]")
                if 2 * j + kk == 0:
                    ax.set_ylabel("y [$\\mu$m]")
                ax.tick_params(labelsize=7)
    fig.suptitle("02  PSF$_{XY}$   cyan = FWHM contour;  x/y FWHM split = astigmatism",
                 fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "02_psf_xy.png", dpi=150)
    plt.close(fig)


def fig03_psf_z(cfg: Config, out: Path) -> None:
    fields = cfg.fields()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(fields)))
    for k, Om in enumerate(fields):
        pup = Pupil(cfg, Om)
        z = z_grid(cfg, Om)
        z = np.linspace(z[0], z[-1], max(cfg.n_z, 401))
        tr = scan_frame(cfg, Om, 0.0, pup)              # follow the beam, not the chief ray
        I = np.abs(axial_line(pup, 0.0, z,
                              y0_um=tr["y0_um"], slope=tr["slope"])) ** 2
        I /= I.max()
        zf = parabolic_peak(z, I)
        lab = f"$\\Omega$={np.degrees(Om):.2f}$^\\circ$"
        axes[0][0].plot(z - zf, I, color=colors[k], label=lab)
        axes[0][1].semilogy(z - zf, np.maximum(I, 1e-6), color=colors[k], label=lab)
        I2 = I ** 2
        axes[1][0].plot(z - zf, I2 / I2.max(), color=colors[k], label=lab)
        axes[1][1].semilogy(z - zf, np.maximum(I2 / I2.max(), 1e-8),
                            color=colors[k], label=lab)
        sf = sidelobe_fraction(z, I2)
        fz = fwhm(z, I2)
        axes[1][0].annotate(f"{lab}: FWHM$_z^{{2P}}$={fz:.2f} $\\mu$m, "
                            f"sidelobe {100*sf:.1f}%",
                            xy=(0.02, 0.95 - 0.07 * k), xycoords="axes fraction",
                            fontsize=7, color=colors[k])
    cf03 = closed_form_AB(cfg)
    axes[1][0].annotate(f"sidelobe % is of the plotted {z[-1]-z[0]:.0f} $\\mu$m window;\n"
                        f"plane-interface SA spans {abs(cf03['spherical_um']):.0f} $\\mu$m",
                        xy=(0.02, 0.02), xycoords="axes fraction",
                        fontsize=7, color="0.35")
    for ax, t in ((axes[0][0], "1P  $|E|^2$  linear"), (axes[0][1], "1P  log"),
                  (axes[1][0], "2P  $|E|^4$  linear"), (axes[1][1], "2P  log")):
        ax.set_xlabel("z - z$_{focus}$  [$\\mu$m]")
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("03  PSF$_Z$   axial profile: Bessel needle + sidelobes "
                 "(the out-of-focus background you can measure)", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "03_psf_z.png", dpi=150)
    plt.close(fig)


def fig04_field_sweep(cfg: Config, out: Path) -> None:
    Om_max = max(cfg.Omega_corner, 1e-4)
    Oms = np.linspace(0, Om_max, 25)
    rows = [dov_sine(cfg, o) for o in Oms]
    deg = np.degrees(Oms)

    # wave-optical metrics on a coarser sub-sample (expensive)
    sub = Oms[::6]
    met = [psf_metrics(cfg, o, 0.0, n_lat=101) for o in sub]
    subdeg = np.degrees(sub)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes[0][0].plot(deg, [r["DOV_rad_um"] for r in rows], label="DOV$_{rad}$")
    axes[0][0].plot(deg, [r["DOV_az_um"] for r in rows], label="DOV$_{az}$")
    axes[0][0].set_ylabel("DOV [$\\mu$m]")
    axes[0][0].set_title("Axial scan range vs field")

    axes[0][1].plot(deg, [r["DOV_aniso_um"] for r in rows], color="purple")
    axes[0][1].axhline(0, color="k", lw=0.6)
    axes[0][1].set_ylabel("DOV$_{az}$ - DOV$_{rad}$ [$\\mu$m]")
    axes[0][1].set_title("DOV anisotropy")

    axes[0][2].plot(deg, [r["astig_um"] for r in rows], color="crimson", label="ray")
    ok = [m for m in met if m["astig_resolved"]]
    okd = [d for d, m in zip(subdeg, met) if m["astig_resolved"]]
    bad = [d for d, m in zip(subdeg, met) if not m["astig_resolved"]]
    if ok:
        axes[0][2].plot(okd, [m["astig_wave_um"] for m in ok], "o--", ms=4,
                        color="navy", label="wave (resolved)")
    for d in bad:
        axes[0][2].plot([d], [0], "x", ms=7, color="gray")
    if bad:
        axes[0][2].annotate("x = focal lines buried in the caustic,\n"
                            "     split not measurable at this NA",
                            xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7,
                            color="gray")
    axes[0][2].axhline(0, color="k", lw=0.6)
    axes[0][2].set_ylabel("$d_{az}-d_{rad}$ [$\\mu$m]")
    axes[0][2].set_title("Positional astigmatism  (ray = full aperture)")
    axes[0][2].legend(fontsize=8)

    axes[1][0].plot(subdeg, [m["fwhm_x_2p_um"] for m in met], "-o", ms=4, label="FWHM$_x$")
    axes[1][0].plot(subdeg, [m["fwhm_y_2p_um"] for m in met], "-s", ms=4, label="FWHM$_y$")
    axes[1][0].set_ylabel("2P lateral FWHM [$\\mu$m]")
    axes[1][0].set_title("Lateral resolution vs field")
    axes[1][0].legend(fontsize=8)

    axes[1][1].plot(subdeg, [m["fwhm_z_2p_um"] for m in met], "-o", ms=4, color="teal")
    axes[1][1].set_ylabel("2P axial FWHM [$\\mu$m]")
    axes[1][1].set_title("Axial resolution vs field")

    pk = np.array([m["peak_1p"] for m in met])
    axes[1][2].plot(subdeg, pk / pk[0], "-o", ms=4, color="darkorange",
                    label="peak (Strehl-like)")
    axes[1][2].plot(subdeg, [1 - m["vignette_frac"] for m in met], "-s", ms=4,
                    color="gray", label="pupil throughput")
    axes[1][2].set_ylabel("normalised")
    axes[1][2].set_title("Peak signal & vignetting")
    axes[1][2].legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("object-space field angle $\\Omega$ [deg]")
        ax.grid(alpha=0.3)
        ax.axvline(np.degrees(cfg.Omega_corner), color="r", ls=":", lw=1)
    fig.suptitle("04  Field sweep   red dotted = FOV corner "
                 f"($\\Omega$={np.degrees(cfg.Omega_corner):.2f}$^\\circ$)", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "04_field_sweep.png", dpi=150)
    plt.close(fig)


def _slice_panel(ax, X, Y, dat, thr, title, xlabel, ylabel, cmap="magma"):
    """dat must have shape (len(Y), len(X))."""
    D = np.asarray(dat, dtype=float)
    D = D / max(D.max(), 1e-300)
    im = ax.pcolormesh(X, Y, D, cmap=cmap, shading="auto", vmin=0, vmax=1)
    ax.contour(X, Y, D, levels=[thr], colors="cyan", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def _2p_slice_figure(cfg: Config, out: Path, sl: dict, Om: float,
                     fname: str, title: str, extra_axis=None) -> None:
    thr = cfg.fluor_threshold
    s, z = sl["s"], sl["z"]
    zc = sl.get("z_focus", 0.5 * (z[0] + z[-1]))
    ncol = 4 if extra_axis is not None else 3
    fig, axes = plt.subplots(2, ncol, figsize=(4.1 * ncol, 8))

    # every D below is passed with shape (len(Y), len(X))
    sets = [
        ("XY  (z = focus)", s, s, sl["Ixy"].T, "x [$\\mu$m]", "y [$\\mu$m]"),
        ("XZ  (y = beam axis)", s, z - zc, sl["Ixz"], "x [$\\mu$m]", "z-z$_0$ [$\\mu$m]"),
        ("YZ  (x = 0)", s, z - zc, sl["Iyz"], "y [$\\mu$m]", "z-z$_0$ [$\\mu$m]"),
    ]
    for j, (t, X, Y, D, xl, yl) in enumerate(sets):
        _slice_panel(axes[0][j], X, Y, D, thr, f"{t}\nraw 2P signal", xl, yl)
        Dn = D / max(D.max(), 1e-300)
        Ds = saturate(Dn, cfg.saturation)
        _slice_panel(axes[1][j], X, Y, Ds, thr,
                     f"{t}\nsaturated (s={cfg.saturation})", xl, yl, cmap="viridis")
        frac = 100.0 * (Dn >= thr).mean()
        axes[0][j].annotate(f"above thr: {frac:.2f}% of panel",
                            xy=(0.03, 0.03), xycoords="axes fraction",
                            fontsize=7, color="w")
    if extra_axis is not None:
        extra_axis(axes[0][3], axes[1][3])
    frame = ""
    if abs(np.degrees(Om)) > 1e-6:
        if "scan_slope" in sl:
            frame = (f"   |   frame follows the focus locus (slope "
                     f"{sl['scan_slope']:.4f}); the dose ridge itself leans "
                     f"{sl['dose_slope']:.4f}, so YZ keeps a real residual tilt")
        else:
            frame = (f"   |   lateral origin fixed on the beam, not the chief ray "
                     f"(which would shear these panels by "
                     f"{abs(sl['pupil'].tanO3 * (z[-1] - z[0])):.1f} $\\mu$m)")
    fig.suptitle(f"{title}   $\\Omega$={np.degrees(Om):.2f}$^\\circ$   "
                 f"cyan = fluorescence threshold {thr:.2f}{frame}", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def fig05_2p_instant(cfg: Config, out: Path) -> None:
    Om = cfg.fields()[-1]
    pup = Pupil(cfg, Om)
    z = z_grid(cfg, Om)
    tr = scan_frame(cfg, Om, 1.0, pup)
    Iz = np.abs(axial_line(pup, 1.0, z,
                           y0_um=tr["y0_um"], slope=tr["slope"])) ** 2
    zf = parabolic_peak(z, Iz)
    sl = slices_at(cfg, Om, 1.0, zf)
    sl["Ixy"] = sl["Ixy"] ** 2
    sl["Ixz"] = sl["Ixz"] ** 2
    sl["Iyz"] = sl["Iyz"] ** 2
    sl["z_focus"] = zf

    def extra(ax0, ax1):
        s = sl["s"]
        line = sl["Ixy"][:, sl["Ixy"].shape[1] // 2]
        ax0.plot(s, line / line.max(), label="x cut")
        line2 = sl["Ixy"][sl["Ixy"].shape[0] // 2, :]
        ax0.plot(s, line2 / line2.max(), label="y cut")
        ax0.axhline(cfg.fluor_threshold, color="c", ls=":")
        ax0.set_title("XY cuts through focus", fontsize=9)
        ax0.set_xlabel("[$\\mu$m]"); ax0.legend(fontsize=7); ax0.grid(alpha=0.3)
        Iax = sl["Iyz"].max(axis=1)        # frame-independent: max over the cut
        ax1.plot(sl["z"] - zf, Iax / Iax.max(), color="teal")
        ax1.axhline(cfg.fluor_threshold, color="c", ls=":")
        ax1.set_title("axial cut", fontsize=9)
        ax1.set_xlabel("z-z$_0$ [$\\mu$m]"); ax1.grid(alpha=0.3)

    _2p_slice_figure(cfg, out, sl, Om, "05_2p_instant_slices.png",
                     "05  Instantaneous 2P excitation (TAG at c=+1)", extra)


def fig06_2p_period(cfg: Config, out: Path) -> None:
    Om = cfg.fields()[-1]
    sl = slices_period(cfg, Om)

    def extra(ax0, ax1):
        # where the focus actually goes over one sweep -- the scan line is tilted
        zf, yf = sl["z_focus_of_c"], sl["y_focus_of_c"]
        ax0.plot(zf, yf - yf.mean(), "-o", ms=3, color="darkorange", label="simulated")
        zz = np.linspace(zf.min(), zf.max(), 5)
        ax0.plot(zz, sl["pupil"].tanO3 * (zz - zf.mean()) + (yf - yf.mean()).mean(),
                 "--", lw=1, color="0.4", label="$\\tan\\Omega_3$ (chief ray)")
        ax0.set_title(f"TAG scan is tilted: walks {sl['walk_um']:.2f} $\\mu$m sideways\n"
                      f"focus slope {sl['scan_slope']:.4f} vs "
                      f"$\\tan\\Omega_3$ {sl['pupil'].tanO3:.4f};  "
                      f"dose ridge {sl['dose_slope']:.4f}", fontsize=9)
        ax0.set_xlabel("focus depth z [$\\mu$m]")
        ax0.set_ylabel("lateral y - $\\langle y\\rangle$ [$\\mu$m]")
        ax0.legend(fontsize=7); ax0.grid(alpha=0.3)
        # dose vs depth taken as the max over the transverse cut, so it does not
        # depend on where the lateral origin was put
        z = sl["z"]
        prof = sl["Iyz"].max(axis=1)
        ax1.plot(z - 0.5 * (z[0] + z[-1]), prof / prof.max(), color="teal")
        ax1.axhline(cfg.fluor_threshold, color="c", ls=":")
        ax1.set_title("period-integrated axial dose\n(max over the transverse cut)",
                      fontsize=9)
        ax1.set_xlabel("z-z$_0$ [$\\mu$m]"); ax1.grid(alpha=0.3)

    _2p_slice_figure(cfg, out, sl, Om, "06_2p_period_slices.png",
                     "06  TAG-period-integrated 2P excitation (dose)", extra)


def fig07_pupil(cfg: Config, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    Om_c = cfg.Omega_corner
    p0 = Pupil(cfg, 0.0)
    pc = Pupil(cfg, Om_c)

    kmax = p0.kax.max()
    axes[0][0].imshow(np.abs(p0.amp).T, origin="lower", cmap="viridis",
                      extent=[-kmax, kmax, -kmax, kmax])
    axes[0][0].set_title("pupil amplitude (on axis)\nGaussian $\\times$ aplanatic", fontsize=9)

    # Sequential map, NOT a cyclic one: this is an absolute optical phase in rad
    # (here 7.6 -> 23.0), and J0 is monotonic across the pupil. A cyclic map
    # (twilight) returns to dark at both ends and paints false rings on it.
    ph = np.where(p0.mask, cfg.phi_amp * p0.tag_profile, np.nan)
    im = axes[0][1].imshow(ph.T, origin="lower", cmap="viridis",
                           extent=[-kmax, kmax, -kmax, kmax])
    plt.colorbar(im, ax=axes[0][1], fraction=0.046).set_label("rad", fontsize=7)
    axes[0][1].set_title(f"TAG phase at c=+1\n$\\varphi_{{amp}}$={cfg.phi_amp:.1f} rad", fontsize=9)

    # radial cut: J0 vs parabola
    r = np.linspace(0, cfg.R_mm, 400)
    rt = r / cfg.M_net
    x = cfg.k_r * rt
    axes[0][2].plot(r, j0(x), label="$J_0(k_r r)$ (true)")
    axes[0][2].plot(r, 1 - x ** 2 / 4, "--", label="parabolic $1-x^2/4$")
    axes[0][2].axhline(0, color="k", lw=0.6)
    for eps, st in ((0.05, ":"), (0.10, "-.")):
        rp = cfg.parabolic_radius_mm(eps) * cfg.M_net
        if rp < cfg.R_mm:
            axes[0][2].axvline(rp, color="gray", ls=st, lw=1,
                               label=f"{int(eps*100)}% parabolic limit")
    rz = cfg.r_first_zero_mm * cfg.M_net
    if rz < cfg.R_mm * 1.5:
        axes[0][2].axvline(rz, color="crimson", lw=1, label="$J_0$ first zero")
    w = cfg.beam_radius_mm
    axes[0][2].axvline(w, color="green", lw=1.2, label="beam $1/e^2$ radius")
    axes[0][2].set_xlabel("pupil radius [mm]")
    axes[0][2].set_title("TAG profile vs parabolic region\n(Gaussian preservation)", fontsize=9)
    axes[0][2].legend(fontsize=6.5)
    axes[0][2].grid(alpha=0.3)

    axes[1][0].imshow(np.abs(pc.amp).T, origin="lower", cmap="viridis",
                      extent=[-kmax, kmax, -kmax, kmax])
    axes[1][0].set_title(f"pupil amplitude at $\\Omega$={np.degrees(Om_c):.2f}$^\\circ$\n"
                         f"vignetted {100*pc.vignette_frac:.1f}%", fontsize=9)

    # k_z curvature anisotropy -> the origin of astigmatism
    n3k0 = cfg.n3 * cfg.k0
    kt = np.linspace(0, min(kmax, 0.98 * n3k0), 300)
    axes[1][1].plot(kt, np.sqrt(np.maximum(n3k0 ** 2 - kt ** 2, 0)))
    axes[1][1].set_xlabel("$k_t$ [rad/$\\mu$m]")
    axes[1][1].set_ylabel("$k_{z3}$")
    axes[1][1].set_title("$k_z(k_t)$ is nonlinear\n$\\Rightarrow$ tilting the pupil "
                         "makes $x/y$ curvature differ = astigmatism", fontsize=9)
    axes[1][1].grid(alpha=0.3)

    rows = []
    for lbl, eps in (("1%", 0.01), ("5%", 0.05), ("10%", 0.10)):
        rp = cfg.parabolic_radius_mm(eps)
        rows.append([lbl, f"{rp:.3f}", f"{2*rp:.3f}", f"{2*rp*cfg.M_net:.3f}"])
    axes[1][2].axis("off")
    tbl = axes[1][2].table(cellText=rows,
                           colLabels=["err", "$r_p$ TAG [mm]", "$2r_p$ [mm]",
                                      "$2r_p$ at pupil [mm]"],
                           loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    axes[1][2].set_title(
        f"parabolic region  ($k_r$={cfg.k_r:.3f} rad/mm, "
        f"$f_{{TAG}}$={cfg.tag_frequency_hz/1e3:.0f} kHz)\n"
        f"beam at TAG plane: $w$={cfg.beam_radius_mm/cfg.M_net:.3f} mm, "
        f"$k_r w$={cfg.k_r*cfg.beam_radius_mm/cfg.M_net:.2f}", fontsize=9)

    kzoom = 1.35 * cfg.n1 * cfg.k0 * np.sin(cfg.theta10 + abs(Om_c))
    for ax in (axes[0][0], axes[0][1], axes[1][0]):
        ax.set_xlabel("$k_x$ [rad/$\\mu$m]", fontsize=8)
        ax.set_ylabel("$k_y$ [rad/$\\mu$m]", fontsize=8)
        ax.set_xlim(-kzoom, kzoom)
        ax.set_ylim(-kzoom, kzoom)
        ax.tick_params(labelsize=7)
    fig.suptitle("07  Pupil diagnostics", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "07_pupil_diagnostics.png", dpi=150)
    plt.close(fig)


def fig08_metrics(cfg: Config, out: Path) -> None:
    fields = cfg.fields()
    cs = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    keys = ["fwhm_x_2p_um", "fwhm_y_2p_um", "fwhm_z_2p_um", "peak_1p", "sidelobe_frac"]
    data = {k: np.zeros((len(fields), cs.size)) for k in keys}
    veff = np.zeros((len(fields), cs.size))
    for i, Om in enumerate(fields):
        for j, c in enumerate(cs):
            m = psf_metrics(cfg, Om, c, n_lat=101)
            for k in keys:
                data[k][i, j] = m[k]
            veff[i, j] = effective_volume(cfg, Om, c, n_lat=49, n_zv=61)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [
        ("fwhm_x_2p_um", "2P FWHM$_x$ [$\\mu$m]"),
        ("fwhm_y_2p_um", "2P FWHM$_y$ [$\\mu$m]"),
        ("fwhm_z_2p_um", "2P FWHM$_z$ [$\\mu$m]"),
        ("peak_1p", "peak intensity (a.u.)"),
        ("sidelobe_frac", "axial sidelobe fraction"),
    ]
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(fields)))
    for a, (k, lbl) in zip(axes.ravel(), panels):
        for i, Om in enumerate(fields):
            y = data[k][i]
            if k == "peak_1p":
                y = y / data[k][0, 2]
            a.plot(cs, y, "-o", ms=4, color=colors[i],
                   label=f"$\\Omega$={np.degrees(fields[i]):.2f}$^\\circ$")
        a.set_xlabel("TAG phase factor  $\\cos\\omega t$")
        a.set_ylabel(lbl)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
        if k == "sidelobe_frac":
            # This metric is discontinuous by construction: it splits the profile
            # at the first minima either side of the GLOBAL max, so when two lobes
            # swap dominance the "main lobe" changes identity and the number jumps.
            # The jumps are real multi-lobe behaviour, not noise, but the metric is
            # a poor summary of it -- read figure 03 for the actual profiles.
            a.annotate("jumps = the two axial lobes swapping dominance\n"
                       "(metric re-identifies the 'main' lobe; see fig 03).\n"
                       "Fraction is of the plotted scan window, not of the\n"
                       f"{abs(closed_form_AB(cfg)['spherical_um']):.0f} $\\mu$m "
                       "spherical-aberration caustic.",
                       xy=(0.03, 0.03), xycoords="axes fraction",
                       fontsize=6.5, color="0.35")
    a = axes.ravel()[5]
    for i in range(len(fields)):
        a.plot(cs, veff[i], "-o", ms=4, color=colors[i],
               label=f"$\\Omega$={np.degrees(fields[i]):.2f}$^\\circ$")
    a.set_xlabel("TAG phase factor  $\\cos\\omega t$")
    a.set_ylabel("$V_{eff}$ [$\\mu$m$^3$]")
    a.set_title("2P effective volume (FCS-measurable)", fontsize=9)
    a.grid(alpha=0.3)
    a.legend(fontsize=7)
    fig.suptitle("08  Measurable PSF metrics vs TAG phase and field angle", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out / "08_psf_metrics.png", dpi=150)
    plt.close(fig)


def fig09_focal_surface(cfg: Config, out: Path) -> None:
    wx_max, wy_max = cfg.omega_pair
    n = 21
    wx = np.linspace(-wx_max, wx_max, n)
    wy = np.linspace(-wy_max, wy_max, n)
    Zr = np.zeros((n, n))
    Za = np.zeros((n, n))
    for i, a in enumerate(wx):
        for j, b in enumerate(wy):
            Om = np.arccos(np.clip(np.cos(a) * np.cos(b), -1, 1))
            g = depths_at_theta(cfg, Om, cfg.theta10)
            Zr[i, j] = g["depth_rad_um"]
            Za[i, j] = g["depth_az_um"]
    X, Y = np.meshgrid(np.degrees(wx), np.degrees(wy), indexing="ij")

    fig = plt.figure(figsize=(16, 5.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_surface(X, Y, Zr, cmap="viridis", alpha=0.9)
    ax.set_title("radial focal surface $z_{rad}$", fontsize=9)
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.plot_surface(X, Y, Za, cmap="plasma", alpha=0.9)
    ax.set_title("azimuthal focal surface $z_{az}$", fontsize=9)
    for a in fig.axes:
        a.set_xlabel("$\\omega_x$ [deg]", fontsize=8)
        a.set_ylabel("$\\omega_y$ [deg]", fontsize=8)
        a.set_zlabel("depth [$\\mu$m]", fontsize=8)
    ax = fig.add_subplot(1, 3, 3)
    im = ax.pcolormesh(X, Y, Za - Zr, cmap="coolwarm", shading="auto")
    plt.colorbar(im, ax=ax).set_label("$z_{az}-z_{rad}$ [$\\mu$m]", fontsize=8)
    ax.set_title("positional astigmatism map", fontsize=9)
    ax.set_xlabel("$\\omega_x$ [deg]"); ax.set_ylabel("$\\omega_y$ [deg]")
    fig.suptitle("09  Focal surfaces: field curvature dome + astigmatism split "
                 "(measurable on a flat sample)", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / "09_focal_surface.png", dpi=150)
    plt.close(fig)


def fig10_validation(cfg: Config, out: Path, res: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # ---- V1: Airy check (uniform pupil amplitude)
    c1 = _variant(cfg, n2=cfg.n1, n3=cfg.n1)
    pup = Pupil(c1, 0.0, tag_on=False, uniform_amp=True)
    zf = res["V1"]["z_found_um"]
    s, E = transverse_line(pup, 0.0, zf, "x", 4.0, 801)
    I = np.abs(E) ** 2
    axes[0].plot(s, I / I.max(), label="sim")
    axes[0].axvline(res["V1"]["fwhm_airy_um"] / 2, color="r", ls=":", label="Airy FWHM/2")
    axes[0].axvline(-res["V1"]["fwhm_airy_um"] / 2, color="r", ls=":")
    axes[0].set_title(f"V1 uniform medium, no TAG\nFWHM {res['V1']['fwhm_um']:.4f} vs "
                      f"Airy {res['V1']['fwhm_airy_um']:.4f} $\\mu$m "
                      f"({res['V1']['fwhm_err_pct']:+.2f}%)", fontsize=9)
    axes[0].set_xlabel("x [$\\mu$m]"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # ---- V2/V2b: the fold comparison
    v2, v2b = res["V2"], res["V2b"]
    names = ["wave\n(Gauss)", "wave\n(uniform)", "md\nclosed form", "md$\\times\\cos^2$",
             "paraxial"]
    vals = [v2b["dz_wave_gauss_um"], v2b["dz_wave_uniform_um"], v2b["dz_md_um"],
            v2b["dz_md_cos2_um"], v2b["dz_paraxial_um"]]
    cols = ["navy", "steelblue", "crimson", "orange", "gray"]
    axes[1].bar(range(5), vals, color=cols)
    for i, v in enumerate(vals):
        axes[1].annotate(f"{v:.3f}", (i, v), ha="center",
                         va="top" if v < 0 else "bottom", fontsize=7)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(names, fontsize=7)
    axes[1].set_ylabel("$\\Delta z$ per unit $V_T$  [$\\mu$m/(1/m)]")
    axes[1].set_title(f"V2b finite-NA fold, uniform medium (NA={v2b['NA']:.3f})\n"
                      f"md over-predicts by {v2b['md_over_pct']:+.1f}%   |   "
                      f"V2 paraxial check {v2['err_pct']:+.3f}%", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    # ---- V3: J0 vs parabolic axial profiles (uniform medium, as in run_validation)
    c3 = _variant(cfg, n2=cfg.n1, n3=cfg.n1, L1_mm=0.0, L2_mm=0.0)
    lo, hi = wave_scan_window(c3, 0.0)
    pad = 0.6 * (hi - lo) + 10.0
    z = np.linspace(lo - pad, hi + pad, 1201)
    for parab, style, lbl in ((True, "--", "parabolic"), (False, "-", "full $J_0$")):
        p = Pupil(c3, 0.0, parabolic_tag=parab)
        for c, col in ((0.0, "k"), (1.0, "crimson")):
            I = np.abs(axial_line(p, c, z)) ** 2
            axes[2].plot(z, I / I.max(), style, color=col, lw=1.1,
                         label=f"{lbl}, c={c:.0f}")
    v3 = res["V3"]
    axes[2].set_title(f"V3 Bessel shortening (uniform medium)\n"
                      f"$\\Delta z(J_0)$={v3['dz_J0_um']:+.3f} vs "
                      f"parabolic {v3['dz_parabolic_um']:+.3f}\n"
                      f"ratio {v3['ratio_vs_parabolic']:.3f} (vs parabolic), "
                      f"{v3['ratio_vs_paraxial']:.3f} (vs paraxial)", fontsize=9)
    axes[2].set_xlabel("z [$\\mu$m]"); axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

    # ---- V4: astigmatism, wave vs ray
    v4 = res["V4"]
    c4 = v4["cfg"]
    m = psf_metrics(c4, np.radians(v4["Omega_deg"]), 0.0, n_lat=121)
    axes[3].plot(m["z"], m["wx_of_z"], "-o", ms=3, label="FWHM$_x$(z)")
    axes[3].plot(m["z"], m["wy_of_z"], "-s", ms=3, label="FWHM$_y$(z)")
    axes[3].axvline(m["z_best_x_um"], color="C0", ls=":")
    axes[3].axvline(m["z_best_y_um"], color="C1", ls=":")
    axes[3].set_title(f"V4 astigmatism, NA={v4['NA']:.2f}, "
                      f"$\\Omega$={v4['Omega_deg']:.2f}$^\\circ$\n"
                      f"wave {v4['astig_wave_um']:+.2f} vs ray {v4['astig_ray_um']:+.2f} $\\mu$m",
                      fontsize=9)
    axes[3].set_xlabel("z [$\\mu$m]"); axes[3].set_ylabel("lateral FWHM [$\\mu$m]")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    fig.suptitle("10  Validation: what the wave model must reproduce, "
                 "and where it disagrees with the ray closed form", fontsize=11)
    _stamp(fig, cfg)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out / "10_validation.png", dpi=150)
    plt.close(fig)


# =============================================================================
# main
# =============================================================================
def summary_text(cfg: Config, res: dict) -> str:
    cf = closed_form_AB(cfg)
    ct = closed_form_AB_tangent(cfg)
    g0 = dov_sine(cfg, 0.0)
    gc = dov_sine(cfg, cfg.Omega_corner)
    L = [
        "TAG unified pupil simulation -- summary",
        "=" * 60,
        f"preset            : {cfg.name}",
        f"NA (sine)         : {cfg.NA:.4f}   theta10 = {np.degrees(cfg.theta10):.3f} deg",
        f"NA (tangent, old) : {ct['NA']:.4f}   theta10 = {ct['theta10_deg']:.3f} deg",
        f"wavelength        : {cfg.wavelength_um} um",
        f"TAG               : f={cfg.tag_frequency_hz/1e3:.1f} kHz  v={cfg.sound_speed_m_s} m/s",
        f"                    k_r={cfg.k_r:.4f} rad/mm  phi_amp={cfg.phi_amp:.2f} rad",
        f"                    J0 first zero at r={cfg.r_first_zero_mm:.3f} mm (TAG plane)",
        f"                    beam at TAG plane w={cfg.beam_radius_mm/cfg.M_net:.3f} mm"
        f"  ->  k_r*w = {cfg.k_r*cfg.beam_radius_mm/cfg.M_net:.3f}",
        f"grid              : {cfg.n_grid} pts, pupil radius {cfg.pupil_radius_px} px",
        "",
        "closed forms (sine condition)",
        f"  A  = {cf['A']:.6f}   B1 = {cf['B1']:.6e}   B2 = {cf['B2']:.6e}",
        f"  marginal-ray depth d0 = {cf['d0_um']:.3f} um",
        f"  paraxial       depth  = {cf['d_paraxial_um']:.3f} um"
        f"   -> plane-interface SA = {cf['spherical_um']:+.1f} um",
        f"  dz per unit V_T  md {cf['dz_per_VT_um']:.5f}   "
        f"md*cos^2 {cf['dz_per_VT_wave_um']:.5f}   paraxial "
        f"{cf['dz_per_VT_paraxial_um']:.5f} um/(1/m)",
        f"  (tangent condition for reference: {ct['dz_per_VT_um']:.5f}, "
        f"{res['sine_vs_tangent']['diff_pct']:+.2f}%)",
        "",
        f"ray-model DOV   [fold_mode = {cfg.fold_mode}]",
        f"  on axis        DOV_rad={g0['DOV_rad_um']:.2f}  DOV_az={g0['DOV_az_um']:.2f} um",
        f"  corner Om={np.degrees(cfg.Omega_corner):.2f}d  DOV_rad={gc['DOV_rad_um']:.2f}  "
        f"DOV_az={gc['DOV_az_um']:.2f}  astig={gc['astig_um']:+.2f} um",
        "",
        "validation",
        f"  V0  fast-transform field err {res['V0']['max_rel_field_err']:.1e}, "
        f"focus shift {res['V0']['max_peak_shift_um']:.1e} um",
        "  V6  FWHMz chief-ray vs beam-track: "
        + "  ".join(f"{v['Omega_deg']:.1f}d {v['fwhm_z_chief_um']:.1f}/"
                    f"{v['fwhm_z_track_um']:.1f}" for v in res["V6"]),
        f"  V1  focus dz={res['V1']['dz_um']:+.4f} um, FWHM err "
        f"{res['V1']['fwhm_err_pct']:+.2f}%",
        f"  V2  paraxial limit wave vs closed: {res['V2']['err_pct']:+.3f}%",
        f"  V2b finite NA: md over-predicts scan range by "
        f"{res['V2b']['md_over_pct']:+.1f}%",
        f"  V3  Bessel ratio vs parabolic {res['V3']['ratio_vs_parabolic']:.4f}"
        f"  /  vs paraxial {res['V3']['ratio_vs_paraxial']:.4f}",
        f"  V4  astig wave {res['V4']['astig_wave_um']:+.3f} vs ray "
        f"{res['V4']['astig_ray_um']:+.3f} um",
        f"  V5  wave-peak depth {res['V5']['d_wave_peak_um']:.2f} um "
        f"(marginal {res['V5']['d_marginal_um']:.2f}, "
        f"paraxial {res['V5']['d_paraxial_um']:.2f})",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="current", choices=sorted(PRESETS))
    ap.add_argument("--figures", default="all",
                    help="comma list of figure ids, e.g. 01,03,05  (default all)")
    ap.add_argument("--fast", action="store_true", help="coarse grid for a quick look")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    if args.fast:
        cfg = Config(**{**asdict(cfg), "n_grid": None, "pupil_radius_px": 64,
                        "n_z": 81, "n_time": 13, "auto_sampling": False})
        print("[--fast] sampling guard disabled: axial profiles may show numerical\n"
              "         ripple. Use it to check plumbing, not to read physics off.")
    check_geometry(cfg)
    cfg = apply_sampling_guard(cfg)
    out = Path(args.outdir) if args.outdir else SCRIPT_DIR / f"unified_out_{cfg.name}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = run_validation(cfg)
    (out / "summary.txt").write_text(summary_text(cfg, res), encoding="utf-8")
    (out / "config.json").write_text(
        json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items()}, indent=2), encoding="utf-8")
    print(f"\n[validation done in {time.time()-t0:.1f}s]  -> {out}")
    if args.validate_only:
        return

    want = set(FIGURES) if args.figures == "all" else {
        s.strip().zfill(2) for s in args.figures.split(",") if s.strip()}
    jobs = [
        ("01", fig01_scan_trajectory), ("02", fig02_psf_xy), ("03", fig03_psf_z),
        ("04", fig04_field_sweep), ("05", fig05_2p_instant), ("06", fig06_2p_period),
        ("07", fig07_pupil), ("08", fig08_metrics), ("09", fig09_focal_surface),
    ]
    for key, fn in jobs:
        if key not in want:
            continue
        t = time.time()
        print(f"  [{key}] {FIGURES[key]} ...", end="", flush=True)
        fn(cfg, out)
        print(f" {time.time()-t:.1f}s")
    if "10" in want:
        t = time.time()
        print("  [10] validation ...", end="", flush=True)
        fig10_validation(cfg, out, res)
        print(f" {time.time()-t:.1f}s")
    print(f"\nall done in {time.time()-t0:.1f}s  ->  {out}")


if __name__ == "__main__":
    main()
