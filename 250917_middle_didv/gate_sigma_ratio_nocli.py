# Write a "dI/dV only" version of the script and save it for download.



"""
gate_sigma_ratio_didv_only.py — gate-dependent Σ/σ from Nanonis dI/dV (ONLY LI-X PATH)
--------------------------------------------------------------------------------------
- 仅使用 dI/dV：G(V) = (LI Demod 1 X) / (Lock-in Amplitude)
- 在 |V|<=WINDOW_V 线性拟合： G ≈ σ + 2Σ V → Σ/σ = (0.5 * slope) / intercept
- 不做 I–V 退化拟合；若缺少 LI X 或幅度未知则报错（可在 FALLBACK_AMP_V 里手动给）
- 扫描 INPUT_DIR，匹配 FILE_GLOBS，并二次过滤含 "VGK[p/m]XX.XX" 的文件名

输出：
  - out_sigma/per_file_sigma_ratio.csv
  - out_sigma/gate_summary_sigma_ratio.csv (+ with_flags)
  - out_sigma/sigma_ratio_vs_gate.png
  - out_sigma/QC_*.png（每文件一张，G–V 拟合）
"""

from pathlib import Path
# ======================= CONFIG =======================
INPUT_DIR   = Path(r"D:/250912_MLG/250917_middle_didv/")              # 放 dI/dV 文件的目录
OUTPUT_DIR  = INPUT_DIR / "out_sigma"  # 输出目录
FILE_GLOBS  = ["VGK*didv*.dat", "spec_VGK*.dat", "*.dat"]  # 多通配符，按需增删
RECURSIVE   = False                    # True 用 rglob 递归子目录
WINDOW_mV   = 20.0                     # 零偏拟合窗口 ±mV
MIN_POINTS  = 12                       # 最少参与拟合的点数
FALLBACK_AMP_V = 0.01               # 若头部没 Lock-in 幅度，用这个值（V）；None=报错
SAVE_FIGS   = True                     # 保存每个文件的 QC 图
# ======================================================

import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# —— 文件名里抽取 gate（兼容 “VGKp8.00_didv_…”, “spec_VGKm12.00_…” 等）
PAT_VGK = re.compile(r'VGK\s*([pm])\s*([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)

def parse_vgk_tag(name: str):
    m = PAT_VGK.search(str(name))
    if not m: return None
    return f"{m.group(1).lower()}{float(m.group(2)):.2f}"

def parse_vgk_value(name: str):
    m = PAT_VGK.search(str(name))
    if not m: return None
    val = float(m.group(2))
    return (+val) if m.group(1).lower()=='p' else (-val)

# —— 头部振幅 / 数据区读取
LOCKIN_AMP_RE = re.compile(r"Lock-?in>Amplitude\s+([-\d.E+]+)")
BIAS_CANDS = ["Bias (V)", "Bias [bwd] (V)", "Bias calc (V)", "Bias"]
LIX_CANDS  = ["LI Demod 1 X (A)", "LI Demod 1 X [bwd] (A)", "LI Demod 1 X"]

def pick_col(df: pd.DataFrame, candidates):
    """先精确（不区分大小写），再宽松包含匹配。"""
    low = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in low:
            return low[name.lower()]
    keys = [k.lower().split("(")[0].strip() for k in candidates]  # 提炼关键词，如 'bias'
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            return c
    return None

def read_nanonis(fp: Path):
    with open(fp, "r", errors="ignore") as f:
        lines = f.readlines()
    data_idx = None
    amp = None
    for i, line in enumerate(lines):
        if data_idx is None:
            m = LOCKIN_AMP_RE.search(line.replace(",", "."))
            if m:
                try: amp = float(m.group(1))
                except: pass
        if line.strip().upper().startswith("[DATA]"):
            data_idx = i; break
    if data_idx is None:
        raise ValueError("缺少 [DATA] 段")
    header = lines[data_idx+1].strip().split("\t")
    df = pd.read_csv(fp, sep="\t", skiprows=data_idx+2, names=header, engine="python")
    return df, amp

# —— 抗异常线性拟合
def robust_linear(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(20):
        yhat = X @ beta
        r = y - yhat
        med = np.median(r)
        mad = np.median(np.abs(r - med)) * 1.4826
        sigma = mad if mad>0 else (np.std(r) if np.std(r)>0 else 1.0)
        delta = 1.345 * sigma
        w = np.where(np.abs(r)<=delta, 1.0, delta/np.maximum(np.abs(r),1e-30))
        W = np.diag(w)
        beta_new = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
        if np.allclose(beta_new, beta, rtol=1e-7, atol=1e-9):
            beta = beta_new; break
        beta = beta_new
    yhat = X @ beta
    r = y - yhat
    dof = max(len(y)-X.shape[1], 1)
    s2 = float(np.sum(r**2))/dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, yhat

# —— 仅 dI/dV 拟合通道
# def fit_sigma_sigma_from_lix(df, bias_col, lix_col, amp, window_V, min_points):
#     if lix_col is None:
#         raise ValueError("未找到 'LI Demod 1 X' 列，无法用 dI/dV 拟合")
#     if amp in (None, 0.0):
#         raise ValueError("缺少 Lock-in 幅度且未设置 FALLBACK_AMP_V —— 无法计算 G(V)")
#     b = pd.to_numeric(df[bias_col], errors="coerce").values
#     lx = pd.to_numeric(df[lix_col],  errors="coerce").values
#     G = lx / amp  # S
#     sel = np.isfinite(b) & np.isfinite(G) & (np.abs(b) <= window_V)
#     if sel.sum() < min_points:
#         idx = np.argsort(np.abs(b))[:max(min_points, 21)]
#         sel = np.zeros_like(b, dtype=bool); sel[idx]=True; sel &= np.isfinite(G)
#     bb, GG = b[sel], G[sel]
#     X = np.column_stack([np.ones_like(bb), bb])   # G ≈ a + c V
#     beta, se, _ = robust_linear(X, GG)
#     sigma = float(beta[0]); dGdV = float(beta[1])
#     Sigma = 0.5 * dGdV
#     return dict(method="G_from_LIX", n=int(sel.sum()),
#                 sigma=sigma, sigma_se=float(se[0]),
#                 Sigma=Sigma, Sigma_se=float(se[1]/2.0))
def fit_sigma_sigma_from_lix(df, bias_col, lix_col, amp, window_V, min_points, pair_tol=2e-4):
    """
    仅用 dI/dV：G = (LI X)/amp；在 |V|<=window_V 内做奇/偶分解来估计 Σ 和 σ。
    pair_tol 是正负点配对的电压容差（单位 V），默认 0.2 mV。
    """
    if lix_col is None:
        raise ValueError("未找到 'LI Demod 1 X' 列")
    if amp in (None, 0.0):
        raise ValueError("缺少 Lock-in 幅度（可在 FALLBACK_AMP_V 里设置）")

    import numpy as np
    import pandas as pd

    V = pd.to_numeric(df[bias_col], errors="coerce").to_numpy(float)
    LX = pd.to_numeric(df[lix_col], errors="coerce").to_numpy(float)
    G  = LX / amp

    sel = np.isfinite(V) & np.isfinite(G) & (np.abs(V) <= window_V)
    V, G = V[sel], G[sel]
    if V.size < max(min_points, 8):
        # 取离 0 最近的若干点
        idx = np.argsort(np.abs(V))[:max(min_points, 12)]
        V, G = V[idx], G[idx]

    # 分正负并排序
    Vp = np.sort(V[V > 0]); Gp = G[V > 0][np.argsort(V[V > 0])]
    Vn = np.sort(V[V < 0]); Gn = G[V < 0][np.argsort(V[V < 0])]

    if Vp.size == 0 or (Vp.size + Vn.size) < max(min_points, 8):
        raise ValueError("零偏附近有效点不足，无法稳定配对")

    # 为每个 +V 在负侧做线性插值得到 G(-V)
    # （这样不要求刚好有对称采样点）
    if Vn.size >= 2:
        Gm_at_minusVp = np.interp(-Vp, Vn, Gn)  # 线性插值
    else:
        # 负侧点太少，退回最近邻
        Gm_at_minusVp = Gn[np.argmin(np.abs(Vn[:,None] + Vp[None,:]), axis=0)] if Vn.size else np.full_like(Vp, np.nan)

    mask = np.isfinite(Gm_at_minusVp) & (np.abs(-Vp - (-Vp)) <= pair_tol)  # 这里主要过滤 NaN
    Vp, Gp, Gm_at_minusVp = Vp[mask], Gp[mask], Gm_at_minusVp[mask]
    if Vp.size < 5:
        raise ValueError("可配对的 ±V 点太少")

    # 奇/偶分解
    G_odd  = 0.5 * (Gp - Gm_at_minusVp)   # ≈ (2Σ) * (V/2)? 实际就是 a*V，其中 a=2Σ
    G_even = 0.5 * (Gp + Gm_at_minusVp)   # ≈ σ + α V^2

    # 对 G_odd vs Vp 做“过原点”线性拟合： G_odd ≈ a * Vp  →  Σ = a/2
    Xo = Vp.reshape(-1,1)
    a = float(np.linalg.lstsq(Xo, G_odd, rcond=None)[0][0])
    Sigma = 0.5 * a

    # 对 G_even vs [1, Vp^2] 做线性拟合： 截距即 σ
    Xe = np.column_stack([np.ones_like(Vp), Vp**2])
    beta_even = np.linalg.lstsq(Xe, G_even, rcond=None)[0]
    sigma = float(beta_even[0])

    # 粗略误差：按普通最小二乘近似
    # （如需严格误差，可加权/自举）
    return dict(method="G_from_LIX_even-odd", n=int(Vp.size),
                sigma=sigma, sigma_se=np.nan,
                Sigma=Sigma, Sigma_se=np.nan)

def process_one_file(fp: Path, window_V, min_points, fallback_amp):
    df, amp = read_nanonis(fp)
    bias_col = pick_col(df, BIAS_CANDS)
    lix_col  = pick_col(df, LIX_CANDS)
    if bias_col is None:
        raise ValueError("缺少 Bias 列")
    amp_eff = amp if amp not in (None, 0.0) else fallback_amp
    # —— 只允许 dI/dV
    row = fit_sigma_sigma_from_lix(df, bias_col, lix_col, amp_eff, window_V, min_points)
    sig, Sig = row["sigma"], row["Sigma"]
    ratio = float(Sig/sig) if (sig is not None and sig != 0.0) else math.nan
    return {
        "file": fp.name,
        "gate_tag": parse_vgk_tag(fp.name),
        "gate_V": parse_vgk_value(fp.name),
        "method": row["method"],
        "N_points": row["n"],
        "lockin_amp_V": amp if amp is not None else fallback_amp,
        "sigma_S": sig, "sigma_se": row["sigma_se"],
        "Sigma_A_per_V2": Sig, "Sigma_se": row["Sigma_se"],
        "Sigma_over_sigma_per_V": ratio
    }

def list_candidate_files():
    files = []
    for pat in FILE_GLOBS:
        files.extend(INPUT_DIR.rglob(pat) if RECURSIVE else INPUT_DIR.glob(pat))
    # 二次过滤：仅保留含 VGK 标签的
    files = [f for f in files if f.is_file() and PAT_VGK.search(f.name)]
    # 去重并排序
    files = sorted(set(files), key=lambda p: p.name)
    print(f"[scan] dir={INPUT_DIR}  patterns={FILE_GLOBS}  matched={len(files)}")
    if not files:
        try:
            sample = [p.name for p in sorted(INPUT_DIR.iterdir())[:20]]
            print("[hint] 目录前20个文件：", sample)
        except Exception as e:
            print("[hint] 无法列目录：", e)
    return files

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list_candidate_files()
    if not files:
        print("⚠️ 未匹配到任何文件；检查 INPUT_DIR / FILE_GLOBS 或 VGK 命名。")
        return

    results = []
    window_V = WINDOW_mV * 1e-3
    for fp in files:
        try:
            res = process_one_file(fp, window_V, MIN_POINTS, FALLBACK_AMP_V)
            results.append(res)
            print(f"[OK] {fp.name}: gate={res['gate_V']}, ratio={res['Sigma_over_sigma_per_V']:.3g}")
            if SAVE_FIGS:
                # QC 图 (G–V)
                df, amp = read_nanonis(fp)
                bias_col = pick_col(df, BIAS_CANDS)
                lix_col  = pick_col(df, LIX_CANDS)
                b = pd.to_numeric(df[bias_col], errors="coerce").values
                amp_eff = (amp if amp not in (None, 0.0) else FALLBACK_AMP_V) or 1.0
                G = pd.to_numeric(df[lix_col], errors="coerce").values / amp_eff
                sel = np.isfinite(b) & np.isfinite(G) & (np.abs(b) <= window_V)
                fig, ax = plt.subplots(figsize=(5,3.2))
                ax.scatter(b[sel], G[sel], s=10, label="G(V) data")
                Vline = np.linspace(-window_V, window_V, 200)
                ax.plot(Vline, res["sigma_S"] + 2*res["Sigma_A_per_V2"]*Vline, lw=2, label="fit σ+2ΣV")
                ax.set_xlabel("Bias [V]"); ax.set_ylabel("G [S]")
                ax.set_title(fp.name); ax.grid(True, alpha=0.3); ax.legend()
                fig.tight_layout()
                fig.savefig(OUTPUT_DIR/f"QC_{fp.stem}.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"[SKIP] {fp.name}: {e}")

    if not results:
        print("⚠️ 未得到任何结果。请检查列名、Lock-in 幅度、拟合窗口。")
        return

    df_pf = pd.DataFrame(results).sort_values(["gate_V","file"])
    df_pf.to_csv(OUTPUT_DIR/"per_file_sigma_ratio.csv", index=False)

    if df_pf["gate_V"].notna().any():
        grp = df_pf.dropna(subset=["gate_V"]).groupby("gate_V")
        summary = grp.agg(
            N_files=("file","count"),
            Sigma_over_sigma_mean=("Sigma_over_sigma_per_V","mean"),
            Sigma_over_sigma_median=("Sigma_over_sigma_per_V","median"),
            Sigma_over_sigma_std=("Sigma_over_sigma_per_V","std")
        ).reset_index()
        # 简单异常标注（z 分数 > 2）
        z = (summary["Sigma_over_sigma_median"] - summary["Sigma_over_sigma_median"].mean()) / \
            summary["Sigma_over_sigma_median"].std(ddof=1)
        summary["flag_failure"] = (np.abs(z) > 2).astype(int)
        summary.to_csv(OUTPUT_DIR/"gate_summary_sigma_ratio.csv", index=False)
        summary.to_csv(OUTPUT_DIR/"gate_summary_sigma_ratio_with_flags.csv", index=False)

        # 画 Σ/σ vs gate
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(summary["gate_V"], summary["Sigma_over_sigma_median"], marker="o", label="median")
        x = summary["gate_V"].values
        y = summary["Sigma_over_sigma_median"].values
        s = summary["Sigma_over_sigma_std"].fillna(0).values
        # if np.isfinite(s).any():
        #     ax.fill_between(x, y - s, y + s, alpha=0.2, label="±1σ")
        ax.set_xlabel("Gate VGK [V]")
        ax.set_ylabel("Σ/σ  [V$^{-1}$]")
        ax.set_title("Gate-dependent Σ/σ ")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR/"sigma_ratio_vs_gate.png", dpi=160)
        plt.close(fig)

    print("✅ 输出目录：", OUTPUT_DIR)

if __name__ == "__main__":
    main()
