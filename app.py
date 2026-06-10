"""
偏振态级联计算器 — 琼斯矩阵 · 逐级演化 · 实时分析
原生 Streamlit 风格,单文件。
"""
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
import streamlit as st


# ---- 中文字体 ----
for _f in ["PingFang SC", "Heiti SC", "STHeiti", "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "SimHei"]:
    if any(_f.lower() in name.lower() for name in fm.get_font_names()):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 计算引擎  (琼斯 calculus)
# ============================================================

def Rot(theta):
    """2D 旋转矩阵"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def polarizer(theta):
    """理想线偏振器,透振方向 theta [rad]"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c * c, s * c], [s * c, s * s]], dtype=complex)


def wave_plate(theta, gamma):
    """波片,快轴 theta [rad],相位延迟 gamma [rad]"""
    W0 = np.array([[1, 0], [0, np.exp(1j * gamma)]], dtype=complex)
    R = Rot(theta)
    return R @ W0 @ R.T


def rotator(theta):
    """旋光器,旋转角 theta [rad]"""
    return Rot(theta)


ELEM_META = {
    "qwp":       {"token": "Q", "name": "四分之一波片", "short": "QWP",
                  "color": "#2563eb", "angle": 45.0,  "gamma": 90.0,  "gamma_fixed": True},
    "hwp":       {"token": "H", "name": "半波片",       "short": "HWP",
                  "color": "#0d9488", "angle": 22.5,  "gamma": 180.0, "gamma_fixed": True},
    "retarder":  {"token": "λ", "name": "自定义波片",   "short": "Retarder",
                  "color": "#7c3aed", "angle": 0.0,   "gamma": 60.0,  "gamma_fixed": False},
    "polarizer": {"token": "P", "name": "偏振器",       "short": "Pol",
                  "color": "#d97706", "angle": 0.0,   "gamma": None,  "gamma_fixed": True},
    "rotator":   {"token": "R", "name": "旋光器",       "short": "Rot",
                  "color": "#db2777", "angle": 45.0,  "gamma": None,  "gamma_fixed": True},
}


def element_matrix(elem):
    """元件 -> 琼斯矩阵"""
    t = elem["type"]
    angle = np.deg2rad(float(elem["angle_deg"]))
    if t in ("qwp", "hwp", "retarder"):
        return wave_plate(angle, np.deg2rad(float(elem["retardance_deg"])))
    if t == "polarizer":
        return polarizer(angle)
    return rotator(angle)


def cascade_stages(elements, jv_in):
    """返回逐级偏振态 [入射, 元件1后, ..., 出射] 与总矩阵"""
    states = [jv_in]
    T = np.eye(2, dtype=complex)
    for elem in elements:
        M = element_matrix(elem)
        T = M @ T
        states.append(M @ states[-1])
    return states, T


def jones_from_linear(psi_deg):
    """线偏振:方位角 psi"""
    p = np.deg2rad(psi_deg)
    return np.array([np.cos(p), np.sin(p)], dtype=complex)


def jones_from_circular(right_handed):
    """圆偏振。约定 E(t)=Re{J e^{-iwt}}, S3>0 为右旋"""
    return np.array([1, -1j if right_handed else 1j], dtype=complex) / np.sqrt(2)


def jones_from_ellipse(psi_deg, chi_deg):
    """椭圆偏振:方位角 psi + 椭偏角 chi (chi>0 右旋)"""
    p, c = np.deg2rad(psi_deg), np.deg2rad(chi_deg)
    ex = np.cos(c) * np.cos(p) + 1j * np.sin(c) * np.sin(p)
    ey = np.cos(c) * np.sin(p) - 1j * np.sin(c) * np.cos(p)
    return np.array([ex, ey], dtype=complex)


def analyze(jv):
    """从琼斯矢量提取 Stokes + 椭圆参数"""
    Ex, Ey = jv[0], jv[1]
    S0 = float(abs(Ex) ** 2 + abs(Ey) ** 2)
    if S0 < 1e-12:
        return dict(S0=0.0, s1=0.0, s2=0.0, s3=0.0, psi=0.0, chi=0.0,
                    delta=0.0, hand="消光", Ex=Ex, Ey=Ey)
    S1 = float(abs(Ex) ** 2 - abs(Ey) ** 2)
    S2 = float(2 * np.real(Ex * np.conj(Ey)))
    S3 = float(2 * np.imag(Ex * np.conj(Ey)))
    s1, s2, s3 = S1 / S0, S2 / S0, S3 / S0
    chi = np.rad2deg(0.5 * np.arcsin(np.clip(s3, -1, 1)))
    psi = np.rad2deg(0.5 * np.arctan2(s2, s1)) % 180
    delta = np.rad2deg(np.angle(Ey) - np.angle(Ex))
    delta = (delta + 180) % 360 - 180
    if s3 > 0.01:
        hand = "右旋"
    elif s3 < -0.01:
        hand = "左旋"
    else:
        hand = "线偏振"
    return dict(S0=S0, s1=s1, s2=s2, s3=s3, psi=psi, chi=chi,
                delta=delta, hand=hand, Ex=Ex, Ey=Ey)


# ============================================================
# 可视化
# ============================================================

HAND_COLOR = {"右旋": "#db2777", "左旋": "#2563eb", "线偏振": "#0d9488", "消光": "#94a3b8"}


def is_circular(info):
    """圆偏振:|χ|≈45°,此时长轴方向(方位角 ψ)无定义"""
    return info["S0"] > 1e-9 and abs(info["chi"]) > 44.5


def ellipse_xy(jv, n=400):
    """偏振椭圆轨迹,约定 E(t)=Re{J e^{-iwt}}"""
    Ex, Ey = jv[0], jv[1]
    t = np.linspace(0, 2 * np.pi, n)
    x = np.real(Ex) * np.cos(t) + np.imag(Ex) * np.sin(t)
    y = np.real(Ey) * np.cos(t) + np.imag(Ey) * np.sin(t)
    return x, y


def draw_state(ax, jv, ref_S0=None, fast_axes=None):
    """只画偏振几何(单位圆 + 椭圆 + 相邻快轴);标题/数据由外层放在左侧"""
    info = analyze(jv)
    color = HAND_COLOR[info["hand"]]
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#e2e8f0", lw=0.8))
    # 坐标轴用有限线段(不用 axhline/axvline,避免撑大画布留白)
    ax.plot([-1, 1], [0, 0], color="#e2e8f0", lw=0.8, zorder=0)
    ax.plot([0, 0], [-1, 1], color="#e2e8f0", lw=0.8, zorder=0)

    # 相邻元件的快轴(仅波片有快慢轴,只画 fast axis);画在偏振椭圆之下,
    # 线略长于直径(端点伸出圆外)更好看,token 标在端点之外
    AX_LEN = 1.16   # 快轴半长(>1 即超出单位圆)
    AX_LBL = 1.32   # 标注沿快轴的位置(端点再外侧)
    for ang_deg, ax_color, token, *_ in (fast_axes or []):
        p = np.deg2rad(ang_deg)
        fx, fy = np.cos(p), np.sin(p)
        ax.plot([AX_LEN * fx, -AX_LEN * fx], [AX_LEN * fy, -AX_LEN * fy],
                color=ax_color, lw=1.3, ls=(0, (6, 2, 1, 2)), alpha=0.85,
                zorder=0.5)
        ax.text(AX_LBL * fx, AX_LBL * fy, token, color=ax_color, fontsize=9,
                fontweight="bold", ha="center", va="center", zorder=5)

    # 参考光强:相对入射归一,使椭圆大小反映真实透过强度;
    # 单位圆代表入射满强度,衰减后的态自然缩小,直至消光收为一点。
    ref = info["S0"] if ref_S0 is None else max(float(ref_S0), 1e-12)

    if info["S0"] < 1e-9:
        ax.scatter([0], [0], s=30, color="#94a3b8")
        ax.text(0, -0.35, "消光", ha="center", fontsize=9, color="#64748b")
    else:
        scale = np.sqrt(ref)
        amp = np.sqrt(info["S0"] / ref)  # 相对入射的振幅(≤1)
        x, y = ellipse_xy(jv)
        x, y = x / scale, y / scale
        ax.fill(x, y, color=color, alpha=0.12)
        ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round")
        if info["hand"] == "线偏振":
            # 线偏振:电场沿一条线往复振动 -> 双向箭头
            p = np.deg2rad(info["psi"])
            dx, dy = amp * np.cos(p), amp * np.sin(p)
            ax.annotate("", xy=(dx, dy), xytext=(-dx, -dy),
                        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=2.2,
                                        mutation_scale=14))
        else:
            # 圆/椭圆偏振:旋转方向箭头 + 长轴虚线
            i0, i1 = int(0.05 * len(x)), int(0.10 * len(x))
            ax.annotate("", xy=(x[i1], y[i1]), xytext=(x[i0], y[i0]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                        mutation_scale=12))
            if not is_circular(info):
                # 仅椭圆偏振才有确定的长轴;圆偏振不画方位线
                p = np.deg2rad(info["psi"])
                ax.plot([amp * np.cos(p), -amp * np.cos(p)],
                        [amp * np.sin(p), -amp * np.sin(p)],
                        color="#94a3b8", lw=0.8, ls=(0, (4, 4)))


def fig_to_svg_html(fig, max_width_px):
    """把 matplotlib figure 转成矢量 SVG 内嵌,Retina 屏也锐利"""
    buf = io.StringIO()
    # bbox_inches="tight" 默认还会补 0.1in 白边(pad_inches),这里去掉多余外边距
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0,
                transparent=True)
    b64 = base64.b64encode(buf.getvalue().encode()).decode()
    return (
        f"<div style='text-align:center;line-height:0;margin:0'>"
        f"<img src='data:image/svg+xml;base64,{b64}' "
        f"style='width:100%;max-width:{max_width_px}px;display:block;margin:0 auto'/></div>"
    )


def state_sub(info):
    """单个偏振态的副标题:形态 · 方位角 · 偏振消光比 · 椭偏角"""
    if info["S0"] < 1e-9:
        return f"{info['hand']}\n$\\chi$ = —\n偏振消光比 (PER) = —"
    chi = info["chi"]
    chi_rad = np.deg2rad(abs(chi))
    # 偏振消光比 = 长/短轴强度比 (a/b)^2,以 dB 表示;圆偏振=0 dB,线偏振=∞
    per = "∞" if chi_rad < 1e-3 else f"{-20 * np.log10(np.tan(chi_rad)):.1f} dB"
    if info["hand"] == "线偏振":
        # 线偏振:椭偏角 χ≈0、PER=∞ 无参考意义,不显示
        return f"{info['hand']} · $\\psi$ {info['psi']:.2f}°"
    if is_circular(info):
        # 圆偏振:无方位角
        head = f"{info['hand']}圆偏振"
    else:
        head = f"{info['hand']} · $\\psi$ {info['psi']:.2f}°"
    axis_ratio = 1.0 / np.tan(chi_rad)  # a/b = 1/tan|χ|
    return f"{head}\n$\\chi$ = {chi:+.1f}°\n轴比 $a/b$ = {axis_ratio:.2f}\n消光比 (PER) = {per}"


def draw_state_thumb(ax, jv, ref_S0=None):
    """缩略图版偏振态:只画椭圆 + 单位圆 + 旋转箭头,紧凑无快轴"""
    info = analyze(jv)
    color = HAND_COLOR[info["hand"]]
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#e2e8f0", lw=0.6))
    ax.plot([-1, 1], [0, 0], color="#e2e8f0", lw=0.6, zorder=0)
    ax.plot([0, 0], [-1, 1], color="#e2e8f0", lw=0.6, zorder=0)

    ref = info["S0"] if ref_S0 is None else max(float(ref_S0), 1e-12)

    if info["S0"] < 1e-9:
        ax.scatter([0], [0], s=18, color="#94a3b8")
    else:
        scale = np.sqrt(ref)
        amp = np.sqrt(info["S0"] / ref)
        x, y = ellipse_xy(jv)
        x, y = x / scale, y / scale
        ax.fill(x, y, color=color, alpha=0.12)
        ax.plot(x, y, color=color, lw=1.6, solid_capstyle="round")
        if info["hand"] == "线偏振":
            p = np.deg2rad(info["psi"])
            dx, dy = amp * np.cos(p), amp * np.sin(p)
            ax.annotate("", xy=(dx, dy), xytext=(-dx, -dy),
                        arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.6,
                                        mutation_scale=9))
        else:
            i0, i1 = int(0.05 * len(x)), int(0.10 * len(x))
            ax.annotate("", xy=(x[i1], y[i1]), xytext=(x[i0], y[i0]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                        mutation_scale=7))


def evolution_strip_svg(states, labels, ref_S0, max_width_px=420):
    """横向全偏振演化缩略图:一行显示入射→各级后→出射的所有偏振椭圆"""
    n = len(states)
    if n == 0:
        return ""
    fig = plt.figure(figsize=(n * 1.55, 1.75), constrained_layout=True)
    for i, (jv, label) in enumerate(zip(states, labels)):
        ax = fig.add_subplot(1, n, i + 1)
        draw_state_thumb(ax, jv, ref_S0=ref_S0)
        short = label.replace("初始偏振", "入射").replace("第", "").replace("级后", "")
        ax.set_title(short, fontsize=7, color="#64748b", pad=1)
    html = fig_to_svg_html(fig, max_width_px)
    plt.close(fig)
    return html


def adjacent_fast_axes(state_idx, elements):
    """该偏振态左右相邻元件的快轴(仅波片有);token 带序号(第几个元件即几)
    返回 [(angle_deg, color, token), ...]"""
    specs = []
    for ei in (state_idx - 1, state_idx):
        if 0 <= ei < len(elements):
            e = elements[ei]
            if e["type"] in ("qwp", "hwp", "retarder"):
                m = ELEM_META[e["type"]]
                specs.append((float(e["angle_deg"]), m["color"],
                              f"{m['token']}{ei + 1}"))
    return specs


def elements_cache_key(elements):
    """基于全部元件配置的不可变哈希键,用于 state_svg 缓存失效。"""
    return tuple(
        (e["id"], e["type"], round(float(e.get("angle_deg", 0)), 6),
         round(float(e.get("retardance_deg", 0)), 6) if "retardance_deg" in e else None)
        for e in elements
    )


@st.cache_data(show_spinner=False, max_entries=256)
def state_svg(jv, label, width_px=420, ref_S0=None, fast_axes=None, *, _cache_key=None):
    """渲染单个偏振态为内嵌 SVG:左侧标题+数据,右侧偏振几何图。
    输出仅由入参决定,故缓存:改某一级参数时,未变化的上游态直接命中缓存,
    只重绘真正变化的图(matplotlib 渲染是这里唯一的性能瓶颈)。
    _cache_key: 元件配置哈希,来自 elements_cache_key(),保证类型切换时缓存失效。"""
    info = analyze(jv)
    fig = plt.figure(figsize=(4.0, 2.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.36, 0.64])
    axL = fig.add_subplot(gs[0])
    axL.axis("off")
    axR = fig.add_subplot(gs[1])
    draw_state(axR, jv, ref_S0=ref_S0, fast_axes=fast_axes)
    # 左栏:标题(上)+ 数据(下),左对齐;整体略左移几个像素
    x0 = -0.2
    axL.text(x0, 0.64, label, fontsize=13, fontweight="bold", color="#0f172a",
             ha="left", va="center", transform=axL.transAxes)
    axL.text(x0, 0.48, state_sub(info), fontsize=10, color="#475569",
             ha="left", va="top", transform=axL.transAxes, linespacing=1.7)
    html = fig_to_svg_html(fig, width_px)
    plt.close(fig)
    return html


def connector_html(meta, num):
    """两态之间的引导线,指向右侧元件卡片;num 为元件序号(第几个)"""
    color = meta["color"]
    return (
        f"<div style='display:flex;align-items:center;height:52px;padding:0 4px'>"
        f"<div style='width:11px;height:11px;border-radius:50%;background:{color};"
        f"flex:0 0 auto;box-shadow:0 0 0 3px {color}22'></div>"
        f"<div style='flex:1;border-top:2px dashed {color};margin:0 4px;opacity:.65'></div>"
        f"<div style='color:{color};font-weight:800;font-size:13px;white-space:nowrap'>"
        f"{meta['token']}{num}&nbsp;▸</div>"
        f"</div>"
    )


# ============================================================
# 工具
# ============================================================

def fmt_c(z):
    return f"{z.real:+.3f}{z.imag:+.3f}i"


def jones_latex(jv):
    return r"\begin{pmatrix}" + fmt_c(jv[0]) + r"\\" + fmt_c(jv[1]) + r"\end{pmatrix}"


def matrix_latex(M):
    return (r"\begin{pmatrix}" + f"{fmt_c(M[0,0])} & {fmt_c(M[0,1])}"
            + r"\\" + f"{fmt_c(M[1,0])} & {fmt_c(M[1,1])}" + r"\end{pmatrix}")


def elem_summary(elem):
    if elem["type"] in ("qwp", "hwp", "retarder"):
        return f"θ={elem['angle_deg']:.3f}°, Γ={elem['retardance_deg']:.3f}°"
    return f"θ={elem['angle_deg']:.3f}°"


def fmt_angle(deg, unit_rad, signed=False):
    """按当前单位格式化角度(三位小数)"""
    if unit_rad:
        v = np.deg2rad(deg)
        return (f"{v:+.3f}" if signed else f"{v:.3f}") + " rad"
    return (f"{deg:+.3f}" if signed else f"{deg:.3f}") + "°"


def angle_input(label, key, canon_deg, lo_deg, hi_deg, unit_rad, help=None):
    """可输入的角度框,支持角度/弧度切换;内部规范量始终是角度(°),三位小数。
    步进由侧栏「粗调/精调」开关统一控制(粗调 1° / 精调 0.1°)。"""
    step_deg = st.session_state.get("angle_step_deg", 0.1)
    if unit_rad:
        shown = st.number_input(
            f"{label} (rad)",
            min_value=round(float(np.deg2rad(lo_deg)), 3),
            max_value=round(float(np.deg2rad(hi_deg)), 3),
            value=round(float(np.deg2rad(canon_deg)), 3),
            step=round(float(np.deg2rad(step_deg)), 3), format="%.3f",
            key=f"{key}__rad", help=help,
        )
        return float(np.rad2deg(shown))
    return st.number_input(
        f"{label} (°)",
        min_value=float(lo_deg), max_value=float(hi_deg),
        value=round(float(canon_deg), 3),
        step=step_deg, format="%.3f", key=f"{key}__deg", help=help,
    )


def read_widget_angle(key, fallback_deg, unit_rad):
    """从已渲染过的角度输入框读取当前值(规范化为°)。
    元件卡片在脚本靠后才渲染,但用户输入的新值在重运行开始时已存入
    session_state,故计算偏振态前先在此取最新值,避免"显示慢一拍"。
    新加元件首帧 widget 尚未建立,则退回 fallback。
    注:key 命名须与 angle_input 保持一致(__deg / __rad)。"""
    k = f"{key}__rad" if unit_rad else f"{key}__deg"
    if k in st.session_state:
        v = float(st.session_state[k])
        return float(np.rad2deg(v)) if unit_rad else v
    return float(fallback_deg)


def add_element(etype):
    m = ELEM_META[etype]
    elem = {"id": st.session_state.next_id, "type": etype, "angle_deg": m["angle"]}
    if etype in ("qwp", "hwp", "retarder"):
        elem["retardance_deg"] = m["gamma"]
    st.session_state.elements.append(elem)
    st.session_state.next_id += 1


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="偏振计算器", page_icon="🔬", layout="wide")

# 收紧竖直行距,让阶梯更紧凑、一屏看到更多元件(仅作用于主区,不影响侧栏)
st.markdown(
    "<style>"
    "[data-testid='stMainBlockContainer'] [data-testid='stVerticalBlock']{gap:0.45rem}"
    "[data-testid='stMainBlockContainer']{padding-top:2.5rem;padding-bottom:2rem}"
    "[data-testid='stMainBlockContainer'] [data-testid='stImage']{margin-bottom:0}"
    "[data-testid='stDialog'] h1,[data-testid='stDialog'] h2{text-align:center}"
    # 「📖 计算原理」按钮微调:整体下移几个像素,与标题视觉更协调
    ".st-key-theory_btn{transform:translateY(11px)}"
    "</style>",
    unsafe_allow_html=True,
)
# 卡片头部小按钮内边距:让 icon 居中
st.markdown(
    "<style>"
    "[class*=\"st-key-type_btn\"] button,"
    "[class*=\"st-key-del_\"] button"
    "{padding-left:0.4rem!important;padding-right:0.4rem!important;"
    "display:inline-flex!important;justify-content:center!important;align-items:center!important}"
    "</style>",
    unsafe_allow_html=True,
)

if "elements" not in st.session_state:
    st.session_state.elements = []
    st.session_state.next_id = 1

# ---------- 侧栏:入射光源 + 元件库 ----------
with st.sidebar:
    st.header("角度单位")
    unit = st.radio("角度单位", ["角度制 (°)", "弧度制 (rad)"],
                    horizontal=True, label_visibility="collapsed")
    unit_rad = unit.startswith("弧度")
    # 切换单位时清掉旧单位的输入框,使其按规范值(°)重新初始化
    if st.session_state.get("_angle_unit") != unit:
        for _k in [k for k in st.session_state if k.endswith(("__rad", "__deg"))]:
            del st.session_state[_k]
        st.session_state["_angle_unit"] = unit
    # 步进:精调 0.1° / 粗调 1°(全局,作用于所有角度输入框)
    # 开 = 精调 0.1°,关 = 粗调 1°
    fine = st.toggle("精调步进 (0.1°)", value=True, help="开:精调 0.1°　关:粗调 1°")
    st.session_state["angle_step_deg"] = 0.1 if fine else 1.0

    st.header("入射光源")
    mode = st.radio("偏振类型", ["线偏振", "圆偏振", "椭圆偏振"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "线偏振":
        psi = angle_input(r"方位角 $\psi$", "src_psi",
                          st.session_state.get("src_psi", 0.0),
                          0.0, 180.0, unit_rad, help=r"电场振动方向与 $x$ 轴的夹角")
        st.session_state["src_psi"] = psi
        jv_in = jones_from_linear(psi)
    elif mode == "圆偏振":
        sense = st.radio("旋向", ["右旋 (RCP)", "左旋 (LCP)"], horizontal=True)
        jv_in = jones_from_circular(sense.startswith("右旋"))
    else:
        psi = angle_input(r"方位角 $\psi$", "src_psi",
                          st.session_state.get("src_psi", 0.0),
                          0.0, 180.0, unit_rad, help=r"椭圆长轴与 $x$ 轴的夹角")
        st.session_state["src_psi"] = psi
        axis_ratio = st.number_input(
            r"轴比 $a/b$ (长轴/短轴)",
            min_value=1.0, max_value=100.0,
            value=round(float(st.session_state.get("src_axis_ratio", 2.0)), 2),
            step=0.1, format="%.2f", key="src_axis_ratio_w",
            help=r"长轴 $a$ 与短轴 $b$ 之比;$=1$ 为圆偏,越大越接近线偏。",
        )
        st.session_state["src_axis_ratio"] = axis_ratio
        sense = st.radio("旋向", ["右旋 (RCP)", "左旋 (LCP)"],
                         horizontal=True, key="src_ellipse_sense")
        # 轴比 a/b -> 椭偏角 |χ| = arctan(b/a);旋向决定符号(右旋 χ>0)
        chi_mag = np.rad2deg(np.arctan(1.0 / axis_ratio))
        chi = chi_mag if sense.startswith("右旋") else -chi_mag
        jv_in = jones_from_ellipse(psi, chi)

    info_in = analyze(jv_in)
    st.latex(r"\mathbf{J}_{in}=" + jones_latex(jv_in))
    st.markdown(f"**{info_in['hand']}**")
    st.metric(
        r"相位差 $\delta$", fmt_angle(info_in["delta"], unit_rad, signed=True),
        help=r"$\delta = \delta_y - \delta_x$,即 $E_y$ 相对 $E_x$ 的相位差。",
    )
    cc = st.columns(2)
    cc[0].metric(
        r"方位角 $\psi$",
        "—" if is_circular(info_in) else fmt_angle(info_in["psi"], unit_rad),
        help=r"偏振椭圆长轴与 $x$ 轴(水平)的夹角;圆偏振时无定义。",
    )
    cc[1].metric(
        r"椭偏角 $\chi$", fmt_angle(info_in["chi"], unit_rad, signed=True),
        help=r"$\tan\chi$ = 短轴/长轴。$\chi=0$ 线偏,$\chi=\pm45°$ 圆偏;符号决定旋向($>0$ 右旋)。",
    )

    st.header("添加元件")
    c1, c2 = st.columns(2)
    if c1.button("¼ 波片 (QWP)", use_container_width=True):
        add_element("qwp"); st.rerun()
    if c2.button("½ 波片 (HWP)", use_container_width=True):
        add_element("hwp"); st.rerun()
    if c1.button("旋光器 (Rotator)", use_container_width=True):
        add_element("rotator"); st.rerun()
    if c2.button("偏振器 (Polarizer)", use_container_width=True):
        add_element("polarizer"); st.rerun()
    if c1.button("自定义波片 (Retarder)", use_container_width=True):
        add_element("retarder"); st.rerun()


# ---------- 主区 ----------
@st.dialog("计算原理", width="large")
def show_theory():
    st.markdown(r"""

本工具用 **琼斯演算**描述完全偏振光。偏振态是一个二维复矢量,光学元件是 $2\times2$ 复矩阵,
级联即矩阵连乘。

### 1. 琼斯矢量与真实电场

**真实电场**（向 $+z$ 传播的平面波，横向分量）：
$$
\mathbf{E}(z,t)=\mathrm{Re}\left\{
\begin{pmatrix}E_{0x}e^{i\delta_x}\\E_{0y}e^{i\delta_y}\end{pmatrix}
e^{i(kz-\omega t)}
\right\}
=\begin{pmatrix}E_{0x}\cos(kz-\omega t+\delta_x)\\
E_{0y}\cos(kz-\omega t+\delta_y)\end{pmatrix}
$$

**琼斯矢量** 是提取复振幅得到的：
$$
\mathbf{J}=\begin{pmatrix}E_x\\E_y\end{pmatrix}
=\begin{pmatrix}E_{0x}e^{i\delta_x}\\E_{0y}e^{i\delta_y}\end{pmatrix}
$$

> ✅ 琼斯矢量 $\leftrightarrow$ 真实场：固定 $z=0$，$\mathbf{E}(t)=\mathrm{Re}\{\mathbf{J}\,e^{-i\omega t}\}$。
> 给定 $\mathbf{J}$ 即可唯一恢复时域电场轨迹；反之，从电场振幅 $E_{0x},E_{0y}$ 和
> 初相 $\delta_x,\delta_y$ 可唯一构造 $\mathbf{J}$。总光强 $I\propto |E_x|^2+|E_y|^2$。

相位差 $\delta=\delta_y-\delta_x$ 决定偏振形状：$\delta=0$（或 $\pi$）为线偏，
$\delta=\pm90°$ 且等幅为圆偏，其余为椭圆。

> ⚠️ **琼斯演算的局限性。** 琼斯矢量只能描述**完全偏振光**（电场两分量有确定
> 的振幅比与相位差）。部分偏振光和非偏振光（自然光）需要用 **Stokes 矢量 + Mueller 矩阵**
> 来处理，琼斯框架无能为力。此外 琼斯演算不考虑元件对光的吸收损耗以外的退偏效应（如散射、
> 荧光），且假定光束是**单色平面波**。宽谱或非准直光束需额外处理。

### 2. 元件矩阵
**线偏振器**(透振方向 $\theta$):
$$
P(\theta)=\begin{pmatrix}\cos^2\theta & \sin\theta\cos\theta\\
\sin\theta\cos\theta & \sin^2\theta\end{pmatrix}
$$
**波片**(快轴 $\theta$,相位延迟 $\Gamma$;QWP $\Gamma=90°$,HWP $\Gamma=180°$):
$$
W(\theta,\Gamma)=R(\theta)\begin{pmatrix}1&0\\0&e^{i\Gamma}\end{pmatrix}R(-\theta)
$$
**旋光器**(旋转角 $\theta$):
$$
R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
$$

### 3. 级联
光依次穿过 $N$ 个元件,**矩阵从右往左乘**:
$$
\mathbf{J}_{out}=M_N\cdots M_2 M_1\,\mathbf{J}_{in}
$$

### 4. Stokes 参数与椭圆几何
$$
S_0=|E_x|^2+|E_y|^2,\quad S_1=|E_x|^2-|E_y|^2,\quad
S_2=2\,\mathrm{Re}(E_xE_y^*),\quad S_3=2\,\mathrm{Im}(E_xE_y^*)
$$
$$
\psi=\tfrac12\arctan2(S_2,S_1)\ \text{(方位角)},\qquad
\chi=\tfrac12\arcsin\tfrac{S_3}{S_0}\ \text{(椭偏角)}
$$

> ⚠️ **手性是约定相关的。** 本工具采用光学常用约定(Hecht、Born & Wolf 一脉),
> 即时间因子取 $e^{-i\omega t}$、$S_3>0$ 记为**右旋**。
> **IEEE / 天线工程界的右旋定义恰好相反**——若与该类资料对照,左/右旋标签会互换。
> 注意:这只影响"左旋 / 右旋"这个**命名**;琼斯矢量、Stokes 参数、椭偏角 $\chi$ 的
> **数值与符号本身都是确定的**。
>
> 👁️ **观察方向。** 图中偏振椭圆是**迎着光看**（观察者面向光源，视线与光传播方向
> $+z$ 相反）所见的电场矢量末端轨迹。时间因子 $e^{-i\omega t}$ 搭配此观察方向时，
> 矢端顺时针旋转对应 $S_3>0$（右旋），逆时针对应 $S_3<0$（左旋）。
""")


# 标题居中,「计算原理」按钮放在同一行最右
_hl, _hc, _hr = st.columns([1, 5, 1], vertical_alignment="center")
_hc.markdown(
    "<h1 style='text-align:center;margin:0'>偏振模拟器</h1>",
    unsafe_allow_html=True,
)
with _hr:
    if st.button("📖", use_container_width=False, key="theory_btn"):
        show_theory()

st.markdown(
    "<hr style='border:none;height:3px;background:#94a3b8;"
    "margin:0.4rem 0 1.8rem;border-radius:2px'/>",
    unsafe_allow_html=True,
)

tab_work = st.container()


def render_card(idx, elem, meta, elements, unit_rad):
    """右列:可编辑的元件卡片"""
    with st.container(border=True):
        hc = st.columns([0.68, 0.12, 0.20], vertical_alignment="center")
        hc[0].markdown(
            f"<span style='color:{meta['color']};font-weight:800'>{meta['token']}{idx+1}</span>　"
            f"**第 {idx+1} 级 · {meta['name']}**",
            unsafe_allow_html=True,
        )
        # ---- 器件类型切换 (popover 下拉菜单) ----
        type_keys = list(ELEM_META.keys())
        with hc[1].popover("⇄", use_container_width=True):
            for t in type_keys:
                m = ELEM_META[t]
                if st.button(
                    f"{m['token']} — {m['name']}",
                    key=f"type_{elem['id']}_{t}",
                    use_container_width=True,
                ):
                    elem["type"] = t
                    elem["angle_deg"] = m["angle"]
                    if m["gamma"] is not None:
                        elem["retardance_deg"] = m["gamma"]
                    for sfx in ("__deg", "__rad"):
                        st.session_state.pop(f"ang_{elem['id']}{sfx}", None)
                        st.session_state.pop(f"gam_{elem['id']}{sfx}", None)
                    st.rerun()
        if hc[2].button("✕", key=f"del_{elem['id']}", use_container_width=True,
                        help="删除此元件"):
            elements.pop(idx)
            st.rerun()

        lo, hi = (-180.0, 180.0) if elem["type"] == "rotator" else (0.0, 180.0)
        angle_help = {
            "qwp": r"**快轴**与 $x$ 轴(水平)的夹角,逆时针为正。",
            "hwp": r"**快轴**与 $x$ 轴(水平)的夹角,逆时针为正。",
            "retarder": r"**快轴**与 $x$ 轴(水平)的夹角,逆时针为正。",
            "polarizer": r"**透振方向**与 $x$ 轴(水平)的夹角,逆时针为正。",
            "rotator": r"偏振方向被整体**旋转的角度**,逆时针为正。",
        }[elem["type"]]
        elem["angle_deg"] = angle_input(
            r"角度 $\theta$", f"ang_{elem['id']}",
            float(elem["angle_deg"]), lo, hi, unit_rad, help=angle_help,
        )
        if not meta["gamma_fixed"]:
            elem["retardance_deg"] = angle_input(
                r"相位延迟 $\Gamma$", f"gam_{elem['id']}",
                float(elem["retardance_deg"]), 0.0, 360.0, unit_rad,
                help=r"快轴与**慢轴**之间引入的相位延迟 $\Gamma$;"
                     r"QWP=90°,HWP=180°。",
            )


with tab_work:
    elements = st.session_state.elements
    # 元件卡片在本帧靠后才渲染,但用户输入的新值已在 session_state 中。
    # 计算前先同步,否则偏振态会用旧角度,显示"慢一拍"。
    for elem in elements:
        elem["angle_deg"] = read_widget_angle(
            f"ang_{elem['id']}", elem["angle_deg"], unit_rad)
        if not ELEM_META[elem["type"]]["gamma_fixed"]:
            elem["retardance_deg"] = read_widget_angle(
                f"gam_{elem['id']}", elem["retardance_deg"], unit_rad)
    states, T_total = cascade_stages(elements, jv_in)
    labels = ["初始偏振"] + [f"第{i + 1}级后" for i in range(len(elements))]
    ref_S0 = analyze(states[0])["S0"]  # 入射光强,作为各级椭圆缩放的参考

    COLS = [0.62, 0.38]  # 左:偏振态演化  右:光路元件

    if not elements:
        row = st.columns(COLS, gap="medium", vertical_alignment="center")
        with row[0]:
            st.markdown(state_svg(states[0], labels[0], ref_S0=ref_S0,
                                  fast_axes=adjacent_fast_axes(0, elements),
                                  _cache_key=elements_cache_key(elements)),
                        unsafe_allow_html=True)
        row[1].info("在左侧「添加元件」中加入波片、偏振器或旋光器。")
    else:
        # ---- 全偏振演化缩略图 (占满一行) ----
        st.markdown(
            "<hr style='border:none;height:2px;background:#cbd5e1;"
            "margin:0 0 0.5rem;border-radius:1px'/>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.caption("全偏振演化")
            st.markdown(
                evolution_strip_svg(states, labels, ref_S0, max_width_px=900),
                unsafe_allow_html=True,
            )
        st.markdown(
            "<hr style='border:none;height:2px;background:#cbd5e1;"
            "margin:0.5rem 0 0;border-radius:1px'/>",
            unsafe_allow_html=True,
        )

        # ---- 交错阶梯:态 → (引导线 | 元件卡片) → 态 → … ----
        for idx, elem in enumerate(elements):
            meta = ELEM_META[elem["type"]]
            srow = st.columns(COLS, gap="medium", vertical_alignment="center")
            with srow[0]:
                html = state_svg(states[idx], labels[idx], ref_S0=ref_S0,
                                 fast_axes=adjacent_fast_axes(idx, elements),
                                 _cache_key=elements_cache_key(elements))
                if idx == 0:
                    html = f"<div style='margin-top:10px'>{html}</div>"
                st.markdown(html, unsafe_allow_html=True)
            crow = st.columns(COLS, gap="medium", vertical_alignment="center")
            with crow[0]:
                st.markdown(connector_html(meta, idx + 1), unsafe_allow_html=True)
            with crow[1]:
                render_card(idx, elem, meta, elements, unit_rad)
        # 末态
        lrow = st.columns(COLS, gap="medium", vertical_alignment="center")
        with lrow[0]:
            st.markdown(state_svg(states[-1], labels[-1], ref_S0=ref_S0,
                                  fast_axes=adjacent_fast_axes(len(elements), elements),
                                  _cache_key=elements_cache_key(elements)),
                        unsafe_allow_html=True)

        # ---- 清空光路（红色 + 二次确认防误触）----
        crow = st.columns(COLS, gap="medium")
        with crow[1]:
            if not st.session_state.get("confirm_clear"):
                if st.button("🗑 清空光路", type="primary", use_container_width=True):
                    st.session_state.confirm_clear = True
                    st.rerun()
            else:
                st.warning("确定清空元件?此操作不可撤销。")
                cy, cn = st.columns(2)
                if cy.button("确认清空", type="primary", use_container_width=True):
                    st.session_state.elements = []
                    st.session_state.confirm_clear = False
                    st.rerun()
                if cn.button("取消", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

    if elements:
        with st.expander("计算过程 (琼斯数值)"):
            st.markdown("**入射**")
            st.latex(r"\mathbf{J}_0=" + jones_latex(states[0]))
            for i, elem in enumerate(elements):
                meta = ELEM_META[elem["type"]]
                M = element_matrix(elem)
                st.markdown(f"**第 {i+1} 级 · {meta['name']}** ({elem_summary(elem)})")
                st.latex(
                    r"\mathbf{J}_" + str(i + 1) + "=" + matrix_latex(M)
                    + r"\mathbf{J}_" + str(i) + "=" + jones_latex(states[i + 1])
                )
            if elements:
                st.markdown("**总传输矩阵** $\\mathbf{T}=M_N\\cdots M_1$")
                st.latex(r"\mathbf{T}=" + matrix_latex(T_total))
