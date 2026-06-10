# 🔬 偏振模拟器 · Polarization Simulator

> 基于琼斯演算的偏振态级联计算器 — 逐级演化 · 实时可视化  
> *Jones Calculus Cascade Calculator — Step-by-step evolution with real-time visualization*

<img width="1894" height="936" alt="image" src="https://github.com/user-attachments/assets/59690989-b271-40bb-a2e1-5abd9fc1f619" />

---

[English](#english)　|　[中文](#中文)

---

## English

### Overview

An interactive **Jones calculus** simulator built with Streamlit. Define an input polarization state (linear, circular, or elliptical), cascade optical elements (wave plates, polarizers, rotators), and watch the polarization ellipse evolve step by step — all computed and rendered in real time.

### Features

- **Input polarization** — Linear (ψ), Circular (LCP/RCP), or Elliptical (ψ + axis ratio)
- **Optical elements** — QWP, HWP, custom retarder, polarizer, rotator
- **Cascade engine** — Jones matrices multiplied right-to-left, with per-stage state extraction
- **Live, instant feedback** — Tweak any parameter (angle, retardance, input state, element type) and see every polarization state recalculate in real time — no "Run" button needed
- **Complete numerical output** — Jones vectors, Stokes parameters, azimuth ψ, ellipticity χ, axis ratio a/b, and polarization extinction ratio (PER)
- **Evolution strip** — Compact horizontal overview of all polarization states through the cascade
- **Angle units** — Degrees or radians, with fine (0.1°) / coarse (1°) step control
- **Built-in theory reference** — Jones formalism, Stokes parameters, and sign convention notes
- **Bilingual** — Chinese UI with full English support

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/polarization_simulation.git
cd polarization_simulation
pip install -r requirements.txt
```

### Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

1. Choose input polarization type and parameters in the sidebar
2. Add optical elements (QWP, HWP, polarizer, rotator, custom retarder)
3. Adjust each element's angle and retardance interactively
4. Observe the polarization state evolution in real time

### Requirements

- Python ≥ 3.9
- streamlit ≥ 1.28
- numpy ≥ 1.24
- matplotlib ≥ 3.7

### Physics Background

This tool uses the **Jones calculus** for fully polarized light:

| Symbol | Meaning |
|--------|---------|
| **J** | Jones vector (2×1 complex) — the polarization state |
| **M** | Jones matrix (2×2 complex) — an optical element |
| **T** | Total transmission matrix = Mₙ···M₁ (right-to-left) |
| ψ | Azimuth — orientation of the polarization ellipse |
| χ | Ellipticity angle — tan χ = b/a |
| PER | Polarization Extinction Ratio = (a/b)², in dB |

**Convention:** Time factor e⁻ⁱωᵗ, S₃ > 0 → right-handed.  
*See the in-app "📖 Theory" dialog for detailed derivations.*

### Limitations

Jones calculus only describes **fully polarized** monochromatic plane waves. For partially polarized or unpolarized light, use the Stokes–Mueller formalism.

---

## 中文

### 概述

基于 **琼斯演算** 的交互式偏振态模拟器。定义入射偏振态（线偏/圆偏/椭圆偏），级联光学元件（波片、偏振器、旋光器），逐级观察偏振椭圆的实时演化。

### 功能

- **入射光源** — 线偏振 (ψ)、圆偏振 (LCP/RCP)、椭圆偏振 (ψ + 轴比)
- **光学元件** — ¼ 波片 (QWP)、½ 波片 (HWP)、自定义波片、偏振器、旋光器
- **级联引擎** — 琼斯矩阵从右往左乘，逐级提取偏振态
- **实时即时反馈** — 任意参数（角度、相位延迟、入射态、元件类型）改动后，全部偏振态即刻重算刷新，无需点击"运行"按钮
- **完整数值输出** — 琼斯矢量、Stokes 参数、方位角 ψ、椭偏角 χ、轴比 a/b、消光比 PER
- **全偏振演化缩略图** — 一行概览所有偏振态
- **角度单位** — 角度制/弧度制切换，精调 (0.1°)/粗调 (1°) 步进
- **内置原理说明** — 琼斯形式、Stokes 参数、手性约定
- **中英双语** — 中文界面，英文 README 支持

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/polarization_simulation.git
cd polarization_simulation
pip install -r requirements.txt
```

### 使用

```bash
streamlit run app.py
```

浏览器打开 http://localhost:8501：

1. 侧栏选择入射偏振类型及参数
2. 添加光学元件
3. 交互调节各元件的角度与相位延迟
4. 实时观察偏振态逐级演化

### 依赖

- Python ≥ 3.9
- streamlit ≥ 1.28
- numpy ≥ 1.24
- matplotlib ≥ 3.7

### 物理背景

本工具使用 **琼斯演算** 处理完全偏振光：

| 符号 | 含义 |
|------|------|
| **J** | 琼斯矢量 (2×1 复矢量) — 偏振态 |
| **M** | 琼斯矩阵 (2×2 复矩阵) — 光学元件 |
| **T** | 总传输矩阵 = Mₙ···M₁（从右往左乘） |
| ψ | 方位角 — 偏振椭圆长轴方向 |
| χ | 椭偏角 — tan χ = b/a（短轴/长轴） |
| PER | 偏振消光比 = (a/b)²，以 dB 表示 |

**约定：** 时间因子 e⁻ⁱωᵗ，S₃ > 0 为右旋。  
*详见应用内「📖 计算原理」对话框。*

### 局限性

琼斯演算仅描述 **完全偏振** 的单色平面波。部分偏振光或非偏振光需使用 Stokes–Mueller 形式。

---

## License

MIT License — feel free to use, modify, and share.

---

<p align="center">
  <sub>Built with Streamlit · NumPy · Matplotlib</sub>
</p>
