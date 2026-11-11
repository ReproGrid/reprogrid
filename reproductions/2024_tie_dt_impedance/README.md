# 三相数字孪生在线阻抗稳定性

IEEE TIE 10.1109/TIE.2024.3395755。这里保存式 (7)–(11) 辨识算法和式 (2)–(3) 回比分析的独立计算。测试系统包括一台级联 PI 的三相 GFM、LC 滤波器、无源网侧支路及其电流驱动数字孪生。六种 SCR/电容组合来自第一作者的博士论文，原台架工况的状态见报告。

## 运行

在本目录使用 Python 3.12。原始交付结果写在 `results/`，下列命令使用新的目录。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/reproduce.py --out results/replay_new
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/compare_replay.py \
  --primary results --primary-figures figures \
  --replay results/replay_new --out results/replay_new/comparison.json
```

相同版本的 macOS arm64 环境用于精确重跑。比较程序验证协议、核心实现、所有数值文件及图片的 SHA-256，也逐数组核对 NPZ。

## 文件

- [来源审计](sources/source_audit.md)及[冻结参数](configs/protocol.json)
- [主报告](reports/REPORT.md)、[状态](reports/status.json)
- [工况与门槛](results/case_metrics.json)、[逐频点误差](results/identification_summary.csv)和[完整阻抗矩阵](results/impedance_curves.npz)
- [独立生成的三相与 dq 测量记录](results/example_waveforms.npz)，含 RL/SCR=3 的 1.6376 s 共同帧、物理调制激励和孪生输入输出
- [正交激励](figures/orthogonal_excitation.png)、[阻抗误差](figures/identification_error.png)、[阻抗幅相](figures/impedance_bode.png)及[频带 Nyquist](figures/nyquist_band.png)
- [独立标量边带实现](verification/sideband.py)和[重跑校验](results/clean_replay_comparison.json)

记录约定为 PCC 电流流向网侧取正，孪生输入电流流入变流器取正。频率采用同步 dq 坐标，NPZ 中阻抗为 Ω，回比无量纲，所有波形为增量。有限频带内的回比用于裕度诊断。已知模型上的全频绕数、连续极点与采样极点分别保存。
