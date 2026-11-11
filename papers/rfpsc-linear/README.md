# 参考前馈功率同步控制 · 空载小信号试点

论文 DOI 为 [10.1109/TPEL.2020.2970991](https://doi.org/10.1109/TPEL.2020.2970991)。公式和条件来自 [Aalto 作者手稿](https://acris.aalto.fi/ws/portalfiles/portal/41217320/ELEC_Harnefors_etal_Reference_Feedforward_IEEETraPowEle_acceptedauthormanuscript.pdf) 的第 II-B 节、式 (18)–(20) 与 Fig. 3。另有 [motulator 参考示例](https://aalto-electric-drives.github.io/motulator/grid_examples/grid_forming/plot_13kva_rfpsc_gfm.html)。

运行 `python3 run.py` 生成 CSV、汇总 JSON、图表数据与包含 SHA-256 和验证阈值的运行清单。以空载、纯感性无限大电网、固定电压和忽略低通滤波器动态的理论近似为条件。50 Hz 仅用于将归一化时间换算为毫秒。使用同样的 0.01 p.u. 功率参考阶跃比较常规 PSC 和 RFPSC。

三种校验包括直接 dq 方程与传递函数对照、步长减半，以及 RFPSC 一阶解析式核对。上升时间按首次 10% 到首次 90% 的区间计算。过冲以相对稳态幅值的百分比表示。Fig. 5 的 1.0 p.u. 电感只用作线性模型的输入。Fig. 5/6 的硬件波形、带载响应与 motulator 示例仍待独立验证。Table I 的有源电阻欧姆值与标幺值需按原作者的归一化约定核对。

`media/` 提供两段带字幕的讲解短片。运行 `python3 make_videos.py --font /path/to/cjk-font.ttf` 可重制。制片依赖 Pillow 11.3.0 与 imageio-ffmpeg 0.6.0，数值实验本身只用 Python 标准库。
