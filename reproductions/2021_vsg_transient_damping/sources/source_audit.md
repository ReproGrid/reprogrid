# 第十四篇资料核对

这篇 TPEL 论文在本地 2026-11-10 OpenAlex 快照中记录为 96 次引用。单台构网变流器接等值电压源。作者稿给出式 (1)–(7)、表 I 和图 3–4，能独立计算电压跌落后的相平面和稳定区。 筛选快照 · [Aalborg 作者稿](https://vbn.aau.dk/en/publications/an-optimal-damping-design-of-virtual-synchronous-generators-for-t/)

表 I 列出 100 V、2 kW、314 rad/s、12 mH 和 48 mΩ。式 (5)–(6) 的 3/2 系数采用相电压峰值，按 2 kW 算得阻抗基值 7.5 Ω，与表中 0.5 p.u. 和 0.006 p.u. 的舍入一致。增益公式使用的 Pmax、Qmax 未另给数值，本复现采用 2 kW 和 2 kvar。图 2 按正常电压 100 V 计算五组阻抗的趋势。

图 3 采用表 I 主要参数，α=3 rad/s，电压从 1.0 降至 0.6 p.u.。图 4 段落明确给出 Rg=0、Lg=12 mH，按 SCR=2 设计。复现将这两个条件分开，并保留 314 rad/s 与精确 50 Hz、12 mH 与精确 SCR=2 的换算差异。图 3–4 可用 PDF 向量图读回定量核对。 [冻结条件](../configs/protocol.json) · [图形数据](../verification/fig3_4_readback.csv)

公开来源中找到的是论文和图形。作者的 ode45 输入脚本、完整参数扫描网格、时间区间和求解容差仍缺。图 6 台架结果的原始时序、内电压环和实际数字实现尚无可运行材料。 [论文 DOI](https://doi.org/10.1109/TPEL.2021.3074027)
