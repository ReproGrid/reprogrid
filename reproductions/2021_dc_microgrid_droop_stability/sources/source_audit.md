# 第十七篇参数与资料核对

论文在本地 2026-11-10 OpenAlex 快照记录 106 次引用。选取 Fig. 3(c) 二阶等效电路和 Fig. 9(b) 四个下垂支路接同一母线的算例。输入来自 Tables II–VII 和 Section V–VI，功率按各阶跃区间的负荷总值处理。 [Aalborg 作者稿](https://vbn.aau.dk/en/publications/system-level-large-signal-stability-analysis-of-droop-controlled-/)

- **等效模型。** 额定电压 200 V、\(r=0.048\,\Omega\)、\(L=0.05\) mH，电容分别为 25 mF 与 6 mF。Fig. 7–8 原始初始电流与求解配置未公开。相图采用空载稳态初值，另对 70 kW 试验从各案原文最小正负荷平衡点起算。
- **四支路。** 按 Table IV 的每一行录入 \(r_k,L_k,C_k\)。表中电感下标重复写作 \(L_4\)，按对应变流器行读取。各支路电容接入同一理想母线并求和。根据式 (4)、(7) 另取并联等效量 \(r_{eq}=2/3\,\Omega\)、\(L_{eq}\approx0.54234\) mH，并单列表中舍入的 0.67 Ω 和 0.54 mH。不同支路的 \(r_k/L_k\) 并不相同，因此完整与降阶模型都运行。
- **负荷。** 正功率吸收，负功率注入。主模型使用式 (17) 的理想恒功率负荷 \(P_e/u\)。四支路的限流负荷敏感性仅在 \(u<0.9U\) 时把负荷电流改为 \(P_e/(0.9U)\)，是给定膝点的工程情景。结果单列。
- **判断范围。** 180 V、220 V 和 90 V 仅用于核验电压范围。低压终止是理想恒功率方程的数值停止条件。原台架内环、开关、寄生、限流退出和保护后的恢复时刻需要原始工程与测量数据。

原稿图 6–11 是栅格图。本轮采用表格数值、阶跃时序及可核验的分类做比较，保留全波形和参数敏感性。 [冻结参数和容差](../configs/protocol.json) · 本轮筛选记录 · [论文 PDF](https://vbn.aau.dk/ws/files/475857902/System_Level_Large_Signal_Stability_Analysis_of_Droop_Controlled_DC_Microgrids.pdf)
