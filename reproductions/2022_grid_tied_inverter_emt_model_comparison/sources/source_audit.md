# 来源与参数审计

目标是 Sano、Horiuchi 和 Noda 的 IEEE TPEL 37(3) 3462–3472，DOI 10.1109/TPEL.2021.3117633。出版版文本可查阅，公式、节次和图注已核对。出版版在表 II 位置只有标题进入网页提取文本。IEEE 开放 PDF 入口返回 HTTP 418，ResearchGate 直链在本地返回 HTTP 403，浏览器显示设备验证。学校仓储仅提供元数据。已在来源清单登记复核入口。 [IEEE 条目](https://ieeexplore.ieee.org/document/9560075/) · [作者收录](https://sites.google.com/view/papers-by-noda/int-journal) · [出版版阅读页](https://www.researchgate.net/publication/358932924_Comparison_and_Selection_of_Grid-Tied_Inverter_Models_for_Accurate_and_Efficient_EMT_Simulations) · [学校仓储记录](https://cir.nii.ac.jp/crid/2120870839734640896)

| 证据 | 已确认输入 | 用途 |
|---|---|---|
| TPEL Sec. II、III-B/C | 三相两电平桥臂、串联滤波电感及阻尼电容支路、同步三角载波、式 (1) 端点电压、AV 电压跟踪参考 | 从方程实现 SW、VI 和 AV |
| TPEL Sec. IV | SW 2 µs、VI 10 µs、AV 100 µs、CCI/SCI 600 µs。XTAP 用 2S-DIRK | 固定对照步长，并单列积分器敏感性 |
| TPEL Sec. IV-C/D/E | 4.5 kHz 载波与线路谐振算例、25 µs 死区例、10% 电网五次谐波 | 复现高频边带和电网注入五次谐波的机制。死区例待补 |
| 参考文献 [3]，CIRED 2019 Fig. 1、Table 1 | 200 V 线电压、50 Hz、4.2 kVA、两侧线路各 1.0 mH、中点电容 1.2 µF | 三相小系统的线路拓扑与元件 |
| 参考文献 [3]，CIRED 2019 Fig. 2、Table 2 | 370 V 直流、1.1 mH 滤波电感、20 µF 与 4.7 Ω 串联阻尼支路、200:200 V Δ–Δ 变压器、0.17 mH 漏感、4.2 kW | 三相小系统的逆变器、滤波与变压器 |
| 参考文献 [3] 的谐波试验 | 研究载波 ± 2 倍基频，试验载波扫 4–10 kHz，6 kHz 邻近其谐振条件 | 预先选定 6 kHz 作复算诊断，并采用其拓扑扫频 |

三相小系统按 CIRED 2019 的图、表搭建，直流源固定 370 V。按平衡三线制把 Δ–Δ 变压器折到同侧，漏感与逆变器侧线路电感串联。三段串联电阻分别设为 0.1 Ω，电阻整体乘 0.5、1、2 和电容乘 0.8、1、1.2 分开核验。在 50 Hz 连续相量下求 4.2 kW、零无功的桥端指令，之后固定给所有模型和扫频。指令经过同步保持后会出现相位滞后，实际稳态有功另行报告。 [CIRED 原始论文及参数表](https://www.cired-repository.org/bitstreams/993c93d6-b75c-4cdf-9845-a58762162e44/download) · [冻结协议](../configs/protocol.json)

原文 Fig. 11–15 的闭环对照待取得表 II 数值、原 XTAP 初态和开关/采样实现、PI 与 PLL 配置及原始数据。Fig. 12 给的是比例增益倍数，Fig. 14 的 25 µs 是默认死区五倍。原文 Fig. 13 的 4.5 kHz 线路参数未锁定，因此本复算把 4.5 kHz 作为同载波测试，把 CIRED 台架 6 kHz 作为近谐振测试。CCI 和 SCI 的闭环或原始注流条件未进入本次数值比较。 [分项缺口](../reports/status.json)
