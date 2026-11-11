# 第五篇复现审计：TIE 的 PMSM 延迟抑制滑模观测器

**决定：缺关键输入，暂缓。** 公开稿可还原连续观测器、双采样补偿思路和电机参数；现有资料无法锁定 Fig. 7–10 的闭环数值实现或 Fig. 12–15 的 DSP 条件。没有用估计增益、惯用 PI 或自行选择的积分器产生替代曲线。未作论文结论失效或造假判断。

Gong、Hu、Gao、Wang、Yan，*An Improved Delay-Suppressed Sliding-Mode Observer for Sensorless Vector-Controlled PMSM*，[DOI](https://doi.org/10.1109/TIE.2019.2952824)，TIE 67(7), 5913–5923（2019 在线，2020 卷期）。依据 [White Rose 作者接收稿](https://eprints.whiterose.ac.uk/id/eprint/162707/)，与 [约克大学出版记录](https://pure.york.ac.uk/portal/en/publications/an-improved-delay-suppressed-sliding-mode-observer-for-sensorless/) 对照。

| 核对项 | 可直接提取 | 未锁定输入及后果 |
|---|---|---|
| SMO 增益 | §III Eq. (8) 为双曲正切切换函数；Fig. 7 扫描 m=1、0.25、0.1、0.01，后文选 m=0.01。Eqs. (15)–(17) 用电流误差容差 ξ 得到 min∣F∣=tanh(mξ)，据此选择 k。 | 未找到 ξ、运行实际 k 或其调度。m 是曲线形状参数；从 m 无法求出 k。四条反电势与 THD 曲线不具备唯一可执行的增益输入。 |
| PI 参数 | Fig. 6 给出外层速度环、内层 d/q 电流环和 SVPWM 结构。 | 未找到这三组比例/积分增益、积分更新、初始值、限幅和抗饱和设置。闭环速度与电流暂态会随这些量变化。 |
| 离散化 | §II–III 给出连续状态式、Eq. (18) 的反正切与 Eq. (19) 的微分；表 I 给出开关周期 100 μs。 | 周期不能确定观测器状态差分式、角度展开/速度差分、各环实际更新次序或仿真求解设置。 |
| 双采样和延迟 | §IV Eqs. (20)–(22) 描述周期起点和执行时刻双采样与电流线性外推；表 II 列出 15 个延迟值，报告平均约 34.1 μs。 | §V 将计算延迟补偿留给 DSP 实验；缺同一工况下的 ADC/PWM 时间戳、同步输入和原始波形，也未披露零斜率等边界处理。延迟均值不能替代这些输入。 |

表 I 可核对：Rs=0.6383 Ω，Ld=Lq=2 mH，p=4，J=0.013 kg·m²，B=0.0035（原表未列单位），磁链 0.085 Wb，直流母线 310 V，C′e=1.67，额定/拐点角速度 314/52.3 rad/s，额定转矩 5 N·m，额定母线电流 6 A。§V-B 报告 TMS320F28335、10 kHz IGBT 逆变器和编码器参照。

检索范围：DOI 和标题精确检索；核对 White Rose、约克大学和可访问作者版；按标题/DOI 搭配 supplement、code、data、GitHub、Zenodo、figshare、DataPort 搜索。未找到能追溯到该论文的代码或原始数据；IEEE DataPort 直接抓取受限。此结论限于上述公开入口。

恢复条件：获得可追溯的实际 k/ξ、PI 和数字更新配置后，先重放仿真部分。对延迟补偿作原实验定量比较还需要带同步时间戳的采样和参考角度/电流记录。

[来源及哈希](../sources/source_manifest.json) · Agent 记录 · [第六篇目录说明](../../2022_dmm_resonant_boost/README.md)
