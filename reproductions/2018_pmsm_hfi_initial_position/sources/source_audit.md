# 第十三篇资料核对

本轮继续筛选电机与实时仿真的小系统。Aalborg 的 TPEL 稿同时给出锁定转子参数、观测器设计式以及可从向量图定量核对的频响与误锁例子，因此进入执行。候选库 2026-11-10 的 OpenAlex 快照记录 72 次引用。 候选库 · [仓储记录](https://vbn.aau.dk/en/publications/high-frequency-voltage-injection-methods-and-observer-design-for-/)

表 I 给出 Rs=0.961 Ω、Ld=17.8 mH、Lq=78.4 mH。第 IV 节使用锁定转子、理想开关、5 kHz PWM、10 kHz 采样和 0.4 μs 仿真步长，明确给出三种观测器的设计带宽和检测信号连续 20 ms 的阈值。图 8 指定 ζ=5 和 2π rad/s，图 9 指定 90° 电角度与零初值。这些数据足以核验观测器频响与脉冲检测器的稳态规律。

原稿未列出第 IV 节的注入幅值、直流母线、RTVI 注入频率、BPF/LPF 系数和时标对齐，也未给出实际非零初值及观测器离散更新规则。公开来源未定位到原 S-Function、开关波形或台架数据。独立闭环用 24 V 注入、540 V 母线、ζ=5、1 rad/s 估计速度初值及每组三拍一次的误差信号更新。两种计算分别使用分段平均电压和按开关事件精确积分的理想 SVPWM。0.4 μs 是论文仿真设置，独立开关实现的积分时标由开关事件决定。 [冻结配置](../configs/protocol.json)

可执行证据为式 (8)、式 (14)–(27)、图 8 及图 9 的锁轴类别。图 10–13 的逐点复跑依赖缺失的时序和观测器实现。台架的磁极极性判别需要磁饱和或实测电流响应。 [作者稿](https://vbn.aau.dk/ws/files/280258862/High_Frequency_Voltage_Injection_Methods_and_Observer_Design_for_Initial_Position_Detection_of_Permanent_Magnet_Synchronous_Machines.pdf) · [期刊 DOI](https://doi.org/10.1109/TPEL.2017.2773094)
