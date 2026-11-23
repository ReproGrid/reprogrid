# 原文参数、复现层级与缺口

- 第 24 篇采用 Aalborg 可访问的 IEEE 正刊 PDF。OpenAlex 2026-11-10 快照记为 464 引，归一化引用百分位为 0.9977。计算对象是一台电流控制变流器经 R–L 支路接电网。[论文](https://vbn.aau.dk/ws/files/337054418/08820019.pdf)
- Fig. 2 理想电流源配 `Rline=0.1 pu`、`Xline=0.28 pu`，50 Hz。电压与电流按 Table I、II 和 III-A 的定义。正常运行注入 `(Id,Iq)=(1,0)`，严重跌落注入 `(0,−1)`。初态从式 (7) 求得，频率误差及积分初值为零。Fig. 1 的滤波电感 0.096 pu 属于完整开关系统，Fig. 2 降阶对象已取输出电流受控。[原始 PDF](https://vbn.aau.dk/ws/files/337054418/08820019.pdf) · [冻结协议](../configs/protocol.json)
- Fig. 13 按式 (6) 的 PI 状态形式和式 (11) 的相角速度形式独立积分。以弧度作为相角单位，速度除以 2π 后与原图的 Hz 纵轴比较。对 Case III 单独绘制比例模式。Fig. 12、15、16 的阈值切换另外用 45–55 Hz 有界的采样模型执行。[代码](../src/model.py) · [图形读取](../verification/read_figures.py)
- Fig. 12 从限幅前的频率取绝对值再求导和低通。式 (15) 和文字又以 ROCOF 幅值描述触发。基线采用框图顺序和带符号滤波后取幅值，另测先对频率导数取幅值再滤波。积分器切换具体实现未给出，分别核对清零与保持存量但暂时旁路。100 µs 是重建采样间隔，50 µs 为精化。原文的 5 / 0.5 Hz/s 阈值和 200 ms 滤波保持作为基线。也单列以角频率单位解释同一门槛数字的诊断。[冻结协议](../configs/protocol.json)
- Fig. 15–16 的 2.5–3.1 s 滤去故障前恒定时段，保留 600 ms 事件和恢复后轨迹。基线设零网压相位跳变、瞬时电流指令切换。另算 2、5、10 ms 线性电流过渡，并保留支路的 `Lline di/dt`。测得的实际电流过渡和输出频率限幅的抗饱和实现仍缺。[结果与事件记录](../results/metrics.json)
- Table III 的实验基值为 GCP 110 V RMS、3.6 kW、50 Hz，表中 R / L 标幺值与 Table I 一致。实验的控制固件、PLL 同步输入、电网相位和原始记录仍缺。对称故障以外的序分量提取、开关级 PLECS 波形及实验未纳入本次定量核验。[作者机构记录](https://vbn.aau.dk/en/publications/design-oriented-transient-stability-analysis-of-pll-synchronized-/)
