# 来源与选题核对

这一轮优先检查已筛选库中公开全文、参数和数值参照都可取得的小系统。单机 VSG 论文为 TPWRS，OpenAlex 冻结快照有 184 次引用，GFM/稳定性主题排序第 6。单台变流器接无穷大母线就能检验摆动方程、内环和线路共振对稳定性的影响。

| 候选 | 当前核对结论 |
| --- | --- |
| TPWRS 单机 VSG 参数约束 | UCD 作者稿给出式 (1)–(8)、三个验证工况、Bode 和 EMT 图。关键模型及有限幅值对照可做。原开关模型和原始数据待补。 |
| TPEL PMSM MAESO-MPCC | 公开出版版表 I–IV 齐备。原速度反转的观测器带宽、机械参数、速度 PI 和比较控制器整定仍有缺口。 |
| TSTE 光伏与构网储能 SISO | Strathclyde 作者稿与附录控制增益已取得。后续可核对小信号端口模型与完整工况。 |
| TPWRS 构网电流参考饱和 | HAL 作者稿可读。本轮先处理摆动方程和线路动态，限流与故障过程未排入执行。 |

## 参数与观察层级

- 原稿 §III 给出 10 kV、50 Hz、1 MW、8.165 kV 相峰值、零初始相角、实线电阻和电感、惯量、LC 及电压/电流环 PI。算例参数取图 2–4 图注。
- 式 (2) 的内环等效时间常数在算例里没有数值。基准按电感与电流环比例增益之比取值，对正文给出的 0.5–5 ms 另作扫描。固定内环时间常数还隐含内电流 PI 的近似极零抵消。
- 算例 2 印刷值保留原样。推断分支只使用文中 414.2 rad/s 的二阶交越频率解两个正电抗根。论文的增益裕度与向量 Bode 曲线用于独立判别。
- 图 5 提供 0.5 MW 功率参考阶跃。图注未给阶跃时刻，使用算例 1 功率曲线约 0.50 s 的上升位置对齐。图中启动段、电压与无功环实现、虚拟电感内部结构、采样/PWM 时序与开关模型文件仍需原始资料。
- Bode 图与功率图数值来自 PDF 向量坐标。它们用于核对图形与文本，数据精度受印刷曲线的采样和坐标量化约束。

[TPWRS 作者稿](https://researchrepository.ucd.ie/server/api/core/bitstreams/a5bfced2-4ddf-4865-8019-53e747ab1bb1/content) · [TPEL PMSM 出版版存档](https://hdl.handle.net/11697/212481) · [TSTE 作者稿](https://strathprints.strath.ac.uk/90812/1/Zhang-etal-interaction-Modeling-and-Stability-Analysis-of-Gridforming-Energy-Storage-System-Based-on-SISO_Transfer_Functions.pdf) · [限流论文 HAL 存档](https://hal.science/hal-03966325)
