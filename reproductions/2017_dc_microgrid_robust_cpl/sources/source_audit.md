# 来源与建模口径

- 主来源是作者公开手稿。Fig. 3 给出 9 个带源与负载的母线和 9 条纯电阻支路。Fig. 6 给出 8 个母线、3 个源、5 路负载和 9 条纯电阻支路。Table I 的标题写了五母线，§IV.A 直接将其用于 Fig. 3 的九母线例。逐条支路与元件的 SI 数值见配置文件。[作者稿](https://arxiv.org/pdf/1702.07580) · [固定配置](../configs/protocol.json)
- 电路以式 (1)–(3) 为准。式 (4) 下方的组装说明将源参考系数写作 `Bkk=1`，并在母线耦合系数中省略了 `1/Cb`。直接从支路方程得 `1/Ls` 和 `1/(Rline Cb)`。每个母线节点有一个等效 `Cb`。负荷增量项按支路微分得到 `p/(Cl v_l²)`。支路独立实现、有限差分与功率平衡结果保留在数值记录。[作者稿](https://arxiv.org/pdf/1702.07580) · [电路实现](../src/network.py) · [数值记录](../results/metrics.json)
- 图 4 横轴到 25.672 s，图上的红线最后一档是 17 kW。图 5 最后一档是 20 kW。图 7 在 6 s 施加 17 kW。图 8 从 2 s 显示，最后一档为 20 kW。图 7–8 的坐标和图例标的是母线 7 所接 CPL 的端电压，图注写成母线电压。分别保留端电压和母线电压数值。阶跃时刻取图中标示的秒数并用矢量线读取误差核对。[矢量读取脚本](../src/read_paper_figures.py) · [读取结果](../results/pdf_vector_readback.json)
- 图 4–5 使用原文统一负荷。图 7–8 每个阶跃也对五路负荷同时施加，起点由首档负荷的高压平衡点确定。各路原始命令和初态缺失。图 8 末档的五路各 20 kW 在正文直接写明，可以独立复算末档平衡点和极点。§IV.A 分别提到 `d=0.06` 和 `d=0.10 Ω`，因此这两组都执行了判据和物理平衡点计算。[作者稿](https://arxiv.org/pdf/1702.07580)
- 八母线 Table II 的式 (7) 按 `[0, δ上限]` 核对。作者段落说明以 1 s⁻¹ 增量试探上限，未提供该表的具体下限或停机容差。式 (9)、(10) 使用公式中的绝对上界，其可行性与选定下限无关。该表的硬件耗时依赖 MATLAB CVX、求解器与主机环境，本报告另列实际运行耗时。[作者稿](https://arxiv.org/pdf/1702.07580)
- 下界敏感性采用 §IV.A 运行区间推得的 `δ下界=36.895 s⁻¹`。在此单列假设下，以相同拓扑复算 Table II 边界，1 Ω 对 220 s⁻¹ 仍无共同二次证书，0.3 Ω 对 170 s⁻¹ 可以获得证书。主表仍采用从 0 开始的区间。[脚本](../verification/lower_delta_sensitivity.py) · [结果与证书](../verification/lower_delta_sensitivity.json)
- 第一作者博士论文第 2 章的 Fig. 2.7 同样注明 440 V，Table 2.2 同样列 220、51、201 s⁻¹。检索 DOI、作者公开页面、arXiv、学位论文与 GitHub 后，能获得图和参数表，原 Simulink、CVX 和原始输出文件尚未找到。[作者论文列表](https://www.wzhanglab.site/publications/) · [博士论文](https://etd.ohiolink.edu/acprod/odb_etd/ws/send_file/send?accession=osu1512049527948171&disposition=inline)
