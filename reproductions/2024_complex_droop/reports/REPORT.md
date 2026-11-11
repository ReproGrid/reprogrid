# 第七篇计算复现 · 复杂下垂控制的稳定条件

复现了单台并网变流器的算例 I–III。在 $r_g=0.08$ pu、$\alpha=1$ 时，四阶模型首次失稳的 $\eta/\omega_0=0.09977586$。两套实现通过 18/18 项检查，干净环境重跑的 51 个数组逐元素一致。

He 等，*Quantitative Stability Conditions for Grid-Forming Converters With Complex Droop Control*，IEEE TPEL 39(9), 10834–10852，2024。计算和公式核对基于与期刊 DOI 对应的 arXiv v3 作者稿。[DOI](https://doi.org/10.1109/TPEL.2024.3404251) · [作者稿](https://arxiv.org/abs/2310.09933v3)

## 电压跌落和模型阶次

按 Eqs. (15)–(22) 建立 2、4、8、12 阶模型。主实现在网同步坐标下组装复数矩阵并用 Radau 积分。独立实现由电路和控制器的实数两轴方程组成，用 BDF 积分。最大平衡残差 9.02e-13 pu/s，Jacobians 的最大归一化差 9.33e-11。十五条跌落轨迹的全状态最大绝对差 1.56e-06 pu。[数值结果](../results/case_metrics.json) · [独立推导](../verification/derivation.md)

Fig. 4 中网压在 0.5 s 从 1 pu 降至 0.5 pu。$\eta/\omega_0$ 从 0.02 提高到 0.06 时，二阶与十二阶的最大参考电压矢量差从 0.0518 增至 0.1215 pu。四种阶次、两个增益和两个电压分量的作者图对照中，最大 RMSE 为 0.000678 pu。[阶次对照图](../figures/model_order_voltage_dip.png)

## 四阶模型稳定边界

对每个 $\alpha$ 求跌落后的平衡点及四阶 Jacobian 首次极点交越。用 Schur 消元和 Routh–Hurwitz 独立求得的交越值与直接极点法最大相差 7.88e-15。下表为 $\alpha=1$ 的无量纲 $\eta/\omega_0$，原图数值来自坐标轴标定的矢量路径。

| $r_g$ (pu) | 极点交越 | Fig. 5 仿真边界 | Eq. (26a,b) 充分上确界 | Fig. 5 解析边界 |
|---:|---:|---:|---:|---:|
| 0.08 | 0.099776 | 0.100000 | 0.037590 | 0.035832 |
| 0.20 | 0.479060 | 0.476214 | 0.111489 | 0.102922 |

Eq. (26a,b) 的上确界采用 $c_1$ 从下方趋近 Eq. (23) 的裕量、$\epsilon$ 从上方趋近 3。原作者图的具体 $c_1$ 和 $\epsilon$ 未披露。$r_g=0.20$ 时的数值交越随 $\alpha$ 非单调，网格上的最大值出现在 $\alpha=3.45$。

在 Fig. 6 的 0.099 和 0.101 工况，四阶主导极点实部分别为 -0.722 和 +1.132 s⁻¹，跌落后分别出现振荡衰减和增长。二阶模型在这两个工况均稳定。原图最初 0.20 s 对照的最大 RMSE 为 0.0104 pu。作者图的后段触及绘图区底边，因此该窗口用于定量比较。[稳定边界和响应图](../figures/stability_boundary_and_response.png) · [图形读回指标](../results/published_curve_comparison.json)

## 极限环和平衡点

算例 III 的跌落前条件有三个平衡点，二阶和十二阶模型都只有 $0.59537+j0.81516$ pu 这一支局部稳定，也与 Fig. 7 标出的初值相符。首次运行因预设平衡点唯一而停下。协议已明确按极点选择跌落前的稳定分支。

跌落后，$\alpha=3$ 时二阶唯一平衡点的主导极点实部为 59.318 s⁻¹。25 个初值的二阶轨迹收敛到同一周期轨道，二阶与十二阶的周期分别为 0.48166 s 和 0.46270 s。二阶跌落后最大电压 1.00943 pu，位于理论上界 1.06837 pu 内。Fig. 7 的二阶与十二阶电压幅值 RMSE 分别为 0.000292 和 0.000337 pu。

$\alpha=1$ 时二阶平衡点局部稳定。25 个初值在 3 s 后的最大距离为 2.19e-07 pu，Eq. (23) 的充分条件裕量为 -0.3559，因而该条件未给出证书。[相图与周期](../figures/limit_cycle_and_phase_portraits.png) · [Fig. 7 原时标对照](../figures/fig7_published_curve_comparison.png)

## 作者稿中影响复现的细节

- 算例 I 和 III 印作 $\varphi=\arctan(l_g/r_g)$，同时将 $l_g$ 以 s/rad 给出。主结果采用 Table III 的阻抗角 $\arg(r_g+j\omega_g l_g)$，Fig. 5(b) 随 $r_g$ 重算角度。按印刷式字面代入时，Fig. 5(a) 的 $\alpha=1$ 交越变为 0.163049，算例 III 的 $\alpha=1$ 主导实部变为 +24.057 s⁻¹，$\alpha=3$ 的上界变为 1.16279 pu。[角度敏感性](../results/rotation_angle_diagnostic.json)
- 命题 4 的判别式“当且仅当”需补充三重根。显式构造的正电压三次式为 $(s-1)^3$，判别式为 0，只有一个不同的正根，对应唯一平衡点。[构造与推导](../verification/derivation.md)
- 命题 5 的物理时间特征多项式常数项缺少 $\eta^2$。算例 III 的直接 Jacobian 行列式为 3829.36，印刷项为 6.06243，乘回因子后与行列式一致。因子为正，后续行列式符号判据仍成立。[代数证据](../results/algebra_evidence.json)
- Appendix A 的经典下垂反例也可复核。其四次稳态残差在非负电压域的最小值为 0.124685，所以该组参数无经典下垂平衡点。复杂下垂稳态的最大方程残差为 3.61e-16。[代数证据](../results/algebra_evidence.json)

## 数值复核范围

$\eta/\omega_0=0.101$ 的四阶工况在延长到 5 s 的比较中最敏感。收紧 BDF 容差把全状态差从 1.03e-05 降至 1.56e-06 pu。协议保留 1e-05 pu 的通过门槛和两次运行记录。计算范围为连续平均模型与算例 I–III。原始实验测量和控制采样配置未取得，限流算例、Eq. (26c,d) 及完整吸引域需要单独验证。

[复跑命令](../README.md) · [来源核对](../sources/source_audit.md) · [主运行清单](../results/run_manifest.json) · [干净环境校验](../results/clean_replay_comparison.json)
