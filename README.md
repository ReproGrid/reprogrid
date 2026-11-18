# ReproGrid · 论文复现库

面向电力电子与控制的独立复现项目。目录按构网控制与稳定性、PMSM 电流控制、实时仿真和数字孪生分组。每篇论文独立记录算法、证据、优点与局限、缺失信息以及 2–3 个视频条目。

[项目网站（中英文）](https://reprogrid.org)

## 本地浏览

需要 Node.js 20+ 和 Python 3.9+，网站构建与首篇试点无需第三方依赖。其余计算实验使用各自锁定的 Python 3.12 依赖。

```sh
node scripts/build.mjs
python3 -m http.server 8000 --directory dist
```

打开 <http://localhost:8000>，或访问英文入口 <http://localhost:8000/en/>。页面顶部可切换语言。两种语言共用计算结果与下载文件。

## 复跑首篇试点

```sh
python3 papers/rfpsc-linear/run.py --out /tmp/reprogrid-rfpsc
diff -r -x '._*' papers/rfpsc-linear/results /tmp/reprogrid-rfpsc
```

[RFPSC 试点说明](papers/rfpsc-linear/README.md)记录来源、实验范围和结果定义。


## 已验证的计算复现 / Verified computational reproductions

| 方向 / Track | 论文 / Paper | 证据范围 / Scope |
| --- | --- | --- |
| 构网控制 / GFM | [RFPSC](reproductions/2020_rfpsc) | 解析与平均模型 / Analytical and averaged models |
| 构网稳定 / GFM stability | [PSC transient stability](reproductions/2018_psc_transient) | 标量模型与限流边界 / Scalar model and current threshold |
| 构网稳定 / GFM stability | [Robust analytical PSC tuning](reproductions/2019_psc_robust) | 频响、附录公式与高无功边界 / Frequency response, formula checks and high reactive injection |
| 构网控制 / GFM | [DO-GFM](reproductions/2024_do_gfm) | 平均模型、PWM 对照与弱网原图差距 / Averaged model, PWM replay and weak-grid plot gap |
| 实时仿真 / Real-time simulation | [DMM](reproductions/2022_dmm_resonant_boost) | 谐振升压离线验证 / Offline resonant boost replay |
| 构网稳定 / GFM stability | [Complex droop](reproductions/2024_complex_droop) | 算例 I–III 与阻抗角 / Cases I–III and impedance angle |
| 数字孪生 / Digital twin | [Online impedance identification](reproductions/2024_tie_dt_impedance) | 六组三相平均模型 / Six averaged three-phase benchmarks |
| 构网稳定 / GFM stability | [VSG parameter constraints](reproductions/2019_vsg_parameter_constraints) | 传递函数复跑与算例 3 差异 / Transfer-function replay and Case 3 discrepancy |
| 构网稳定 / GFM stability | [VSG transient damping](reproductions/2021_vsg_transient_damping) | 电压跌落与暂态稳定增益区 / Voltage dip and transient gain region |
| 构网稳定 / GFM stability | [GFM current reference saturation](reproductions/2022_gfm_current_saturation) | 18 组全电压跌落工况、第一摆与电流角上界 / 18 full-sag cases, first-swing boundary and current-angle bound |
| 电机控制 / Motor control | [PMSM high-frequency injection](reproductions/2018_pmsm_hfi_initial_position) | 观测器、归一化和 q 轴误锁 / Observers, normalization and false q-axis lock |
| 构网稳定 / GFM stability | [Parallel VSG acceleration damping](reproductions/2021_parallel_vsg_acceleration) | 频响、极点与小负荷阶跃 / Bode response, poles and small load step |
| 构网稳定 / GFM stability | [Synchronverter equilibria](reproductions/2021_synchronverter_equilibria) | 平衡点、极点与 P–Q 稳定域 / Equilibria, poles and P–Q stability regions |
| 微网稳定 / Microgrid stability | [DC microgrid droop stability](reproductions/2021_dc_microgrid_droop_stability) | 负荷上限和有限阶跃 / Load limits and finite load steps |
| 电机控制 / Motor control | [PMSM current-constrained control](reproductions/2021_pmsm_dob_current_constraint) | 独立方程模型与表 II 差异 / Equation model and Table II discrepancy |
| 实时仿真 / Real-time simulation | [LLC direct mapping](reproductions/2021_dmm_llc_resonant_converter) | 整流模式、0.6 秒序列及离线定点核对 / Rectifier modes, 0.6 s sequence and offline fixed-point checks |
| 实时仿真 / Real-time simulation | [Grid-tied inverter EMT](reproductions/2022_grid_tied_inverter_emt_model_comparison) | 电压插值、载波边带与近谐振步长 / Voltage interpolation, carrier sidebands and time steps near resonance |

各目录包含冻结配置、开源实现、计算结果与验证记录。以上范围均为计算证据，原硬件实验未得到定量复现。

[PMSM 延时抑制滑模观测器](reproductions/2020_pmsm_delay_smo)已完成资料审计，因缺少关键输入暂缓仿真。

另外三篇资料审计记录了复跑缺口：[DC–DC 健康指标数字孪生](reproductions/2021_buck_dt_health/reports/REPORT.md)、[参数鲁棒性 PMSM 预测电流控制](reproductions/2019_pmsm_mpcc_robust/reports/REPORT.md)、[MAESO 辅助 PMSM 预测电流控制](reproductions/2023_pmsm_maeso_mpcc/reports/REPORT.md)。可在[投稿页](https://reprogrid.org/submit/)提供可分享的资料。

## 增加论文

1. 在 [候选队列](research/candidate_queue.json)按 DOI 登记元数据、状态、来源和待检验主张。
2. 在 [目录映射](research/catalog.json)补充唯一短路径、中文题名、描述与标签。研究领域由映射定义。在 [翻译数据](research/localization.json)补充对应短路径的英文信息和候选测试目标的中文说明。
3. 有运行证据时，在 `papers/<实验名>/` 添加实现、配置、结果和 `report.json`，并为论文目录项关联该实验目录。中文和英文报告里各配置同一组两到三个视频，支持本地 MP4/WebM 或外部 HTTPS 链接。大量媒体可以托管在独立视频服务，网页保留链接。
4. 重新运行构建，核对论文详情页、参数来源和视频。运行清单记录可复核的校验值与验证阈值。

[期刊配置](platform_spec.json)记录目录优先收录的期刊。原始论文、引用数据和第三方实现遵循各自来源的条款。项目自有软件使用 [MIT 许可证](LICENSE)。
