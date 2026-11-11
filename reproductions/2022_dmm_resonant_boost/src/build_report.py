"""Build the evidence report and its status from verified reproduction outputs."""
from hashlib import sha256
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def digest(name):
    return sha256((ROOT/name).read_bytes()).hexdigest()


def main():
    p = read('configs/protocol.json')
    m = read('results/metrics.json')
    run = read('results/run_manifest.json')
    clean = read('results/clean_replay_comparison.json')
    topology = read('results/topology_summary.json')
    probes = read('results/region_probes.json')
    fig = read('results/fig314_comparison.json')
    ref = read('results/step_refinement.json')
    window = read('results/fig314_step_diagnostic.json')['per_step_ns']
    assert run['passed'] and clean['passed'] and run['protocol_sha256']==digest('configs/protocol.json')
    assert all(digest(f'results/{name}')==expected for name,expected in run['outputs'].items())
    assert all(digest(name)==expected for name,expected in run['code_sha256'].items())
    modes = sorted({x['modes'] for x in topology.values()})
    w = sorted({tuple(x['W_shape']) for x in topology.values()})
    plus = sorted({x['positive_output_mode_count'] for x in topology.values()})
    assert len(modes)==len(w)==len(plus)==1, 'Report expects a common cardinality across gate pairs.'
    wstr = '×'.join(map(str,w[0]))
    fstr = '×'.join(map(str,topology['0']['F_shape']))
    visited = len(m['visited_diode_masks'])
    max_err = max(m['trajectory_relative_l2'])
    dot_err = fig['replay_vs_published_dmm_at_dots_relative_l2']
    curve_err = fig['replay_vs_published_emtp_curve_relative_l2']
    count = sum(run['checks'].values())
    rel_table = '\n'.join(f"| {s} | {e:.3e} | {x:.3e} |" for s,e,x in
                          zip(m['state_order'],m['trajectory_relative_l2'],m['trajectory_max_abs_error']))
    r = m['network_residuals']
    rstat = run['elapsed_seconds']
    fine = ref['published_edges']['pairwise_relative_l2']
    same = ref['fixed_edges']['pairwise_relative_l2']
    diagnostic = '、'.join(f"{int(s)} ns：{window[s]['published_dmm_marker_relative_l2']:.2%}"
                         for s in map(str,p['refinement']['steps_ns']))
    report = f"""# 第六篇计算复现：DMM 谐振升压变流器

**结论：支持直接映射求解在冻结电阻开关模型上的软件等价性。** 100 ns 下完成 {m['time_steps']:,} 步，五个状态相对 L₂ 误差最大 **{max_err:.3g}**。原文 **10 个可行组合、14×6 判别矩阵**没有在所声明的全输入域中重现；实际 PWM 轨迹访问了 **{visited} 个**二极管位型。原始 FPGA 指标未复现。

Chalangar、Ould-Bachir、Sheshyekani、Mahseredjian，*Methods for the Accurate Real-Time Simulation of High-Frequency Power Converters*，[DOI](https://doi.org/{p['doi']})，TIE 69(9), 9613–9623（2021 在线，2022 卷期）。依据[论文作者版](https://www.researchgate.net/publication/355029015_Methods_for_the_Accurate_Real-Time_Simulation_of_High-Frequency_Power_Converters) §II–III、§VI-C 及[第一作者学位论文](https://publications.polymtl.ca/9473/) §3.6，Fig. 3.12–3.14。

## 冻结的输入

原图：150 V 输入、100 kHz；Lg=1 mH、Lr=32 μH、Cs=0.32 nF、Cr=3.2 nF、Co=0.1 mF、RL=250 Ω。学位论文的 PWM：S1 导通区间为 0.54–5.00 μs，S2 为 0–1.43 μs。100 ns 网格的首个不早于原边沿的时点分别是 0.60/1.50 μs。各支路极性见[协议](../configs/protocol.json)。

RBC 专用导通、关断阻值和初始化未披露；本次采用原文 EVBC 表 II 的 1 mΩ/100 kΩ、初始储能与历史量为零、第一步施加输入源。所有六个半导体均用有限两态电阻。该假设冻结于主运行前，未用原图曲线拟合。[来源审计](../sources/source_audit.md)

## 同一步长的独立对照

主实现用精确有理数消元生成 W 与 F，在线只做点积和查表。独立参考用含支路电流的 tableau 系统直接解 KCL 和本构方程，按二极管压差迭代换态；参考不读取主实现的映射。两套方法各自推进历史量，共用电路、开关和步长输入。

| 状态 | 相对 L₂ 误差（无量纲） | 最大绝对误差（A 或 V） |
|---|---:|---:|
{rel_table}

- {probes['probes_total']:,} 个状态探针含 {probes['strict_constructed_witnesses']} 个严格见证，远离换向边界的位型全部一致。主轨迹有 {m['trajectory_mask_mismatches_near_transition']} 个位型差异，全部在换向边界附近；这些点的最小二极管压差绝对值最高为 {m['min_diode_voltage_at_mask_mismatches_max_v']:.3g} V。
- 归一化 KCL 残差 {r['max_relative_kcl']:.3g}，梯形残差 {r['max_relative_trapezoid_residual']:.3g}。DMM 在线线性求解 0 次；参考共 {m['reference_linear_solves_total']:,} 次，单步最多 {m['reference_linear_solves_per_step_max']} 次。
- 主数值检查 {count}/{len(run['checks'])} 通过。独立 Python 环境的数组逐元素相同，映射表、误差和步长精化指标一致。主运行中 DMM 循环 {rstat['dmm_steps']:.2f} s、参考 {rstat['tableau_steps']:.2f} s；这是主机计时，仿真时间步为 100 ns。

## 原文映射规模核对

| 统计域 | 二极管可行或访问位型 | W / F |
|---|---|---|
| 论文 §VI-C 报告 | 10 个可行组合 | 14×6 / 5×6 |
| 本次全历史量域，每个固定门极组合 | {modes[0]} 个严格可行位型 | {wstr} / {fstr} |
| 加 v_o>0，每个固定门极组合 | {plus[0]} 个严格可行位型 | 未按该子域编译 W |
| 本次 38 ms PWM 轨迹 | {visited} 个不同二极管位型，跨门极合计 {m['visited_gate_diode_pairs']} 个门极/二极管配对 | 只统计访问 |

可行域差异可独立重建：给定任何四个二极管压差，可从 Fig. 3.12 的连线反解四个自由节点电压。Lg、Lr、Cr、Co 的四列历史电流注入满秩，可在固定 Vg=150 V 时让这些电压满足 KCL。要求 v_o>0 只排除 D1 和 D3 同时导通的四种组合。精确消元和独立 tableau 见证都支持这一区分。实际轨迹访问数恰好是 10；原文没有公开用于得到 10 个组合和 14 行的域约束或映射表，无法确认其计数口径。[推导](../verification/derivation.md) · [见证数据](../results/region_probes.json)

## 原图与步长

从学位论文 Fig. 3.14 读回 {fig['dmm_markers']} 个 DMM 散点及 EMTP 参考线，按标注时刻对齐，不拟合相位或幅值。100 ns 重放的电流极值 {fig['replay_min_max_a'][0]:.3f}/{fig['replay_min_max_a'][1]:.3f} A；论文散点 {fig['published_dmm_min_max_a'][0]:.3f}/{fig['published_dmm_min_max_a'][1]:.3f} A。峰值接近；按散点和参考线计算的相对 L₂ 差距分别是 **{dot_err:.2%}** 和 **{curve_err:.2%}**。这是图上读数对照，不能代替 Table VII 的硬件与原始参考信号比较。

前 2 ms 的 100→50 ns 精化中，谐振电流相对差 {fine['100ns_vs_50ns'][1]:.2%}；固定两条门极边沿为 100 ns 网格的实际时点后为 {same['100ns_vs_50ns'][1]:.2%}。缩小步长时同时改变开关实际触发时点会影响波形。延长到原图时窗的各步长散点误差：{diagnostic}。这些结果保留同一初始条件和门极取样规则，未定位原图差异的唯一原因。

## 交付与未验证范围

原 EMTP 工程、RBC 专用开关阻值、原始初值/边沿处理、裁剪后的 W/TCAM 以及 FPGA 定点缩放、舍入、逻辑和原始测试波形未取得。未验证 FPGA 100 ns 实时期限、Table VII 的五项硬件误差、EVBC 或 NTT/SDD。上述可复现性差异不提供造假证据。

[复跑说明](../README.md) · [主运行清单](../results/run_manifest.json) · [独立环境核对](../results/clean_replay_comparison.json) · [五状态波形](../figures/rbc_waveforms.png) · [原图对照](../figures/fig314_comparison.png) · [步长精化](../results/step_refinement.json) · [原图时窗精化](../results/fig314_step_diagnostic.json) · [Agent 耗时](agent_timing.json)
"""
    (ROOT/'reports/REPORT.md').write_text(report)
    status = {
        'doi':p['doi'],'state':'report_ready',
        'outcome':'offline_DMM_equivalence_supported_with_map_size_and_published_trace_gaps',
        'scope':'Published resonant-boost circuit and thesis PWM; float64 trapezoidal RSM, FME sign mapping, independent iterative tableau, clean replay, and published plot readback.',
        'scope_zh':'谐振升压 DMM 的离线等价性、可行域规模、步长精化及原图对照；未验证 FPGA 实时期限和 EVBC。',
        'code_executed':True,'all_paper_claims_reproduced':False,'hardware_validation':False,
        'quantitative_original_experiment_match':False,'clean_environment_replay_passed':True,
        'cross_implementation_passed':True,'checks_passed':count,'checks_total':len(run['checks']),
        'key_results':{'step_ns':m['step_ns'],'steps':m['time_steps'],'max_same_step_relative_l2':max_err,
                       'full_history_domain_masks_per_gate':modes[0],'positive_output_masks_per_gate':plus[0],
                       'visited_diode_masks':visited,'W_shape_per_gate':list(w[0]),'F_shape':topology['0']['F_shape'],
                       'published_plot_marker_relative_l2':dot_err},
        'critical_gaps':[
            'No RBC-specific switch resistance, initialization, event quantization or complete EMTP case.',
            'The published 10 feasible combinations and 14×6 W are not reproduced over the stated unrestricted history domain. Author RBC domain restrictions or lookup tables were not available.',
            'The 100 ns replay reproduces the main shape but differs quantitatively from thesis Fig. 3.14; only plot readback is available.',
            'Original FPGA timing, fixed-point accuracy and resources, and EVBC/NTT/SDD claims are outside verified scope.'
        ],
        'anomaly_evidence':[
            {'type':'reported_map_size_not_recovered_under_stated_domain',
             'source':'Section VI-C and thesis Section 3.6; 64 independent witnesses at Vg=150 V',
             'finding':f'{modes[0]} open regions and {wstr} W per gate for unrestricted histories; {plus[0]} with vo>0. The chosen trajectory visits {visited} masks; the author scope may differ.'},
            {'type':'published_trace_quantitative_gap',
             'source':'Chalangar thesis Fig. 3.14 vector plot',
             'finding':f'At the reported time coordinates the 100 ns replay differs by {dot_err:.2%} relative L2 at the DMM markers; peaks are close. Input and discretization gaps prevent attribution.'}
        ],
        'misconduct_allegation':None,'report_sha256':digest('reports/REPORT.md'),
        'primary_manifest_sha256':digest('results/run_manifest.json')
    }
    (ROOT/'reports/status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
