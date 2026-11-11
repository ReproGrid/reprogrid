"""Cross-check both implementations and build the report from their recorded data."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def write(path, obj):
    (ROOT / path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def compare_independent(run, verification, cfg):
    threshold = cfg["numerics"]["cross_implementation_scalar_tolerance"]
    if run["source_sha256"] != verification["source_sha256"]:
        raise ValueError("The two implementations used different source manuscripts")
    cases = {"type_i_line_open": "type_i_line_open", "type_ii_uncleared": "type_ii_uncleared",
             "type_ii_early": "type_ii_clear_0p5", "type_ii_late": "type_ii_clear_0p7"}
    comparisons = {}
    with np.load(ROOT / "verification/trajectories.npz", allow_pickle=False) as independent:
        for branch, verify_name in (("table_pu", "stated_pu"), ("table_si", "table_SI_simulation")):
            original = run["branches"][branch]
            verified = verification["branches"][verify_name]
            type_i = verified["typeI"]
            type_ii = verified["typeII"]
            values = {
                "type_i_pre_deg": type_i["initial_angle_deg"],
                "type_i_post_deg": type_i["post_sep_angle_deg"],
                "type_ii_cca_deg": type_ii["CCA_deg"],
                "type_ii_cct_s": type_ii["CCT_s"],
                "fault_slip_period_s": type_ii["uncleared_slip_period_s"],
            }
            scalar_diff = {k: abs(original[k] - value) for k, value in values.items()}
            trajectories = {}
            for name, verify_case in cases.items():
                case = original["cases"][name]
                summary = {"basin_agrees": True, "current_limit_event_agrees": True}
                if name.startswith("type_ii"):
                    case_key = name.rsplit("_", 1)[-1]
                    independent_case = type_ii["cases"][case_key]
                    summary["basin_agrees"] = case["basin_index"] == independent_case["attractor_basin_index"]
                    for k, v in (("clear_angle_deg", "clear_angle_deg"),
                                 ("first_current_limit_crossing_s", "first_current_limit_absolute_s")):
                        a, b = case[k], independent_case[v]
                        summary[k + "_error"] = abs(a-b) if a is not None and b is not None else None
                        if (a is None) != (b is None):
                            raise ValueError(f"Event mismatch in {branch}/{name}/{k}")
                        if a is not None:
                            scalar_diff[f"{name}_{k}"] = abs(a-b)
                with np.load(ROOT / "results" / case["npz"], allow_pickle=False) as primary:
                    if not np.array_equal(primary["time_s"], independent["time_s"]):
                        raise ValueError("Sample times differ")
                    summary["samples"] = len(primary["time_s"])
                    summary["fields"] = {}
                    for field in ("angle_rad", "power_pu", "current_pu"):
                        target = independent[f"{verify_name}_{verify_case}_{field}"]
                        if target.shape != primary[field].shape:
                            raise ValueError(f"Different trajectory shape for {field}")
                        summary["fields"][field] = float(np.max(np.abs(target - primary[field])))
                trajectories[name] = summary
            comparisons[branch] = {"scalar_absolute_differences": scalar_diff, "trajectories": trajectories}
    waveform_errors = {field: max(t["fields"][field] for b in comparisons.values() for t in b["trajectories"].values())
                       for field in ("angle_rad", "power_pu", "current_pu")}
    max_scalar_error = max(v for b in comparisons.values() for v in b["scalar_absolute_differences"].values())
    all_basins_agree = all(t["basin_agrees"] for b in comparisons.values() for t in b["trajectories"].values())
    return {"tolerance": threshold, "cases": 8, "samples_per_case": 5001,
            "max_waveform_error": waveform_errors, "max_scalar_error": max_scalar_error,
            "branches": comparisons, "source_sha256": run["source_sha256"],
            "primary_manifest_sha256": digest("results/run_manifest.json"),
            "verification_manifest_sha256": digest("verification/results.json"),
            "verification_trajectory_sha256": digest("verification/trajectories.npz"),
            "passed": max(max(waveform_errors.values()), max_scalar_error) <= threshold and all_basins_agree}


def main():
    run = load("results/run_manifest.json")
    verification = load("verification/results.json")
    replay = load("results/clean_replay_comparison.json")
    cfg = load("configs/protocol.json")
    cross = compare_independent(run, verification, cfg)
    write("results/cross_implementation.json", cross)
    primary = run["branches"]["table_pu"]
    gates = [run["numerical_checks_passed"], all(primary["printed_target_matches"].values()),
             verification["all_internal_numeric_checks_passed"], verification["all_reported_main_targets_passed"],
             cross["passed"], replay["passed"], run["protocol_sha256"] == digest("configs/protocol.json"),
             run["implementation_sha256"] == digest("src/reproduce.py"),
             verification["implementation_sha256"] == digest("verification/verifier.py"),
             verification["protocol_sha256"] == digest("verification/protocol.json")]
    if not all(gates):
        raise RuntimeError("Evidence checks failed. Inspect result manifests and cross_implementation.json.")
    si = run["branches"]["table_si"]
    tests = sum(len(b) for b in run["checks"].values())
    case_rows = []
    for name, label in (("type_i_line_open", "Type-I 断线"), ("type_ii_uncleared", "Type-II 故障持续"),
                        ("type_ii_early", "Type-II 0.5 s 后清除"), ("type_ii_late", "Type-II 0.7 s 后清除")):
        case = primary["cases"][name]
        clearing = f'{case["clear_angle_deg"]:.3f}°' if case["clear_angle_deg"] is not None else "—"
        first = f'{case["first_current_limit_crossing_s"]:.6f} s' if case["first_current_limit_crossing_s"] is not None else "观测窗内未达到"
        basin = {None: "持续相位漂移", 0: "收敛至原吸引区", 1: "滑移一周后再同步"}[case["basin_index"]]
        case_rows.append(f'| {label} | {clearing} | {basin} | {case["sampled_peak_current_pu"]:.3f} | {first} |')
    case_table = "\n".join(case_rows)
    max_refinement = max(v for b in run["branches"].values() for c in b["cases"].values()
                         for v in c["max_refinement_differences"].values())
    late = primary["cases"]["type_ii_late"]
    uncleared = primary["cases"]["type_ii_uncleared"]
    period = primary["fault_slip_period_s"]
    experimental_si = verification["branches"]["table_SI_experiment"]["typeII"]
    sensitivity = "\n".join(f'| {p["Xgnd_pu"]:.2f} | {p["CCA_deg"]:.3f}° | {p["CCT_s"]:.6f} s |'
                            for p in verification["fault_severity_sensitivity"])
    text = f"""# PSC 暂态稳定分析的计算复现

Heng Wu 与 Xiongfei Wang，*Design-Oriented Transient Stability Analysis of Grid-Connected Converters With Power Synchronization Control*，IEEE Transactions on Industrial Electronics。[DOI](https://doi.org/10.1109/TIE.2018.2875669) · [作者接收稿](https://arxiv.org/abs/1804.09394v2)

单变流器接无穷大母线的一阶 PSC 模型复现了 Type-I 的单调收敛和 Type-II 的临界角、临界时间及吸引区变化。用表 I 标幺电抗和 SI 增益得到 **{primary["type_ii_cca_deg"]:.3f}°、{primary["type_ii_cct_s"]:.6f} s**。0.7 s 算例在标量模型中滑移一周后再同步，但在清除时已经跨过 1.8 pu 电流阈值，完整限流系统的再同步仍未验证。

## 参数与方法

式 (5) 在标幺功率和物理秒下写为 `dδ/dt = 9.3 × (Pref − Pe(δ))`。主实现按 Fig. 1、Fig. 3 建立节点导纳，内部发送节点通过 KCL 消元。故障期间线路 2 仍连接电网，发送端接入接地支路。清除时一起移除线路 2 与接地支路。电流取完整网络的 PCC 注入电流，功率由电压与电流相量计算。Fig. 9/10 的扰动起点约为 3 s，按文中持续时间设置 3.5 s 和 3.7 s 清除。

表 I 的 220 kV 按相电压 RMS 理解，与标幺电感列及式 (2) 的相峰值口径一致。对应的电流基准为 {run["current_base_RMS_A"]:.3f} A RMS。主分支直接用标幺电抗，另用表中舍入的 SI 电感按 50 Hz 换算复算。两支均使用 SI 增益 9.3×10⁻⁹ rad/(W·s)。表中 0.01 pu 的增益与按常用三相功率和基频归一化得到的 0.02960 不同，原文未说明增益基准。

| 指标 | 论文报告值 | 表 I 标幺电抗 | 表 I 舍入 SI 电感 |
|---|---:|---:|---:|
| Type-I 断线前稳定角 | 26.4° | {primary["type_i_pre_deg"]:.4f}° | {si["type_i_pre_deg"]:.4f}° |
| Type-I 断线后稳定角 | 60.5° | {primary["type_i_post_deg"]:.4f}° | {si["type_i_post_deg"]:.4f}° |
| Type-II 临界清除角 | 108° | {primary["type_ii_cca_deg"]:.4f}° | {si["type_ii_cca_deg"]:.4f}° |
| Type-II 临界清除时间 | 0.58 s | {primary["type_ii_cct_s"]:.6f} s | {si["type_ii_cct_s"]:.6f} s |

## 轨迹与适用边界

下表使用标幺电抗。清除角来自连续的未折叠相角。首次达到阈值时刻取绝对仿真时间，含拓扑切换后的电流跳变。

| 场景 | 清除角 | 理想 PSC 模型行为 | 采样峰值电流 (pu) | 首次达到或跨过 1.8 pu |
|---|---:|---|---:|---:|
{case_table}

Type-I 由 {primary["type_i_pre_deg"]:.3f}° 单调收敛到 {primary["type_i_post_deg"]:.3f}°。Type-II 早清除回到 {primary["type_ii_post_deg"]:.3f}°。晚清除跨过原吸引区边界，走向 {late["target_angle_deg"]:.3f}°，即同一电气平衡点加 360°。恰在临界角时，精确标量解停在不稳定平衡点。超过临界时间后，是否还能再同步取决于故障后的网络和控制模式。

故障持续时，标量模型的角速度始终为正，完整滑移周期为 {period:.6f} s，首次越过 2π 在 {uncleared["first_2pi_crossing_s"]:.4f} s。作者稿 Fig. 10(a) 读图约在 4.0、5.1、6.3 s 回绕，间隔约 1.15 s。这个图示时间差可用来定位后续审查，原始波形和控制器配置不足以确定成因。两套计算同样在持续故障的首次回绕前检测到 1.8 pu 越界。

0.7 s 算例在 {late["first_current_limit_crossing_s"]:.1f} s 清除时立即跨过阈值，随后达到约 {late["sampled_peak_current_pu"]:.3f} pu。Fig. 4 描述了达到限流后切入矢量电流控制与 PLL。原始限流检测基准、滤波、切换逻辑和详细内环未给足，不能从本次标量轨迹确认其实际切换时刻或切换后的行为。重合闸、完整 EMT 波形及硬件数据的定量复放也因条件不足跳过。

![重建相图](../figures/phase_portraits.png)

![标幺参数工况的相角、功率和电流](../figures/trajectories_table_pu.png)

![舍入 SI 电感工况](../figures/trajectories_table_si.png)

## 独立核对与复跑

- 主实现通过 **{tests} 项**数值检查。DOP853 与 Radau 的 8 组轨迹最大绝对差为 {max_refinement:.3g}（在各自单位下）。用数值事件与定积分分别求 CCT，两者一致。
- 独立验证由戴维南等效推导网络，再用半角代换后的闭式流、定积分及求根计算。两套实现按同一时间网格比较每个工况 **{cross["samples_per_case"]:,} 点**，包含拓扑切换时刻。相角、功率、电流最大差分别为 {cross["max_waveform_error"]["angle_rad"]:.3g} rad、{cross["max_waveform_error"]["power_pu"]:.3g} pu、{cross["max_waveform_error"]["current_pu"]:.3g} pu。
- 新建 Python 虚拟环境，按相同版本重跑 8 组数据，数组最大差为 {replay["array_max_difference"]:.3g}。本次复跑使用同一台机器。

独立验证额外改变故障接地电抗，保持清除后的拓扑不变。下表是额外敏感性计算。临界角由清除后曲线决定，临界时间还依赖故障期间的角速度。

| 接地电抗 (pu) | CCA | CCT |
|---:|---:|---:|
{sensitivity}

论文实验部分沿用标幺几何并报告 108°、0.58 s。另按表 III 的物理数值及 50 V 相电压 RMS 假设计算，得到 {experimental_si["CCA_deg"]:.3f}°、{experimental_si["CCT_s"]:.6f} s，0.7 s 清除仍在原吸引区。该分支仅展示电压口径及表列换算的敏感性。原实验的实际电压和限流设置仍待核实。

[冻结协议](../configs/protocol.json) · [来源审计](../sources/source_audit.md) · [运行清单](../results/run_manifest.json) · [独立验证数据](../verification/results.json) · [交叉核对](../results/cross_implementation.json) · [新环境重跑](../results/clean_replay_comparison.json) · [Agent 与耗时](agent_timing.json)
"""
    (ROOT / "reports/REPORT.md").write_text(text)
    status = {"doi": cfg["doi"], "state": "report_ready", "outcome": "scalar_psc_claims_reproduced_with_current_limit_scope_boundary",
              "scope": cfg["scope"], "numerical_checks_passed": True, "checks_passed": tests, "checks_total": tests,
              "full_sequence_cases": 8, "cross_implementation_passed": cross["passed"],
              "clean_environment_replay_passed": replay["passed"], "all_paper_claims_reproduced": False,
              "hardware_validation": False, "quantitative_original_experiment_match": False,
              "key_results": {"type_i_angles_deg": [primary["type_i_pre_deg"], primary["type_i_post_deg"]],
                              "cca_deg": primary["type_ii_cca_deg"], "cct_s": primary["type_ii_cct_s"],
                              "late_clearing_peak_current_pu": late["sampled_peak_current_pu"]},
              "critical_gaps": ["PCC current exceeds 1.8 pu under the stated base in late and uncleared scalar trajectories.",
                                "Original EMT and hardware raw traces, inner-loop settings and current limit transition details are unavailable.",
                                "SI inductance and gain normalization conventions require documented inference."],
              "anomaly_evidence": [
                  {"type": "model_scope_boundary", "source": "Author manuscript Sec. III-A and Table II, Fig. 10",
                   "finding": "Late ideal-scalar case crosses reported current threshold at clearance, so its subsequent recapture is conditional."},
                  {"type": "figure_trajectory_difference", "source": "Fig. 10(a), visual axis reading",
                   "finding": "Approx. 1.15 s between plotted wraps versus 1.4582 s for the ideal scalar model. Cause unresolved."}],
              "misconduct_allegation": None, "report_sha256": digest("reports/REPORT.md"),
              "primary_manifest_sha256": digest("results/run_manifest.json")}
    write("reports/status.json", status)
    print(json.dumps({"state": status["state"], "checks": tests, "max_cross_errors": cross["max_waveform_error"]}))


if __name__ == "__main__":
    main()
