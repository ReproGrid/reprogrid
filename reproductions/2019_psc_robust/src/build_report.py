"""Generate the research report and cross-implementation evidence from saved runs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT/path).read_text())


def digest(path):
    return sha256((ROOT/path).read_bytes()).hexdigest()


def write(path, content):
    (ROOT/path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def compare(run, config):
    pairs = {f"{case}_{loop}": f"{case}_{loop}" for case in run["fig5"] for loop in ("gp", "gc", "gd")}
    for a, b in (("psc", "psc"), ("vsm0", "droop"), ("vsm0_kd_30pct", "droop_reduced_dc_gain")):
        for loop in ("gp", "gc", "gd"):
            pairs[f"fig7_8_{a}_{loop}"] = f"fig7_{b}_{loop}"
    errors = {}
    with np.load(ROOT/"results/frequency_responses.npz", allow_pickle=False) as primary, \
         np.load(ROOT/"verification/frequency_responses.npz", allow_pickle=False) as independent:
        if not np.array_equal(primary["w_pu"], independent["w_pu"]):
            raise ValueError("Frequency grids differ")
        count = len(primary["w_pu"])
        for left, right in pairs.items():
            if primary[left].shape != (count,) or independent[right].shape != (count,):
                raise ValueError(f"Frequency response shape differs: {left}, {right}")
            error = abs(primary[left]-independent[right])
            errors[left] = {"independent_key": right, "abs_complex_error_max": float(np.max(error)),
                            "scaled_complex_error_max": float(np.max(error/np.maximum(1, abs(independent[right]))))}
    maximum = max(v["abs_complex_error_max"] for v in errors.values())
    tolerance = config["numerics"]["cross_frequency_abs_tolerance"]
    result = {"samples_per_response": count, "response_count": len(errors), "tolerance": tolerance,
              "max_absolute_complex_error": maximum, "responses": errors,
              "primary_manifest_sha256": digest("results/run_manifest.json"),
              "independent_result_sha256": digest("verification/results.json"),
              "independent_frequency_sha256": digest("verification/frequency_responses.npz"),
              "passed": maximum <= tolerance}
    write("results/cross_implementation.json", result)
    return result


def fig5_variant_section(comparison, reference):
    printed_key = "figure_labels"
    inferred_key = "flip_upper_iq_and_round_scr3_inductance"
    printed, inferred = (comparison["cases"][key] for key in (printed_key, inferred_key))
    p, q = (comparison["claims"][key] for key in (printed_key, inferred_key))
    pdf = {case["scr_label"]: case for case in reference["curves"]
           if case["condition"] == "high_active" and case["loop"] == "dc_link"}
    rows = []
    for scr in (1, 3, 10):
        key = f"fig5_active_scr{scr}"
        a, b = printed[key], inferred[key]
        rows.append(
            f'| {scr} | {a["gp"]["gain_margin"]:.3f} / {b["gp"]["gain_margin"]:.3f} | '
            f'{a["gp"]["phase_margin_deg"]:.2f}° / {b["gp"]["phase_margin_deg"]:.2f}° | '
            f'{a["gd"]["gain_margin"]:.3f} / {b["gd"]["gain_margin"]:.3f} | '
            f'{a["gd"]["phase_margin_deg"]:.2f}° / {b["gd"]["phase_margin_deg"]:.2f}° | '
            f'{a["gd"]["peak_sensitivity"]:.3f} / {b["gd"]["peak_sensitivity"]:.3f} / '
            f'{pdf[scr]["visible_peak_sensitivity_estimate"]:.3f} |'
        )
    table = "\n".join(rows)
    verdict = comparison["validation"]

    def rank(key):
        return " → ".join(map(str, comparison["claims"][key]["high_active_dc_sensitivity_ascending_scr"]))

    return f"""## Fig. 5 参数标注与曲线的差异

Aalto 同行评审作者稿的 Fig. 5 上排标注 `(id0, iq0) = (0.95, +0.2)`。按式 (8)，正号对应吸收无功，负号对应注入无功。上排六条原图矢量曲线与 `iq0 = −0.2` 更吻合。虚线另与 `L = 0.33 pu` 更吻合，对应有效 SCR 约 3.0303。先单独变号和单独取整，再组合测试，四套输入均有独立状态空间核对。组合后十二条轨迹的最大逐曲线 RMS 距离为 **{verdict["max_combined_vector_rms_pu"]:.3g} pu**。在相同的曲线一致电感下，仅改符号就使上排六条曲线的 RMS 距离至少缩小 **{verdict["minimum_upper_rms_improvement_factor_from_sign_at_curve_inductance"]:.0f} 倍**。证据限于所核对的 Aalto 稿件和测试的参数假设，作者绘图输入及最终出版版图面仍待确认。[PDF 矢量核对](../verification/fig5_vector_comparison.json) · [四套参数与数值结果](../results/fig5_condition_comparison.csv)

上排各单元格为图注工况 / 曲线反推工况。最后一列追加从 PDF 线段估计的灵敏度峰值。

| SCR | 有功 GM | 有功 PM | 直流侧 GM | 直流侧 PM | 直流侧灵敏度峰值 |
|---|---:|---:|---:|---:|---:|
{table}

- **第 V-B 节的灵敏度排序依赖符号。** 按峰值从小到大，图注给出 SCR {rank(printed_key)}，曲线反推给出 SCR {rank(inferred_key)}。仅变号时为 SCR {rank("flip_upper_iq")}，仅将电感取整时为 SCR {rank("round_scr3_inductance")}。因此，论文用 SCR=3 最低灵敏度峰值解释直流电压响应的频域依据，与原图及负号工况一致。该段实验的有功电流约为 0.6 pu，图 5 上排为 0.95 pu，硬件性能排序仍以原文实验记录为准。
- **第 III-C 节的稳定性和裕度趋势在两套工况下均复现。** 六个工作点的有功环和直流侧环均稳定，有功环最小增益裕度为 {min(p["minimum_active_gain_margin"], q["minimum_active_gain_margin"]):.3f} 倍。两排有功环最小相位裕度都出现在 SCR=10，符合第 V-A 节对强网相位裕度偏小的解释。两套工况中，各有 {p["dc_gain_margin_below_four_count"]} 个直流侧增益裕度低于 4 倍。SCR=1 高无功注入时，直流侧相位裕度为 {q["weak_grid_high_reactive_dc_phase_margin_deg"]:.2f}°。
- **弱网带宽降幅的数值随符号变化。** SCR=1 的上排有功闭环 −3 dB 带宽由 {p["weak_grid_active_bandwidth_pu"]:.5f} pu 变为 {q["weak_grid_active_bandwidth_pu"]:.5f} pu。下排高无功注入工况仍约 {q["weak_grid_high_reactive_bandwidth_pu"]:.5f} pu，带宽下降的结论成立。Fig. 7–8 沿用例 2 的电流工况，附录裕度下界来自相应的冻结滤波器条件。

![Fig. 5 图注与曲线反推工况的 Nyquist 叠图](../figures/fig5_parameter_overlay.png)

![Fig. 5 上排直流侧裕度与灵敏度峰值](../figures/fig5_robustness_comparison.png)
"""


def main():
    run = load("results/run_manifest.json")
    config = load("configs/protocol.json")
    source = load("sources/source_manifest.json")
    independent = load("verification/results.json")
    replay = load("results/clean_replay_comparison.json")
    fig5_comparison = load("results/fig5_condition_comparison.json")
    fig5_reference = load("verification/fig5_vector_comparison.json")
    fig8_independent = load("results/fig8_independent_comparison.json")
    cross = compare(run, config)
    independent_gates = [value for key,value in independent["gates"].items()
                         if key.endswith("_pass") or key.endswith("_stable")]
    if not all((run["numerical_checks_passed"], replay["passed"], cross["passed"], *independent_gates,
                run["protocol_sha256"] == digest("configs/protocol.json"),
                run["implementation_sha256"] == digest("src/reproduce.py"),
                run["source_sha256"] == source["sha256"],
                independent["protocol_sha256"] == digest("verification/protocol.json"),
                fig5_comparison["passed"],
                replay["fig5_comparison_json_byte_equal"],
                replay["fig8_independent_json_byte_equal"],
                all(replay["fig5_comparison_figure_byte_equal"].values()),
                fig8_independent["passed"],
                fig8_independent["implementation_sha256"] == digest("verification/verify_fig8.py"),
                fig5_reference["pdf_sha256"] == source["sha256"],
                fig5_comparison["inputs_sha256"]["variants"] == digest("configs/fig5_variants.json"),
                fig5_comparison["inputs_sha256"]["implementation"] == digest("src/compare_fig5_conditions.py"),
                fig5_comparison["inputs_sha256"]["paper_vector_evidence"] == digest("verification/fig5_vector_comparison.json"))):
        raise RuntimeError("Evidence checks failed. Inspect the primary, replay, and independent manifests.")
    rows = []
    for case in run["fig5"].values():
        group = "有功工况" if case["op"][0] > case["op"][1] and case["op"][1] >= 0 else "无功注入"
        gp, gd = case["gp"], case["gd"]
        rows.append(f'| {group} | {case["scr"]} | {gp["gain_margin"]:.3f} | {gp["phase_margin_deg"]:.2f}° | '
                    f'{gd["gain_margin"]:.3f} | {gd["phase_margin_deg"]:.2f}° | {case["active_bandwidth_pu"]:.5f} |')
    margin_table = "\n".join(rows)
    vsm = run["fig7_8"]
    psc_pm = vsm["psc"]["gd"]["phase_margin_deg"]
    vsm_pm = vsm["vsm0"]["gd"]["phase_margin_deg"]
    vsm30_pm = vsm["vsm0_kd_30pct"]["gd"]["phase_margin_deg"]
    rci = run["fig5"]["fig5_reactive_scr1"]
    active = run["fig5"]["fig5_active_scr1"]
    alt = run["table_I_Ra_sensitivity"]
    appendix_case = next(r for r in run["appendix_audit"] if r["scr"]==1 and r["iq"] == -.95)
    zero_case = run["appendix_audit"][-1]
    worst_reactive = run["reactive_sweep"]["strongest_injection"]
    error_step = max(r["linear_nonlinear_max_relative_to_step"] for r in run["local_steps"].values())
    error_refinement = max(r["DOP853_Radau_max_relative_to_step"] for r in run["local_steps"].values())
    checks = len(run["checks"])
    variant_checks = len(fig5_comparison["checks"])
    sign_section = fig5_variant_section(fig5_comparison, fig5_reference)
    inferred_steps = fig5_comparison["curve_inferred_local_steps"]
    inferred_error = max(x["linear_nonlinear_max_relative_to_step"] for x in inferred_steps.values())
    inferred_refinement = max(x["DOP853_Radau_max_relative_to_step"] for x in inferred_steps.values())
    text = f"""# PSC 鲁棒解析整定的计算复现

Harnefors、Hinkkanen、Riaz、Rahman 与 Zhang，*Robust Analytic Design of Power-Synchronization Control*，IEEE Transactions on Industrial Electronics 66(8)，5810–5819，2019。[DOI](https://doi.org/{source["doi"]}) · [Aalto 作者稿]({source["landing_url"]})

按式 (13)、(19)、(25) 设置 `Ra=0.2 pu`、`Kp=0.2 pu` 和 `Kd=1/(4√2)=0.17678 pu`，重建了 Fig. 5、7、8 的连续模型结果。Fig. 5 的图注与曲线反推工况均稳定。上排无功电流的符号改变了直流侧灵敏度排序。SCR=1、高无功注入时，直流侧环相位裕度降到 **{rci["gd"]["phase_margin_deg"]:.2f}°**。两套计算均发现附录 II、IV 的若干中间表达式不一致，重新推导后，文中的增益裕度下界和式 (24) 仍成立于相应假设下。

## Fig. 5 与高无功注入

模型采用单变流器、纯电感和无穷大母线，忽略 PWM 延迟。频率以 50 Hz 对应角频率归一化。相量工作点由 `Vg·exp(−jδ₀)=V−jLi₀` 确定。两组电流采用图中标注的 (0.95, 0.2) 和 (0.2, −0.95)，模长约 0.970824 pu。原文正文称 1 pu。频响使用完整 `Ha(s)=Ra·s/(s+0.1)`。

| 工况 | SCR | 有功 GM | 有功 PM | 直流侧 GM | 直流侧 PM | 有功闭环 −3 dB 带宽 (pu) |
|---|---:|---:|---:|---:|---:|---:|
{margin_table}

六例的有功 GM 都高于 2。有限滤波带宽下，直流侧 GM 可低于 4，符合原文对定理适用条件的说明。SCR=1 时，高无功注入把有功闭环带宽由 **{active["active_bandwidth_pu"]:.5f}** 降到 **{rci["active_bandwidth_pu"]:.5f} pu**，尽管直流侧 GM 仍为 {rci["gd"]["gain_margin"]:.2f}。扩展算例固定 SCR=1、id=0，按平衡方程随 iq 改变 Vg。当 iq=−0.999 pu、Vg=0.001 pu 时，有功带宽约 {worst_reactive["bandwidth_gc_pu"]:.6f} pu，直流侧 PM 约 {worst_reactive["dc_phase_margin_deg"]:.2f}°。151 个扫参点的直流侧极点均在左半平面。

![Fig. 5 Nyquist 曲线](../figures/fig5_nyquist.png)

![高无功注入扩展算例](../figures/reactive_boundary.png)

{sign_section}
## Fig. 7–8 的 VSM 对照

例 2、3 沿用例 1 的 ωb=0.1 pu，这项继承在原文中未重述。SCR=2、id=0.7、iq=−0.7 时，PSC、零惯量 VSM 以及 VSM 的 30% 直流侧增益三种配置，直流侧相位裕度分别为 **{psc_pm:.2f}°、{vsm_pm:.2f}°、{vsm30_pm:.2f}°**。这支持 30% 增益对照。实际电容须增至 3.3 倍的波形比较缺少原始系统配置。

Fig. 8 计算使用 5 s 的物理惯量和 1 rad/s 的阻尼滤波频率。有功环相位裕度从零惯量的 {vsm["vsm0"]["gp"]["phase_margin_deg"]:.2f}° 降到 {vsm["vsm_H5"]["gp"]["phase_margin_deg"]:.2f}°，加入 KD=50 pu 的阻尼后回到 {vsm["vsm_H5_KD50"]["gp"]["phase_margin_deg"]:.2f}°。两种 H=5 s 配置在原 PSC 推荐 Kd 下串接直流侧环均有右半平面极点，独立状态模型复核了频响和极点。[Fig. 8 独立核验](../results/fig8_independent_comparison.json)

![Fig. 7–8 Nyquist 对照](../figures/fig7_8_nyquist.png)

## 附录公式核查

令 α=Ra/(ω₁L)，并用标幺增益 Kd。下列推导使用冻结的电流滤波扰动（ωb=0）。

- **附录 II** 式 (39) 的第一个分母 `D=1+b−aα²` 与数值交点一致。代入式 (10) 给出 `D=1−2aα²−(Ra|i₀|/V)²`，所以式 (39) 第二个等号和式 (40) 漏掉无功项。SCR=1、高无功工况给出 **GM={appendix_case["p_gm_numeric"]:.6f}**，印刷式 (40) 给出 {appendix_case["p_gm_printed_eq40"]:.6f}。在 D>0 的负实轴交点适用域，`1+α²−D=α²(Vg/V)²≥0`，所以 GM≥2 的下界仍成立。
- **附录 IV** 由式 (14) 可得直流侧相交频率 `ωπ/ω₁=√((1+a+b)/(2+a))`。同一 SCR=1 高无功工况为 {appendix_case["dc_w_pi_numeric"]:.6f} pu，印刷频率为 {appendix_case["dc_w_pi_printed"]:.6f} pu。式 (42) 第一项的分母还需乘 2。零电流且 L=√2 Ra/ω₁ 时，数值 GM 为 **{zero_case["dc_gm_numeric"]:.0f}**，印刷式 (42) 给出 **{zero_case["dc_gm_printed_eq42"]:.0f}**。
- 修正后 `GM=[(1−b)/(2(2+a)α)+α/2]/Kd`。在正项和有效交点条件下，用算术–几何均值不等式得到原式 (43)，因此式 (24) 的充分增益上界保留。固定 a、b 时等号要求 `L=Ra/ω₁·√((2+a)/(1−b))`。原稿给出的 √2 Ra/ω₁ 对应 a=b=0。

首轮自动检查把印刷式 (40) 当作相等性参考，检测到不符并退出。复核确认是公式化简差异，重跑保持冻结协议和数值容差，同时保存印刷值与重新推导值。[首轮清单](../logs/first_run_manifest.json) · [附录数值对照](../results/appendix_audit.csv) · [独立推导](../verification/independent_derivation.md)

## 时域、复跑与资料缺口

- 12 组理想 dq 小阶跃覆盖 Fig. 5 六工况的有功和直流侧能量参考。线性和非线性输出的最大差为阶跃幅值的 {100*error_step:.4f}%。DOP853 与 Radau 的最大差为阶跃幅值的 {100*error_refinement:.5f}%。
- 曲线反推工况也完成了 12 组小阶跃。线性和非线性输出的最大差为阶跃幅值的 {100*inferred_error:.4f}%，求解器精化差为 {100*inferred_refinement:.5f}%。[曲线反推工况时域数据](../results/fig5_curve_inferred_local_steps.npz)
- 主运行 **{checks} 项检查**、图 5 参数对照 **{variant_checks} 项检查**均通过。主实现与独立模型对照 **{cross["response_count"]} 条频响，每条 {cross["samples_per_response"]} 点**，最大复数绝对差 {cross["max_absolute_complex_error"]:.3g}。新建环境重跑的数值数组最大差为 {replay["array_max_difference"]:.3g}，四张 CSV、图 5 对照 JSON 和两幅新图逐字节一致。
- Table I 的 Ra 列给出 4.4 Ω、12.6 Ω 基阻和 0.2 pu。计算得到 {alt["Ra_pu"]:.6f} pu，已另存这一取值并按式 (13) 重新整定 Kp 后的六工况裕度。原稿未给出可确认实验实际 Ra 的补充记录。
- Figs. 9–17 的开关波形、硬件测量以及电流限幅未复放。缺少原始波形、限流配置、有效阻抗及硬件延迟记录。小阶跃对照的模型假设记录在冻结协议中。

![小阶跃线性与非线性对照](../figures/local_step_response.png)

[冻结协议](../configs/protocol.json) · [来源审计](../sources/source_audit.md) · [主运行清单](../results/run_manifest.json) · [独立核验](../verification/verification_report.md) · [交叉核对](../results/cross_implementation.json) · [干净环境重跑](../results/clean_replay_comparison.json) · [Agent 与耗时](agent_timing.json)
"""
    (ROOT/"reports/REPORT.md").write_text(text)
    status = {"doi": config["doi"], "state": "report_ready",
              "outcome": "computational_claims_reproduced_with_documented_intermediate_formula_corrections",
              "scope": config["scope"], "all_paper_claims_reproduced": False, "hardware_validation": False,
              "quantitative_original_experiment_match": False,
              "checks_passed": checks+variant_checks, "checks_total": checks+variant_checks,
              "baseline_checks_total": checks, "fig5_variant_checks_total": variant_checks,
              "cross_implementation_passed": True, "clean_environment_replay_passed": True,
              "fig8_independent_passed": True,
              "key_results": {"fig5_reactive_scr1_dc_phase_margin_deg": rci["gd"]["phase_margin_deg"],
                              "fig5_upper_dc_peak_sensitivity_best_scr_printed": fig5_comparison["claims"]["figure_labels"]["high_active_dc_sensitivity_ascending_scr"][0],
                              "fig5_upper_dc_peak_sensitivity_best_scr_curve_inferred": fig5_comparison["claims"]["flip_upper_iq_and_round_scr3_inductance"]["high_active_dc_sensitivity_ascending_scr"][0],
                              "fig7_dc_phase_margins_deg": [psc_pm, vsm_pm, vsm30_pm],
                              "appendix_iv_zero_current_numeric_gm": zero_case["dc_gm_numeric"],
                              "appendix_iv_zero_current_printed_gm": zero_case["dc_gm_printed_eq42"]},
              "critical_gaps": ["Quantitative hardware replay requires original traces, limiter settings and experimental delay/impedance data.",
                                "Table I Ra in ohms is inconsistent with the listed pu value.",
                                "Fig. 7–8 filter bandwidth is inherited from Example 1."],
              "anomaly_evidence": [{"type": "intermediate_formula_inconsistency", "source": "Appendices II and IV",
                                    "finding": "Eq39 last equality, Eq40, Appendix IV crossover, and Eq42 first term disagree with direct transfer evaluation. Main bounds still follow from the corrected derivation."},
                                   {"type": "parameter_normalization_inconsistency", "source": "Table I",
                                    "finding": "4.4/12.6 = 0.349206 pu, versus stated 0.2 pu."},
                                   {"type": "parameter_label_and_curve_inconsistency", "source": "Aalto accepted manuscript, Fig. 5",
                                    "finding": "The upper row is labeled iq0=+0.2 pu. Its vector loci agree with iq0=-0.2 pu, with L=0.33 pu for the SCR=3 dashed trace. The sign changes which SCR minimizes the DC-link sensitivity peak.",
                                    "evidence": "verification/fig5_vector_comparison.json"}],
              "misconduct_allegation": None, "report_sha256": digest("reports/REPORT.md"),
              "primary_manifest_sha256": digest("results/run_manifest.json"),
              "fig5_condition_comparison_sha256": digest("results/fig5_condition_comparison.json")}
    write("reports/status.json", status)
    print(json.dumps({"state": status["state"], "checks": checks+variant_checks, "cross_frequency_responses": cross["response_count"],
                      "cross_max_absolute_error": cross["max_absolute_complex_error"]}))


if __name__ == "__main__":
    main()
