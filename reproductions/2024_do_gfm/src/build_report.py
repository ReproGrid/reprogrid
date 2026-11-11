"""Cross-check both implementations and render the evidence report from saved outputs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE/"results"
VERIFY = HERE/"verification"


def read(path):
    return json.loads((HERE/path).read_text())


def digest(path):
    return sha256((HERE/path).read_bytes()).hexdigest()


def write(path,data):
    (HERE/path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def compare_coordinates(cfg,run,verifier):
    tolerance = cfg["numerics"]["trajectory_tolerance_pu"]
    cases = {}
    with np.load(VERIFY/"ideal_grid_frame.npz",allow_pickle=False) as second:
        for name in run["scenarios"]:
            with np.load(RESULTS/f"ideal_{name}.npz",allow_pickle=False) as primary:
                time = second[f"{name}__time_s"]
                if not np.array_equal(primary["time_s"],time):
                    raise ValueError(f"Time grid mismatch: {name}")
                fields = {}
                for k in primary.files:
                    if k=="time_s":
                        continue
                    other = second[f"{name}__{k}"]
                    if other.shape!=primary[k].shape:
                        raise ValueError(f"Output shape mismatch: {name}/{k}")
                    fields[k] = float(np.max(abs(primary[k]-other)))
                cases[name] = {"samples":len(time),"max_abs_by_field":fields}
    maximum = max(v for case in cases.values() for v in case["max_abs_by_field"].values())
    verified = {(x["mode"],x["p0"]):x for x in verifier["frequency_equilibria"]}
    frequency = []
    for case in run["frequency_static"]:
        other = verified[case["mode"],case["p0"]]
        error = max(abs(case[a]-other[b]) for a,b in (("p_derivative_numeric","p_derivative"),
                      ("v_derivative_numeric","v_derivative"),("p_at_49_hz","p_at_49_hz"),("v_at_49_hz","v_at_49_hz")))
        frequency.append({"mode":case["mode"],"p0":case["p0"],"max_abs_difference":error})
    return {"passed":maximum<=tolerance and max(r["max_abs_difference"] for r in frequency)<=tolerance,
            "max_abs_trace_difference":maximum,"trajectory_tolerance":tolerance,
            "cases":cases,"frequency_equilibria":frequency,"main_manifest_sha256":digest("results/run_manifest.json"),
            "verification_manifest_sha256":digest("verification/results.json")}


def main():
    cfg = read("configs/protocol.json")
    source = read("sources/source_manifest.json")
    run = read("results/run_manifest.json")
    verifier = read("verification/results.json")
    replay = read("results/clean_replay_comparison.json")
    author = read("results/author_replay_comparison.json")
    published = read("results/metrics.json")
    previous = read("logs/first_run_manifest.json")
    plateau = read("diagnostics/steady_state_consistency.json")
    solver = read("verification/solver_diagnostic.json")
    cross = compare_coordinates(cfg,run,verifier)
    write("results/cross_implementation.json",cross)
    gates = [
        run["numerical_checks_passed"],verifier["passed"],replay["passed"],author["passed"],cross["passed"],
        run["protocol_sha256"]==verifier["protocol_sha256"]==previous["protocol_sha256"]==digest("configs/protocol.json"),
        run["implementation_sha256"]==digest("src/reproduce.py"),
        verifier["implementation_sha256"]==digest("verification/verifier.py"),
        previous["implementation_sha256"]==digest("logs/reproduce_initial.py"),
        run["source_sha256"]==source["sha256"],
        published["source_hashes"]["protocol"]==digest("protocol.json"),
        published["source_hashes"]["script"]==digest("reproduce.py"),
        plateau["source_sha256"]["results/metrics.json"]==digest("results/metrics.json"),
        all(author["files"][name]["original_sha256"]==digest("results/"+name) for name in author["files"]),
    ]
    if not all(gates):
        raise RuntimeError(f"Reproduction evidence checks failed: {gates}")

    def weak_metrics(ideal, prior, source_kind, label):
        if source_kind=="ideal":
            series = run["scenarios"][ideal]["steps"]
            return (label,series[2]["transition_10_90_ms"],series[3]["transition_10_90_ms"],
                    series[2]["plateau_p"],series[2]["plateau_q"])
        series = published["runs"][prior][source_kind]
        p = series[2]["last_50_ms_mean"]
        return (label,series[2]["rise_or_fall_10_90_ms"],series[3]["rise_or_fall_10_90_ms"],p["p_g_pu"],p["q_g_pu"])

    rows = []
    for ideal,prior,display in (("rfpsc_weak","rfpsc","RFPSC"),("do13_weak","do_gfm","DO-GFM")):
        for kind,label in (("ideal","方程平均模型"),("switched_steps","motulator PWM"),("paper_steps","出版图数字化")):
            name,rise,fall,p,q = weak_metrics(ideal,prior,kind,f"{display} · {label}")
            rows.append(f"| {name} | {rise:.2f} | {fall:.2f} | {p:.4f} | {q:.4f} |")
    table = "\n".join(rows)
    def row_for(mode,p):
        return next(r for r in run["frequency_static"] if r["mode"]==mode and r["p0"]==p)
    do12 = row_for("do12",0.)
    do13 = row_for("do13",.5)
    strong = run["scenarios"]["do13_strong"]["steps"][2]
    sweeps = run["inductance_sweep"]
    mode = run["scenarios"]["mode_fig10"]
    jac = max(r["jacobian_max_abs_error"] for r in run["linearization_cases"])
    local = max(r["max_abs_error_relative_to_step"] for r in run["local_steps"].values())
    refinement = max(v for r in run["scenarios"].values() for v in r["refinement_max_abs_by_field"].values())
    diag = plateau["runs"]["do_gfm"]["published"][2]
    with np.load(RESULTS/"ideal_mode_fig10.npz",allow_pickle=False) as m:
        clipped_times = m["time_s"][m["current_ref_raw"]>1.3]
        clip_interval = [float(clipped_times[0]),float(clipped_times[-1])] if len(clipped_times) else None
    clip_description = (f"{len(clipped_times)} 个采样点触及限幅（{clip_interval[0]:.4f}–{clip_interval[1]:.4f} s）"
                        if clip_interval else "未观测到限幅动作")
    checks = len(run["checks"])
    text = f"""# DO-GFM 扰动观测器控制的计算复现

Nurminen、Mourouvin、Hinkkanen、Kukkola，*Multifunctional Grid-Forming Converter Control Based on a Disturbance Observer*，IEEE TPEL 39(10)，13023–13032，2024。 [DOI](https://doi.org/{source["doi"]}) · [Aalto 出版记录]({source["landing_url"]})

论文方程的连续平均模型完成复现。强弱网下的 DO-GFM 与 RFPSC 有功响应相近。电感失配扫描、50→49 Hz 斜坡及 GFM/GFL 切换支持原文的计算机制。已有的 motulator PWM 对照也已重新运行。弱网出版波形的上升时间和无功平台仍有可量化差异，归因所需的硬件数据不足。

## 独立实现与验证

主程序在额定频率坐标下积分 Eq. (18) 的状态 z=Vhat+ko·Lhat·i；核对程序在实际电网频率坐标下直接用 Eq. (10) 和对象电流导数。两套程序共用冻结协议，由同一 agent 编写，均从论文方程实现。RFPSC 由 Appendix C 另行实现。输入与工作点采用表 I–II 的标幺口径，理想对象为单变流器、纯电感和无穷大母线。

- **{checks} 项主数值检查**通过。7 个时域工况用 DOP853 和 Radau 精化，最大绝对差 {refinement:.3g}。两坐标实现的轨迹最大差 {cross["max_abs_trace_difference"]:.3g}。Eq. (19) 的连续坐标投影残差最大 {verifier["max_eq19_projection_error"]:.3g}。
- 48 个小信号工况以解析模型对照物理方程的数值 Jacobian，最大差 {jac:.3g}。匹配电感时，极点和参考传递函数符合 Eq. (16)、(21)–(25)。8 组小阶跃的线性/非线性输出最大差为阶跃幅值的 {100*local:.5f}%。
- 固定 Lhat=0.15 pu，扫描 L=0.15–1.00 pu 及五个有功工作点。两种增益共 {sum(r["count"] for r in sweeps.values())} 个采样工作点，未发现右半平面闭环极点。Eq. (13) 对应 Eq. (29) 的电压误差耦合项实部在数值精度内为零。扫描结论限定在上述工作点和理想模型。
- 首轮 Jacobian 检查定位到本实现的 Eq. (12) 线性化在失配时误用了真实 Ug 方向。改用观测器的 uhat0 后通过，协议和阈值未变。原始代码和首轮失败清单已保留。[首轮清单](../logs/first_run_manifest.json)
- 首轮跨坐标校验在强网空载尾段检测到默认 Radau 路径约 {solver["initial_radau_vs_tight_dop853_max_current_pu"]:.3g} pu 的电流漂移。显式 RK45 与收紧的 DOP853 在该工况相差 {solver["rk45_vs_tight_dop853_max_current_pu"]:.3g} pu。最终核验用 RK45，保留原精化容差和最大步长。失败轨迹与中止的隐式求解尝试均已保存。[求解器诊断](../verification/solver_diagnostic.json)

![强弱网阶跃](../figures/independent_tracking.png)

![电感失配闭环极点](../figures/inductance_sweep.png)

## 频率斜坡及印刷表达式核对

Fig. 9 的初始 P=0.5 pu。在相同 50→49 Hz 斜坡下，稳态 P 为 **{run["fig9_late_p_pu"]:.6f} pu**，吻合正文的约 0.6 pu。积分 Eq. (1) 保留了频率斜坡。若直接用瞬时 ωg 微分 Eq. (2)，电压导数须包含 j·L·i·d(ωg)/dt，因此下述印刷式核对只取零频增益。

匹配电感时，取单位频率偏差，Eq. (21a) 给出稳态估计误差 `t = j·(v0+ko·L·i0)/(ko+j·ω0)`。令 `E_v = Re(conj(v0)·t)/|v0|`。代入 Eq. (22a)–(23a) 得有功稳态斜率 `D(0)+(βpv/αp)·(E_v−Q(0))`，其中 D、Q 按 Eq. (26a,b) 定义。印刷 Eq. (16a)/(26c) 对应 `−D(0)−(βpv/αp)·Q(0)`。

| 工况 | 印刷式 dP/dω (pu/pu) | Eq. (21)–(23) 重推 | 非线性稳态差分 |
|---|---:|---:|---:|
| Eq. (12)，P₀=0，L=Lhat=0.15 | {do12["p_derivative_eq16_26_printed"]:.6f} | {do12["p_derivative_from_eq21_23"]:.6f} | {do12["p_derivative_numeric"]:.6f} |
| Eq. (13)，P₀=0.5，L=Lhat=0.15 | {do13["p_derivative_eq16_26_printed"]:.6f} | {do13["p_derivative_from_eq21_23"]:.6f} | {do13["p_derivative_numeric"]:.6f} |

第一行 βpv=0，差异简化为 D 项符号。另一坐标实现的稳态求根独立确认了斜率。Eq. (26b) 电压零频斜率一致。该问题影响直接使用印刷式预测有功频率增益，Fig. 9 的 RFPSC 型频降增功率结论仍得到数值支持。

## 模式切换

按 Fig. 10 的时刻与表 II 增益，0.2 s 切入 GFL，0.4 s 将 Q 指令改为 −0.8 pu，0.8 s 返回 GFM。GFL 区间的有功最大偏差 {run["fig10_gfl_max_active_deviation_pu"]:.3g} pu。切换共享观测器状态，状态及电流连续。返回 GFM 时，电压指令相量跳变 {mode["switches"][-1]["voltage_command_vector_jump"]:.4f} pu，随后 {clip_description}。限幅值采用 Fig. 8 报告的 1.3 pu，约束对象是电流参考。

![频率与模式切换](../figures/frequency_and_mode.png)

## 弱网框架与出版波形对照

弱网 L=0.89 pu。框架使用固定 motulator 提交、10 kHz 采样和 5 kHz PWM，保持原复跑配置。以下为同一指令步进的 10%–90% 时间及额定指令最后 50 ms 的平台值。出版图采用矢量数字化和首次指令边沿对齐。

| 控制与数据 | 0.66→1 上升 (ms) | 1→0 下降 (ms) | P 平台 (pu) | Q 平台 (pu) |
|---|---:|---:|---:|---:|
{table}

两套独立控制律的有功轨迹差为强网 {run["do13_vs_rfpsc_active_rmse_pu"]["strong"]:.5f} pu RMS、弱网 {run["do13_vs_rfpsc_active_rmse_pu"]["weak"]:.5f} pu RMS。强网同一指令序列的 DO-GFM 额定上升时间 {strong["transition_10_90_ms"]:.2f} ms，可与 Fig. 7(a) 报告的约 6 ms 作量级比较。弱网 DO-GFM 原图额定平台的 P/Q 与两种仿真均有偏差。按原图 P/Q 与名义 R+jX 推算，稳态端电压约 {diag["terminal_voltage_from_pq_pu"]:.4f} pu，图上的指令约 {diag["last_50_ms_mean"]["u_c_ref_pu"]:.4f} pu。推算依赖名义对象和稳态近似，现有数据不足以把指令电压差归因到硬件。

![motulator 和出版图叠加](../results/fig5b_fig6b_comparison.png)

## 重跑、资料缺口与交付

- 新建 Python 环境按锁定版本重跑，独立模型的数值数组最大差为 {replay["array_max_difference"]:.3g}，电感扫描 CSV 字节一致。motulator 的 6 个基准输出逐字节一致，3 个出版图数字化数据文件也重新提取并核对一致。
- 旧框架结果保留独立来源标签。实验的 Eq. (19) 离散实现、实际电阻、有效输出基波、死区、传感延迟和电网模拟器动态缺少原始输入。Figs. 8、11 的定量硬件轨迹以及保护和 LCL 工况已记录跳过。

[代码与复跑入口](../README.md) · [冻结协议](../configs/protocol.json) · [来源审计](../sources/source_audit.md) · [主运行清单](../results/run_manifest.json) · [坐标交叉核对](../results/cross_implementation.json) · [干净环境重跑](../results/clean_replay_comparison.json) · [框架复跑核对](../results/author_replay_comparison.json) · [Agent 与耗时](agent_timing.json)
"""
    (HERE/"reports/REPORT.md").write_text(text)
    status = {
        "doi":cfg["doi"],"state":"report_ready",
        "outcome":"equation_level_reproduction_with_printed_frequency_gain_discrepancy_and_hardware_trace_gap",
        "scope":cfg["scope"],"scope_zh":"DO-GFM/RFPSC 理想时域、观测器小信号与电感失配、频率斜坡、GFM/GFL 切换及独立框架重跑。",
        "all_paper_claims_reproduced":False,"hardware_validation":False,"quantitative_original_experiment_match":False,
        "checks_passed":checks,"checks_total":checks,"cross_implementation_passed":cross["passed"],
        "clean_environment_replay_passed":replay["passed"],"author_framework_replay_passed":author["passed"],
        "key_results":{"fig9_final_power_pu":run["fig9_late_p_pu"],"fig10_gfl_max_active_deviation_pu":run["fig10_gfl_max_active_deviation_pu"],
                       "voltage_command_return_jump_pu":mode["switches"][-1]["voltage_command_vector_jump"],
                       "no_load_do12_dP_domega_printed":do12["p_derivative_eq16_26_printed"],
                       "no_load_do12_dP_domega_numeric":do12["p_derivative_numeric"]},
        "critical_gaps":["Exact laboratory Eq. (19) discretization, terminal voltage, plant resistance and device delays are unavailable.",
                         "Fig. 6(b) weak-grid power rise and reactive plateau differ quantitatively from both reproduction layers.",
                         "Fig. 8 fault ride-through, Fig. 11 hardware DC-bus control and LCL validation remain outside verified scope."],
        "anomaly_evidence":[{"type":"printed_frequency_gain_discrepancy","source":"Eq. (16a), Eq. (26a,c) versus Eq. (21)–(23)",
                             "finding":"At zero current with Eq. (12), the printed DC P/f slope is +2.5; direct state-equation derivation and two steady solves give −2.5."},
                            {"type":"published_trace_quantitative_gap","source":"Fig. 6(b) versus the frozen weak-grid models",
                             "finding":"The rated step rise is about 109.6 ms in the vector trace and 64.9/67.0 ms in the ideal/PWM replays. Raw hardware inputs are unavailable."}],
        "misconduct_allegation":None,"report_sha256":digest("reports/REPORT.md"),
        "primary_manifest_sha256":digest("results/run_manifest.json")}
    write("reports/status.json",status)
    print(json.dumps({"state":status["state"],"checks":checks,"cross_max_absolute_error":cross["max_abs_trace_difference"]}))


if __name__=="__main__":
    main()
