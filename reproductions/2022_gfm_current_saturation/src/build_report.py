"""Verify a clean replay and regenerate the report from frozen CCT outputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean


CASE = Path(__file__).resolve().parents[1]
RESULTS = CASE / "results"
ARTIFACTS = ("cct_results.json", "rated_trajectories.npz", "cct_comparison.png", "rated_first_swing.png")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path, help="Output directory from a separate run")
    args = parser.parse_args()
    if args.replay.resolve() == RESULTS.resolve():
        raise ValueError("The replay must use a separate output directory")
    checks = {file: {"baseline_sha256": digest(RESULTS / file), "replay_sha256": digest(args.replay / file)}
              for file in ARTIFACTS}
    passed = all(row["baseline_sha256"] == row["replay_sha256"] for row in checks.values())
    (RESULTS / "replay_check.json").write_text(json.dumps(
        {"passed": passed, "byte_identical": passed, "files": checks}, indent=2,
    ) + "\n")
    if not passed:
        raise RuntimeError("The replay files differ from the archived result")
    data = json.loads((RESULTS / "cct_results.json").read_text())
    if not data["passed"] or data["protocol_sha256"] != digest(CASE / "configs/protocol.json") or (
        data["implementation_sha256"] != digest(CASE / "src/reproduce.py")
    ):
        raise RuntimeError("Regenerate the results with the current source and protocol")
    grouped = {table: [row for row in data["table_rows"] if row["table"] == table]
               for table in ("II", "III", "IV")}
    rated = {table: max(rows, key=lambda row: row["p_ref_pu"]) for table, rows in grouped.items()}
    comparison = []
    for table, rows in grouped.items():
        paper = rated[table]
        eac = mean(abs(row["formula_ms"] - row["paper_dynamic_ms"]) for row in rows)
        damped = mean(abs(row["damped_manifold_ms"] - row["paper_dynamic_ms"]) for row in rows)
        comparison.append(
            f"| {table} | {paper['formula_ms']:.2f} | {paper['damped_manifold_ms']:.2f} "
            f"| {paper['paper_dynamic_ms']} | {eac:.2f} → {damped:.2f} |"
        )
    table_rows = "\n".join(comparison)
    angle = data["angle_bound"]
    errors = data["errors"]
    base, shifted = rated["II"]["damped_manifold_ms"], rated["IV"]["damped_manifold_ms"]
    text = f"""# 构网变流器电流参考饱和的暂态稳定复现

Rokrok、Qoria、Bruyere、Francois、Guillaud，IEEE TPWRS，37(2)，1519–1531，2022。采用 HAL 作者稿 PDF 第 4–9 页的式 (1)–(14)、表 I–IV 与 Fig. 9。 [DOI](https://doi.org/{data["doi"]}) · [HAL 来源](https://hal.science/hal-03966325v1) · [冻结协议](../configs/protocol.json)

## 临界清除时间

电压正常时用式 (3) 初始化低角平衡点。100% 对称跌落期间取 P=0，恢复电压后按式 (5) 取饱和功角曲线。等面积结果按式 (7)–(9) 和 (10)–(12) 计算。18 个工况相对印刷预测列的最大偏差为 **{errors["formula_vs_printed_max_ms"]:.3f} ms**。面积数值积分与反向鞍点稳定流形分别复核无阻尼结果，最大偏差 {errors["formula_vs_energy_or_saddle_max_ms"]:.3g} ms。

加入式 (1) 的 IP 阻尼后，用相对预故障平衡点的积分状态 z 写成 dδ/dt=ωb[z+kp(P₀−P)]、dz/dt=(P₀−P)/(2H)。清除时保持 δ、z 连续，按恢复后的功率重算角速度。反向积分从饱和曲线的右侧鞍点出发，与故障轨迹求交。全部 18 组在所得边界前后各 0.25 ms 正向积分，分别在鞍点前转向、越过鞍点角。步长、容差及初始鞍点偏移加密后，最大 CCT 差为 {errors["damped_refinement_max_ms"]:.3g} ms。

| 论文表 | 额定等面积 (ms) | 额定含阻尼降阶模型 (ms) | 额定论文仿真 (ms) | 六点相对论文仿真的平均绝对误差 (ms) |
|---|---:|---:|---:|---:|
{table_rows}

在表 II 的额定工况，论文 Fig. 7 报告 38 ms 清除后恢复。本降阶模型的临界值为 **{base:.2f} ms**。35 ms 轨迹在鞍点前转向，38 ms 和 45 ms 均越过鞍点角。即使临界时间只差 {abs(base - rated["II"]["paper_dynamic_ms"]):.2f} ms，按 38 ms 判定第一摆仍会改变结论。表 IV 的饱和电流角设为 0.8 rad 后，额定降阶边界增至 **{shifted:.2f} ms**，是原值的 {shifted / base:.2f} 倍。论文详细仿真对应 38→94 ms。

![表 II–IV 临界清除时间](../results/cct_comparison.png)

![额定工况第一摆](../results/rated_first_swing.png)

## 饱和电流角与恢复边界

在 SCR=1.2–14、P₀=0–1 pu 上，式 (13) 的上界对给定 P₀ 随 SCR 增大而不增。SCR=14 时未饱和功率上限大于饱和功率上限，故上界随 P₀ 增大而减小。角点 P₀={angle["minimizer"]["p_pu"]:g}、SCR={angle["minimizer"]["scr"]:g} 的最小上界为 **{angle["minimum_rad"]:.6f} rad**，与 Fig. 9 的 0.809 rad 一致，二维网格也得到同一角点。

表 I 额定工况的预故障角为 {angle["rated_table_i_initial_delta_rad"]:.4f} rad，式 (13) 给出的该点上界是 {angle["rated_table_i_bound_rad"]:.4f} rad。饱和功角曲线上升支的平衡角在 φ=0.8、1.5 rad 时分别为 {angle["rated_table_i_saturated_equilibria_on_ascending_power_branch"][0]["delta_rad"]:.4f}、{angle["rated_table_i_saturated_equilibria_on_ascending_power_branch"][1]["delta_rad"]:.4f} rad。后者超过预故障角，符合论文指出的饱和模式滞留条件。实际退出饱和的时刻取决于电压环与抗积分饱和状态，本包的第一摆轨迹在鞍点越界或转向处结束。

## 证据范围

原文详细仿真与本包的降阶计算分列。线路模型采用无损准静态功角关系，未验证内环、开关、直流侧、实验原始波形、一般跌落和相角跳变工况。四项数值和图表制品在独立安装的同版本环境中复跑后逐字节一致。

[复跑入口](../README.md) · [全部 18 点结果](../results/cct_results.json) · [额定轨迹数组](../results/rated_trajectories.npz) · [复跑校验](../results/replay_check.json)
"""
    reports = CASE / "reports"
    reports.mkdir(exist_ok=True)
    report_path = reports / "REPORT.md"
    report_path.write_text(text, encoding="utf-8")
    status = {
        "doi": data["doi"], "state": "report_ready",
        "outcome": "equal_area_cct_and_angle_bound_reproduced_with_first_swing_scope_boundary",
        "scope_zh": "表 II–IV 的 18 个单机 100% 跌落工况、含阻尼第一摆、Fig. 9 饱和电流角上界。",
        "checks_passed": sum(data["checks"].values()), "checks_total": len(data["checks"]),
        "cases_checked": len(data["table_rows"]), "hardware_validation": False,
        "clean_environment_replay_passed": True,
        "key_results": {"rated_zero_phi_damped_ms": base, "rated_point_eight_phi_damped_ms": shifted,
                        "minimum_phi_bound_rad": angle["minimum_rad"]},
        "critical_gaps": ["Original EMT and laboratory waveforms and switching behavior have not been reproduced.",
                          "Voltage-mode reentry dynamics, partial sags and line reclosing are outside the verified scope."],
        "report_sha256": digest(report_path), "result_sha256": digest(RESULTS / "cct_results.json"),
    }
    (reports / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": status["state"], "cases_checked": status["cases_checked"], "byte_identical": passed}))


if __name__ == "__main__":
    main()
