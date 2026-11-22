# 第 21 篇来源和实现核对

Li、Gu 和 Green 的论文《Revisiting Grid-Forming and Grid-Following Inverters: A Duality Theory》，IEEE TPWRS，DOI 10.1109/TPWRS.2022.3151851。本次锁定作者稿 v3 和补充工程 commit `8f73752332d6c2144570ac101e556f1859c06930`，复算 Fig. 8–10 的单台逆变器并网案例。 [作者稿](https://arxiv.org/abs/2105.13094v3) · [作者项目](https://github.com/Future-Power-Networks/Simplus-Grid-Tool/tree/8f73752332d6c2144570ac101e556f1859c06930)

| 输入 | 解析脚本 | 原始时域工程 |
|---|---|---|
| 基频、滤波 LC | 50 Hz，ω₀Lf=0.05 p.u.，ω₀Cf=0.02 p.u. | 同值 |
| GFM 电流环、额定电压环 | 500 Hz、250 Hz，电压积分增益另乘 20 | 同值，PI 用前向 Euler |
| GFM 下垂和低通 | 0.05 p.u.、15 Hz | 同值，低通为前向 Euler 传递函数 |
| GFL 电流环、PLL | 250 Hz、15 Hz，PLL 比例为 ωPLL、积分为 ωPLL²/4 | 同值，PLL 前增加 1 kHz 低通 |
| 同步参考工作点 | GFM 为 V=1、I=0。GFL 为 V=1、负载方向 I=−0.5 | GFM vcdq≈[1,0]，GFL 三组实际 vd≈1.028–1.033 |
| 变流器侧 Rf | GFM 0.01 p.u.，GFL 0.005 p.u. | GFM 0.005 p.u.，GFL 网强例约 1e−9 p.u.，两组增益例 0.01 p.u. |
| 网络与增益变化 | 每组 7 点，X/R=5 | 2.0 s 改变，2.2 s 恢复，绘图减 1.8 s |
| 离散电路与输出 | 解析连续时间模型 | 100 µs，R2020b，powergui Tustin/Backward Euler，abc 受控电压延迟一个采样 |

以上设置由 `Case2_Sim.slx`、链接库 `SimplexPS.slx`、两个 CalcSwing 脚本、`Case2_SimPara.m` 与 `Plot_Case2.m` 核对。增益缩放中比例支路乘缩放系数，积分支路经 magnitude² 再乘，Fig. 10 的 0.6 倍带宽对应 Ki 的 0.36 倍。GFL PLL 和电流带宽案例的展示频率分别再通过一级和两级 100 Hz 低通。 [原文件哈希和自动提取值](source_audit.json)

Fig. 8 的网强变化采用实际两段串联电感与跨接其中一段的断路器。GFM 主段 X=0.1、被跨接段 X=0.2 p.u.，GFL 主段 X=0.4、被跨接段 X=0.2 p.u.，各段 R=X/5。闭合电阻为 0.01 p.u.，开路保留 1e6 p.u. 电阻缓冲。复建保留两段电流的状态，在改变跨接电阻时继续积分。四组增益例另有 10000 p.u. 电阻的 100 µs 负荷脉冲。极小 Lc=1e−9 p.u. 在独立交流模型中并入外部线路，此项为显式近似。

CalcSwingPLL.m 的 `Y_inner_n = I*G_PLL*Lf/2` 直接作为非对角导纳。原脚本此处保留了该表达式。复算按原式生成原图极点，另计算该项除以 Z_inner 的敏感性。这项差异对 60 Hz PLL 工况的主导极点实部有可见影响，报告保留了两组数值。

六个 MAT 文件每个包含 30001 个采样点、8 组信号，起止为 0 和 3 s。MAT 时间及单位按作者绘图处理。模型初始化通过电路和控制状态的周期固定点求解，未用暂态幅值调节元件、脉冲或控制增益。对记录输入的同步环重放只使用原始 igd、vcq 和固定的原始参数。

原始 R2020b 电力系统专用工具箱的 TBE 切换历史更新仍待逐步重建。本地未发现可执行 MATLAB 或 Octave。独立实现采用梯形电路和前向 Euler 控制，另计算切换点改用一次后向 Euler、以及固定电压精确传播的敏感性。所有时域幅值对照按各自的模型条件报告。 [输入协议](../configs/protocol.json)
