# 独立平衡点与稳定边界核对

把二阶方程除以 $\eta$，可写成

$$
\dot u/\eta=(A+jC-B|u|^2)u+D.
$$

其中 $A+jC=e^{j\varphi}(\varsigma^{\star *}-y)+\alpha+j\omega_\Delta/\eta$，$B=\alpha/v_\star^2$，$D=e^{j\varphi}y v_g$。令 $s=|u|^2$，则平衡点满足

$$
B^2 s^3-2ABs^2+(A^2+C^2)s-|D|^2=0.
$$

在作者 Appendix A 的 $v_g>0$、$\alpha>0$ 条件下，原点处多项式为负，正无穷处为正。每个正根给出 $u=-D/(A+jC-Bs)$。独立实数实现直接联立两个轴的稳态方程。

四阶模型用相同平衡点和网侧电流 $i=Y(u-v_g)$。设 $B_u$ 为 dVOC 项对电压的实数导数，省略乘在其前的 $\eta$，令

$$
T=\lambda I-\eta B_u,\qquad U=\lambda I+Z/l_g.
$$

对网侧电流作 Schur 消元得到特征多项式 $\det(TU+\eta R_\varphi/l_g)$。单独展开四阶系数并用 Routh–Hurwitz 判据求交越，与完整 4×4 Jacobian 的极点比较。

命题 4 的重根情形可由一个显式输入构造。取 $\alpha=v^\star=v_g=1$、$z=0.8+j0.6$、$\varphi=\arg z$、$\omega_\Delta=0$，旋转后的归一化设定值为 $1.5+j\sqrt{3}/2$。此时 $A=1.5$、$C=\sqrt{3}/2$、$B=1$、$D=1$，三次式等于 $(s-1)^3$。判别式为 0，唯一不同的正根为 1，得到一个平衡点 $u=-0.5+j\sqrt{3}/2$。因此命题 4 的严格判别式表述需另外处理三重根。

命题 5 中显示的物理时间 Jacobian 在沿平衡电压方向和正交方向旋转后，令 $a=\alpha |u_s|^2/v_\star^2$，两个对角项分别为 $\eta(A-3a)$ 和 $\eta(A-a)$，两个非对角项为 $\mp\eta C$。所以行列式为 $\eta^2[(A-2a)^2-a^2+C^2]$。作者稿显示的多项式常数项缺少外部 $\eta^2$。正增益下该因子保留常数项符号。
