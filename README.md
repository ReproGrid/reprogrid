# ReproGrid · 论文复现库

面向电力电子与控制的独立复现项目。目录按构网控制与稳定性、PMSM 电流控制、实时仿真和数字孪生分组。每篇论文独立记录算法、证据、优点与局限、缺失信息以及 2–3 个视频条目。

[项目网站（中英文）](https://reprogrid.org)

## 本地浏览

需要 Node.js 20+ 和 Python 3.9+，网站和数值实验均无运行时第三方依赖。

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

[RFPSC 试点说明](papers/rfpsc-linear/README.md)记录来源、实验范围和结果定义。其他六篇仍处于候选或暂缓阶段。

## 增加论文

1. 在 [候选队列](research/candidate_queue.json)按 DOI 登记元数据、状态、来源和待检验主张。
2. 在 [目录映射](research/catalog.json)补充唯一短路径、中文题名、描述与标签。研究领域由映射定义。在 [翻译数据](research/localization.json)补充对应短路径的英文信息和候选测试目标的中文说明。
3. 有运行证据时，在 `papers/<实验名>/` 添加实现、配置、结果和 `report.json`，并为论文目录项关联该实验目录。中文和英文报告里各配置同一组两到三个视频，支持本地 MP4/WebM 或外部 HTTPS 链接。大量媒体可以托管在独立视频服务，网页保留链接。
4. 重新运行构建，核对论文详情页、参数来源和视频。运行清单记录可复核的校验值与验证阈值。

[期刊配置](platform_spec.json)记录目录优先收录的期刊。原始论文、引用数据和第三方实现遵循各自来源的条款。项目自有软件使用 [MIT 许可证](LICENSE)。
