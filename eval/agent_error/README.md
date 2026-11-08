# Agent 出错以后:错误恢复与失败报告评测

复现公开基准,不自建测试集。外部代码和数据放在 `external/agent_error/`(不进仓库,只引用);
这个目录只放我们写的适配脚本、配置和汇总。

| 基准 | 测什么 | 链路环节 | 来源 |
|---|---|---|---|
| ToolMaze | 工具出错后能否恢复:显式/隐式 × 暂时/持续(P1–P4),P0 无扰动 | 执行工具、观察回填 | [论文](https://arxiv.org/abs/2606.05806) · [代码](https://github.com/Zhudongsheng75/ToolMaze) |
| CRITICTOOL | 模型自己犯的错(选错工具、参数错)能否发现和改正 | 生成调用 | [论文](https://arxiv.org/abs/2506.13977) · [代码](https://github.com/Shellorley0513/CriticTool) |
| Failure-Transparent Agents | 工具失败后会不会谎称成功、编造细节;"证据契约"提示词 | 最终回答 | [论文](https://arxiv.org/html/2609.35732)(数据未发布,只复现方法) |

RAG 检索不可靠时的拒答(RGB)放在 `eval/rag/`。

## 目录

```
external/agent_error/toolmaze/     ToolMaze 代码 + data/(scripts/e0_setup.ps1 下载)
external/agent_error/critictool/   CRITICTOOL 代码 + CriticTool-Dataset/
external/agent_error/results/      运行结果(不进仓库)
eval/agent_error/configs/          我们的运行配置
```

## 接入 ToolMaze 时处理的问题(读源码发现)

1. **P0 与 P1–P4 的系统提示词不同**(`evaluation/agents/openai_agent.py`):P1–P4 多了"工具可能失败"的故障处理指引,
   默认设置下 P0 与 P2 的差异混杂了扰动和提示词两个因素。正式实验把提示词作为独立因素
   (无提示 / 故障提示 / 证据契约)与 P0–P4 交叉,P0 列即过度拒绝的对照。
2. **上下文**:官方 `max_tokens` 32000、最多 25 轮,工具结果原样拼进对话。本地改为 `max_tokens` 2048,
   并用固定 `num_ctx 8192` 的模型变体(OpenAI 兼容接口不能逐请求传 `num_ctx`,否则 Ollama 静默截断)。
3. **暂时性扰动按工具名计数**(`sandbox.py::_should_perturb`):同名工具第二次调用无论参数都放行,
   "恢复"可能是碰巧;分析时读轨迹确认模型是否识别了坏数据。
4. **判分看轨迹和最终回答**,工具基本无状态,部分依赖 LLM 裁判 → 裁判用不同家族(硅基流动上的 DeepSeek),人工抽查校准。
5. **判分按路径而不是按结果**(E0 发现):P0 要求轨迹包含标准路径的每个工具;用功能等价的"一步到位"工具
   (如 `book_flight_with_status_check` 代替 `build_flight_booking_request` + 预订)会被判失败,即使确认码已拿到。
   与论文对比用官方判分;失败回合另做人工审计,报告"路径不同但结果正确"的比例。
6. **判分是纯规则的**:`judge.py` 只保存了 LLM 客户端,公开代码里从未调用;判分确定、零成本。
   LLM 裁判(硅基流动 DeepSeek-V3.2)改为用于失败回合的"结果是否正确"审计,并用人工标注校准(Cohen's κ)。
7. **并发**:本地 Ollama 串行处理,并发设 1;耗时用服务端计时,不用客户端墙钟时间。

## 运行

```
scripts\run_e0.bat        E0:安装、下载数据、建模型变体、跑 5 道 P0 题(日志 external\logs\e0.log)
scripts\run_e1.bat        E1:官方设置复现 C1,100 题 × P0–P4(日志 external\logs\e1.log,可中断后重跑续跑)
python eval\agent_error\summarize_e1.py   从结果文件重算 Wilson 区间、主要比较、"声称重试"清单 → eval\results\e1_summary.md
```

## E0 结果(2026-11-05,管线检查)

本地 qwen3i-8k,C1 前 5 道 P0 题:官方判分 4/5;热启动每题 6.1–10.7 s;每轮输出均值约 74 token;
显存增加 3,677 MiB(预测 3,789 MiB)。详见 `eval/results/predictions.md`。

## E1 结果(2026-11-05,官方设置复现 C1,100 题 × 5 模式,本地 qwen3i-8k)

| 模式 | TSR [95% Wilson] | PRR | RC | 命中率 |
|---|---|---|---|---|
| P0 无扰动 | 70% [60, 78] | — | — | — |
| P1 显式·暂时 | 9% [5, 16] | 10/91 = 11% | 0.90 | 91% |
| P2 显式·持续 | 63% [53, 72] | 76/92 = 83% | 0.32 | 92% |
| P3 隐式·暂时 | 0% [0, 4] | 0/90 = 0% | 1.00 | 90% |
| P4 隐式·持续 | 13% [8, 21] | 21/90 = 23% | 0.86 | 90% |

主要比较(事先指定):隐式 vs 显式,配对 200 对,59 : 0,精确 McNemar p ≈ 3.5 × 10⁻¹⁸;PRR 相对下降 75%(论文报告约 37%)。
4B 模型的两种失败:显式错误一律放弃(P1),隐式错误一律信任(P3);另有 22 个回合声称重试过但其实没有。
总耗时 68.6 分钟。详细对照与局限见 `eval/results/predictions.md`;上表的区间、配对计数和 22 个回合的清单
(P1 12、P2 9、P4 1)由 `summarize_e1.py` 生成,见 `eval/results/e1_summary.md`。
