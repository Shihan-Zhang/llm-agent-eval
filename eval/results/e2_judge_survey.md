# E2 阶段 C:调查与规则层试算(由 eval/agent_error/judge/survey.py 生成,勿手改)

## 1. expected_result 的格式

共 5500 条(C1 每个文件 1 条;C2/C3 每题 2 条有效路径、C4 6 条);全部是结构化对象,没有单个值。

| 层级 | 模式 | 格式 | 条数 |
|---|---|---|---|
| C1 | P0 | tool_calls 列表 | 100 |
| C1 | P1 | tool_calls 列表 | 100 |
| C1 | P2 | tool_calls 列表 | 100 |
| C1 | P3 | tool_calls 列表 | 100 |
| C1 | P4 | tool_calls 列表 | 100 |
| C2 | P0 | {status, message, results} | 200 |
| C2 | P1 | tool_calls 列表 | 200 |
| C2 | P2 | tool_calls 列表 | 200 |
| C2 | P3 | tool_calls 列表 | 200 |
| C2 | P4 | tool_calls 列表 | 200 |
| C3 | P0 | {status, message, results} | 200 |
| C3 | P1 | tool_calls 列表 | 200 |
| C3 | P2 | tool_calls 列表 | 200 |
| C3 | P3 | tool_calls 列表 | 200 |
| C3 | P4 | tool_calls 列表 | 200 |
| C4 | P0 | {status, message, results} | 600 |
| C4 | P1 | tool_calls 列表 | 600 |
| C4 | P2 | tool_calls 列表 | 600 |
| C4 | P3 | tool_calls 列表 | 600 |
| C4 | P4 | tool_calls 列表 | 600 |

含 is_perturbed 调用的条数(按模式):P1 1100,P2 1100,P3 1100,P4 1100。P1–P4 是带故障的参考轨迹,正确值取同一题的 P0。

P0 叶子值类型(启发式:按字段名与正则):

| 类型 | 最后一步 | 全部步骤 | 最后一步的例子 |
|---|---|---|---|
| 短标签(≤3 词) | 3951/8332 = 47.4% | 28534/66349 = 43.0% | `message: "Hotel booked successfully"`; `hotel_name: "Hilton Tokyo"` |
| ID | 2293/8332 = 27.5% | 14654/66349 = 22.1% | `booking_id: "HOTEL8D99FB"`; `confirmation_code: "891392"` |
| 自由文本(>3 词) | 923/8332 = 11.1% | 5298/66349 = 8.0% | `message_preview: "Financial snapshot saved as document NOTE18E3D: BT..."`; `message: "Meeting room booked successfully"` |
| 数字 | 723/8332 = 8.7% | 13765/66349 = 20.7% | `size_mb: 300`; `attendees: 5` |
| 日期时间 | 362/8332 = 4.3% | 2844/66349 = 4.3% | `check_in_date: "2025-07-01"`; `check_out_date: "2025-07-05"` |
| 布尔 | 0/8332 = 0.0% | 1012/66349 = 1.5% |  |
| 空值 | 80/8332 = 1.0% | 242/66349 = 0.4% | `recipients: null`; `tags: null` |

## 2. ① API 失败回合:只看最终回答

失败回合 149 个(基线 80、完整回执 69)。C 类匹配上参考 ID 也不说明结果正确:回放沙盒下最终 ID 基本固定。

| 类别 | 回合 | 基线 | 完整回执 |
|---|---|---|---|
| A 没有最终回答 | 27 | 15 | 12 |
| B 最终回答含损坏值 | 22 | 14 | 8 |
| C 含某条路径全部最后一步参考 ID | 33 | 18 | 15 |
| D 其余(要 LLM 裁判) | 67 | 33 | 34 |

D 细分:不带失败措辞(启发式) 55;带失败措辞(启发式) 12;所有路径最后一步都没有 ID 类字段 9;有参考 ID,一个都不含 32;有参考 ID,只含部分 26

## 3. 脱敏要处理的痕迹(① API 推理文件)

字段:assistant.content 2087,assistant.metadata 1724,assistant.metadata.thought 1724,assistant.role 2087,assistant.tool_call 1954,assistant.type 2087,tool.call_id 1954,tool.content 1954,tool.metadata 1954,tool.metadata.perturbation_status 1954,tool.name 1954,tool.role 1954,user.content 160,user.role 160,顶层.messages 160,顶层.mode 160,顶层.task_id 160,顶层.tokens 160

工具消息 perturbation_status:baseline/clean 852,baseline/perturbed 107,full/clean 875,full/perturbed 120

助手文字含回执词汇的回合:基线 0/80,完整回执 20/80

| 中位数 | 基线 | 完整回执 |
|---|---|---|
| 输入 token | 35917 | 41653.5 |
| 工具调用数 | 9 | 10 |
| 消息数 | 20 | 22 |

用户提示两组完全相同的配对:80/80

## 4. 规则层在 ① 全部回合上的落点

| 判定 | ToolMaze 失败:基线 | ToolMaze 失败:回执 | ToolMaze 通过:基线 | ToolMaze 通过:回执 | 合计 |
|---|---|---|---|---|---|
| L1 最终回答含损坏值 → 失败 | 14 | 8 | 0 | 0 | 22 |
| L2 损坏值进入参数 → 失败 | 5 | 2 | 0 | 0 | 7 |
| L2 缺必需动作 → 失败 | 14 | 11 | 0 | 0 | 25 |
| L2 必需参数为空 → 失败 | 30 | 28 | 0 | 11 | 69 |
| L2 全部满足 → 通过 | 1 | 2 | 0 | 0 | 3 |
| L3 交给 LLM 裁判 | 16 | 18 | 0 | 0 | 34 |
| 合计 | 80 | 69 | 0 | 11 | 160 |

L3 的问题(一个回合可有多个,取问题最少的那条有效路径):ToolMaze 失败 · 多余动作 6;ToolMaze 失败 · 参考值在参数里找不到 26;ToolMaze 失败 · 替代工具完成动作 9;ToolMaze 失败 · 证据类值溯源不到 14;ToolMaze 失败 · 弱值与参考不一致 13;ToolMaze 失败 · 弱值无参考可比 1

## 5. 人工核对清单

L1/L2 判失败的 121 个回合(不含点名的)里随机抽 20 个(种子 20261109),加点名的 5 个;L3 判正确的回合待 LLM 裁判后全部补进来。

| 回合 | 组 | 来源 | ToolMaze | 规则判定 |
|---|---|---|---|---|
| C4_task_015_P4 | full | 点名 | 失败 | L3 交给 LLM 裁判(参考值在参数里找不到、弱值与参考不一致) |
| C4_task_052_P4 | baseline | 点名 | 失败 | L3 交给 LLM 裁判(参考值在参数里找不到、弱值与参考不一致) |
| C4_task_052_P4 | full | 点名 | 失败 | L2 全部满足 → 通过 |
| C3_task_039_P3 | full | 点名 | 通过 | L2 必需参数为空 → 失败 |
| C4_task_014_P3 | full | 点名 | 通过 | L2 必需参数为空 → 失败 |
| C3_task_001_P3 | full | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_013_P4 | baseline | 随机 | 失败 | L1 最终回答含损坏值 → 失败 |
| C3_task_026_P4 | baseline | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_042_P3 | baseline | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_043_P4 | full | 随机 | 失败 | L1 最终回答含损坏值 → 失败 |
| C3_task_050_P3 | baseline | 随机 | 失败 | L1 最终回答含损坏值 → 失败 |
| C3_task_051_P4 | full | 随机 | 失败 | L1 最终回答含损坏值 → 失败 |
| C3_task_052_P3 | baseline | 随机 | 失败 | L2 损坏值进入参数 → 失败 |
| C3_task_060_P3 | baseline | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_071_P3 | baseline | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_084_P4 | full | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C3_task_092_P4 | baseline | 随机 | 失败 | L1 最终回答含损坏值 → 失败 |
| C3_task_097_P3 | full | 随机 | 失败 | L2 缺必需动作 → 失败 |
| C4_task_001_P4 | baseline | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C4_task_001_P4 | full | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C4_task_017_P4 | full | 随机 | 通过 | L2 必需参数为空 → 失败 |
| C4_task_071_P4 | full | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C4_task_083_P3 | full | 随机 | 失败 | L2 必需参数为空 → 失败 |
| C4_task_084_P4 | baseline | 随机 | 失败 | L2 缺必需动作 → 失败 |
| C4_task_096_P4 | full | 随机 | 失败 | L2 缺必需动作 → 失败 |

## 6. 管线检查:schema 只写 type: object 的参数

ToolMaze 工具定义里这类参数 111 个(都没有 properties)。各次运行里模型传的值:

| 运行 | 空 {} | 非空对象 | 其他 |
|---|---|---|---|
| ① API Qwen3.5-35B-A3B(硅基流动) | 793 | 0 | 0 |
| ① 本地 qwen3i-8k(Ollama) | 0 | 186 | 0 |

API 一律为空、本地一律非空:空对象来自 API 端(推测是服务端对无 properties 的 object 做受约束解码或解析),不是模型行为。API 运行里这类动作都带着空数据执行,第 4 节的必需参数为空多数源于此。
