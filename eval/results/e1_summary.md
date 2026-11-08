# E1 汇总(由 eval/agent_error/summarize_e1.py 生成,勿手改)

结果目录:`external/agent_error/results/e1/qwen3i-8k/fc/c1`;判分文件 500 个,推理文件 500 个。

| 模式 | 通过 / 总数 | TSR [95% Wilson] | 命中率 | PRR(上游) | RC(上游) |
|---|---|---|---|---|---|
| P0 | 70/100 | 70% [60%, 78%] | — | — | — |
| P1 | 9/100 | 9% [5%, 16%] | 91/100 | 11.0% | 0.90 |
| P2 | 63/100 | 63% [53%, 72%] | 92/100 | 82.6% | 0.32 |
| P3 | 0/100 | 0% [0%, 4%] | 90/100 | 0.0% | 1.00 |
| P4 | 13/100 | 13% [8%, 21%] | 90/100 | 23.3% | 0.86 |

## 主要比较(事先指定):隐式 vs 显式

配对 200 对((P1, P3) 与 (P2, P4),每题各 1 对)。只有显式通过 59 对,只有隐式通过 0 对;精确 McNemar 双侧 p = 3.47e-18(= 2 × P(Binomial(59, 1/2) ≤ 0))。

## 声称重试过但没有(描述性,启发式)

规则:最终回答匹配 `retr(y|ied|ies|ying)`,且至少一个被注入故障的工具只调用了 1 次。共 22 个回合:P1 12、P2 9、P4 1。

| 模式 | 回合 | 只调用 1 次的故障工具 |
|---|---|---|
| P1 | C1_task_001 | book_flight_with_status_check |
| P1 | C1_task_006 | create_document |
| P1 | C1_task_009 | compute_trade_amount |
| P1 | C1_task_018 | estimate_trading_cost |
| P1 | C1_task_019 | generate_payment_invoice |
| P1 | C1_task_025 | convert_datetime |
| P1 | C1_task_034 | build_checkout_from_hold |
| P1 | C1_task_035 | get_available_coupons |
| P1 | C1_task_039 | build_room_booking_request |
| P1 | C1_task_043 | get_stock_level |
| P1 | C1_task_050 | estimate_trading_cost |
| P1 | C1_task_073 | generate_invoice_from_cart |
| P2 | C1_task_006 | create_document |
| P2 | C1_task_018 | estimate_trading_cost |
| P2 | C1_task_019 | generate_payment_invoice |
| P2 | C1_task_035 | get_available_coupons |
| P2 | C1_task_041 | order_items |
| P2 | C1_task_050 | estimate_trading_cost |
| P2 | C1_task_068 | place_buy_order_with_risk_check |
| P2 | C1_task_081 | assess_trade_risk |
| P2 | C1_task_096 | build_device_restart_request |
| P4 | C1_task_050 | estimate_trading_cost |
