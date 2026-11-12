# 无属性对象参数的最小复现(由 eval/agent_error/schema_probe/probe.py summarize 生成,勿手改)

- 日期 2026-11-09;请求取自 `external\agent_error\results\e2\api\main\baseline\Qwen_Qwen3.5-35B-A3B\fc\c4\inferences\C4_task_010_P3_inference.json` 第 13 条消息之前(任务 C4_task_010 P3,基线组;此前工具结果都未注入)。
- ① 原运行里这一步调用 `archive_weatherapi_alert_card_native`,参数 `alert_card` 传成 {}。
- 请求参数:{"temperature": 0, "max_tokens": 16000, "parallel_tool_calls": false, "timeout": 300, "extra_body": {"enable_thinking": false}};每个变体 3 次。

| 变体 | 模型 | 调用的工具 | `alert_card` | 原始 arguments(截断) |
|---|---|---|---|---|
| A 原 schema #1 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 空 {} | `{"alert_card": {}, "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
| A 原 schema #2 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 空 {} | `{"alert_card": {}, "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
| A 原 schema #3 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 空 {} | `{"alert_card": {}, "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
| B additionalProperties: true #1 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| B additionalProperties: true #2 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| B additionalProperties: true #3 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| C 按参考补 properties #1 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| C 按参考补 properties #2 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| C 按参考补 properties #3 | Qwen/Qwen3.5-35B-A3B | `archive_weatherapi_alert_card_native` | 非空对象 | `{"alert_card": {"alert_card_id": "wapi_new_york", "city_label": "New York", "celsius_now": 15, "sky_text": "Rain", "advice_level": "normal"}, "title": "Post-pur` |
| D 原 schema,DeepSeek-V3.2 #1 | deepseek-ai/DeepSeek-V3.2 | `archive_weatherapi_alert_card_native` | 其他(str) | `{"alert_card": "", "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
| D 原 schema,DeepSeek-V3.2 #2 | deepseek-ai/DeepSeek-V3.2 | `archive_weatherapi_alert_card_native` | 其他(str) | `{"alert_card": "", "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
| D 原 schema,DeepSeek-V3.2 #3 | deepseek-ai/DeepSeek-V3.2 | `archive_weatherapi_alert_card_native` | 其他(str) | `{"alert_card": "", "title": "Post-purchase weather archive", "user_id": "user_123", "priority": "medium", "channel": "email"}` |
