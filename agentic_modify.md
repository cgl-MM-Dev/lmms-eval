# Agentic 测评逻辑高并发架构修改清单

本文档详细记录了在 `lmms-eval` 框架中引入支持高并发 Batch 推理的多轮交互（Agentic）测评逻辑的各项架构级修改。

## 🎯 架构设计理念
传统的 Agentic 测评往往在单个任务内部写死 `for` 循环来处理多轮对话，这会导致底层推理引擎（如 vLLM）的 Batch Size 退化为 1，严重拖慢测评速度。
本次修改引入了 **“无状态模型 + 有状态请求队列（Stateful Request Queue） + 环境模拟器”** 的解耦架构。使得不同任务、不同轮次的请求能够被凑成一个大 Batch 交给模型统一推理。

---

## ✅ 核心修改 To-Do List

- [x] **`lmms_eval/api/instance.py`**
  - [x] 继承基础 `Instance` 类，新增 `AgenticInstance` 数据类。
  - [x] 增加状态属性：`current_step`, `max_steps`, `history_trace`, `previous_round_info`, `is_done`。
  - [x] 绑定环境函数：增加 `env_step_fn` 属性（承接任务侧的 `doc_to_text`）。
  - [x] 实现 `step(model_response: str) -> bool` 核心状态机推进方法：
    - 记录模型输出到 `history_trace`。
    - 调用 `env_step_fn` 获取环境反馈 `(visuals, next_context, terminal_signal, updated_outputs, next_round_info)`。
    - 动态重构下一步的 `arguments`（例如针对包含新图像的环境反馈，动态重写 `doc_to_messages` 函数拼接图文 prompt）。
    - 处理终止信号并将最终轨迹存入 `resps`。

- [x] **`lmms_eval/api/task.py`**
  - [x] 在 `ConfigurableTask` 与 `ConfigurableMessagesTask` 的 `construct_requests` 逻辑中新增对 `OUTPUT_TYPE == "generate_until_agentic"` 的识别。
  - [x] 解析任务配置中的 `max_agentic_steps`。
  - [x] 实例化 `AgenticInstance` 时，将包装好的 `doc_to_text` 作为 `env_step_fn` 传入。

- [x] **`lmms_eval/evaluator.py`**
  - [x] 改造主评估调度循环中针对不同 `reqtype` 的处理分支。
  - [x] 新增针对 `generate_until_agentic` 的 **Stateful Request Queue** 逻辑：
    - 初始化 `active_requests = cloned_reqs` 队列。
    - 构建 `while active_requests:` 循环。
    - 批量调用模型推理：`resps = lm.generate_until(active_requests)`，最大化利用底层框架吞吐量。
    - 遍历当前批次的回复，调用 `req.step(resp_text)` 推进每个请求的状态。
    - 将未结束（`is_done == False`）的请求保留到下一轮队列 `next_active_requests` 中继续参与下一次 Batch 推理。
    
- [x] **`lmms_eval/models/chat/url_model.py` (模型层兼容修补)**
  - [x] 修改 `generate_until(self, requests)` 入参处理逻辑。
  - [x] 增加防御性判断 `if not isinstance(requests, list): requests = [requests]`，确保框架层面传来的单请求也能被正确列表化并处理，防止发生 `AttributeError` 等兼容性崩溃。

- [x] **`lmms_eval/test_lmm_eval/arc_agi_2/` (具体 Agentic 任务适配)**
  - [x] 针对 ARC-AGI-2 数据集复杂的嵌套 NumPy Array 和图像 Bytes 进行解包，转为 `jsonl` + `media` 目录的形式以避免编码阻塞。
  - [x] 更新 `utils.py` 中 `_to_rgb_image` 逻辑，兼容基于路径的 Agentic 测试环境图像加载。

