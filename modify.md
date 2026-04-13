# lmms-eval Agentic 测评逻辑修改记录

本文档汇总了为了支持 Agentic（多轮环境交互式）任务而在 `lmms-eval` 中进行的架构和代码修改。本实现摒弃了 `lmms-eval-ref` 中会导致 Batch Size 退化为 1 的单例阻塞循环，而是采用了**状态机请求队列 (Stateful Request Queue)**，完美保留了并发推理（Batching）性能。

## 1. 核心架构设计：引入 `AgenticInstance`
**文件:** `lmms_eval/api/instance.py`

- 在 `Instance` 类的 `request_type` Literal 中增加了 `"generate_until_agentic"`。
- 新增 `AgenticInstance(Instance)` 子类：
  - **职责**: 封装了任务环境的状态模拟。它保存了当前轮次、最大步数、历史轨迹、以及当前环境的回调函数 `env_step_fn`。
  - **核心方法**: `step(model_response: str) -> bool`。该方法接受模型的输出，调用 `env_step_fn`（通常是 task 的 `doc_to_text`），获取新的 `next_context` 和 `visuals`。然后它会自动覆盖自身的 `arguments`（模型推理所需的 prompt），并返回当前 episode 是否结束（`is_done`）。结束时自动将完整的 `history_trace` 放入 `resps`。

## 2. 任务解析：拦截 Agentic 请求并创建 `AgenticInstance`
**文件:** `lmms_eval/api/task.py`

- 将 `"generate_until_agentic"` 添加至 `ALL_OUTPUT_TYPES` 列表中。
- 修改 `ConfigurableTask.construct_requests` 和 `ConfigurableMessagesTask.construct_requests`：
  - 当 `self.OUTPUT_TYPE == "generate_until_agentic"` 时，提取 `generation_kwargs` 中的 `max_agentic_steps` (默认 10)，并在深拷贝后将其 `pop` 出去以免影响模型自带的 generate 函数。
  - 获取 `self.config.doc_to_text` 并将其作为 `env_step_fn` 传递。
  - 返回实例化的 `AgenticInstance` 而非普通的 `Instance`。
- 修改 `process_results`：如果类型是 `"generate_until_agentic"`，不再强行对结果调用 `.strip()`，以保留多轮轨迹可能生成的 JSON 字典或嵌套结构。

## 3. 调度器改造：支持高并发的 Agentic 活跃队列 (Active Queue)
**文件:** `lmms_eval/evaluator.py`

- 摒弃了对单条请求套用 `for round_idx in range(...)` 的做法。
- 在 `evaluate`（全局推理部分）和 `evaluate_streaming`（流式推理部分）中修改了 `getattr(lm, reqtype)(cloned_reqs)` 的逻辑：
  - 如果 `reqtype == "generate_until_agentic"`，初始化 `active_requests = cloned_reqs`（全量数据）。
  - 进入 `while active_requests:` 循环：
    1. **批量推理**：直接调用 `lm.generate_until(active_requests)`。所有活着的 agent 都在这一个大 Batch 里并发执行。
    2. **环境状态更新**：遍历模型结果，调用每个 `req.step(resp)`。
    3. **过滤存活请求**：如果 `not is_done`，则将其加入 `next_active_requests`，进入下一轮。
  - 这种设计让 vLLM、SGLang 等后端可以在包含多个进度不同的 Agent 任务时依然满载运行。

## 4. 指标和注册表的兼容
**文件:** `lmms_eval/api/metrics.py` 和 `lmms_eval/api/registry.py`

- 在各类 Metric 函数（如 exact_match）的 `output_type` 白名单中加入了 `"generate_until_agentic"`。
- 在 `registry.py` 中注册该类型，使系统能够正常调度评估函数。

---
*注：由于此方案在 Evaluator 层维持了 `generate_until` 的统一接口调用，`lmms_eval/models/` 目录下的所有模型实现**无需做任何修改**，只要模型原生支持 `generate_until`，即可开箱即用地支持 Agentic 任务。*

## 5. 修复 `evaluate_streaming` 多线程模式下的方法调用错误
**文件:** `lmms_eval/evaluator.py`

- 在 `batch_size == 1` 时流式评测使用 `ThreadPoolExecutor` 和 `process_single_request`。
- 原实现强行通过 `getattr(lm, reqtype)(instance)` 调用，导致在 Agentic 模式下触发 `'URLModel' object has no attribute 'generate_until_agentic'`。
- 已将 `process_single_request` 改写：当 `reqtype == "generate_until_agentic"` 时，使用一个内部的 `while not is_done:` 循环，并且底层调用强制使用 `lm.generate_until([instance])` 传递单个请求实例的列表。这样完美兼容了只实现了 `generate_until` 而不了解 agentic 逻辑的通用模型类。

## 6. 修复自定义模型（如 `url_model`）参数不规范引发的 AttributeError
**文件:** `lmms_eval/models/chat/url_model.py`

- 发现 `url_model` 的 `generate_until` API 没有遵守 lmms-eval 标准的列表输入规范（它期待单一的 `request` 而非 `requests: List[Instance]`）。
- 在并发评测传递单例列表 `[instance]` 时，由于传入的是 list 而抛出了 `AttributeError: 'list' object has no attribute 'arguments'`。
- **修复**: 修改了 `url_model.py` 中的 `generate_until` 方法，使其标准地接收 `requests` 列表输入，并循环处理返回结果列表。这保证了底层与整个并发评测框架（以及所有其他的内置模型）的一致性。
