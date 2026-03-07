"""
基于 JSONL 配置文件驱动的评估脚本。

使用方式:
    python run_from_config.py --config path/to/config.jsonl
    python run_from_config.py --config path/to/config.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """加载 JSON/JSONL 配置文件。

    Args:
        config_path: 配置文件路径，支持 .json 和 .jsonl 格式。

    Returns:
        解析后的配置字典。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # 尝试解析为标准 JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 尝试解析 JSONL（取第一行）
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"配置文件解析失败: {e}") from e

    raise ValueError("配置文件为空或格式不正确")


def normalize_bool(value: Any) -> bool:
    """将多种布尔表示形式统一转换为 Python bool。

    支持: True/False, "yes"/"no", "true"/"false", 1/0
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"yes", "true", "1"}
    return False


def build_model_args(model_cfg: dict[str, Any]) -> str:
    """将模型配置字典转换为 lmms-eval 的 model_args 字符串。

    Args:
        model_cfg: 模型配置字典，包含 base_url、api_key、model_name 等字段。

    Returns:
        逗号分隔的 model_args 字符串。
    """
    # 字段名映射：配置文件字段 -> model_args 参数名
    field_mapping = {
        "base_url": "base_url",
        "api_key": "api_key",
        "model_name": "model_name",
        # 本地模型常用字段
        "pretrained": "pretrained",
        "tensor_parallel_size": "tensor_parallel_size",
        "gpu_memory_utilization": "gpu_memory_utilization",
        "max_pixels": "max_pixels",
        "min_pixels": "min_pixels",
        "max_frame_num": "max_frame_num",
        "threads": "threads",
        "attn_implementation": "attn_implementation",
        "dtype": "dtype",
        "device_map": "device_map",
        "trust_remote_code": "trust_remote_code",
    }

    # 生成参数（会写入 gen_kwargs）
    gen_kwargs_fields = {
        "temperature",
        "top_p",
        "max_new_tokens",
        "top_k",
        "do_sample",
        "repetition_penalty",
    }

    model_args_parts = []
    gen_kwargs_parts = []

    for key, value in model_cfg.items():
        if key == "filter_list":
            continue  # filter_list 通过专门参数传递，不放在 model_args 中
        if key in gen_kwargs_fields:
            gen_kwargs_parts.append(f"{key}={value}")
            continue

        mapped_key = field_mapping.get(key, key)
        if value is None:
            continue
        # 字符串值中若含逗号需要特殊处理
        model_args_parts.append(f"{mapped_key}={value}")

    return ",".join(model_args_parts), ",".join(gen_kwargs_parts)


def determine_model_type(model_cfg: dict[str, Any]) -> str:
    """根据配置推断模型类型。

    Args:
        model_cfg: 模型配置字典。

    Returns:
        模型类型字符串，对应 lmms-eval 的 --model 参数。
    """
    # 优先使用显式指定的模型类型
    if "model_type" in model_cfg:
        return model_cfg["model_type"]

    # 默认使用 url_model 类型，适用于大多数在线模型和部分本地模型
    return "url_model"


def build_command(
    config: dict[str, Any],
    model_name: str,
    model_cfg: dict[str, Any],
    output_path: str,
) -> list[str]:
    """构建单次评估的命令行参数列表。

    Args:
        config: 全局配置字典。
        model_name: 模型标识名（用于输出目录区分）。
        model_cfg: 该模型的具体配置。
        output_path: 结果输出根目录。

    Returns:
        命令行参数列表。
    """
    cmd = ["python", "-m", "lmms_eval"]

    # ── 模型参数 ──────────────────────────────────────────────
    model_type = determine_model_type(model_cfg)
    cmd.extend(["--model", model_type])

    model_args_str, gen_kwargs_str = build_model_args(model_cfg)
    if model_args_str:
        cmd.extend(["--model_args", model_args_str])

    if gen_kwargs_str:
        cmd.extend(["--gen_kwargs", gen_kwargs_str])

    # ── 任务参数 ──────────────────────────────────────────────
    task_name = config.get("task_name", "")
    task_path = config.get("task_path", "")

    if task_name:
        cmd.extend(["--tasks", task_name])

    # task_path 对应 --include_path，用于加载自定义任务配置
    if task_path:
        cmd.extend(["--include_path", task_path])

    # ── 评估模式 ──────────────────────────────────────────────
    inference_only = normalize_bool(config.get("inference_only", False))
    evaluate_only = normalize_bool(config.get("evaluate_only", False))

    if inference_only:
        cmd.extend(["--mode", "predict_only"])
    elif evaluate_only:
        cmd.extend(["--mode", "eval_only"])
    else:
        cmd.extend(["--mode", "full"])

    # ── 批量大小 ──────────────────────────────────────────────
    batch_size = config.get("batch_size", 1)
    cmd.extend(["--batch_size", str(batch_size)])

    # ── 流式评估 ──────────────────────────────────────────────
    streaming_eval = normalize_bool(config.get("streaming_eval", False))
    if streaming_eval:
        cmd.append("--streaming_eval")

    inference_threads = config.get("inference_threads", 1)
    cmd.extend(["--inference_threads", str(inference_threads)])

    eval_threads = config.get("eval_threads", 2)
    cmd.extend(["--eval_threads", str(eval_threads)])

    # ── 日志参数 ──────────────────────────────────────────────
    log_samples = normalize_bool(config.get("log_samples", True))
    if log_samples:
        cmd.append("--log_samples")
        log_suffix = config.get("log_samples_suffix", model_name)
        cmd.extend(["--log_samples_suffix", log_suffix])

    # ── 检查点参数 ────────────────────────────────────────────
    enable_checkpointing = normalize_bool(config.get("enable_checkpointing", False))
    if enable_checkpointing:
        cmd.append("--enable_checkpointing")
        checkpoint_interval = config.get("checkpoint_interval", 50)
        cmd.extend(["--checkpoint_interval", str(checkpoint_interval)])

    # ── 输出路径 ──────────────────────────────────────────────
    model_output_path = str(Path(output_path) / model_name)
    cmd.extend(["--output_path", model_output_path])

    # ── 过滤器覆盖 ────────────────────────────────────────────
    filter_list = model_cfg.get("filter_list")
    if filter_list is not None:
        import json as _json
        cmd.extend(["--filter_list", _json.dumps(filter_list)])

    # ── 种子参数 ──────────────────────────────────────────────
    if "seed" in config:
        cmd.extend(["--seed", str(config["seed"])])

    return cmd


def run_evaluation(config_path: str, dry_run: bool = False) -> int:
    """解析配置并依次对每个模型执行评估。

    Args:
        config_path: 配置文件路径。
        dry_run: 若为 True，仅打印命令不执行。

    Returns:
        最终退出码（所有模型均成功则为 0）。
    """
    config = load_config(config_path)

    task_name = config.get("task_name", "unknown_task")
    output_path = config.get("output_path", f"./logs/{task_name}")
    model_configs: dict[str, dict[str, Any]] = config.get("model_config", {})

    if not model_configs:
        print("配置文件中未找到 model_config，请至少配置一个模型。", file=sys.stderr)
        return 1

    print(f"任务名称: {task_name}")
    print(f"输出目录: {output_path}")
    print(f"模型数量: {len(model_configs)}")
    print("=" * 60)

    overall_exit_code = 0

    for model_name, model_cfg in model_configs.items():
        print(f"\n🚀 开始评估模型: {model_name}")

        cmd = build_command(config, model_name, model_cfg, output_path)

        print("📌 执行命令:")
        print("   " + " ".join(cmd))
        print("-" * 60)

        if dry_run:
            print("⚠️  dry_run 模式，跳过实际执行。")
            continue

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(
                f"❌ 模型 {model_name} 评估失败，退出码: {result.returncode}",
                file=sys.stderr,
            )
            overall_exit_code = result.returncode
        else:
            print(f"✅ 模型 {model_name} 评估完成。")

    return overall_exit_code


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于 JSONL/JSON 配置文件驱动 lmms-eval 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            示例:
            python run_from_config.py --config configs/my_eval.json
            python run_from_config.py --config configs/my_eval.jsonl --dry_run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="评估配置文件路径（JSON 或 JSONL 格式）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="仅打印命令，不实际执行",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    exit_code = run_evaluation(args.config, dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()