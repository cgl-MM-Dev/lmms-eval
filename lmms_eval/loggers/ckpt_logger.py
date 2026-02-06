import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import filelock
from loguru import logger as eval_logger

from lmms_eval.utils import (
    handle_non_serializable,
    sanitize_list,
    sanitize_model_name,
    unflatten_dict
)

class CheckpointLogger:
    """
    checkpoint logger that preserves all evaluation context.
    
    Checkpoint format includes:
    - doc_id: Document identifier
    - filter_key: Filter applied to results
    - metrics: All computed metrics for this (doc_id, filter_key) pair
    - resps, filtered_resps, doc, target, etc.: Full evaluation context
    """

    @staticmethod
    def _extract_model_name(model_args: str) -> str:
        """Extracts the model name from the model arguments."""  
        def extract_model_name(model_args: str, key: str) -> str:
            """Extracts the model name from the model arguments using a key."""
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # order does matter, e.g. peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "model_version=", "model_name=", "model_id=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)
        return ""

    def __init__(
        self,
        output_path: str,
        model_name: str,
        enable_checkpointing: bool = True,
        checkpoint_interval: int = 50,
    ):
        """
        Initialize checkpoint logger.

        Args:
            output_path: Base path for saving checkpoints
            model_name: Name of the model being evaluated
            enable_checkpointing: Enable checkpoint-based resume
            checkpoint_interval: Save checkpoint every N samples
        """
        self.output_path = Path(output_path)
        # 如果model_name包含 "="，说明传入的是model_args，需要提取模型名
        if "=" in model_name:
            model_name = self._extract_model_name(model_name)
        self.model_name_sanitized = sanitize_model_name(model_name)
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval

        if not self.enable_checkpointing:
            eval_logger.info("Checkpointing is disabled")
            return

        # Create checkpoint directory
        self.ckpt_dir = self.output_path / self.model_name_sanitized / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Distributed settings
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Track completed (doc_id, filter_key) pairs per task
        # Format: {task_name: {(doc_id, filter_key)}}
        self.completed_samples: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        # Sample buffers for batched writes
        self.sample_buffers: Dict[str, List[dict]] = defaultdict(list)

        # Thread lock for file operations
        self.lock = threading.Lock()

        # Prefix for checkpoint files
        self.checkpoint_prefix = "checkpoint"

        eval_logger.info(f"CheckpointLogger initialized at {self.ckpt_dir}")
        eval_logger.info(f"Rank {self.rank}/{self.world_size}, checkpoint interval: {checkpoint_interval}")

        # Auto-load existing checkpoints
        self._load_all_checkpoints()

    def _get_sample_file_path(self, task_name: str) -> Path:
        """Get sample file path."""
        if self.world_size > 1:
            return self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task_name}_rank{self.rank}.jsonl"
        return self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task_name}.jsonl"

    def _load_all_checkpoints(self):
        """Load all existing checkpoints from JSONL files."""
        if not self.ckpt_dir.exists():
            return

        # Find all sample files for this rank
        pattern = f"*_samples_*_rank{self.rank}.jsonl" if self.world_size > 1 else "*_samples_*.jsonl"

        for sample_file in self.ckpt_dir.glob(pattern):
            # Extract task name from filename
            filename = sample_file.stem
            parts = filename.split("_samples_")
            if len(parts) != 2:
                continue

            task_name_part = parts[1]
            # Remove rank suffix if exists
            if self.world_size > 1 and task_name_part.endswith(f"_rank{self.rank}"):
                task_name = task_name_part[: -len(f"_rank{self.rank}")]
            else:
                task_name = task_name_part

            # Load completed samples (doc_id, filter_key pairs)
            try:
                with open(sample_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            sample = json.loads(line.strip())
                            doc_id = str(sample.get("doc_id"))
                            filter_key = sample.get("filter_key", "none")
                            
                            if doc_id:
                                self.completed_samples[task_name].add((doc_id, filter_key))
                        except json.JSONDecodeError:
                            continue

                eval_logger.info(
                    f"[{task_name}] Loaded checkpoint: "
                    f"{len(self.completed_samples[task_name])} completed samples"
                )
            except Exception as e:
                eval_logger.warning(f"Failed to load checkpoint for {task_name}: {e}")

    def load_historical_metrics(self, task_name: str) -> Tuple[dict, list]:
        """
        load historical metrics and samples from checkpoint file.
        
        Args:
            task_name: the name of the task
            
        Returns:
            (historical_metrics, historical_samples)
            - historical_metrics: dict mapping (metric, filter_key) -> list of values
            - historical_samples: list of sample dicts (for log_samples)
        """
        if not self.enable_checkpointing:
            return {}, []
        
        sample_file = self._get_sample_file_path(task_name)
        
        if not sample_file.exists():
            eval_logger.debug(f"[{task_name}] No checkpoint file found")
            return {}, []
        
        historical_metrics = defaultdict(list)
        historical_samples = []
        seen_samples = set()  # Track (doc_id, filter_key) pairs
        
        try:
            with open(sample_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        doc_id = str(sample.get("doc_id"))
                        filter_key = sample.get("filter_key", "none")
                        
                        # 避免重复加载相同的 (doc_id, filter_key)
                        sample_key = (doc_id, filter_key)
                        if sample_key in seen_samples:
                            continue
                        seen_samples.add(sample_key)
                        
                        # 提取所有指标（排除元数据字段）
                        excluded_fields = {
                            "doc_id", "doc", "target", "resps", "filtered_resps", "doc_hash", "arguments", "filter_key"
                        }
                        
                        # 提取指标并使用正确的filter_key
                        for key, value in sample.items():
                            if key not in excluded_fields:
                                # 使用checkpoint中保存的filter_key
                                historical_metrics[(key, filter_key)].append(value)
                        
                        # 保存完整样本（用于log_samples）
                        historical_samples.append(sample)
                        
                    except json.JSONDecodeError as e:
                        eval_logger.warning(f"Failed to parse checkpoint line: {e}")
                        continue
            
            eval_logger.info(
                f"[{task_name}] Loaded historical data: "
                f"{len(seen_samples)} samples, "
                f"{len(historical_metrics)} metric types, "
                f"{sum(len(v) for v in historical_metrics.values())} data points"
            )
            
            return dict(historical_metrics), historical_samples
            
        except Exception as e:
            eval_logger.error(f"Failed to load historical data for {task_name}: {e}")
            return {}, []

    def is_sample_completed(self, task_name: str, doc_id: str, filter_key: str) -> bool:
        """
        Check if a (doc_id, filter_key) sample has been completed.
        
        Args:
            task_name: Name of the task
            doc_id: Document ID
            filter_key: Filter key used
            
        Returns:
            True if this sample is already completed
        """
        if not self.enable_checkpointing:
            return False
        return (str(doc_id), filter_key) in self.completed_samples.get(task_name, set())

    def log_sample(self, task_name: str, sample: dict, filter_key: str):
        """
        Log a sample with its filter_key.

        Args:
            task_name: Name of the task
            sample: Sample dictionary containing doc_id, responses, and metrics
            filter_key: The filter key used for this sample
        """
        if not self.enable_checkpointing:
            return

        doc_id = sample.get("doc_id")
        if doc_id is None:
            eval_logger.warning(f"Sample missing doc_id, skipping checkpoint")
            return

        # Convert doc_id to string for consistency
        doc_id = str(doc_id)

        with self.lock:
            # Add filter_key to sample before saving
            sample_with_filter = sample.copy()
            sample_with_filter["filter_key"] = filter_key
            
            # Add to buffer
            self.sample_buffers[task_name].append(sample_with_filter)

            # Mark as completed
            self.completed_samples[task_name].add((doc_id, filter_key))

            # Save checkpoint if buffer is full
            if len(self.sample_buffers[task_name]) >= self.checkpoint_interval:
                self._flush_samples(task_name)

    def _flush_samples(self, task_name: str, force: bool = False):
        """Flush samples to disk."""
        if not self.enable_checkpointing:
            return

        if not force and len(self.sample_buffers[task_name]) < self.checkpoint_interval:
            return

        if len(self.sample_buffers[task_name]) == 0:
            return

        sample_file = self._get_sample_file_path(task_name)
        lock_file = sample_file.with_suffix(".lock")

        try:
            # Use file lock for multi-process safety
            with filelock.FileLock(str(lock_file), timeout=10):
                # Write samples as JSONL
                with open(sample_file, "a", encoding="utf-8") as f:
                    for sample in self.sample_buffers[task_name]:
                        # Clean up sample data
                        cleaned_sample = self._clean_sample(sample)

                        # Write as JSONL
                        sample_json = json.dumps(
                            cleaned_sample,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        f.write(sample_json + "\n")

                # Clear buffer
                num_saved = len(self.sample_buffers[task_name])
                self.sample_buffers[task_name].clear()

                eval_logger.debug(
                    f"[{task_name}] Saved checkpoint: {num_saved} samples, "
                    f"total {len(self.completed_samples[task_name])} completed"
                )

        except filelock.Timeout:
            eval_logger.warning(f"Failed to acquire lock for {sample_file}")
        except Exception as e:
            eval_logger.error(f"Failed to save checkpoint for {task_name}: {e}")

    def _clean_sample(self, sample: dict) -> dict:
        """Clean sample data while preserving filter_key."""
        cleaned = sample.copy()

        # Sanitize resps and filtered_resps
        if "resps" in cleaned:
            cleaned["resps"] = sanitize_list(cleaned["resps"])

        if "filtered_resps" in cleaned:
            cleaned["filtered_resps"] = sanitize_list(cleaned["filtered_resps"])

        if "doc" in cleaned and isinstance(cleaned["doc"], dict):
            doc = cleaned["doc"].copy()
            # 如果 doc 中有 metadata 字段
            if "metadata" in doc:
                flat_metadata = doc.pop("metadata")
                if flat_metadata and isinstance(flat_metadata, dict):
                    # 使用反扁平化
                    nested_metadata = unflatten_dict(flat_metadata)
                    if nested_metadata:
                        doc["metadata"] = nested_metadata
            cleaned["doc"] = doc

        return cleaned

    def flush(self, task_name: Optional[str] = None):
        """
        Force save all buffered samples.

        Args:
            task_name: Specific task to flush, or None for all tasks
        """
        if not self.enable_checkpointing:
            return

        tasks = [task_name] if task_name else list(self.sample_buffers.keys())

        for task in tasks:
            if task in self.sample_buffers:
                self._flush_samples(task, force=True)

    def get_remaining_docs(self, task_name: str, all_doc_ids: List[str]) -> List[str]:
        """
        Get list of documents that need to be processed.
        
        Note: This returns doc_ids that have NO completed samples yet.
        A doc_id is considered "remaining" if it has not been processed for ANY filter.

        Args:
            task_name: Name of the task
            all_doc_ids: List of all document IDs

        Returns:
            List of document IDs that haven't been completed
        """
        if not self.enable_checkpointing:
            return all_doc_ids

        # Get all completed doc_ids (regardless of filter_key)
        completed_doc_ids = set(
            doc_id for doc_id, _ in self.completed_samples.get(task_name, set())
        )
        
        remaining = [
            str(doc_id) for doc_id in all_doc_ids 
            if str(doc_id) not in completed_doc_ids
        ]

        if len(remaining) < len(all_doc_ids):
            eval_logger.info(
                f"[{task_name}] Resume from checkpoint: "
                f"{len(all_doc_ids) - len(remaining)}/{len(all_doc_ids)} docs already completed"
            )

        return remaining

    def merge_distributed_checkpoints(self, task_name: str):
        """Merge checkpoints from all ranks (only for rank 0)."""
        if not self.enable_checkpointing or self.world_size == 1 or self.rank != 0:
            return

        eval_logger.info(f"[{task_name}] Merging checkpoints from {self.world_size} ranks")

        merged_file = (
            self.output_path / self.model_name_sanitized / 
            f"{self.checkpoint_prefix}_samples_{task_name}.jsonl"
        )

        merged_samples = []
        seen_samples = set()  # Track (doc_id, filter_key) pairs

        # Merge samples from all ranks
        for rank in range(self.world_size):
            rank_sample_file = (
                self.ckpt_dir / 
                f"{self.checkpoint_prefix}_samples_{task_name}_rank{rank}.jsonl"
            )

            if not rank_sample_file.exists():
                continue

            try:
                with open(rank_sample_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            sample = json.loads(line.strip())
                            doc_id = str(sample.get("doc_id"))
                            filter_key = sample.get("filter_key", "none")

                            # Deduplicate by (doc_id, filter_key)
                            sample_key = (doc_id, filter_key)
                            if sample_key in seen_samples:
                                continue
                            seen_samples.add(sample_key)

                            merged_samples.append(sample)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                eval_logger.warning(f"Failed to load samples from rank {rank}: {e}")

        # Save merged results
        try:
            with open(merged_file, "w", encoding="utf-8") as f:
                for sample in merged_samples:
                    f.write(
                        json.dumps(sample, default=handle_non_serializable, 
                                 ensure_ascii=False) + "\n"
                    )

            eval_logger.info(
                f"[{task_name}] Merged checkpoint: "
                f"{len(merged_samples)} samples from {self.world_size} ranks"
            )
        except Exception as e:
            eval_logger.error(f"Failed to save merged checkpoint: {e}")

    def cleanup_checkpoints(self, task_name: Optional[str] = None, 
                          keep_merged: bool = True):
        """Clean up checkpoint files."""
        if not self.enable_checkpointing or self.rank != 0:
            return

        tasks = [task_name] if task_name else list(self.completed_samples.keys())

        for task in tasks:
            # Remove rank-specific checkpoints
            if self.world_size > 1:
                for rank in range(self.world_size):
                    sample_file = (
                        self.ckpt_dir / 
                        f"{self.checkpoint_prefix}_samples_{task}_rank{rank}.jsonl"
                    )
                    lock_file = sample_file.with_suffix(".lock")

                    for file in [sample_file, lock_file]:
                        if file.exists():
                            file.unlink()

            # Remove merged checkpoints if requested
            if not keep_merged:
                merged_file = (
                    self.output_path / self.model_name_sanitized / 
                    f"{self.checkpoint_prefix}_samples_{task}.jsonl"
                )
                if merged_file.exists():
                    merged_file.unlink()

        eval_logger.info(f"Cleaned up checkpoints for tasks: {tasks}")

    def get_progress(self, task_name: str, total_samples: int) -> dict:
        """
        Get progress information for a task.

        Args:
            task_name: Name of the task
            total_samples: Total number of samples (doc_id × filters)

        Returns:
            Dictionary with progress information
        """
        if not self.enable_checkpointing:
            return {
                "task_name": task_name,
                "total_samples": total_samples,
                "completed_samples": 0,
                "progress": 0.0,
            }

        completed = len(self.completed_samples.get(task_name, set()))

        return {
            "task_name": task_name,
            "total_samples": total_samples,
            "completed_samples": completed,
            "progress": completed / max(total_samples, 1),
        }