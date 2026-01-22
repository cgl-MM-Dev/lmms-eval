import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import filelock
from loguru import logger as eval_logger

from lmms_eval.utils import (
    handle_non_serializable,
    sanitize_list,
    sanitize_model_name,
)


class CheckpointLogger:
    """
    Simple checkpoint logger for resumable inference and evaluation.

    Saves samples incrementally in the same format as evaluation_tracker.
    Single parameter control for both inference and evaluation checkpoints.
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
            enable_checkpointing: Enable checkpoint-based resume (single parameter)
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

        # Create checkpoint directory (same structure as evaluation_tracker)
        self.ckpt_dir = self.output_path / self.model_name_sanitized / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Distributed settings
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Track completed documents per task
        self.completed_docs: Dict[str, Set[str]] = defaultdict(set)

        # Sample buffers for batched writes (task_name -> list of samples)
        self.sample_buffers: Dict[str, List[dict]] = defaultdict(list)

        # Thread lock for file operations
        self.lock = threading.Lock()

        # prefix name for checkpoint files
        self.checkpoint_prefix = "checkpoint"

        eval_logger.info(f"CheckpointLogger initialized at {self.ckpt_dir}")
        eval_logger.info(f"Rank {self.rank}/{self.world_size}, checkpoint interval: {checkpoint_interval}")

        # Auto-load existing checkpoints
        self._load_all_checkpoints()

    def _get_sample_file_path(self, task_name: str) -> Path:
        """Get sample file path (same format as evaluation_tracker)."""
        if self.world_size > 1:
            return self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task_name}_rank{self.rank}.jsonl"
        return self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task_name}.jsonl"

    def _get_completed_file_path(self, task_name: str) -> Path:
        """Get completed doc IDs tracking file path."""
        if self.world_size > 1:
            return self.ckpt_dir / f"{self.checkpoint_prefix}_completed_{task_name}_rank{self.rank}.txt"
        return self.ckpt_dir / f"{self.checkpoint_prefix}_completed_{task_name}.txt"

    def _load_all_checkpoints(self):
        """Load all existing checkpoints from checkpoint directory."""
        if not self.ckpt_dir.exists():
            return

        # Find all completed files for this rank
        pattern = f"*_completed_*_rank{self.rank}.txt" if self.world_size > 1 else "*_completed_*.txt"

        for completed_file in self.ckpt_dir.glob(pattern):
            # Extract task name from filename
            # Format: {checkpoint_prefix}_completed_{task_name}_rank{rank}.txt or {checkpoint_prefix}_completed_{task_name}.txt
            filename = completed_file.stem
            parts = filename.split("_completed_")
            if len(parts) != 2:
                continue

            task_name_part = parts[1]
            # Remove rank suffix if exists
            if self.world_size > 1 and task_name_part.endswith(f"_rank{self.rank}"):
                task_name = task_name_part[: -len(f"_rank{self.rank}")]
            else:
                task_name = task_name_part

            # Load completed doc IDs
            try:
                with open(completed_file, "r") as f:
                    for line in f:
                        doc_id = line.strip()
                        if doc_id:
                            self.completed_docs[task_name].add(doc_id)

                eval_logger.info(f"[{task_name}] Loaded checkpoint: " f"{len(self.completed_docs[task_name])} completed docs")
            except Exception as e:
                eval_logger.warning(f"Failed to load checkpoint for {task_name}: {e}")

    def is_doc_completed(self, task_name: str, doc_id: str) -> bool:
        """Check if a document has been completed."""
        if not self.enable_checkpointing:
            return False
        return doc_id in self.completed_docs.get(task_name, set())

    def log_sample(self, task_name: str, sample: dict):
        """
        Log a sample (combines inference and evaluation results).

        This method matches the format of evaluation_tracker.save_results_samples.

        Args:
            task_name: Name of the task
            sample: Sample dictionary containing doc_id, responses, and metrics
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
            # Add to buffer
            self.sample_buffers[task_name].append(sample)

            # Mark as completed
            self.completed_docs[task_name].add(doc_id)

            # Save checkpoint if buffer is full
            if len(self.sample_buffers[task_name]) >= self.checkpoint_interval:
                self._flush_samples(task_name)

    def _flush_samples(self, task_name: str, force: bool = False):
        """Flush samples to disk (same format as evaluation_tracker)."""
        if not self.enable_checkpointing:
            return

        if not force and len(self.sample_buffers[task_name]) < self.checkpoint_interval:
            return

        if len(self.sample_buffers[task_name]) == 0:
            return

        sample_file = self._get_sample_file_path(task_name)
        completed_file = self._get_completed_file_path(task_name)
        lock_file = sample_file.with_suffix(".lock")

        try:
            # Use file lock for multi-process safety
            with filelock.FileLock(str(lock_file), timeout=10):
                # Write samples (same format as evaluation_tracker.save_results_samples)
                with open(sample_file, "a", encoding="utf-8") as f:
                    for sample in self.sample_buffers[task_name]:
                        # Clean up sample data (matching evaluation_tracker logic)
                        cleaned_sample = self._clean_sample(sample)

                        # Write as JSONL
                        sample_json = json.dumps(
                            cleaned_sample,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        f.write(sample_json + "\n")

                # Write completed doc IDs
                new_completed = []
                for sample in self.sample_buffers[task_name]:
                    doc_id = str(sample.get("doc_id"))
                    if doc_id not in new_completed:
                        new_completed.append(doc_id)

                with open(completed_file, "a", encoding="utf-8") as f:
                    for doc_id in new_completed:
                        f.write(f"{doc_id}\n")

                # Clear buffer
                num_saved = len(self.sample_buffers[task_name])
                self.sample_buffers[task_name].clear()

                eval_logger.debug(f"[{task_name}] Saved checkpoint: {num_saved} samples, " f"total {len(self.completed_docs[task_name])} completed")

        except filelock.Timeout:
            eval_logger.warning(f"Failed to acquire lock for {sample_file}")
        except Exception as e:
            eval_logger.error(f"Failed to save checkpoint for {task_name}: {e}")

    def _clean_sample(self, sample: dict) -> dict:
        """Clean sample data (matching evaluation_tracker.save_results_samples logic)."""
        cleaned = sample.copy()

        # Add input field if arguments exist
        if "arguments" in cleaned and len(cleaned["arguments"]) > 0:
            cleaned["input"] = cleaned["arguments"][0]

        # Sanitize resps and filtered_resps
        if "resps" in cleaned:
            cleaned["resps"] = sanitize_list(cleaned["resps"])

        if "filtered_resps" in cleaned:
            cleaned["filtered_resps"] = sanitize_list(cleaned["filtered_resps"])

            # Remove duplicate resps if they match filtered_resps
            if "resps" in cleaned:
                if cleaned["filtered_resps"] == cleaned["resps"][0] or cleaned["filtered_resps"] == cleaned["resps"]:
                    cleaned.pop("resps")

        # Remove arguments to save space
        if "arguments" in cleaned:
            cleaned.pop("arguments")

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

        Args:
            task_name: Name of the task
            all_doc_ids: List of all document IDs

        Returns:
            List of document IDs that haven't been completed
        """
        if not self.enable_checkpointing:
            return all_doc_ids

        completed_set = self.completed_docs.get(task_name, set())
        remaining = [str(doc_id) for doc_id in all_doc_ids if str(doc_id) not in completed_set]

        if len(remaining) < len(all_doc_ids):
            eval_logger.info(f"[{task_name}] Resume from checkpoint: " f"{len(all_doc_ids) - len(remaining)}/{len(all_doc_ids)} already completed")

        return remaining

    def merge_distributed_checkpoints(self, task_name: str):
        """
        Merge checkpoints from all ranks (only for rank 0).

        Args:
            task_name: Name of the task to merge
        """
        if not self.enable_checkpointing:
            return

        if self.world_size == 1:
            return

        if self.rank != 0:
            return

        eval_logger.info(f"[{task_name}] Merging checkpoints from {self.world_size} ranks")

        merged_file = self.output_path / self.model_name_sanitized / f"{self.checkpoint_prefix}_samples_{task_name}.jsonl"
        merged_completed_file = self.output_path / self.model_name_sanitized / f"{self.checkpoint_prefix}_completed_{task_name}.txt"

        merged_samples = []
        merged_completed = set()

        # Merge samples and completed docs from all ranks
        for rank in range(self.world_size):
            rank_sample_file = self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task_name}_rank{rank}.jsonl"
            rank_completed_file = self.ckpt_dir / f"{self.checkpoint_prefix}_completed_{task_name}_rank{rank}.txt"

            # Load samples
            if rank_sample_file.exists():
                try:
                    with open(rank_sample_file, "r", encoding="utf-8") as f:
                        for line in f:
                            sample = json.loads(line.strip())
                            merged_samples.append(sample)
                except Exception as e:
                    eval_logger.warning(f"Failed to load samples from rank {rank}: {e}")

            # Load completed doc IDs
            if rank_completed_file.exists():
                try:
                    with open(rank_completed_file, "r") as f:
                        for line in f:
                            doc_id = line.strip()
                            if doc_id:
                                merged_completed.add(doc_id)
                except Exception as e:
                    eval_logger.warning(f"Failed to load completed docs from rank {rank}: {e}")

        # Save merged results
        try:
            # Save merged samples
            with open(merged_file, "w", encoding="utf-8") as f:
                for sample in merged_samples:
                    f.write(json.dumps(sample, default=handle_non_serializable, ensure_ascii=False) + "\n")

            # Save merged completed docs
            with open(merged_completed_file, "w") as f:
                for doc_id in sorted(merged_completed):
                    f.write(f"{doc_id}\n")

            eval_logger.info(f"[{task_name}] Merged checkpoint: " f"{len(merged_samples)} samples, {len(merged_completed)} completed docs")
        except Exception as e:
            eval_logger.error(f"Failed to save merged checkpoint: {e}")

    def cleanup_checkpoints(self, task_name: Optional[str] = None, keep_merged: bool = True):
        """
        Clean up checkpoint files.

        Args:
            task_name: Specific task to clean, or None for all tasks
            keep_merged: Keep merged checkpoints (for rank 0 only)
        """
        if not self.enable_checkpointing:
            return

        if self.rank != 0:
            return

        tasks = [task_name] if task_name else list(self.completed_docs.keys())

        for task in tasks:
            # Remove rank-specific checkpoints
            if self.world_size > 1:
                for rank in range(self.world_size):
                    sample_file = self.ckpt_dir / f"{self.checkpoint_prefix}_samples_{task}_rank{rank}.jsonl"
                    completed_file = self.ckpt_dir / f"{self.checkpoint_prefix}_completed_{task}_rank{rank}.txt"
                    lock_file = sample_file.with_suffix(".lock")

                    for file in [sample_file, completed_file, lock_file]:
                        if file.exists():
                            file.unlink()

            # Remove merged checkpoints if requested
            if not keep_merged:
                merged_sample_file = self.output_path / self.model_name_sanitized / f"{self.checkpoint_prefix}_samples_{task}.jsonl"
                merged_completed_file = self.output_path / self.model_name_sanitized / f"{self.checkpoint_prefix}_completed_{task}.txt"

                for file in [merged_sample_file, merged_completed_file]:
                    if file.exists():
                        file.unlink()

        eval_logger.info(f"Cleaned up checkpoints for tasks: {tasks}")

    def get_progress(self, task_name: str, total_docs: int) -> dict:
        """
        Get progress information for a task.

        Args:
            task_name: Name of the task
            total_docs: Total number of documents

        Returns:
            Dictionary with progress information
        """
        if not self.enable_checkpointing:
            return {
                "task_name": task_name,
                "total_docs": total_docs,
                "completed_docs": 0,
                "progress": 0.0,
            }

        completed = len(self.completed_docs.get(task_name, set()))

        return {
            "task_name": task_name,
            "total_docs": total_docs,
            "completed_docs": completed,
            "progress": completed / max(total_docs, 1),
        }
