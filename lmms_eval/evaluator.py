import collections
import inspect
import itertools
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import Image, Sequence
from loguru import logger as eval_logger
from tqdm import tqdm

import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lmms_eval.llm_judge.launcher import get_launcher
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.loggers.ckpt_logger import CheckpointLogger
from lmms_eval.models import get_model
from lmms_eval.tasks import Task, TaskManager, get_task_dict
from lmms_eval.utils import (
    create_iterator,
    get_datetime_str,
    get_git_commit_hash,
    handle_non_serializable,
    hash_string,
    make_table,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
    unflatten_dict,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    launcher_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    enable_checkpointing: bool = False, # 是否启用检查点记录器
    checkpoint_interval: int = 50, # checkpoint间隔
    output_path: Optional[str] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    filter_list: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    distributed_executor_backend: str = "accelerate",
    cli_args=None,
    force_simple: bool = False,
    mode: str = "full",
    streaming_eval: bool = False,
    inference_threads: int = 1,
    eval_threads: int = 2,
    **kwargs,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lmms_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :param enable_checkpointing: bool
        Enable checkpoint-based resume for both inference and evaluation (single parameter)
    :param checkpoint_interval: int
        Save checkpoint every N samples (default: 50)
    :return
        Dictionary of results
    """
    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    assert distributed_executor_backend in {"accelerate", "torchrun"}, f"Invalid distributed executor backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'."

    # 初始化 checkpoint logger
    checkpoint_logger = None
    if enable_checkpointing and output_path:
        checkpoint_logger = CheckpointLogger(
            output_path=output_path,
            model_name=model_args if isinstance(model_args, str) else model,
            enable_checkpointing=True,
            checkpoint_interval=checkpoint_interval,
        )

    # 仅评估模式：强制使用虚拟模型
    if mode == "eval_only":
        eval_logger.info("=" * 50)
        eval_logger.info("In EVAL_ONLY mode, only metrics are calculated; you must provide the model outputs yourself")
        eval_logger.info("Make sure your task YAML includes 'doc_to_answer' configuration")
        eval_logger.info("=" * 50)

        # 覆盖模型参数
        model = "virtual_model"
        batch_size = 8
        if model_args is None:
            model_args = ""

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(f"generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if launcher_args is not None:
        launcher_args = simple_parse_args_string(launcher_args)
        launcher_name = launcher_args.pop("name")
        eval_launcher = get_launcher(launcher_name)(**launcher_args)
    else:
        eval_launcher = None

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lmms_eval.models.get_model(model, force_simple).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif isinstance(model, lmms_eval.api.model.lmms):
        lm = model
    task_type = "simple" if lm.is_simple else "chat"
    task_dict = get_task_dict(tasks, task_manager, task_type)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if mode == "predict_only":
                    eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # 覆盖任务的 filter_list 配置，如果命令行提供了 filter_list 参数
                if filter_list is not None:
                    import json as _json
                    parsed = _json.loads(filter_list) if isinstance(filter_list, str) else filter_list
                    task_obj.set_config(key="filter_list", value=parsed)
                    # 同步重建 _filters
                    from lmms_eval.filters import build_filter_ensemble
                    task_obj._filters = []
                    for fc in parsed:
                        components = [[fn["function"], {k: v for k, v in fn.items() if k != "function"}] for fn in fc["filter"]]
                        task_obj._filters.append(build_filter_ensemble(fc["name"], components))
                    eval_logger.info(f"Overriding filter_list for task {task_name}")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 选择评估模式
    if streaming_eval:
        eval_logger.info("=" * 50)
        eval_logger.info("Streaming evaluation enabled: inference and evaluation run in parallel")
        eval_logger.info(f"Running on rank {global_rank} (world_size {world_size})")
        eval_logger.info("=" * 50)

        results = evaluate_streaming(
            lm=lm,
            task_dict=task_dict,
            batch_size=batch_size,
            limit=limit,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            bootstrap_iters=bootstrap_iters,
            write_out=write_out,
            log_samples=True if mode == "predict_only" else log_samples,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            verbosity=verbosity,
            distributed_executor_backend=distributed_executor_backend,
            cli_args=cli_args,
            eval_server_launcher=None,
            mode=mode,
            inference_threads=inference_threads,
            eval_threads=eval_threads,
            checkpoint_logger=checkpoint_logger,
            output_path=output_path,
        )
    else:
        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            bootstrap_iters=bootstrap_iters,
            write_out=write_out,
            log_samples=True if mode == "predict_only" else log_samples,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            verbosity=verbosity,
            distributed_executor_backend=distributed_executor_backend,
            cli_args=cli_args,
            eval_server_launcher=eval_launcher,
            mode=mode,
        )

    if global_rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lmms_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = datetime_str
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm: "LM",
    task_dict,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    distributed_executor_backend: str = "accelerate",
    eval_server_launcher: Optional[Union[str, Callable]] = None,
    mode: str = "full",
    cli_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :return
        Dictionary of results
    """

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    eval_logger.info(f"Running on rank {global_rank} (local rank {local_rank})")

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    name_to_task = {}
    if not log_samples:
        if not all("bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys() for task_output in eval_tasks):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    if distributed_executor_backend == "accelerate" and not hasattr(lm, "accelerator"):
        lm.accelerator = Accelerator()

    for task_output in eval_tasks:
        task: Task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args

        name_to_task[task_name] = task

        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"
        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=global_rank,
            world_size=world_size,
            cache_requests=cache_requests,  # later we will add them
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
        eval_logger.debug(f"Task: {task_output.task_name}; number of requests on this rank: {len(task._instances)}")
        if write_out:
            eval_logger.warning(
                "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
                "Use --log_samples instead for saving model outputs and debugging. "
                "The write_out flag only prints the first few documents and impacts performance."
            )
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            elif distributed_executor_backend == "torchrun":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = torch.zeros(world_size * 1, dtype=instances_rnk.dtype, device=lm.device)
                dist.all_gather_into_tensor(gathered_item, instances_rnk)
                gathered_item = gathered_item.cpu().detach().numpy().tolist()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = "loglikelihood" if task.OUTPUT_TYPE == "multiple_choice" else task.OUTPUT_TYPE
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    # 如果是 VirtualModel，传递任务实例
    if mode == "eval_only" and hasattr(lm, "set_parent_tasks") and callable(lm.set_parent_tasks):
        lm.set_parent_tasks(task_dict)

    ### Run LMM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)  # Choiszt run generate until

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                lm.accelerator.wait_for_everyone()
            elif distributed_executor_backend == "torchrun":
                dist.barrier()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    # Cleaning lm's cuda memory if you are launching llm as judge in local
    lm.clean()
    RANK = global_rank
    WORLD_SIZE = world_size
    if eval_server_launcher is not None and RANK == 0:
        eval_server_launcher.launch()

    if world_size > 1:
        if distributed_executor_backend == "accelerate":
            lm.accelerator.wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            if cli_args is not None and not cli_args.process_with_media:
                doc_iterator = create_iterator(enumerate(task.eval_docs_no_media), rank=RANK, limit=int(limit) if limit else None, world_size=WORLD_SIZE)
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
            doc_iterator_for_counting = itertools.islice(range(len(task.test_docs())), RANK, limit, WORLD_SIZE) if task.has_test_docs() else itertools.islice(range(len(task.validation_docs())), RANK, limit, WORLD_SIZE)
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=total_docs, desc=f"Postprocessing", disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                try:
                    metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])
                    eval_success = True
                except Exception as e:
                    eval_logger.error(f"Error in process_results for doc_id={doc_id}: {e}")
                    metrics = {}
                    eval_success = False

                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        # If image is not in key
                        if "image" not in key:
                            # If audio is also not the value
                            if isinstance(value, dict) and "array" in value:
                                continue
                            else:
                                saved_doc[key] = value
                    filtered_arguments = []
                    for req in requests:
                        # check if req.args is a list of tuples, and each item in the list is a serializable object
                        for value in req.args:
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                filtered_arguments.append(value)
                            # else:
                            #     filtered_arguments.append(_handle_non_serializable(value))
                    infer_success = all(getattr(req, "success", True) is not False for req in requests)
                    example = {
                        "doc_id": doc_id,
                        "doc": saved_doc,
                        "target": target,
                        "arguments": filtered_arguments,
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        # 取所有 requests 的 success 值，任一失败则整体为 False
                        "success": infer_success and eval_success,
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        # Removing prompt hash and target hash here
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                if eval_success:
                    for metric, value in metrics.items():
                        task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)

            pbar.close()

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)

                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))

            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

        dist.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        if eval_server_launcher is not None:
            eval_server_launcher.clean()
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if m in _higher_is_better and _higher_is_better[m] is not None and _higher_is_better[m] != h:
                            eval_logger.warning(f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None.")
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(group_agg.items())} if (bool(group_agg) & show_group_table) else {}),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)
    else:
        results_dict = None

    if WORLD_SIZE > 1:
        # if muti-gpu, wait for all processes to finish
        if distributed_executor_backend == "accelerate":
            # this should work for torchrun as well since it internally calls torch.distributed.barrier()
            Accelerator().wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()
        else:
            raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    return results_dict

@positional_deprecated
def evaluate_streaming(
    lm: "LM",
    task_dict,
    batch_size: int = 1,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    distributed_executor_backend: str = "accelerate",
    cli_args=None,
    eval_server_launcher: Optional[Union[str, Callable]] = None,
    mode: str = "full",
    inference_threads: int = 1,
    eval_threads: int = 4,
    checkpoint_logger: Optional["CheckpointLogger"] = None,
    output_path: Optional[str] = None,
):
    """
    真正的流式评估：推理和评估完全并行执行
    
    核心设计：
    1. 推理线程：不断批量推理，将结果放入队列
    2. 评估线程：从队列取出结果，立即计算指标
    3. 两个线程完全独立，互不阻塞
    """
    import queue
    import threading
    import filelock
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # ========== 初始化 ==========
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    RANK = global_rank
    WORLD_SIZE = world_size
    if batch_size != "auto":
        batch_size = int(batch_size)
    else:
        eval_logger.info(f"Auto batch size not implemented in streaming eval, defaulting to batch size 1")
        batch_size = 1

    model_folder = "model"
    if hasattr(lm, "_resolve_model_name_for_cache"):
        # lmms_eval models usually have this method 
        model_folder = lm._resolve_model_name_for_cache()
    elif hasattr(lm, "model_name") and getattr(lm, "model_name"):
        model_folder = getattr(lm, "model_name")
    elif hasattr(lm, "model") and isinstance(getattr(lm, "model"), str):
        model_folder = getattr(lm, "model")
    elif hasattr(lm, "lm") and hasattr(lm.lm, "_resolve_model_name_for_cache"):
        # in case it is cached LM wrapper
        model_folder = lm.lm._resolve_model_name_for_cache()

    eval_logger.info(f"[Rank {RANK}] Starting streaming evaluation with parallel inference and evaluation")

    # 存储结果的数据结构
    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    configs = collections.defaultdict(dict)
    samples = collections.defaultdict(list)
    padding_requests = collections.defaultdict(int)
    num_fewshot = collections.defaultdict(int)

    # 任务初始化
    eval_tasks = get_task_list(task_dict)

    if distributed_executor_backend == "accelerate" and not hasattr(lm, "accelerator"):
        lm.accelerator = Accelerator()

    # 如果是 VirtualModel，传递任务实例
    if mode == "eval_only" and hasattr(lm, "set_parent_tasks") and callable(lm.set_parent_tasks):
        lm.set_parent_tasks(task_dict)
    
    # 构建任务映射：task_name -> task_output
    task_name_to_output = {}
    # 按请求类型聚合所有任务的实例
    requests = collections.defaultdict(list)
    for task_output in eval_tasks:
        task: Task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args
        task_name_to_output[task_name] = task_output
        
        # 记录任务元数据
        if type(task) == tuple:
            group_name, task = task
        else:
            group_name = None
        
        if task is None:
            continue
        
        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())
        
        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        # 构建请求
        limit_size = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit_size,
            rank=global_rank,
            world_size=world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
    
        # Resume Logic
        original_instances = task._instances
        eval_completed_doc_ids = set()
        inferred_responses = {}
        
        if checkpoint_logger and checkpoint_logger.enable_checkpointing:
            historical_metrics, historical_samples = checkpoint_logger.load_historical_metrics(task_name)
            for sample in historical_samples:
                doc_id = sample.get("doc_id")
                if doc_id is not None:
                    eval_completed_doc_ids.add(str(doc_id))
                    task_output.logged_samples.append(sample)
                    
            for (metric_key, filter_key), values in historical_metrics.items():
                task_output.sample_metrics[(metric_key, filter_key)].extend(values)
        elif output_path is not None:
            # 1. evaluation stream
            eval_file_path = os.path.join(output_path, model_folder, "evaluation", f"evaluation_stream_{task_name}.jsonl")
            if os.path.exists(eval_file_path):
                try:
                    with open(eval_file_path, 'r', encoding='utf-8') as ef:
                        for line in ef:
                            if not line.strip(): continue
                            data = json.loads(line)
                            if 'doc_id' in data:
                                eval_completed_doc_ids.add(str(data['doc_id']))
                                task_output.logged_samples.append(data)
                                
                                # Extract metrics
                                filter_key = data.get("filter_key", "none")
                                excluded_fields = {"doc_id", "doc", "target", "resps", "filtered_resps", "doc_hash", "arguments", "filter_key", "success"}
                                for key, value in data.items():
                                    if key not in excluded_fields:
                                        task_output.sample_metrics[(key, filter_key)].append(value)
                except Exception as e:
                    eval_logger.warning(f"Failed to read evaluation stream: {e}")
                    
        # 2. inference stream
        if output_path is not None:
            infer_file_path = os.path.join(output_path, model_folder, "inference", f"inference_stream_{task_name}.jsonl")
            if os.path.exists(infer_file_path):
                try:
                    with open(infer_file_path, 'r', encoding='utf-8') as f_inf:
                        for line in f_inf:
                            if not line.strip(): continue
                            data = json.loads(line)
                            if 'doc_id' in data and 'resps' in data:
                                inferred_responses[str(data['doc_id'])] = data['resps']
                except Exception as e:
                    eval_logger.warning(f"Failed to read inference stream: {e}")
        
        filtered_instances = []
        for inst in original_instances:
            # a) Skip completed documents
            if str(inst.doc_id) in eval_completed_doc_ids:
                continue
                
            # b) Load already inferred responses
            if str(inst.doc_id) in inferred_responses:
                inst.resps = inferred_responses[str(inst.doc_id)]
                inst._already_inferred = True
                filtered_instances.append(inst)
            else:
                # c) Keep instance for both inference and evaluation
                filtered_instances.append(inst)
        
        task._instances = filtered_instances
        
        for instance in filtered_instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        eval_logger.info(f"[Rank {RANK}] Task {task_name}: {len(filtered_instances)} instances")
    
    # 计算padding（确保多卡同步）
    if world_size > 1:
        for task_output in eval_tasks:
            task = task_output.task
            if distributed_executor_backend == "accelerate":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            elif distributed_executor_backend == "torchrun":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = torch.zeros(world_size * 1, dtype=instances_rnk.dtype, device=lm.device)
                dist.all_gather_into_tensor(gathered_item, instances_rnk)
                gathered_item = gathered_item.cpu().detach().numpy().tolist()
            
            reqtype = "loglikelihood" if task.OUTPUT_TYPE == "multiple_choice" else task.OUTPUT_TYPE
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[reqtype] += numpad
    
    # 流式处理：推理+评估（并行实现）
    eval_logger.info(f"[Rank {RANK}] Starting parallel streaming inference and evaluation")
    
    for reqtype, reqs in requests.items():
        eval_logger.info(f"[Rank {RANK}] Processing {len(reqs)} {reqtype} requests")
        
        # 创建克隆请求（处理 repeats）
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)
        
        # 应用padding
        if (world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([reqs[-1]] * reqs[-1].repeats)
        
        total_eval_requests = len(cloned_reqs)
        
        evaluation_queue = queue.Queue()
        
        real_inf_reqs = []
        for req in cloned_reqs:
            if getattr(req, "_already_inferred", False):
                evaluation_queue.put(req)
            else:
                real_inf_reqs.append(req)
        
        cloned_reqs = real_inf_reqs
        total_inf_requests = len(cloned_reqs)
        
        # 创建共享队列
        # max_queue_size = max(100, total_requests // 10)  # 最多缓存10%的结果
        # evaluation_queue = queue.Queue(maxsize=max_queue_size)


        def _save_inference_logging_if_needed(inst):
            if getattr(inst, '_already_inferred', False): return
            if getattr(inst, '_already_inferred', False): return
            if output_path is not None:
                try:
                    save_dir = os.path.join(output_path, model_folder, "inference")
                    os.makedirs(save_dir, exist_ok=True)
                    infer_file = os.path.join(save_dir, f"inference_stream_{inst.task_name}.jsonl")
                    with filelock.FileLock(str(infer_file) + ".lock", timeout=10):
                        with open(infer_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "doc_id": inst.doc_id,
                                "resps": inst.resps,
                                "task_name": inst.task_name,
                                "arguments": inst.arguments,
                                "success": getattr(inst, "success", True)
                            }, default=str) + "\n")
                except Exception as e:
                    pass # ignore logging errors

        def _save_evaluation_logging_if_needed(t_name, ex):
            if output_path is not None:
                try:
                    save_dir = os.path.join(output_path, model_folder, "evaluation")
                    os.makedirs(save_dir, exist_ok=True)
                    eval_file = os.path.join(save_dir, f"evaluation_stream_{t_name}.jsonl")
                    with filelock.FileLock(str(eval_file) + ".lock", timeout=10):
                        with open(eval_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(ex, default=str) + "\n")
                except Exception as e:
                    pass # ignore logging errors

        # ========== 推理线程：批量推理，结果放入队列 ==========
        def inference_worker():
            """推理线程：不断推理，将结果放入队列"""
            if batch_size == 1:
                # 使用线程池并发处理每个请求
                try:
                    eval_logger.info(f"[Rank {RANK}] Inference thread started with {inference_threads} workers")
                    
                    # 创建进度条
                    pbar_inference = tqdm(
                        total=total_inf_requests,
                        desc=f"[Rank {RANK}] Inference",
                        disable=(RANK != 0),
                        position=0
                    )
                    
                    def process_single_request(instance):
                        """处理单个请求的函数"""
                        try:
                            # 将单个instance包装成列表传给模型
                            response = getattr(lm, reqtype)(instance).strip()
                            instance.resps.append(response)
                            return instance
                        except Exception as e:
                            eval_logger.error(f"[Rank {RANK}] Error processing request: {e}")
                            instance.resps.append(f"error: {e}")
                            instance.success = False
                            return instance
                    
                    # 使用线程池并发处理所有请求
                    with ThreadPoolExecutor(max_workers=inference_threads) as executor:
                        # 提交所有任务
                        future_to_instance = {
                            executor.submit(process_single_request, instance): instance
                            for instance in cloned_reqs
                        }
                        
                        # 按完成顺序处理结果
                        for future in as_completed(future_to_instance):
                            try:
                                processed_instance = future.result()
                                evaluation_queue.put(processed_instance)
                                _save_inference_logging_if_needed(processed_instance)
                            except Exception as e:
                                instance = future_to_instance[future]
                                eval_logger.error(f"[Rank {RANK}] Request failed: {e}")
                                instance.resps.append(f"error: {e}")
                                instance.success = False
                                evaluation_queue.put(instance)
                                _save_inference_logging_if_needed(instance)
                            
                            pbar_inference.update(1)
                    
                    pbar_inference.close()
                    evaluation_queue.put(None)  # 结束信号
                    eval_logger.info(f"[Rank {RANK}] Inference thread completed")
                    
                except Exception as e:
                    eval_logger.error(f"[Rank {RANK}] Inference thread error: {e}")
                    evaluation_queue.put(None)
            else:
                # 批量推理
                try:
                    eval_logger.info(f"[Rank {RANK}] Inference thread started")
                    
                    # 创建进度条
                    pbar_inference = tqdm(
                        total=total_inf_requests,
                        desc=f"[Rank {RANK}] Inference",
                        disable=(RANK != 0),
                        position=0
                    )

                    num_batches = (total_inf_requests + batch_size - 1) // batch_size
                    for batch_idx in range(num_batches):
                        # 获取当前批次
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, total_inf_requests)
                        batch_reqs = cloned_reqs[start_idx:end_idx]
                        
                        eval_logger.debug(
                            f"[Rank {RANK}] Processing batch {batch_idx + 1}/{num_batches} "
                            f"({len(batch_reqs)} requests)"
                        )
                        
                        # 推理当前批次
                        try:
                            batch_resps = getattr(lm, reqtype)(batch_reqs)
                            # 立即将每个结果放入队列
                            for resp, instance in zip(batch_resps, batch_reqs):
                                instance.resps.append(resp)
                                evaluation_queue.put(instance)
                                _save_inference_logging_if_needed(instance)
                        except Exception as e:
                            eval_logger.error(
                                f"[Rank {RANK}] Error in batch {batch_idx + 1}: {e}"
                            )
                            # 即使出错，也要放入空结果，保持队列同步
                            for instance in batch_reqs:
                                instance.resps.append("")
                                getattr(instance, "success", False) # dummy
                                instance.success = False
                                evaluation_queue.put(instance)
                                _save_inference_logging_if_needed(instance)
                        # 更新进度条
                        pbar_inference.update(len(batch_reqs))
                    
                    pbar_inference.close()
                    evaluation_queue.put(None)
                    eval_logger.info(f"[Rank {RANK}] Inference thread completed")
                    
                except Exception as e:
                    eval_logger.error(f"[Rank {RANK}] Inference thread error: {e}")

        # ========== 评估线程：使用线程池并发评估 ==========
        def evaluation_worker():
            """评估线程：使用线程池并发计算指标"""
            eval_logger.info(f"[Rank {RANK}] Evaluation thread started with pool size {eval_threads}")
            
            pbar_evaluation = tqdm(
                total=total_eval_requests,
                desc=f"[Rank {RANK}] Evaluation",
                disable=(RANK != 0),
                position=1
            )
            
            task_docs = {}
            for task_name, task_output in task_name_to_output.items():
                task = task_output.task
                
                if cli_args is not None and not cli_args.process_with_media:
                    docs = list(task.eval_docs_no_media)
                else:
                    docs = list(task.test_docs() if task.has_test_docs() else task.validation_docs())
                
                task_docs[task_name] = {i: doc for i, doc in enumerate(docs)}
            
            for task_name, task_output in task_name_to_output.items():
                task_output._streaming_instances_by_doc = collections.defaultdict(list)
                task_output._streaming_expected_instances = {}
                
                task = task_output.task
                for inst in task.instances:
                    doc_id = inst.doc_id
                    if doc_id not in task_output._streaming_expected_instances:
                        task_output._streaming_expected_instances[doc_id] = sum(
                            1 for i in task.instances if i.doc_id == doc_id
                        )
            
            pending_docs = collections.defaultdict(lambda: collections.defaultdict(list))
            
            def evaluate_doc(task_name, doc_id, doc_instances, doc):
                """评估单个文档的所有实例"""
                task_output = task_name_to_output[task_name]
                task = task_output.task
                
                # 在评估时局部应用filter
                if hasattr(task, "_filters") and task._filters:
                    for f in task._filters:
                        f.apply(doc_instances, getattr(task, "task_docs", None))
                else:
                    for inst in doc_instances:
                        inst.filtered_resps["none"] = inst.resps

                doc_instances.sort(key=lambda x: x.idx)
                
                infer_success = all(getattr(inst, "success", True) is not False for inst in doc_instances)
                
                # 遍历每个filter
                for filter_key in doc_instances[0].filtered_resps.keys():
                    eval_success = True
                    metrics = {}
                    
                    if infer_success:
                        filtered_results = [inst.filtered_resps[filter_key] for inst in doc_instances]
                        try:
                            metrics = task.process_results(unflatten_dict(doc), filtered_results)
                        except Exception as e:
                            eval_logger.error(f"[Rank {RANK}] Error processing doc_id {doc_id} for task {task_name}: {e}")
                            eval_success = False
                    else:
                        eval_success = False
                        
                    if eval_success and metrics:
                        for metric, value in metrics.items():
                            task_output.sample_metrics[(metric, filter_key)].append(value)
                    
                    if log_samples:
                        target = task.doc_to_target(doc)
                        saved_doc = {}
                        for key, value in doc.items():
                            if "image" not in key:
                                if isinstance(value, dict) and "array" in value:
                                    continue
                                else:
                                    saved_doc[key] = value
                        
                        filtered_arguments = []
                        for inst in doc_instances:
                            for value in inst.args:
                                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                    filtered_arguments.append(value)

                        example = {
                            "doc_id": doc_id,
                            "filter_key": filter_key,
                            "doc": saved_doc,
                            "target": target,
                            "arguments": filtered_arguments,
                            "resps": [inst.resps for inst in doc_instances],
                            "filtered_resps": [inst.filtered_resps.get(filter_key, inst.resps) for inst in doc_instances],
                            "success": infer_success and eval_success,
                            "doc_hash": hash_string(
                                json.dumps(
                                    doc_instances[0].doc,
                                    indent=2,
                                    default=handle_non_serializable,
                                    ensure_ascii=False,
                                )
                            ),
                        }
                        example.update(metrics)
                        task_output.logged_samples.append(example)
                        _save_evaluation_logging_if_needed(task_name, example)

                        if checkpoint_logger:
                            checkpoint_logger.log_sample(task_name, example, filter_key)

            with ThreadPoolExecutor(max_workers=eval_threads) as pool:
                futures = {}
                
                while True:
                    instance = evaluation_queue.get()
                    
                    if instance is None:
                        eval_logger.info(f"[Rank {RANK}] Evaluation thread received end signal")
                        evaluation_queue.task_done()
                        break
                    
                    try:
                        doc_id = instance.doc_id
                        task_name = instance.task_name
                        
                        if task_name not in task_name_to_output:
                            eval_logger.warning(f"[Rank {RANK}] Unknown task: {task_name}, skipping")
                            continue
                            
                        task_output = task_name_to_output[task_name]
                        task = task_output.task
                        
                        if doc_id not in task_docs[task_name]:
                            eval_logger.warning(f"[Rank {RANK}] Doc {doc_id} not found for task {task_name}")
                            continue
                            
                        doc = task_docs[task_name][doc_id]
                        pending_docs[task_name][doc_id].append(instance)
                        
                        expected_count = task_output._streaming_expected_instances[doc_id]
                        current_count = len(pending_docs[task_name][doc_id])
                        
                        if current_count == expected_count:
                            doc_instances = pending_docs[task_name].pop(doc_id)
                            future = pool.submit(evaluate_doc, task_name, doc_id, doc_instances, doc)
                            futures[future] = (task_name, doc_id)
                            
                    except Exception as e:
                        eval_logger.error(f"[Rank {RANK}] Error processing instance: {e}")
                    finally:
                        evaluation_queue.task_done()
                        pbar_evaluation.update(1)
                
                for future in as_completed(futures):
                    task_name, doc_id = futures[future]
                    try:
                        future.result()
                        eval_logger.debug(f"[Rank {RANK}] Completed evaluation for task={task_name}, doc_id={doc_id}")
                    except Exception as e:
                        eval_logger.error(f"[Rank {RANK}] Error in evaluation task {task_name}/{doc_id}: {e}")

            pbar_evaluation.close()
            if checkpoint_logger:
                eval_logger.debug(f"[Rank {RANK}] Flushing checkpoints in evaluation thread")
                for task_name in task_name_to_output.keys():
                    checkpoint_logger.flush(task_name)
            eval_logger.info(f"[Rank {RANK}] Evaluation thread completed")

        # ========== 启动推理和评估线程（完全并行）==========
        inference_thread = threading.Thread(target=inference_worker, name=f"Inference-{RANK}")
        evaluation_thread = threading.Thread(target=evaluation_worker, name=f"Evaluation-{RANK}")
        
        # 启动两个线程
        inference_thread.start()
        evaluation_thread.start()
        
        # 等待两个线程完成
        inference_thread.join()
        evaluation_thread.join()

        # 刷新所有任务的checkpoint
        if checkpoint_logger:
            eval_logger.info(f"[Rank {RANK}] Flushing checkpoints for all tasks")
            for task_name in task_name_to_output.keys():
                checkpoint_logger.flush(task_name)
        
        # 同步点：确保所有rank完成
        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                lm.accelerator.wait_for_everyone()
            elif distributed_executor_backend == "torchrun":
                dist.barrier()
    
    # ========== 清理模型内存 ==========
    lm.clean()
    
    if eval_server_launcher is not None and RANK == 0:
        eval_server_launcher.launch()
    
    if world_size > 1:
        if distributed_executor_backend == "accelerate":
            lm.accelerator.wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()
    
    # ========== 清理临时数据结构 ==========
    for task_output in eval_tasks:
        if hasattr(task_output, '_streaming_expected_instances'):
            del task_output._streaming_expected_instances
    
    # ========== 聚合结果（多卡场景）==========
    if WORLD_SIZE > 1:
        eval_logger.info(f"[Rank {RANK}] Gathering results across ranks")
        
        for task_output in eval_tasks:
            # 聚合样本
            if log_samples:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = task_output.logged_samples
                
                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )
                
                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))
            
            # 聚合指标
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))
        
        dist.barrier()
    
    # 合并分布式checkpoint（仅rank 0）
    if RANK == 0 and checkpoint_logger and WORLD_SIZE > 1:
        eval_logger.info("Merging distributed checkpoints")
        for task_output in eval_tasks:
            checkpoint_logger.merge_distributed_checkpoints(task_output.task_name)

    
    # ========== 计算聚合指标（仅 Rank 0）==========
    if RANK == 0:
        if eval_server_launcher is not None:
            eval_server_launcher.clean()
        
        eval_logger.info("Calculating aggregate metrics")
        
        # 计算每个任务的聚合指标
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        
        # 整合结果
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)
        
        # 计算分组指标
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )
        
        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)
        
        # 收集 higher_is_better 信息
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h
                        
                        if m in _higher_is_better and _higher_is_better[m] is not None and _higher_is_better[m] != h:
                            eval_logger.warning(
                                f"Higher_is_better values for metric {m} in group {group} are not consistent. "
                                "Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better
        
        # 构建结果字典
        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(group_agg.items())} if (bool(group_agg) & show_group_table) else {}),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        
        if log_samples:
            results_dict["samples"] = dict(samples)
        
        eval_logger.info("Streaming evaluation completed successfully")
        
    else:
        results_dict = None
    
    # 最终同步
    if WORLD_SIZE > 1:
        if distributed_executor_backend == "accelerate":
            Accelerator().wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()
    
    return results_dict


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
