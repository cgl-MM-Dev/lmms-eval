from typing import List, Tuple
from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("virtual_model")
class VirtualModel(lmms):
    """
    Virtual model for evaluation-only mode.
    
    This model doesn't perform actual inference but retrieves pre-existing
    responses from the dataset using doc_to_answer function.
    
    Example usage:
    python -m lmms_eval \
        --model virtual_model \
        --tasks mmbench_en \
        --output_path ./results
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        **kwargs
    ):
        super().__init__()
        self._batch_size = batch_size
        self._rank = 0
        self._world_size = 1
        
        # 存储任务实例的引用
        self._parent_tasks = {}
        
        eval_logger.info("VirtualModel initialized")
        eval_logger.info("This model will use pre-existing answers from dataset")

    
    def set_parent_tasks(self, task_dict):
        """
        Set task instances for accessing doc_to_answer method.
        This should be called by the evaluator before generating.
        
        Args:
            task_dict: Dictionary mapping task names to Task instances
        """
        self._parent_tasks = task_dict
        eval_logger.info(f"VirtualModel loaded {len(task_dict)} task instances")
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Return pre-existing answers from documents instead of generating new ones.
        """
        results = []
        
        for request in requests:
            # Extract request components based on request type
            if hasattr(request, 'arguments'):
                args = request.arguments
                if len(args) == 6:
                    _, _, _, doc_id, task, split = args
                else:
                    eval_logger.error(f"Unexpected arguments length: {len(args)}")
                    results.append("")
                    continue
            else:
                eval_logger.error(f"Unknown request format: {type(request)}")
                results.append("")
                continue
            
            # Get the corresponding task instance
            if task not in self._parent_tasks:
                eval_logger.error(
                    f"Task '{task}' not found in task instances. "
                    f"Available tasks: {list(self._parent_tasks.keys())}"
                )
                results.append("")
                continue
            
            belong_to_task = self._parent_tasks[task]
            
            doc = belong_to_task.dataset[split][doc_id]
            
            # Retrieve pre-existing answer using doc_to_answer
            try:
                answer = belong_to_task.doc_to_answer(doc)
                
                # Handle different return types
                if answer is None:
                    eval_logger.warning(
                        f"doc_to_answer returned None for doc_id={doc_id} in task={task}"
                    )
                    answer = ""
                elif isinstance(answer, list):
                    # If doc_to_answer returns a list, join it or take first element
                    answer = answer[0] if len(answer) > 0 else ""
                
                results.append(str(answer))
            except Exception as e:
                eval_logger.error(
                    f"Error in doc_to_answer for doc_id={doc_id} in task={task}: {e}"
                )
                results.append("")
    
        return results
    
    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """
        Multi-round generation using pre-existing answers.
        """
        return self.generate_until(requests)
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Not supported in eval_only mode.
        """
        raise NotImplementedError(
            "loglikelihood is not supported in virtual_model mode. "
            "This mode only supports generation tasks."
        )
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size