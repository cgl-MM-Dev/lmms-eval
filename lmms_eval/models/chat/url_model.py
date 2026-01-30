import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import aiohttp
import requests
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.protocol import ChatMessages
from lmms_eval.api.model import lmms
from accelerate import Accelerator, DistributedType
from openai import OpenAI
from concurrent.futures import as_completed

WORKERS = int(os.getenv("WORKERS", "1"))

@register_model("url_model")
class URLModel(lmms):
    """
    URL-based model that sends requests to an external API endpoint.
    
    This model converts chat messages to OpenAI format and sends them to
    an external URL endpoint for inference.
    
    Args:
        model_name: Model name (used in API calls)
        base_url: The URL endpoint to send requests to (required)
        api_key: API key for authentication (optional)
        timeout: Request timeout in seconds (default: 600)
        max_retries: Number of retries on failure (default: 3)
        batch_size: Batch size for processing (default: 1)
        max_pixels: Maximum pixels for image resolution (default: 1605632)
        min_image_pixels: Minimum pixels for image (default: 28)
        fps: Frames per second for video processing (optional)
        nframes: Number of frames to extract (default: 32)
    """
    is_simple = False

    def __init__(
        self,
        model_name: str = "/home/devdata/pre-trained/Qwen-Family/Qwen3-VL-4B-Instruct/",
        base_url: str = None,
        api_key: str = None,
        timeout: int = 600,
        max_retries: int = 3,
        batch_size: int = 1,
        max_pixels: int = 1605632,
        min_image_pixels: int = 28,
        fps: Optional[int] = None,
        nframes: Optional[int] = 32,
        disable_log_stats: bool = False,
        **kwargs,
    ):
        """Initialize URL model."""
        super().__init__()
        
        self.model_name = model_name
        self.base_url = base_url or os.getenv("URL_MODEL_BASE_URL")
        self.api_key = api_key or os.getenv("URL_MODEL_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size_per_gpu = batch_size
        self.max_pixels = max_pixels
        self.min_image_pixels = min_image_pixels
        self.fps = fps
        self.nframes = nframes
        self.disable_log_stats = disable_log_stats
        if not self.base_url:
            raise ValueError("base_url must be provided")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)          
            
        # Initialize accelerator for distributed settings
        accelerator = Accelerator()
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
 
    def generate_request(self, request: Instance) -> Tuple[list[dict], dict]:
        """
        Generate a single request to the URL model.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
        raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages = ChatMessages(messages=raw_messages)
        
        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_image_pixels,
            "max_frames": 768,  # Default max frames
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        
        messages = chat_messages.to_openai_messages(video_kwargs=video_kwargs)
    
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=gen_kwargs.get("temperature", 0),
                    top_p=gen_kwargs.get("top_p", 0.95),
                    max_tokens=gen_kwargs.get("max_new_tokens", 4096),
                )
                response = response.choices[0].message.content.strip()

                # 检查响应是否有效
                if response is None or response.strip() == "" or "error" in response.strip().lower():
                    raise ValueError("Invalid response received from URL model")

                return response
            except Exception as e:
                eval_logger.warning(f"{doc_id}: Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        eval_logger.error(f"All attempts failed for {doc_id}")
        return "[ERROR]"
    
    def generate_until(self, requests) -> List[str]:
        """
        Generate responses for a batch of requests.
        
        Args:
            requests: List of Instance objects
            
        Returns:
            List of generated responses
        """
        result = []
        
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        # 使用线程池并行处理请求
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # 提交所有任务
            future_to_idx = {executor.submit(self.generate_request, request): idx for idx, request in enumerate(requests)}
            
            # 按提交顺序收集结果
            results_dict = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    response = future.result()
                    results_dict[idx] = response
                except Exception as e:
                    eval_logger.error(f"Error processing request {idx}: {str(e)}")
                    results_dict[idx] = "[ERROR: Processing failed]"
                pbar.update(1)
        
        # 按原始顺序排列结果
        result = [results_dict[i] for i in range(len(requests))]
        pbar.close()
        return result

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Not implemented for URL model."""
        raise NotImplementedError("URL model does not support loglikelihood")

    def generate_until_multi_round(self, requests) -> List[str]:
        """Not implemented for URL model."""
        raise NotImplementedError("Multi-round generation not implemented for URL model")
    


