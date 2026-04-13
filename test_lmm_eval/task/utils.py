import datetime
import json
import os
import re
from collections import defaultdict
from PIL import Image

from loguru import logger as eval_logger
try:
    from lmms_eval.llm_judge import get_server, ServerConfig, Request, get_judge_model
except ImportError:
    pass

# _JUDGE_MODEL_INSTANCE = None

# def get_judge_model():
#     """获取配置中对应的LLM Judge服务端实例"""
#     global _JUDGE_MODEL_INSTANCE
#     if _JUDGE_MODEL_INSTANCE is not None:
#         return _JUDGE_MODEL_INSTANCE

#     judge_config_str = os.environ.get("LLM_JUDGE_CONFIG")
#     if not judge_config_str:
#         return None
    
#     config_dict = json.loads(judge_config_str)
    
#     # Configure environment variables for API key and base URL if provided
#     api_type = config_dict.get("api_type", "openai")
    
#     if api_type == "openai":
#         if "api_key" in config_dict:
#             os.environ["OPENAI_API_KEY"] = config_dict["api_key"]
#         if "base_url" in config_dict:
#             os.environ["OPENAI_API_URL"] = config_dict["base_url"]
            
#     server_config = ServerConfig(
#         model_name=config_dict.get("model_name", ""),
#         temperature=config_dict.get("temperature", 0.0),
#         max_tokens=config_dict.get("max_tokens", 1024),
#         top_p=config_dict.get("top_p", None)
#     )
    
#     _JUDGE_MODEL_INSTANCE = get_server(api_type, config=server_config)
#     return _JUDGE_MODEL_INSTANCE

# 直接在模块加载时初始化LLM Judge实例
judge_server = get_judge_model("qwen_judge")

nlg_type = ["BLEU", "ROUGE", "METEOR", "BERTScore"]

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pass
def process_content(content_text, images_list, videos_list=None):
    """
    处理user角色的content，将<image>和<video>标签替换为实际的图片和视频内容字典
    
    Args:
        content_text (str): 原始的content文本，可能包含<image>或<video>标签
        images_list (list): 图片路径列表
        videos_list (list): 视频路径列表
        
    Returns:
        list: 转换后的content列表
    """
    content = []
    image_idx = 0
    video_idx = 0
    videos_list = videos_list or []
    
    # 使用正则表达式分割 <image> 和 <video> 标签
    parts = re.split(r'(<image>|<video>)', content_text)
    
    for part in parts:
        if part == '<image>':
            if images_list and image_idx < len(images_list):
                content.append({
                    "type": "image",
                    "url": Image.open(images_list[image_idx]).convert("RGB")  # 直接加载为 PIL Image
                })
                image_idx += 1
        elif part == '<video>':
            if videos_list and video_idx < len(videos_list):
                content.append({
                    "type": "video",
                    "url": videos_list[video_idx]
                })
                video_idx += 1
        elif part:
            content.append({
                "type": "text",
                "text": part
            })
    
    return content

def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    images_sample = doc.get("images", [])
    videos_sample = doc.get("videos", [])
    
    messages = []
    for msg in doc.get("messages", []):
        if isinstance(msg.get("content"), str):
            if msg.get("role") == "user":
                messages.append({
                    "role": msg.get("role"),
                    "content": process_content(msg["content"], images_sample, videos_sample)
                })
            else:
                messages.append({
                    "role": msg.get("role"),
                    "content": [{"type": "text", "text": msg["content"]}]
                })
        else:
            messages.append(msg)

    return messages

def doc_to_answer(doc, lmms_eval_specific_kwargs=None):
    for msg in doc["messages"]:
        if msg["role"] == "assistant":
            answer = msg["content"]
    return answer

def _compute_bleu(pred, ref):
    try:
        import sacrebleu
        return sacrebleu.sentence_bleu(pred, [ref]).score
    except Exception as e:
        eval_logger.warning(f"BLEU compute failed: {e}")
        return 0.0

def _compute_rouge_l(pred, ref):
    try:
        from rouge import Rouge
        rouge = Rouge(metrics=["rouge-l"])
        scores = rouge.get_scores([pred], [ref], avg=True)
        return scores["rouge-l"]["f"] * 100
    except Exception as e:
        eval_logger.warning(f"ROUGE-L compute failed: {e}")
        return 0.0

def process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary mapping each NLG metric name to {"uid": doc["uid"], "score": value}
    """
    pred_ans = results[0].split("<answer>")[-1].split("</answer>")[0].strip()
    gt_ans = doc_to_answer(doc).split("<answer>")[-1].split("</answer>")[0].strip()

    if judge_server:
        eval_logger.info("Using LLM Judge for evaluation...")
        # Customize this depending on your evaluation setup context
        req = Request(
            messages=[{
                "role": "user",
                "content": f"Please evaluate this prediction.\nGround Truth: {gt_ans}\nPrediction: {pred_ans}\nReturn '1' if correct, '0' if incorrect."
            }]
        )
        try:
            resp = judge_server.evaluate(req)
            score = 1.0 if '1' in resp.content else 0.0
            return {
                "LLM_JUDGE": {"id": doc.get("id"), "metric": "LLM_JUDGE", "score": score}
            }
        except Exception as e:
            eval_logger.warning(f"LLM Judge evaluation failed: {e}")

    bleu = _compute_bleu(pred_ans, gt_ans)
    rouge_l = _compute_rouge_l(pred_ans, gt_ans)

    # return {
    #     nlg_type[0]: {"uid": doc.get("uid"), "metric": nlg_type[0], "score": bleu},
    #     nlg_type[1]: {"uid": doc.get("uid"), "metric": nlg_type[1], "score": rouge_l},
    #     nlg_type[2]: {"uid": doc.get("uid"), "metric": nlg_type[2], "score": meteor},
    #     nlg_type[3]: {"uid": doc.get("uid"), "metric": nlg_type[3], "score": bertscore},
    # }
    return {
        "BLEU": {"id": doc.get("id"), "metric": "BLEU", "score": bleu},
        "ROUGE": {"id": doc.get("id"), "metric": "ROUGE", "score": rouge_l},
        "LLM_JUDGE": {"id": doc.get("id"), "metric": "LLM_JUDGE", "score": score}
    }

def aggregate_results(results):
    """
    Aggregate each metric results into a single score (mean).
    Returns a dictionary with average scores for each NLG metric.
    """
    # 虽然计算了四个指标，但每次只计算一个指标的结果
    # 四个指标的结果在evaluator主函数中分别有一个results传入，即该函数调用了4次
    category2score = defaultdict(list)
    
    for result in results:
        id = result["id"]
        metric = result["metric"]
        score = result["score"]
        if metric and score is not None:
            category2score[metric].append(score)
        else:
            eval_logger.warning(f"Missing metric or score for uid: {uid}")
    
    category2avg_score = {}
    for category, scores in category2score.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        category2avg_score[category] = avg_score
        eval_logger.info(f"{category}: {avg_score:.2f}")
    
    return category2avg_score