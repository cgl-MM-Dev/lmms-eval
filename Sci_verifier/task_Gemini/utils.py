import sys
import os
import re
import json
import base64
import io
from openai import OpenAI
from PIL import Image

# Add path for DMXAPI or use direct OpenAI call for evaluation
# The user's script used this path
sys.path.append("/home/devdata/Dataset/SciGen-Verify/benchmark/wzq/data-process-toolkits")

try:
    from datatool.apis.dmx_api import DMXAPI
except ImportError:
    # Fallback if import fails, though user environment should have it
    DMXAPI = None
    print("Warning: Could not import DMXAPI. Explanation evaluation might fail if not handled manually.")

# Set API Key from environment or hardcode as in the user script (though env is better)
# os.environ['DMX_API_KEY'] = 'sk-hgIEgl56AYxtQRhh79Pa2qLafITX0XeCxZXCXXxqKYuSXMhr'

API_KEY = os.environ.get('DMX_API_KEY', 'sk-gSitTfg3Qsq0FyqPW1dezCxeUxAW3Q0zAVQSDl3buY06kO2Y')
EVAL_MODEL_NAME_EXP = "glm-4.7"
EVAL_MODEL_NAME_EDIT = "doubao-seed-2-0-lite-260215"
MAX_IMAGE_PIXELS = int(os.environ.get("SCIGEN_MAX_IMAGE_PIXELS", "200704"))
MAX_IMAGE_BYTES = int(os.environ.get("SCIGEN_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
IMAGE_URL_AS_DATA_URI = os.environ.get("SCIGEN_IMAGE_DATA_URI", "1") not in ("0", "false", "False")

def image_to_base64(image_path):
    # This might not be needed if lmms-eval handles image loading, 
    # but we might need it if we were constructing the request manually.
    # lmms-eval framework handles image loading for the model being evaluated.
    pass

def load_image_with_limits(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return None
    Image.MAX_IMAGE_PIXELS = None
    try:
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            pixels = w * h
            if pixels > MAX_IMAGE_PIXELS:
                scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                img = img.resize((new_w, new_h), resample)
            quality = 92
            current_img = img
            image_bytes = None
            for _ in range(8):
                buf = io.BytesIO()
                current_img.save(buf, format="JPEG", quality=quality, optimize=True)
                image_bytes = buf.getvalue()
                if len(image_bytes) <= MAX_IMAGE_BYTES:
                    break
                if quality > 60:
                    quality -= 10
                else:
                    new_w = max(int(current_img.size[0] * 0.9), 1)
                    new_h = max(int(current_img.size[1] * 0.9), 1)
                    current_img = current_img.resize((new_w, new_h), resample)
            if image_bytes is None:
                return None
            if len(image_bytes) > MAX_IMAGE_BYTES:
                scale = (MAX_IMAGE_BYTES / len(image_bytes)) ** 0.5
                new_w = max(int(current_img.size[0] * scale), 1)
                new_h = max(int(current_img.size[1] * scale), 1)
                current_img = current_img.resize((new_w, new_h), resample)
                buf = io.BytesIO()
                current_img.save(buf, format="JPEG", quality=max(quality, 60), optimize=True)
                image_bytes = buf.getvalue()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

def image_to_base64(image_path: str) -> str:
    if not os.path.exists(image_path):
        return None
    try:
        try:
            with open(image_path, "rb") as f:
                header = f.read(8)
            if header[:2] == b"\xff\xd8":
                mime_type = "image/jpeg"
            elif header[:4] == b"\x89PNG":
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"
        except Exception:
            mime_type = "image/jpeg"
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(image_path) as img:
            w, h = img.size
            pixels = w * h
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            if pixels > MAX_IMAGE_PIXELS:
                scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
                new_w = max(int(w * scale), 1)
                new_h = max(int(h * scale), 1)
                img = img.resize((new_w, new_h), resample)
            if mime_type == "image/jpeg":
                img = img.convert("RGB")
            quality = 92
            current_img = img
            image_data = None
            for _ in range(8):
                buf = io.BytesIO()
                if mime_type == "image/jpeg":
                    current_img.save(buf, format="JPEG", quality=quality, optimize=True)
                else:
                    current_img.save(buf, format="PNG", optimize=True)
                image_data = buf.getvalue()
                if len(image_data) <= MAX_IMAGE_BYTES:
                    break
                if mime_type == "image/jpeg" and quality > 60:
                    quality -= 10
                else:
                    new_w = max(int(current_img.size[0] * 0.9), 1)
                    new_h = max(int(current_img.size[1] * 0.9), 1)
                    current_img = current_img.resize((new_w, new_h), resample)
            if len(image_data) > MAX_IMAGE_BYTES:
                scale = (MAX_IMAGE_BYTES / len(image_data)) ** 0.5
                new_w = max(int(current_img.size[0] * scale), 1)
                new_h = max(int(current_img.size[1] * scale), 1)
                current_img = current_img.resize((new_w, new_h), resample)
                buf = io.BytesIO()
                if mime_type == "image/jpeg":
                    current_img.save(buf, format="JPEG", quality=max(quality, 60), optimize=True)
                else:
                    current_img.save(buf, format="PNG", optimize=True)
                image_data = buf.getvalue()
        base64_str = base64.b64encode(image_data).decode("utf-8")
        if IMAGE_URL_AS_DATA_URI:
            return f"data:{mime_type};base64,{base64_str}"
        return base64_str
    except Exception:
        return None


def process_content(content_text, images_list):
    """
    处理user角色的content，将<image>标签替换为实际的图片路径字典

    Args:
        content_text (str): 原始的content文本，可能包含<image>标签
        images_list (list): 图片路径列表

    Returns:
        list: 转换后的content列表
    """
    content = []
    image_idx = 0

    parts = re.split(r'(<image>)', content_text)

    for part in parts:
        if part == '<image>':
            if images_list and image_idx < len(images_list):
                img_path = images_list[image_idx]
                compressed_image = load_image_with_limits(img_path)
                if compressed_image is None:
                    image_idx += 1
                    continue
                content.append({
                    "type": "image",
                    "url": compressed_image
                })
                image_idx += 1
        elif part:
            content.append({
                "type": "text",
                "text": part
            })

    return content

def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    Extracts the user prompt from the dataset item.
    """
    messages = doc.get("messages", [])
    user_prompt = ""
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content')
            if isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        user_prompt += part.get('text', '')
            else:
                user_prompt = str(content)
            # We assume the last user message is the prompt or accumulate?
            # User script accumulates all user parts? 
            # "user_prompt += ..." inside the loop over messages.
            # But usually there is only one user message prompt for the task.
            # Let's stick to the script logic: it iterates all messages and extracts user content.
    
    # Update prompt with instructions
    return user_prompt

def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """
    Returns the list of images.
    lmms-eval handles PIL conversion if we return paths.
    """
    return doc.get("images", [])

def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    """
    converts doc to a list of messages for chat models.
    """
    messages = doc.get("messages", [])
    images = doc.get("images", [])
    
    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content')
            if isinstance(content, str):
                processed_content = process_content(content, images)
            elif isinstance(content, list):
                processed_content = []
                for part in content:
                    if part.get('type') == 'text':
                        text = part.get('text', '')
                        processed_content.extend(process_content(text, images))
                    elif part.get('type') == 'image':
                        image_url = part.get("url")
                        compressed_image = load_image_with_limits(image_url) if isinstance(image_url, str) else image_url
                        if compressed_image is None:
                            continue
                        processed_content.append({
                            "type": "image",
                            "url": compressed_image
                        })
            else:
                processed_content = [{"type": "text", "text": str(content)}]
            
            user_msg = {
                "role": "user",
                "content": processed_content
            }
            return [user_msg]
    
    return [{"role": "user", "content": [{"type": "text", "text": ""}]}]

def extract_answer_from_content(content):
    """Extract answer from JSON response: {"answer": true/false, ...}"""
    if not content:
        return None
    text = str(content).strip()

    json_obj = extract_json_object(text)
    if isinstance(json_obj, dict) and "answer" in json_obj:
        val = json_obj.get("answer")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            v = val.strip().lower()
            if v == "true":
                return True
            if v == "false":
                return False

    content_str = text.lower()

    try:
        match = re.search(r'<\s*a\s*>\s*(true|false)\s*<\s*/\s*a\s*>', content_str, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip() == 'true'
    except:
        pass

    if content_str == "true": return True
    if content_str == "false": return False
    
    if "true" in content_str and "false" not in content_str: return True
    if "false" in content_str and "true" not in content_str: return False
    
    return None

def extract_explanation(text):
    if not text:
        return None
    json_obj = extract_json_object(str(text))
    if isinstance(json_obj, dict) and "explanation" in json_obj:
        val = json_obj.get("explanation")
        if isinstance(val, str):
            val = val.strip()
            if val:
                return val
        elif val is not None:
            val = str(val).strip()
            if val:
                return val
    try:
        match = re.search(r'<\s*ex\s*>(.*?)<\s*/\s*ex\s*>', str(text), re.IGNORECASE | re.DOTALL)
        if match: return match.group(1).strip()
    except: pass
    return None

def extract_edit(text):
    if not text:
        return None
    json_obj = extract_json_object(str(text))
    if isinstance(json_obj, dict):
        if "edit" in json_obj:
            val = json_obj.get("edit")
            if isinstance(val, str):
                val = val.strip()
                if val:
                    return val
            elif val is not None:
                val = str(val).strip()
                if val:
                    return val
        if "ed" in json_obj:
            val = json_obj.get("ed")
            if isinstance(val, str):
                val = val.strip()
                if val:
                    return val
            elif val is not None:
                val = str(val).strip()
                if val:
                    return val
    try:
        match = re.search(r'<\s*ed\s*>(.*?)<\s*/\s*ed\s*>', str(text), re.IGNORECASE | re.DOTALL)
        if match:
            val = match.group(1).strip()
            if val:
                return val
    except:
        pass
    try:
        match = re.search(r'"ed"\s*:\s*"([\s\S]*?)"', str(text), re.IGNORECASE)
        if match:
            val = match.group(1).strip()
            if val:
                return val
    except:
        pass
    text_str = str(text).strip()
    return text_str if text_str else None

def extract_edit_by_regex(text):
    return extract_edit(text)


def extract_json_object(text):
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        match = re.search(r'\{[\s\S]*\}', str(text))
        if not match:
            return None
        obj = json.loads(match.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None

def get_ground_truth_answer(doc):
    """From 2_评估answer.py"""
    messages = doc.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # The GT usually just says "True" or "False" or includes it?
            # Script 2 uses extract_answer_from_content on GT too.
            val = extract_answer_from_content(content)
            if val is not None: return val
    return None

def doc_to_target_answer(doc, lmms_eval_specific_kwargs=None):
    return get_ground_truth_answer(doc)

def build_explanation_eval_prompt(instruction, gt_explanation, pred_explanation):
    """From 3_评估explanation_dmx.py"""
    return f"""You are a strict evaluator for a scientific image verification system.
Your task is to compare the "Predicted Explanation" generated by a model against the "Ground Truth Explanation".

**Context (Instruction):**
{instruction}

**Ground Truth Explanation (Standard Answer):**
{gt_explanation}

**Predicted Explanation (To be Evaluated):**
{pred_explanation}

**Task:**
Determine if the "Predicted Explanation" identifies the **same core discrepancy** as the "Ground Truth Explanation".
Focus on semantic meaning and logical consistency.

**Judgement Criteria:**
- Return `true` if the predicted explanation identifies the error described in the ground truth.
- Return `false` if the prediction is irrelevant or incorrect.

**Output Format (JSON only):**
{{
    "result": <boolean, true or false>
}}"""

def build_edit_eval_prompt(instruction, gt_edit, pred_edit):
    return f"""You are a senior scientific image editor.
You are reviewing a suggested "Edit Plan" to fix an incorrect image based on a "Generation Context" and the provided images.

**Background Context:**
- The **Generation Context** below includes the original instruction and the generated image result (represented by <image>).
- The generated image is **incorrect** and contains errors or deviations from the instruction requirements.
- A **Ground Truth Edit Plan** is provided as a reference, which describes how to correctly fix the errors in the generated image.

**Generation Context (Instruction & Incorrect Image):**
{instruction}

**Ground Truth Edit Plan (Standard Reference for Fixing):**
{gt_edit}

**Predicted Edit Plan (To be Evaluated):**
{pred_edit}

**Task:**
Judge whether the "Predicted Edit Plan" is a valid and correct fix for the incorrect image.
Compare it with the "Ground Truth Edit Plan" and the visual context provided in the generation context.
- Return `true` if the predicted edit is semantically consistent with the ground truth or effectively fixes the issue.
- Return `false` if the predicted edit is irrelevant, incorrect, contradictory to the ground truth, or fails to address the core problem.

**Output Format (JSON only):**
{{
    "result": <boolean, true or false>
}}"""

def pil_image_to_data_uri(img):
    if img is None:
        return None
    try:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
        base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"
    except Exception:
        return None

def call_exp_eval_api(prompt, model=EVAL_MODEL_NAME_EXP):
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://www.dmxapi.cn/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            reasoning_effort="medium",
            max_tokens=16384,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Eval API failed: {e}")
        return None

def call_edit_eval_api(prompt, images, model=EVAL_MODEL_NAME_EDIT):
    try:
        content = process_content(prompt, images)
        content_parts = []
        for part in content:
            if part.get("type") == "text":
                content_parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                data_uri = pil_image_to_data_uri(part.get("url"))
                if data_uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        valid_image_count = sum(1 for p in content_parts if p.get("type") == "image_url")
        if valid_image_count <= 0 and images:
            for img in images:
                data_uri = pil_image_to_data_uri(load_image_with_limits(img))
                if data_uri:
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            valid_image_count = sum(1 for p in content_parts if p.get("type") == "image_url")

        if valid_image_count <= 0:
            return None

        client = OpenAI(api_key=API_KEY, base_url="https://www.dmxapi.cn/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": content_parts}],
            model=model,
            reasoning_effort="medium",
            max_tokens=16384,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Edit eval API failed: {e}")
        return None

def extract_json_result(response_text):
    """From 3_评估explanation_dmx.py"""
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_obj = json.loads(match.group(0))
            result = json_obj.get("result", None)
            if isinstance(result, bool): return result
            if isinstance(result, str): return result.lower() == "true"
        return None
    except Exception:
        return None

def compute_format_and_success_stats(pred_answer, pred_explanation, pred_edit):
    answer_extract_success = 1 if isinstance(pred_answer, bool) else 0
    answer_extract_fail = 1 - answer_extract_success

    true_exp_success = None
    true_exp_fail = None
    if pred_answer is True:
        true_exp_success = 1 if bool(pred_explanation) else 0
        true_exp_fail = 1 - true_exp_success

    false_exp_success = None
    false_exp_fail = None
    false_edit_success = None
    false_edit_fail = None
    false_both_success = None
    false_both_fail = None
    if pred_answer is False:
        false_exp_success = 1 if bool(pred_explanation) else 0
        false_exp_fail = 1 - false_exp_success
        false_edit_success = 1 if bool(pred_edit) else 0
        false_edit_fail = 1 - false_edit_success
        false_both_success = 1 if (bool(pred_explanation) and bool(pred_edit)) else 0
        false_both_fail = 1 - false_both_success

    if pred_answer is None:
        format_error = 1
    elif pred_answer is True:
        format_error = 0 if bool(pred_explanation) else 1
    else:
        format_error = 0 if (bool(pred_explanation) and bool(pred_edit)) else 1

    return {
        "format_error": format_error,
        "format_ok": 1 - format_error,
        "infer_answer_extract_success": answer_extract_success,
        "infer_answer_extract_fail": answer_extract_fail,
        "infer_true_explanation_extract_success": true_exp_success,
        "infer_true_explanation_extract_fail": true_exp_fail,
        "infer_false_explanation_extract_success": false_exp_success,
        "infer_false_explanation_extract_fail": false_exp_fail,
        "infer_false_edit_extract_success": false_edit_success,
        "infer_false_edit_extract_fail": false_edit_fail,
        "infer_false_explanation_edit_both_success": false_both_success,
        "infer_false_explanation_edit_both_fail": false_both_fail,
        "eval_success_rate": answer_extract_success,
        "eval_fail_rate": answer_extract_fail,
    }

def process_results(doc, results):
    """
    Computes metrics: Answer Accuracy, Explanation Validity.
    """
    # results is a list of strings (generated responses)
    pred_text = results[0]
    
    # 1. Parse Prediction
    gt_answer = get_ground_truth_answer(doc)
    evaluation_result = {}
    judge_stats = {
        "exp_judge_extract_success": None,
        "exp_judge_extract_fail": None,
        "edit_judge_extract_success": None,
        "edit_judge_extract_fail": None,
        "exp_judge_api_success": None,
        "edit_judge_api_success": None,
    }

    if pred_text is None or str(pred_text).strip() == "" or str(pred_text).strip() == "[ERROR]":
        stats = compute_format_and_success_stats(None, None, None)
        evaluation_result["answer_correct"] = None
        evaluation_result["predicted_answer"] = None
        evaluation_result["ground_truth_answer"] = gt_answer
        evaluation_result["answer_score"] = None
        evaluation_result["exp_is_valid"] = None
        evaluation_result["edit_is_valid"] = None
        evaluation_result["reason"] = "Inference API failed"
        evaluation_result.update(stats)
        evaluation_result.update(judge_stats)
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": None, "exp_is_valid": None, "edit_is_valid": None, **stats, **judge_stats}

    pred_answer = extract_answer_from_content(pred_text)
    pred_explanation = extract_explanation(pred_text)
    pred_edit = extract_edit(pred_text)
    pred_edit_regex = extract_edit_by_regex(pred_text)
    stats = compute_format_and_success_stats(pred_answer, pred_explanation, pred_edit_regex)
    evaluation_result.update(stats)
    evaluation_result.update(judge_stats)

    if gt_answer is None:
        evaluation_result["answer_correct"] = None
        evaluation_result["predicted_answer"] = pred_answer
        evaluation_result["ground_truth_answer"] = None
        evaluation_result["answer_score"] = None
        evaluation_result["exp_is_valid"] = None
        evaluation_result["edit_is_valid"] = None
        evaluation_result["reason"] = "Ground truth answer missing"
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": None, "exp_is_valid": None, "edit_is_valid": None, **stats, **judge_stats}

    acc = 0
    if pred_answer is not None and gt_answer is not None and pred_answer == gt_answer:
        acc = 1

    evaluation_result["answer_correct"] = bool(acc == 1)
    evaluation_result["predicted_answer"] = pred_answer
    evaluation_result["ground_truth_answer"] = gt_answer
    evaluation_result["answer_score"] = 100 if acc == 1 else 0
    if pred_answer is None:
        evaluation_result["answer_note"] = "Failed to parse prediction"
        evaluation_result["exp_is_valid"] = None
        evaluation_result["edit_is_valid"] = None
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": acc, "exp_is_valid": None, "edit_is_valid": None, **stats, **judge_stats}

    exp_valid = 0
    edit_valid = None
    if acc != 1:
        evaluation_result["exp_is_valid"] = False
        evaluation_result["reason"] = "Answer is incorrect"
        evaluation_result["edit_is_valid"] = False
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": acc, "exp_is_valid": exp_valid, "edit_is_valid": 0, **stats, **judge_stats}

    if not pred_explanation:
        evaluation_result["exp_is_valid"] = False
        exp_valid = 0
        evaluation_result["edit_is_valid"] = False
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": acc, "exp_is_valid": exp_valid, "edit_is_valid": 0, **stats, **judge_stats}

    instruction = doc.get("question", "")
    eeg_gt = doc.get("explanation_edit_generation", {})
    gt_explanation = eeg_gt.get("explanation", "No explanation provided.")
    prompt = build_explanation_eval_prompt(instruction, gt_explanation, pred_explanation)
    resp_text = call_exp_eval_api(prompt)
    if not resp_text:
        judge_stats["exp_judge_api_success"] = 0
        evaluation_result.update(judge_stats)
        evaluation_result["exp_is_valid"] = None
        evaluation_result["edit_is_valid"] = None
        evaluation_result["reason"] = "Explanation eval API failed"
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": acc, "exp_is_valid": None, "edit_is_valid": None, **stats, **judge_stats}
    judge_stats["exp_judge_api_success"] = 1

    is_valid = extract_json_result(resp_text)
    if is_valid is None:
        judge_stats["exp_judge_extract_success"] = 0
        judge_stats["exp_judge_extract_fail"] = 1
        evaluation_result.update(judge_stats)
        evaluation_result["exp_is_valid"] = None
        evaluation_result["edit_is_valid"] = None
        evaluation_result["reason"] = "Failed to parse explanation eval result"
        doc["evaluation_result"] = evaluation_result
        return {"answer_correct": acc, "exp_is_valid": None, "edit_is_valid": None, **stats, **judge_stats}
    judge_stats["exp_judge_extract_success"] = 1
    judge_stats["exp_judge_extract_fail"] = 0
    evaluation_result.update(judge_stats)

    if is_valid:
        exp_valid = 1
    evaluation_result["exp_is_valid"] = bool(exp_valid == 1)

    if gt_answer is True:
        edit_valid = None
        evaluation_result["edit_is_valid"] = None
    elif pred_answer is True:
        edit_valid = 0
        evaluation_result["edit_is_valid"] = False
        evaluation_result["edit_reason"] = "Prediction is True but GT is False"
    else:
        gt_edit = eeg_gt.get("edit", "")
        image_paths = doc.get("images", [])
        if not pred_edit:
            edit_valid = 0
            evaluation_result["edit_is_valid"] = False
        elif not image_paths:
            edit_valid = 0
            evaluation_result["edit_is_valid"] = False
            evaluation_result["edit_reason"] = "No images provided"
        else:
            edit_prompt = build_edit_eval_prompt(instruction, gt_edit, pred_edit)
            edit_resp_text = call_edit_eval_api(edit_prompt, image_paths)
            if not edit_resp_text:
                judge_stats["edit_judge_api_success"] = 0
                evaluation_result.update(judge_stats)
                edit_valid = None
                evaluation_result["edit_is_valid"] = None
                evaluation_result["edit_reason"] = "Edit eval API failed or no valid image after compression"
            else:
                judge_stats["edit_judge_api_success"] = 1
                edit_is_valid = extract_json_result(edit_resp_text)
                if edit_is_valid is None:
                    judge_stats["edit_judge_extract_success"] = 0
                    judge_stats["edit_judge_extract_fail"] = 1
                    evaluation_result.update(judge_stats)
                    edit_valid = None
                    evaluation_result["edit_is_valid"] = None
                    evaluation_result["edit_reason"] = "Failed to parse edit eval result"
                else:
                    judge_stats["edit_judge_extract_success"] = 1
                    judge_stats["edit_judge_extract_fail"] = 0
                    evaluation_result.update(judge_stats)
                    edit_valid = 1 if edit_is_valid else 0
                    evaluation_result["edit_is_valid"] = bool(edit_valid == 1)

    doc["evaluation_result"] = evaluation_result
    return {"answer_correct": acc, "exp_is_valid": exp_valid, "edit_is_valid": edit_valid, **stats, **judge_stats}

def aggregate_results(results):
    """Averages the results"""
    valid_results = [x for x in results if isinstance(x, (int, float, bool))]
    if not valid_results:
        return 0
    return sum(valid_results) / len(valid_results)

def scigen_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return doc.get("images", [])

def scigen_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc_to_text(doc)
