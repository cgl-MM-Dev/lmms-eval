import sys
import os
import re
import json
import base64
import io
import ast
from openai import OpenAI
from PIL import Image

sys.path.append("/home/devdata/Dataset/SciGen-Verify/benchmark/wzq/data-process-toolkits")

try:
    from datatool.apis.dmx_api import DMXAPI
    from datatool.apis.silicon_api import SILICONAPI
except ImportError:
    DMXAPI = None
    SILICONAPI = None
    print("Warning: Could not import DMXAPI or SILICONAPI.")

API_KEY = os.environ.get('SILICON_API_KEY', 'sk-CYgRzEZ8OPW6Etsb5Nn4RLQ9JygQDT1qRGJxyOJNkDkGvVpZ')
EVAL_MATCH_MODEL = "glm-4.7"
MAX_IMAGE_PIXELS = int(os.environ.get("SCIGEN_MAX_IMAGE_PIXELS", "200704"))
MAX_IMAGE_BYTES = int(os.environ.get("SCIGEN_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
IMAGE_URL_AS_DATA_URI = os.environ.get("SCIGEN_IMAGE_DATA_URI", "1") not in ("0", "false", "False")

RUBRIC_GEN_TEMPLATE_TEXT = r"""You are an expert evaluator for Large Language Models, specializing in **instruction-following**. Your task is to analyze a given user instruction and generate a detailed **evaluation checklist** (or "rubric").
This checklist will be used by a human or an AI evaluator to judge whether a *subsequent* LLM response strictly and accurately follows all directives in the original instruction.
The goal is to identify and isolate every single **"critic key point"** or **constraint** . You must deconstruct the instruction into testable components.
--
### Instructions for Checklist Generation
1. **Deeply Analyze the [User Instruction]:** Read the instruction carefully. Deconstruct it into all its component parts. Identify:
* **Explicit Constraints:** Direct commands (e.g., quantities, formats, specific content).
* **Implicit Constraints:** Implied tasks (e.g., answering all sub-questions, maintaining context).
* **Stylistic Constraints:** formatting requirements.
* **Negative Constraints:** Things to explicitly avoid.
2. **Categorize Key Points:** Generate a markdown-formatted checklist. You **must** categorize each key point into one of the following four levels of importance.
3. **Format:** Use clear, simple language for each checklist item. Each item should be a single, verifiable question or statement.
### Checklist Structure
You must follow this exact markdown structure for your output:
## 1. Hard Constraints
*(These are non-negotiable, pass/fail key points. Failure here means the instruction was not followed. This is where most "critic key points" like exact numbers belong.)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item] 
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 2. Core Task Fulfillment
*(These relate to the main purpose or topic of the instruction. Did the response successfully complete the primary task's goal?)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item] 
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 3. Optional Criteria (Style & Quality)
*(These are secondary instructions for style or formatting. Failing these makes the response lower quality but not an outright failure of the core instruction.)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 4. Pitfall Criteria (Explicit Violations)
*(These explicitly list what the response **must not** do. They are the inverse of essential criteria and catch common errors or explicit negative constraints.)*
* `[ ]` **Pitfall:** [Description of the violation to check for] 
* `[ ]` **Pitfall:** [Description of the violation to check for]
--
### Example Task
**[User Instruction]:** "Please generate 5 bullet points explaining the benefits of hydration. Be concise and use a professional tone. Do not mention any specific brands of water."
### Example Checklist Output
## 1. Essential Criteria (Hard Constraints)
* `[ ]` **Count:** Does the response contain *exactly* 5 points? 
* `[ ]` **Format:** Are the 5 points presented as bullet points?
* `[ ]` **Negative Constraint:** Does the response avoid mentioning *any* specific water brands?
## 2. Important Criteria (Core Task Fulfillment)
* `[ ]` **Topic:** Do all 5 points describe the "benefits of hydration"?
* `[ ]` **Conciseness:** Are the points concise (e.g., short sentences, not long paragraphs)?
## 3. Pitfall Criteria (Explicit Violations)
* `[ ]` **Pitfall (Count):** Response generates fewer or more than 5 points. 
* `[ ]` **Pitfall (Brand):** Response mentions a brand name (e.g., "Evian," "Fiji").
* `[ ]` **Pitfall (Topic):** Response discusses unrelated topics (e.g., nutrition, exercise).
* `[ ]` **Pitfall (Tone):** Response uses casual, informal, or slang language.
"""

RUBRIC_GEN_TEMPLATE_IMAGE = r"""You are an expert evaluator for Large Language Models, specializing in **instruction-following** and **Visual Reasoning**. Your task is to analyze a given user instruction alongside a provided **Generated Image** and create a detailed **evaluation checklist** (or "rubric").
This checklist will be used to judge whether the *subsequent* LLM response strictly and accurately follows all directives in the original instruction and clearly reflects the content of the image.
The goal is to identify and isolate every single **"critic key point"** or **constraint** based on BOTH the text instruction and the visual content. You must deconstruct the task into testable components.
--
### Instructions for Checklist Generation
1. **Deeply Analyze the [User Instruction] AND [Generated Image]:** Read the instruction carefully and observe the image. Deconstruct them into components. Identify:
* **Explicit Constraints:** Direct commands (e.g., quantities, formats, specific visible content, spatial relationships).
* **Implicit Constraints:** Implied tasks (e.g., answering all sub-questions, maintaining context based on image).
* **Stylistic Constraints:** formatting requirements.
* **Negative Constraints:** Things to explicitly avoid.
2. **Categorize Key Points:** Generate a markdown-formatted checklist. You **must** categorize each key point into one of the following four levels of importance.
3. **Format:** Use clear, simple language for each checklist item. Each item should be a single, verifiable question or statement.
### Checklist Structure
You must follow this exact markdown structure for your output:
## 1. Hard Constraints
*(These are non-negotiable, pass/fail key points. Failure here means the instruction was not followed. This is where most "critic key points" like exact numbers belong.)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item] 
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 2. Core Task Fulfillment
*(These relate to the main purpose or topic of the instruction. Did the response successfully complete the primary task's goal?)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item] 
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 3. Optional Criteria (Style & Quality)
*(These are secondary instructions for style or formatting. Failing these makes the response lower quality but not an outright failure of the core instruction.)*
* `[ ]` **[Criteria Title]:** [Verifiable checklist item]
## 4. Pitfall Criteria (Explicit Violations)
*(These explicitly list what the response **must not** do. They are the inverse of essential criteria and catch common errors or explicit negative constraints.)*
* `[ ]` **Pitfall:** [Description of the violation to check for] 
* `[ ]` **Pitfall:** [Description of the violation to check for]
"""

BATCH_JUDGE_MATCH_TEMPLATE = r"""You are an expert evaluator for the Rubric benchmark. Your task is to perform a batch semantic alignment between a set of "Candidate Rubric Rules" and a set of "Gold Standard Rules".
### Instruction:
Compare each Candidate Rule against the entire Gold Rules list. Determine if there is a semantic match.
### Strict Matching Criteria A "Hit" (YES) requires: 
1. Specific Intent Match: The Candidate Rule must check the EXACT SAME constraint as the Gold Rule (e.g., if Gold checks "Structure", Candidate must check "Structure", not just "Quality"). 
2. Scope Match: The Candidate Rule must not be significantly broader or vaguer than the Gold Rule.
### Automatic Rejection Criteria (NO) 
- Vague vs Specific: If Candidate says "Is the explanation good/detailed?" and Gold says "Does it mention Concept X?", this is NO. 
- Different Dimension: If Candidate checks "Content" and Gold checks "Structure/Formatting", this is NO. 
- Partial Overlap: If Candidate checks "Relevance" but maps it to a Gold Rule about "Completeness", this is NO (unless the correct Gold Rule is missing).
### Evaluation Policy (Must Follow) 
Semantic equivalence means the Candidate Rule would accept and reject the same set of responses as the Gold Rule in practice. Any broadening, weakening, or generalization of constraints counts as a scope mismatch. Do NOT combine partial overlaps across multiple Gold Rules to justify a YES. If "hit" is NO, return an empty list for hit_gold_rule_indices. 

CRITICAL JSON RULES:
1. Wrap ALL keys and string values in DOUBLE QUOTES (""). Never use single quotes ('').
2. Escape all literal double quotes inside strings using backslash (e.g., "Word").
3. Escape all newlines inside strings with \n. Do not output literal line breaks inside strings.
4. Separate items in lists and dictionaries with COMMAS (,). Do not miss the comma.
### Input Data 
Gold Rules List: {gold_rules}
Candidate Rules to Evaluate: {candidate_rules}
### Output Format:
Output a strictly valid JSON object where the key "match_results" contains a list of objects, one for each Candidate Rule in the exact order provided. Do not include markdown formatting like ```json or any other commentary.
Example:
{{
  "match_results": [
    {{ "candidate": "Rule 1 text", "hit": "YES", "hit_gold_rule_indices": [0, 2] }},
    {{ "candidate": "Rule 2 text", "hit": "NO", "hit_gold_rule_indices": [] }}
  ]
}}"""

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


def process_content(content_text, images_list):
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
    pass

def doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    return doc.get("images", [])

def doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    messages = doc.get("messages", [])
    images = doc.get("images", [])
    
    template = RUBRIC_GEN_TEMPLATE_IMAGE if len(images) > 1 else RUBRIC_GEN_TEMPLATE_TEXT
    
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": template}]
    }
    
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
                        if compressed_image is not None:
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
            return [system_msg, user_msg]
    
    return [system_msg, {"role": "user", "content": [{"type": "text", "text": ""}]}]


def extract_candidate_rules(markdown_text):
    if not markdown_text or not isinstance(markdown_text, str):
        return []
    
    pattern = r"\*\s+(?:`\[\s?\]`|\[\s?\])\s+(?:\*\*(.*?)\*\*:\s*)?(.*)"
    matches = re.findall(pattern, markdown_text)
    
    rules = []
    for title, content in matches:
        full_rule = f"{title}: {content}" if title else content
        if full_rule.strip():
            rules.append(full_rule.strip())
        
    if not rules:
        simple_pattern = r"(?:`\[\s?\]`|\[\s?\])\s*(.*)"
        simple_matches = re.findall(simple_pattern, markdown_text)
        rules = [m.strip() for m in simple_matches if m.strip()]
        
    return rules

def extract_json_block(text):
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        return text[start_idx:end_idx+1]
    return ""
    
def call_eval_api(prompt, model=EVAL_MATCH_MODEL):
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://www.dmxapi.cn/v1")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=8192,
            temperature=0.1,
            extra_body={"enable_thinking":False}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Eval API failed: {e}")
        return None

def process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred] (the string generated by gemini containing the candidate rubric)
    """
    pred_text = results[0]
    
    # Extract candidate rules using regex
    candidate_rules = extract_candidate_rules(pred_text)
    
    # Extract gold rules
    gold_rules = doc.get("gold_rules", [])
    if not gold_rules and "criteria" in doc:
        for criterion in doc["criteria"]:
            gold_rules.extend(criterion.get("points", []))
    
    if not gold_rules:
        # Without gold rules we can't calculate metrics
        doc["evaluation_result"] = {
            "error": "No gold rules found",
            "candidate_rules": candidate_rules
        }
        return {"RubricRecall": 0.0, "HallucinationRate": 0.0, "StructuralF1": 0.0}
        
    if not candidate_rules:
        # Model failed to generate any recognizable rules
        doc["evaluation_result"] = {
            "error": "No candidate rules generated/extracted",
            "raw_pred": pred_text
        }
        return {"RubricRecall": 0.0, "HallucinationRate": 1.0, "StructuralF1": 0.0}
        
    # Format and call API for matching
    formatted_gold = "\n".join([f"{i}. {rule}" for i, rule in enumerate(gold_rules)])
    formatted_candidate = "\n".join([f"{j}. {rule}" for j, rule in enumerate(candidate_rules)])
    prompt = BATCH_JUDGE_MATCH_TEMPLATE.format(gold_rules=formatted_gold, candidate_rules=formatted_candidate)
    
    resp_text = call_eval_api(prompt)
    
    match_results = []
    if resp_text:
        json_str = extract_json_block(resp_text)
        if json_str:
            json_str = json_str.replace('\n', ' ')
            try:
                parsed = json.loads(json_str, strict=False)
                match_results = parsed.get("match_results", [])
            except json.JSONDecodeError:
                try:
                    content_fix = json_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
                    parsed = ast.literal_eval(content_fix)
                    match_results = parsed.get("match_results", [])
                except:
                    # Fallback to regex
                    hits = re.findall(r'[\'"]hit[\'"]\s*:\s*[\'"](YES|NO)[\'"]', json_str, re.IGNORECASE)
                    indices_raw = re.findall(r'[\'"]hit_gold_rule_indices[\'"]\s*:\s*\[(.*?)\]', json_str)
                    if len(hits) == len(candidate_rules) and len(indices_raw) >= len(candidate_rules):
                        for i in range(len(candidate_rules)):
                            hit_val = hits[i].upper()
                            idx_str = indices_raw[i].strip()
                            indices_list = [int(num) for num in re.findall(r'\d+', idx_str)]
                            match_results.append({
                                "candidate": candidate_rules[i],
                                "hit": hit_val,
                                "hit_gold_rule_indices": indices_list
                            })

    # If extraction failed, count everything as un-matched
    if not match_results or len(match_results) != len(candidate_rules):
        match_results = [{"candidate": cand, "hit": "NO", "hit_gold_rule_indices": []} for cand in candidate_rules]
        
    doc["evaluation_result"] = {
        "candidate_rules": candidate_rules,
        "match_results": match_results,
        "raw_response": resp_text
    }

    # Calculate Metrics
    M = len(gold_rules)
    K = len(candidate_rules)
    
    hit_gold_rule_indices = set()
    hallucination_count = 0
    
    for res in match_results:
        hit_val = res.get("hit", "NO").upper()
        indices = res.get("hit_gold_rule_indices", [])
        
        # u_k logic: 1 if it matched NO gold rules, else 0
        if hit_val == "YES" and len(indices) > 0:
            for idx in indices:
                if 0 <= idx < M:
                    hit_gold_rule_indices.add(idx)
        else:
            hallucination_count += 1
            
    # H: number of gold rules matched at least once
    H = len(hit_gold_rule_indices)
    
    rubric_recall = H / M if M > 0 else 0.0
    hallucination_rate = hallucination_count / K if K > 0 else 0.0
    precision = 1.0 - hallucination_rate
    
    if (rubric_recall + precision) > 0:
        structural_f1 = (2 * rubric_recall * precision) / (rubric_recall + precision)
    else:
        structural_f1 = 0.0
        
    doc["metrics"] = {
        "RubricRecall": rubric_recall,
        "HallucinationRate": hallucination_rate,
        "StructuralF1": structural_f1
    }

    return {
        "RubricRecall": rubric_recall,
        "HallucinationRate": hallucination_rate,
        "StructuralF1": structural_f1
    }

def aggregate_rubric_recall(results):
    valid = [x for x in results if isinstance(x, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0

def aggregate_hallucination_rate(results):
    valid = [x for x in results if isinstance(x, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0

def aggregate_structural_f1(results):
    valid = [x for x in results if isinstance(x, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0
