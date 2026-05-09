import torch, numpy as np, time, cv2, os, json, re
from transformers import TextIteratorStreamer
from typing import Generator
from threading import Thread
from PIL import Image

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    import llava.conversation as conversation_lib
except ImportError:
    print("Warning: LLaVA not installed.")


# Model name shown in source attribution
MODEL_NAME = "llava-fastvithd_0.5b"

# Scalar math: e.g. '45 * 12', '100/4+3'
MATH_PATTERN = re.compile(
    r'\d+\s*[\+\-\*\/\^\%]\s*\d+(?:\s*=\s*[\?]?)?',
    re.IGNORECASE
)

# Matrix problems: bracket notation, C=AB, A=, B=
MATRIX_PATTERN = re.compile(
    r'\[|matrix\s*multiply|C\s*=\s*AB|A\s*=|B\s*=',
    re.IGNORECASE
)

# Web search triggers: news, current, price, who is, what is (factual but time-sensitive)
WEBSEARCH_PATTERN = re.compile(
    r'news|search|latest|current|price|who is|what is|where is|when was|today|yesterday|now',
    re.IGNORECASE
)


def detect_math(text: str):
    """Return the scalar math expression string if found, else None."""
    match = MATH_PATTERN.search(text)
    if match:
        expr = match.group(0).strip().rstrip('=?').strip()
        if expr and len(expr) >= 3:
            return expr
    return None


def detect_matrix(text: str):
    """Return True if text describes a matrix multiplication problem."""
    return bool(MATRIX_PATTERN.search(text))


def detect_web_search(text: str):
    """Return True if the text suggests a need for live web data."""
    return bool(WEBSEARCH_PATTERN.search(text))


class EdgeAgent:
    def __init__(self, model_path="checkpoints/llava-fastvithd_0.5b_stage3", device=None, torch_dtype=None):
        # --- Universal Auto-Detection Logic ---
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        if torch_dtype is None:
            # CPU and older GPUs handle float32 best; CUDA/MPS shine with float16
            torch_dtype = torch.float32 if device == "cpu" else torch.float16

        print(f"\U0001F34E [IRIS OS] Universal Engine booting on: {device.upper()} ({torch_dtype})")
        
        self.device = device
        self.dtype = torch_dtype
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, model_base=None,
            model_name=get_model_name_from_path(model_path),
            device=device, torch_dtype=torch_dtype
        )
        
        # Ensure vision tower is on the correct device
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch_dtype)
        
        # Push model to device (builder handles some, but we ensure it here)
        self.model.to(device=device, dtype=torch_dtype)
        
        # Update model name for attribution
        global MODEL_NAME
        MODEL_NAME = "llava-fastvithd_1.5b" if "1.5b" in model_path.lower() else "llava-fastvithd_0.5b"

        self.warmup()
        self.warmup()

    def warmup(self):
        print("\U0001F525 Warming up inference engine...")
        print("\u2705 Engine Ready.")

    def generate_stream(self, image: np.ndarray, prompt: str, stop_event=None) -> Generator[str, None, None]:
        from tools import AVAILABLE_TOOLS

        STOP_TOK_A = '<|im' + '_end|>'
        STOP_TOK_B = '</s>'

        conv = conversation_lib.conv_templates["qwen_2"].copy()

        rule_instruction = (
            "You are a high-precision OCR and fact-checking agent.\n"
            "Examine the image and extract the EXACT question text written in it.\n"
            "You MUST return ONLY a JSON object with these two fields.\n"
            "{\n"
            "  \"text_in_image\": \"the literal question\",\n"
            "  \"answer\": \"your factual response\"\n"
            "}\n"
        )

        qs = DEFAULT_IMAGE_TOKEN + '\n' + rule_instruction + "What is the question and answer for this image?"
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        # Prefixing the assistant response to force JSON structure
        formatted_prompt = conv.get_prompt() + '{\n  "text_in_image": "'

        input_ids = tokenizer_image_token(
            formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = process_images([pil_img], self.image_processor, self.model.config)[0]
        # Explicitly cast to half (float16) for GPU inference
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[pil_img.size],
            streamer=streamer,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            if stop_event and stop_event.is_set():
                break
            generated_text += new_text
            if STOP_TOK_A in new_text or STOP_TOK_B in new_text:
                break

        generated_text = generated_text.replace(STOP_TOK_A, '').replace(STOP_TOK_B, '').strip()
        
        # Heuristic to rebuild valid JSON from the pre-filled prefix
        full_json_str = '{\n  "text_in_image": "' + generated_text
        if not full_json_str.strip().endswith('}'):
            # Close quotes and braces if missing
            if '"' in generated_text and 'answer' not in generated_text:
                 # It likely failed to emit the second field, try to add it
                 full_json_str += '",\n  "answer": "Scanning failed."'
            full_json_str += '\n}'

        text_in_image = ""
        llm_answer = ""

        try:
            payload = json.loads(full_json_str)
            text_in_image = payload.get("text_in_image", "").strip()
            llm_answer    = payload.get("answer", "").strip()
        except json.JSONDecodeError:
            m1 = re.search(r'"text_in_image":\s*"(.*?)"', full_json_str, re.DOTALL)
            if m1: text_in_image = m1.group(1).strip()
            m2 = re.search(r'"answer":\s*"(.*?)"', full_json_str, re.DOTALL)
            if m2: llm_answer = m2.group(1).strip()

        # --- Python-Side Agentic Router ---
        # Priority: matrix > scalar math > web search > direct LLM answer
        if detect_matrix(text_in_image) and "matrix" in AVAILABLE_TOOLS:
            matrix_input = full_json_str + ' ' + llm_answer
            result = AVAILABLE_TOOLS["matrix"](matrix_input)
            yield f"\U0001f4ac Answer:\n{result}"
            yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) + Tool (matrix/numpy)"
        elif detect_math(text_in_image) and "calculator" in AVAILABLE_TOOLS:
            math_expr = detect_math(text_in_image)
            result = AVAILABLE_TOOLS["calculator"](math_expr)
            yield f"\U0001f4ac Answer: {result}"
            yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) + Tool (calculator)"
        elif detect_web_search(text_in_image) and "web_search" in AVAILABLE_TOOLS:
            yield f"[\U0001f310 Searching Web...] {text_in_image}"
            result = AVAILABLE_TOOLS["web_search"](text_in_image)
            yield f"\n\U0001f4ac Answer (Live):\n{result}"
            yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) + Tool (web_search/tavily_ai)"
        else:
            yield f"\U0001f4ac Answer: {llm_answer}"
            yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME})"