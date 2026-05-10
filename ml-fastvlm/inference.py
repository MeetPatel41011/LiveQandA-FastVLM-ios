import torch, numpy as np, time, cv2, os, json, re
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from typing import Generator
from threading import Thread, Event
from PIL import Image

class StopOnEventCriteria(StoppingCriteria):
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

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

        print(f"\U0001F9E0 [IRIS OS] Booting Brain Engine (Qwen2.5-1.5B-Instruct)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.brain_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        self.brain_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", 
            torch_dtype=torch_dtype, 
            device_map={"": device}
        )

        self.warmup()
        self.warmup()

    def warmup(self):
        print("\U0001F525 Warming up inference engine...")
        print("\u2705 Engine Ready.")

    def generate_stream(self, image: np.ndarray, prompt: str, stop_event=None) -> Generator[str, None, None]:
        from tools import AVAILABLE_TOOLS

        if stop_event is None:
            stop_event = Event()

        STOP_TOK_A = '<|im' + '_end|>'
        STOP_TOK_B = '</s>'

        # ==========================================
        # PHASE 1: VLM Text Extraction (FastVLM)
        # ==========================================
        yield f"\n[\U0001f4f7 OCR Scanner] Reading image...\n"
        
        conv = conversation_lib.conv_templates["qwen_2"].copy()
        vlm_prompt = "What is the exact text written in this image? Just transcribe it."
        qs = DEFAULT_IMAGE_TOKEN + '\n' + vlm_prompt
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = process_images([pil_img], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        vlm_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[pil_img.size],
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEventCriteria(stop_event)])
        )

        with torch.no_grad():
            vlm_output_ids = self.model.generate(**vlm_kwargs)
        
        if stop_event.is_set():
            return
            
        extracted_text = self.tokenizer.decode(vlm_output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Remove common conversational prefixes the VLM might add
        extracted_text = extracted_text.replace("The text in the image is:", "").replace("The text says:", "").strip()
        
        if len(extracted_text) < 2:
            yield f"\n\U0001f4ac Direct Answer: No readable text detected. Please hold the paper clearly in the frame."
            return

        # ==========================================
        # PHASE 2: Brain Reasoning (Qwen2.5-1.5B)
        # ==========================================
        yield f"[\U0001F9E0 Brain] Analyzing text: '{extracted_text}'...\n"
        
        rule_instruction = (
            "You are an expert, fast-thinking AI Orchestrator.\n"
            f"Extracted Text from User's Image: '{extracted_text}'\n"
            "Rules:\n"
            "1. Answer the question presented in the extracted text.\n"
            "2. Speed & Reasoning: Keep your reasoning ruthlessly short (max 15 words).\n"
            "3. Quality: If a term is ambiguous (e.g., 'Gemini', 'Apple'), be comprehensive and mention the major contexts.\n"
            "4. Tool Selection: You have access to tools. Choose ONE tool if needed:\n"
            "   - 'calculator': Use for ALL math equations (e.g., '12 * 45', '100 / 4').\n"
            "   - 'matrix': Use for matrix multiplication.\n"
            "   - 'web_search': ONLY use for live, real-time data (weather, news) or highly specific facts. NEVER search for generic phrases.\n"
            "   - 'none': Use your own knowledge for general facts, history, science, coding.\n"
            "Output your reasoning first, then a JSON object at the exact end.\n"
            "JSON Format: {\"tool_needed\": \"none\"|\"web_search\"|\"calculator\"|\"matrix\", \"tool_query\": \"<exact equation or search query>\", \"answer\": \"<your final direct answer>\"}\n"
        )

        messages = [
            {"role": "system", "content": "You are a helpful and fast analytical AI orchestrator."},
            {"role": "user", "content": rule_instruction}
        ]
        text_prompt = self.brain_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        brain_inputs = self.brain_tokenizer([text_prompt], return_tensors="pt").to(self.device)
        
        streamer = TextIteratorStreamer(self.brain_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **brain_inputs,
            streamer=streamer,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.brain_tokenizer.pad_token_id,
            eos_token_id=self.brain_tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnEventCriteria(stop_event)])
        )

        thread = Thread(target=self.brain_model.generate, kwargs=generation_kwargs)
        thread.start()

        full_raw_text = ""
        try:
            for new_text in streamer:
                if stop_event.is_set():
                    break
                full_raw_text += new_text
                # Stream the "Reasoning" part to the user live
                if "{" not in full_raw_text:
                    yield new_text
                if STOP_TOK_A in new_text or STOP_TOK_B in new_text:
                    break
        except GeneratorExit:
            stop_event.set()
            raise

        full_raw_text = full_raw_text.replace(STOP_TOK_A, '').replace(STOP_TOK_B, '').strip()
        
        # --- Option 1: Log the raw reasoning for analysis ---
        try:
            reasoning_part = full_raw_text.split("{")[0].strip()
            with open("reasoning_logs.txt", "a", encoding="utf-8") as f:
                f.write(f"--- LOG START ---\n{time.strftime('%Y-%m-%d %H:%M:%S')}\nEXTRACTED TEXT: {extracted_text}\nREASONING:\n{reasoning_part}\nRAW_OUTPUT:\n{full_raw_text}\n--- LOG END ---\n\n")
        except Exception as e:
            print(f"Failed to log reasoning: {e}")

        # Extract JSON from the end of the chain-of-thought
        tool_needed = "none"
        tool_query = ""
        llm_answer = ""
        
        try:
            json_start = full_raw_text.rfind("{")
            if json_start != -1:
                json_str = full_raw_text[json_start:]
                if not json_str.endswith("}"): json_str += '"}'
                payload = json.loads(json_str)
                tool_needed = payload.get("tool_needed", "none").lower()
                tool_query = payload.get("tool_query", "")
                llm_answer = payload.get("answer", "")
        except:
            # Fallback regex if JSON is malformed
            m = re.search(r'"answer":\s*"([^"]+)"', full_raw_text)
            if m: llm_answer = m.group(1)
            
            # Fallback tool detection from raw text
            if "tool_needed\": \"calculator" in full_raw_text.lower(): tool_needed = "calculator"
            elif "tool_needed\": \"matrix" in full_raw_text.lower(): tool_needed = "matrix"
            elif "tool_needed\": \"web_search" in full_raw_text.lower(): tool_needed = "web_search"
            
            m_query = re.search(r'"tool_query":\s*"([^"]+)"', full_raw_text)
            if m_query: tool_query = m_query.group(1)

        # --- Agentic Decision Logic ---
        query_to_use = tool_query if tool_query else (extracted_text if extracted_text else llm_answer)
        
        if tool_needed == "calculator" and "calculator" in AVAILABLE_TOOLS:
            yield f"\n[\U0001F522 Decided: Math Required] -> '{query_to_use}'"
            result = AVAILABLE_TOOLS["calculator"](query_to_use)
            yield f"\n\U0001f4ac Final Answer:\n{result}"
            yield f"\n\U0001f4cc Source: Brain Engine ({MODEL_NAME}) + Python Calculator Tool"
            
        elif tool_needed == "matrix" and "matrix" in AVAILABLE_TOOLS:
            yield f"\n[\U0001F4BE Decided: Matrix Math Required] -> '{query_to_use}'"
            result = AVAILABLE_TOOLS["matrix"](query_to_use)
            yield f"\n\U0001f4ac Final Answer:\n{result}"
            yield f"\n\U0001f4cc Source: Brain Engine ({MODEL_NAME}) + NumPy Matrix Tool"
            
        elif tool_needed == "web_search" and "web_search" in AVAILABLE_TOOLS:
            bad_queries = ["what is the question", "what is the question?", "what is the question and answer for this image?"]
            if not query_to_use or query_to_use.lower().strip() in bad_queries:
                # If it's a bad query, just fallback to direct answer
                yield f"\n\U0001f4ac Direct Answer: {llm_answer if llm_answer else full_raw_text.split('{')[0].strip()}"
                yield f"\n\U0001f4cc Source: Brain Engine ({MODEL_NAME}) Internal Knowledge"
            else:
                yield f"\n[\U0001f310 Decided: Search Required] -> '{query_to_use}'"
                result = AVAILABLE_TOOLS["web_search"](query_to_use)
                yield f"\n\U0001f4ac Final Answer:\n{result}"
                yield f"\n\U0001f4cc Source: Brain Engine ({MODEL_NAME}) + Tavily AI"
                
        else: # "none" or tool not available
            if not llm_answer:
                # If it didn't give an answer in JSON, use the reasoning part
                llm_answer = full_raw_text.split("{")[0].strip()
            yield f"\n\U0001f4ac Direct Answer: {llm_answer}"
            yield f"\n\U0001f4cc Source: Brain Engine ({MODEL_NAME}) Internal Knowledge"