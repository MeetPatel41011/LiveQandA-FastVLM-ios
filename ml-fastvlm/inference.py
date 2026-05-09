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

        # Phase 3: Chain-of-Thought Agentic Instructions
        rule_instruction = (
            "You are an expert Vision AI. Your primary job is to read the image and provide the correct direct answer.\n"
            "Rules:\n"
            "1. Read the text or question in the image. If there is NO readable text or question (e.g. just a person's face), set answer to 'No question detected. Please show the text clearly.' and stop.\n"
            "2. Think step-by-step and answer the question using your own knowledge (math, history, science, coding, general facts).\n"
            "3. ONLY set needs_search to true if the question requires live, real-time data (like today's weather, current prices, recent news) OR if you absolutely cannot answer it without the internet.\n"
            "4. NEVER search for generic phrases like 'what is the question'.\n\n"
            "Output your reasoning first, then a JSON object at the exact end.\n"
            "JSON Format: {\"extracted_question\": \"<text read from image>\", \"needs_search\": false, \"search_query\": \"<only if search needed>\", \"answer\": \"<your final direct answer>\"}\n"
        )

        qs = DEFAULT_IMAGE_TOKEN + '\n' + rule_instruction + "Analyze this image and answer the question."
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_tensor = process_images([pil_img], self.image_processor, self.model.config)[0]
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        if stop_event is None:
            stop_event = Event()

        generation_kwargs = dict(
            inputs=input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[pil_img.size],
            streamer=streamer,
            max_new_tokens=256, # Increased for Chain-of-Thought
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            stopping_criteria=StoppingCriteriaList([StopOnEventCriteria(stop_event)])
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
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
        
        # Extract JSON from the end of the chain-of-thought
        extracted_question = ""
        needs_search = False
        llm_answer = ""
        search_query = ""
        
        try:
            json_start = full_raw_text.rfind("{")
            if json_start != -1:
                json_str = full_raw_text[json_start:]
                if not json_str.endswith("}"): json_str += '"}'
                payload = json.loads(json_str)
                extracted_question = payload.get("extracted_question", "")
                needs_search = payload.get("needs_search", False)
                search_query = payload.get("search_query", "")
                llm_answer = payload.get("answer", "")
        except:
            # Fallback regex if JSON is malformed
            m = re.search(r'"answer":\s*"([^"]+)"', full_raw_text)
            if m: llm_answer = m.group(1)
            needs_search = "needs_search\": true" in full_raw_text.lower()

        # --- Agentic Decision Logic ---
        if needs_search and "web_search" in AVAILABLE_TOOLS:
            final_query = search_query if search_query else (extracted_question if extracted_question else llm_answer)
            # Guard against the LLM searching for generic/meta prompts
            bad_queries = ["what is the question", "what is the question?", "what is the question and answer for this image?"]
            if not final_query or final_query.lower().strip() in bad_queries:
                # If it's a bad query, just fallback to direct answer
                yield f"\n\U0001f4ac Direct Answer: {llm_answer if llm_answer else full_raw_text.split('{')[0].strip()}"
                yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) Internal Knowledge"
            else:
                yield f"\n[\U0001f310 Decided: Search Required] -> '{final_query}'"
                result = AVAILABLE_TOOLS["web_search"](final_query)
                yield f"\n\U0001f4ac Final Answer:\n{result}"
                yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) + Tavily AI"
        else:
            if not llm_answer:
                # If it didn't give an answer in JSON, use the reasoning part
                llm_answer = full_raw_text.split("{")[0].strip()
            yield f"\n\U0001f4ac Direct Answer: {llm_answer}"
            yield f"\n\U0001f4cc Source: LLM ({MODEL_NAME}) Internal Knowledge"