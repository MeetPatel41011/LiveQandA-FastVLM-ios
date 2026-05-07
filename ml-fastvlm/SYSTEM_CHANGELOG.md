# Iris OS on FastVLM — System Architecture & Workflow

> This repository is **two layers stacked on top of each other**:
>
> 1. **Apple `ml-fastvlm`** — the upstream research code (the LLaVA-derived
>    Vision Language Model with the FastViTHD encoder + Qwen2 LLM, plus the
>    Swift/MLX iOS/macOS demo app and the CoreML/MLX export tooling).
> 2. **"Iris OS"** — a custom agentic runtime built around the VLM. It adds a
>    live camera ("Sentinel"), JSON-guided decoding, and a deterministic tool
>    router (calculator / matrix / web search). All Iris files live next to the
>    Apple code in this same folder.
>
> The two layers share one Python process. Iris OS imports `llava.*` and calls
> the VLM directly via PyTorch — it does **not** use ONNX, MLX, or CoreML at
> runtime, even though export scripts for those formats exist in this repo.

---

## 1. High-Level Flow (for Junior SWEs)

Imagine the system as a person who looks at a piece of paper, reads what's on
it, and answers the question — out loud — with the right "skill" for the job.

### 1.1 The cast (one sentence each)

| Module | Real-world analogy | What it actually does |
| :--- | :--- | :--- |
| `main.py` (`IrisSystem`) | The brain stem | Boots everything and runs the forever-loop. |
| `sentinel.py` (`Sentinel`) | The eyes | Background camera thread. Scores every frame for sharpness + motion. |
| `inference.py` (`EdgeAgent`) | The cortex | Loads the VLM, prompts it for JSON, then routes to a tool. |
| `tools.py` (`AVAILABLE_TOOLS`) | The hands | Calculator, matrix multiplier, multi-source web search. |
| `llava/` package | The neurons | Apple/LLaVA model code — vision tower + projector + Qwen2 LLM. |
| `checkpoints/llava-fastvithd_0.5b_stage3/` | Long-term memory | The 1.5 GB pretrained weights actually loaded into the model. |

### 1.2 What happens when you run `python main.py`

1. **Boot.** `IrisSystem.__init__` prints a banner, then *lazily* imports
   `EdgeAgent`. The lazy import is intentional: PyTorch + the VLM weights take
   15–30 s to load and we want the boot banner on screen before that stall.
2. **Eyes open.** `Sentinel.start()` spawns a daemon thread. That thread
   continuously reads frames from `cv2.VideoCapture(0)`, computes two cheap
   metrics on each frame (Laplacian variance for **sharpness**, mean absolute
   pixel diff for **motion**), and pushes them into a fixed-size ring buffer
   (`deque(maxlen=15)`). It also draws an OpenCV HUD window so the user sees
   live "Motion / Sharpness / STABLE" overlays.
3. **Wait for a steady picture.** The main loop (`IrisSystem.run`) wakes up
   every 500 ms. It triggers an inference only when **both** are true:
   - `has_new_content(baseline_motion=8.0)` — something *did* move recently
     (so the user just held up a new piece of paper).
   - `is_camera_stable(history=8, max_motion=4.5)` — the last 8 frames are
     calm again (so the paper isn't shaking anymore).
   This double-gate is what stops the system from spamming the LLM dozens of
   times while the paper is being moved into position.
4. **Pick the best frame.** `get_best_frame()` walks the ring buffer and
   returns the *sharpest* frame from the last ~0.5 s — not necessarily the
   most recent one. This is the "magic" that handles the moment a phone
   refocuses or a user briefly shakes their hand.
5. **Ask the brain.** `EdgeAgent.generate_stream(frame, query)` is called.
   Internally it: (a) runs the image through the FastViTHD vision encoder,
   (b) prepends a "you MUST output JSON with two fields" instruction, and (c)
   pre-fills the model's reply with `{"text_in_image": "` so the model is
   physically forced to start producing JSON. This is **guided decoding** —
   we steer a small 0.5 B model into structured output instead of trusting it
   to follow English instructions.
6. **Route.** Once the JSON is parsed (`text_in_image`, `answer`), Python —
   not the LLM — decides what to do next:
   - Looks like matrix algebra (`[`, `C=AB`, `A=`)? → `tools.py: matrix`
   - Looks like scalar arithmetic (`45*12`, `100/4+3`)? → `tools.py: calculator`
   - Looks time-sensitive (`who is`, `news`, `latest`, `price`)? →
     `tools.py: web_search`
   - Otherwise → just print the LLM's own `answer`.
7. **Print, cool down, repeat.** Each yielded line gets streamed to the
   terminal with a `Source:` tag. The loop then enters a 5 s cooldown so the
   user can read the answer before the next cycle starts.

### 1.3 Single-image mode

`predict.py` is the same brain (`EdgeAgent`) without the eyes. It loads one
image from disk, calls `generate_stream` once, and prints the result. Useful
for testing without a camera. `test_vision.py` is essentially the same thing
with a hard-coded prompt.

### 1.4 Where Apple's iOS app fits

The `app/` folder is an **independent** Apple-Silicon-only path. It is a
SwiftUI app that uses the **MLX** runtime (not PyTorch) and a **CoreML**
vision encoder produced by `model_export/export_vision_encoder.py`. Iris OS
does not use this app and the app does not use Iris OS — they are two
parallel ways to consume the same FastVLM weights.

---

## 2. Low-Level Flow (for Senior SWEs)

### 2.1 Process topology

```
                 ┌────────────────────────── main thread ──────────────────────────┐
                 │                                                                  │
   camera thread │  IrisSystem.run()                                                │
   (daemon)      │   ├─ poll every 500 ms                                           │
   ─────────────►│   ├─ gate: has_new_content() ∧ is_camera_stable()                │
   reads cv2,    │   ├─ frame = Sentinel.get_best_frame()  ◄── shared ring buffer   │
   appends to    │   └─ EdgeAgent.generate_stream(frame, prompt)                    │
   deque under   │            │                                                     │
   self.lock     │            ▼                                                     │
                 │     ┌──────────────────┐                                         │
                 │     │ generation thread│  model.generate(...)                    │
                 │     │ (Thread target)  │  → TextIteratorStreamer                 │
                 │     └──────────────────┘                                         │
                 │            │                                                     │
                 │            ▼ stream yields tokens                                │
                 │     parse JSON → router → tools.py                               │
                 └──────────────────────────────────────────────────────────────────┘
```

Three threads run concurrently: (1) the camera daemon in `Sentinel`, (2) the
`Thread(target=self.model.generate, ...)` spawned inside
`EdgeAgent.generate_stream`, and (3) the main Python thread that consumes the
streamer. Synchronisation between (1) and (3) is a single `threading.Lock`
guarding the deque. (2) is one-shot per inference and joined implicitly via
the streamer's StopIteration.

### 2.2 Tier 1 — Perception (`sentinel.py`)

- **Buffer**: `collections.deque(maxlen=15)` of `FrameData(image, timestamp,
  sharpness_score, motion_score)`. At ~30 fps this is ~0.5 s of memory.
- **Sharpness**: `cv2.Laplacian(gray_256, CV_64F).var()`. The image is
  resized to 256×256 first because Laplacian variance is scale-sensitive but
  rank-stable across the buffer, and we only need a *relative* score. Cost
  ≈ 0.5–1 ms on commodity CPUs.
- **Motion**: mean absolute difference between consecutive 64×64 grayscale
  frames. This is intentionally crude — we only care about a coarse "is the
  scene stationary" gate, not optical flow.
- **Stability gate**: `is_camera_stable(history=8, max_motion=4.5)` — average
  motion over the last 8 frames must be below 4.5. The 4.5 (vs. an earlier
  3.0) is calibrated for handheld phones whose tremor is non-zero even when
  the user thinks they are still.
- **Best-frame selection**: `max(buffer, key=lambda x: x.sharpness_score)`,
  with a `blur_threshold=100.0` warning. We deliberately do **not** drop the
  frame when below threshold; we log and pass it through, because at the
  application layer a blurry answer is still better than no answer.
- **HUD side-channel**: the OpenCV `imshow` window draws on a `frame.copy()`,
  so the buffered image stays untouched. The model never sees the overlays.

### 2.3 Tier 2 — Cognition (`inference.py` → `llava/`)

#### 2.3.1 Model load (`EdgeAgent.__init__`)

- **Hardware auto-pick**: `cuda` → `mps` → `cpu`, with `torch.float16` on
  CUDA/MPS and `torch.float32` on CPU. This is the "Universal Engine" line in
  the README.
- **Loader**: `llava.model.builder.load_pretrained_model(...)` with
  `model_name = "llava-fastvithd_0.5b_stage3"`. Because the name contains
  `llava` and not `mpt`/`mistral`/`dclm`, the dispatch in `builder.py` lands
  on `LlavaQwen2ForCausalLM.from_pretrained(...)`.
- **Vision tower**: `MobileCLIPVisionTower("mobileclip_l_1024", ...)` from
  `llava/model/multimodal_encoder/mobileclip_encoder.py`. The trailing
  `_1024` is the input resolution (`int(name.split("_")[-1])`). Internally
  this is `MCi(...)` — a MobileCLIP / FastViT-Hybrid (FastViTHD) backbone in
  `mobileclip/mci.py` whose `reparameterize()` mechanism collapses
  multi-branch train-time blocks (`reparam_conv`, `lkb_reparam`) into single
  conv layers for inference (RepVGG-style). Output is a 4-D
  `B×C×H×W` tensor that `feature_select` reshapes to `B×(H·W)×C`.
- **Projector**: `mlp2x_gelu` MLP from `multimodal_projector/builder.py`,
  mapping `mm_hidden_size=3072 → hidden_size=896` (Qwen2-0.5B's hidden dim).
- **LLM**: `Qwen2ForCausalLM` with config
  `hidden_size=896`, `num_hidden_layers=24`, `num_attention_heads=14`,
  `num_key_value_heads=2`, `vocab_size=151936`, `tie_word_embeddings=true`,
  `bos=151643`, `eos=151645`. (See `checkpoints/llava-fastvithd_0.5b_stage3/
  config.json`.)
- **Glue**: `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal`
  (in `llava_arch.py`) is the splice point. It:
  1. Encodes the image: `image_features = mm_projector(vision_tower(images))`.
  2. Walks `input_ids`, finds every `IMAGE_TOKEN_INDEX = -200` placeholder,
     and substitutes the corresponding image embedding into `inputs_embeds`.
  3. Returns the merged `inputs_embeds` so the LLM never receives a literal
     `<image>` token — it only ever sees a continuous embedding stream.
- **Two warmup calls**: cosmetic; the function body is just two `print`s.
  Real warmup happens implicitly on first `model.generate`.

#### 2.3.2 Prompt construction & guided decoding

```
conv = conversation_lib.conv_templates["qwen_2"].copy()        # qwen2 chatml-ish template
qs   = DEFAULT_IMAGE_TOKEN + "\n" + rule_instruction + prompt  # rule = "MUST output JSON {...}"
conv.append_message(USER,  qs)
conv.append_message(ASST,  None)
formatted = conv.get_prompt() + '{"text_in_image": "'          # ← prefix-fill the answer
```

The trailing `{"text_in_image": "` is the trick. We commit the model to JSON
*before* the first sampled token. Combined with `do_sample=False` (greedy)
and `repetition_penalty=1.2`, this turns a 0.5 B model that would otherwise
hallucinate paragraphs into a fairly reliable JSON extractor.

`tokenizer_image_token` then splits the prompt at every `<image>` and
re-inserts the sentinel id `IMAGE_TOKEN_INDEX = -200`, which is what the
multimodal splicer above looks for. `process_images` (with
`image_aspect_ratio="pad"` from config) pads the BGR frame to a square,
runs it through `CLIPImageProcessor` with `mean=[0,0,0]`, `std=[1,1,1]`,
and produces the `1×3×1024×1024` tensor expected by FastViTHD.

#### 2.3.3 Streaming + JSON repair

- `TextIteratorStreamer(skip_prompt=True, skip_special_tokens=True)` is
  consumed in the *main* thread; `model.generate` runs in a background
  `Thread`. We early-exit on `<|im_end|>` / `</s>` or on `stop_event`.
- After the stream ends we re-prepend the prefilled `{"text_in_image": "`
  and apply a heuristic close (`'"}'`) if the model didn't emit a closing
  brace. We then try `json.loads` and fall back to two regex extractions for
  `text_in_image` and `answer`. This is intentionally tolerant — small
  models often miss the final quote/brace.

#### 2.3.4 Deterministic router

Three regexes (`MATH_PATTERN`, `MATRIX_PATTERN`, `WEBSEARCH_PATTERN`) are
matched against `text_in_image` (the *question*, not the model's answer).
Priority order: **matrix > scalar math > web search > raw LLM answer**.
This is the "Python-Native Routing" design choice: rather than letting a
0.5 B model decide which tool to call (which fails at this scale), tool
selection is a deterministic function of regexes on the extracted question.
The LLM's job is reduced to a single, well-defined sub-task: OCR-style
extraction into a JSON schema.

### 2.4 Tier 3 — Execution (`tools.py`)

| Tool | Implementation | Notes |
| :--- | :--- | :--- |
| `calculator` | `eval(...)` on a regex-filtered numeric expression | Whitelist of `[\d\s+\-*/.()]`. Not sandboxed — adequate for this single-user demo, **not** safe for untrusted input. |
| `matrix` | NumPy `A @ B` over complex numbers | Custom `_parse_complex` handles `i` notation (`1+i`, `4-i`). It extracts all numbers from the JSON+answer blob, splits them in half, infers √n size, reshapes into two square matrices, multiplies, and pretty-prints. Brittle by design; assumes well-formed input. |
| `web_search` | Wikipedia REST → Tavily → DuckDuckGo cascade | (1) If the question starts with "what is / who is / define", hit `en.wikipedia.org/api/rest_v1/page/summary/{slug}` first (free, high quality, no key). (2) Else / on miss, use `TavilyClient` if `TAVILY_API_KEY` is set in `.env`. (3) Last resort, `duckduckgo_search.DDGS.text(...)`. The `dotenv` load is per-call, which is wasteful but harmless. |

`AVAILABLE_TOOLS` is a plain dict at module bottom. Adding a new tool is one
line in `tools.py` plus one detector + branch in `inference.py`.

### 2.5 The Apple iOS / macOS path (parallel universe)

This is **not used by Iris OS** but ships in the same repo for completeness.

- `app/FastVLM/FastVLM.swift` — a hand-ported MLX implementation of the
  FastVLM architecture (Qwen2 attention with multimodal RoPE + a Swift
  vision tower that loads `fastvithd.mlpackage`).
- `app/FastVLM App/FastVLMModel.swift` — a `@MainActor` view-model. Loads
  the model via `VLMModelFactory.shared.loadContainer(...)`, calls
  `MLXLMCommon.generate(...)`, streams tokens to the SwiftUI view, reports
  TTFT in ms.
- `app/get_pretrained_mlx_model.sh` — downloads a pre-converted MLX bundle
  and drops it into `app/FastVLM/model/` (which is gitignored).
- `model_export/export_vision_encoder.py` — converts the PyTorch vision
  tower to **CoreML** (`.mlpackage`) via `torch.jit.trace` →
  `coremltools.convert`. It also patches `config.json`,
  `tokenizer_config.json`, `preprocessor_config.json`, and writes a fresh
  `processor_config.json` so that `mlx-vlm` can auto-load the result.
- `model_export/fastvlm_mlx-vlm.patch` — applied on top of a pinned
  `mlx-vlm` commit to teach it about FastVLM.

### 2.6 The (unused) ONNX path

`prepare_models.py`, `export_models.py`, and `check_onnx.py` are an earlier
attempt to ship the model as ONNX (vision encoder via `torch.onnx.export`,
LLM via `optimum-cli export onnx --task text-generation-with-past`, then
`onnxruntime.quantization.quantize_dynamic` → INT8). They produce
`models/fastvithd.onnx` (~500 MB) and `models/qwen2_0.5b_int8.onnx` (~496
MB). **Nothing in the runtime actually loads these files.** A grep for the
filenames returns only the three scripts that *write* them. They are
artefacts of a planned-but-not-finished ONNX inference path; today they are
just dead weight on disk.

### 2.7 Configuration surface

- `.env` — `TAVILY_API_KEY=...` (required for the Tavily branch of
  `web_search`; the system still works without it via Wikipedia + DDG).
- `pyproject.toml` — pins `torch==2.6.0`, `transformers==4.48.3`,
  `numpy==1.26.4` (NumPy 2.x breaks LLaVA tokenization on Windows — see
  README warning), `coremltools==8.2`, etc.
- Hard-coded constants worth knowing:
  - `EdgeAgent(model_path="checkpoints/llava-fastvithd_0.5b_stage3")` —
    relative path; you must run from `ml-fastvlm/`.
  - `max_new_tokens=100`, `do_sample=False`, `repetition_penalty=1.2`.
  - Cooldown 5 s, motion baseline 8.0, stability max-motion 4.5,
    sharpness blur threshold 100.0.

---

## 3. Connecting the Two Flows (one-glance summary)

```
camera ──► Sentinel(buffer,sharpness,motion)
              │   (junior: "the eyes wait for a steady, sharp picture")
              ▼   (senior: deque + Laplacian + frame-diff under threading.Lock)
       IrisSystem.run() gate
              │   (junior: "is something new AND is it stable?")
              ▼   (senior: has_new_content ∧ is_camera_stable, then 5 s cooldown)
       EdgeAgent.generate_stream(frame, prompt)
              │   (junior: "give me JSON, not a chat reply")
              ▼   (senior: prefix-filled prompt + greedy decode + streamer thread)
       LlavaQwen2ForCausalLM
              │   (junior: "the brain looks at the picture and writes JSON")
              ▼   (senior: FastViTHD → mlp2x_gelu projector → -200 splice → Qwen2)
       JSON {"text_in_image", "answer"}
              │   (junior: "we now know the question on paper")
              ▼   (senior: tolerant json.loads with regex fallback)
       Python router (regex on text_in_image)
              │   (junior: "math? matrix? facts? or just answer it?")
              ▼   (senior: matrix > calculator > web_search > raw LLM)
       tools.py executes  ──►  yield blocks to terminal with Source: tag
```

Same pipeline. Two reading levels.

---

## 4. File-by-File Map (what lives where, what it's worth)

### 4.1 Active runtime (Iris OS)

| File | Role | Status |
| :--- | :--- | :--- |
| `main.py` | Entry point, event loop | Active |
| `sentinel.py` | Camera + ring buffer + HUD | Active |
| `inference.py` | `EdgeAgent`, prompt, router | Active |
| `tools.py` | Calculator / matrix / web_search | Active |
| `predict.py` | Single-image CLI | Active |
| `test_vision.py` | Standalone end-to-end smoke test | Active (developer-only) |
| `.env` | `TAVILY_API_KEY` | Active (gitignored) |

### 4.2 Apple upstream code

| Path | Role | Status |
| :--- | :--- | :--- |
| `llava/` | LLaVA library (model + conv templates + utils) | **Required**, imported by `inference.py` and `predict.py` |
| `llava/model/multimodal_encoder/mobileclip/` | FastViTHD / MobileCLIP backbone | **Required** at runtime |
| `app/` | SwiftUI + MLX iOS/macOS demo app | Optional (only for Apple-device deployment) |
| `model_export/` | CoreML / mlx-vlm export | Optional (only for `app/`) |
| `docs/` | GIFs and PNGs for the README | Documentation only |
| `pyproject.toml`, `LICENSE`, `LICENSE_MODEL`, `ACKNOWLEDGEMENTS`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` | Package + legal | Required |
| `get_models.sh` | Bash one-shot to download all 6 checkpoints | Useful (alternative to `prepare_models.py` step 1) |

### 4.3 Model artefacts on disk

| Path | Size | Used at runtime? |
| :--- | :--- | :--- |
| `checkpoints/llava-fastvithd_0.5b_stage3/` | ~1.5 GB | **Yes** — loaded by `EdgeAgent` |
| `models/fastvithd.onnx` | ~500 MB | No (orphaned ONNX export) |
| `models/qwen2_0.5b_int8.onnx` | ~496 MB | No (orphaned ONNX export) |
| `models/qwen_export/` | ~1 GB | No (intermediate of ONNX export) |
| `checkpoints/__MACOSX/` | small | No — macOS unzip artefact |
| `checkpoints/fastvithd.pth` | 263 B | No — looks like a stub/LFS pointer |
| `llava-fastvithd_0.5b_stage3.zip` | 1.15 GB | No — already extracted |
| `fastvit_t8_reparam.tar` | 16 MB | No — not referenced by any code |

---

## 5. Recommended Cleanup (zero contribution today)

> Confirmed by `grep` against the entire `ml-fastvlm/` tree. None of the
> filenames below appear in any active Python code path
> (`main.py`, `sentinel.py`, `inference.py`, `tools.py`, `predict.py`,
> the `llava/` package, or any imported module).

### 5.1 Safe to delete (pure dead weight, ~3.2 GB combined)

1. **`llava-fastvithd_0.5b_stage3.zip`** (1.15 GB, repo root and
   `ml-fastvlm/` root — yes, two copies). The model is already extracted into
   `checkpoints/llava-fastvithd_0.5b_stage3/`. Once extracted, the zip is
   pure storage cost.
2. **`fastvit_t8_reparam.tar`** (16 MB). Not imported, not referenced by
   any path. Likely a leftover from an alternative encoder experiment.
3. **`models/fastvithd.onnx`** (500 MB), **`models/qwen2_0.5b_int8.onnx`**
   (496 MB), **`models/qwen_export/`** (~1 GB). Outputs of an ONNX path that
   the current runtime does not exercise. Safe to delete unless you intend to
   resurrect the ONNX inference plan from `TASKS_TO_COMPLETE_FIRST.md`.
4. **`checkpoints/__MACOSX/`**. macOS-specific metadata folder created
   automatically when `unzip` runs on Apple HFS volumes. Garbage on Windows.
5. **`checkpoints/fastvithd.pth`** (263 bytes). Far too small to be a real
   weight file; almost certainly a Git LFS pointer or stub. Verify before
   deleting; if it's a stub, remove it.
6. **`__pycache__/`** and **`llava.egg-info/`**. Auto-generated. Already
   covered by `.gitignore` but exist on disk. Safe to delete; will be
   recreated on next run / `pip install -e .`.

### 5.2 Candidates to delete *if* you abandon the ONNX plan

These three scripts only generate the ONNX artefacts above. If you delete
the `.onnx` files (5.1 #3) and have no plan to wire ONNX inference into
`inference.py`, these scripts also become dead code:

7. **`prepare_models.py`** — downloads + ONNX-exports + INT8-quantises.
8. **`export_models.py`** — duplicate of step 1 + 2 of `prepare_models.py`,
   minus the vision encoder (this overlap alone is a code smell).
9. **`check_onnx.py`** — sanity-checks the ONNX I/O shapes; only useful
   alongside the two scripts above.

> Recommendation: keep `prepare_models.py` *only* if you genuinely intend to
> finish the ONNX path described in `TASKS_TO_COMPLETE_FIRST.md` ("8-bit
> Optimization"). Otherwise delete all three.

### 5.3 Documentation / dev-only — keep but be aware

10. **`test_search_isolated.py`** — ad-hoc DuckDuckGo connectivity probe.
    Useful for debugging `web_search` regressions, but not part of the
    system. Move to a `dev/` folder or delete once stable.
11. **`math_test.jpg`, `math_test2.png`, `search_test.jpg`, `test.jpg`,
    `test.png`** — sample images for `predict.py`. Five is overkill;
    consolidate to two (one math, one general) or move into a `samples/`
    folder for clarity.

### 5.4 Security note (not cleanup, but flagged)

- **`.env` is committed to the repository tree** (visible above as 74 bytes
  containing a real-looking `TAVILY_API_KEY`). `.env` *is* listed in
  `.gitignore`, so it should not have been pushed to a remote — but if this
  workspace was ever shared, **rotate that key immediately**.

---

## 6. Known Gaps (mirrors `TASKS_TO_COMPLETE_FIRST.md`)

- **No speech input.** `is_user_speaking` flag in `Sentinel` is a stub.
- **Frame-by-frame, not video.** Each inference is independent; there is no
  temporal context across frames.
- **No 8-bit quantisation in the live path.** The ONNX/INT8 artefacts on
  disk are not consumed; the live model is fp16 (CUDA/MPS) or fp32 (CPU).
- **No conversation memory.** Each cycle is stateless; the cooldown is the
  only history the system has.