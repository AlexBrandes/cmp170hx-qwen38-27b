# Qwen3.8-27B on a CMP 170HX

This runs Qwen3.8-27B on a single NVIDIA CMP 170HX. It uses vLLM with a 4-bit
version of the model and serves an OpenAI-compatible API that accepts text,
images and video. It works on both the 64 GB and 40 GB versions of the card.

On a 40 GB card, one user gets around 200–380 tokens per second, depending on
what's being generated. The full numbers and quality checks are under
[Benchmarks](#benchmarks).

```bash
git clone https://github.com/AlexBrandes/cmp170hx-qwen38-27b
cd cmp170hx-qwen38-27b
./download.sh     # checkpoint and drafter at the pinned revisions (~16 GB)
./start.sh        # sets the power limit (sudo), starts the container, waits for /health
```

The API listens on `127.0.0.1:8000` as model `qwen3.8-27b`. `./stop.sh` stops
it. Settings go in `.env` (see `.env.example`); `GPU=1` picks another card.

## Requirements

- 1× NVIDIA CMP 170HX, 64 GB or 40 GB. `start.sh` reads the card's memory and
  sets the window: 262,144 tokens on 64 GB, 163,840 on 40 GB (`MAX_MODEL_LEN`
  overrides).
- Linux, NVIDIA driver 580 or newer, Docker with the NVIDIA Container Toolkit.
- ~27 GB free disk while `download.sh` runs (~16 GB afterwards: the checkpoint
  and the drafter), plus the engine image and compilation caches; ~26 GB of RAM
  while it builds the INT8 embedding table.
- `sudo` for `host-tune.sh` (or `HOST_TUNE=0`; see below).

## Benchmarks

Measured on a 40 GB CMP 170HX at 170 W, using the
[sparkDash](https://github.com/MiaAI-Lab/sparkDash) benchmark protocol at commit
`3e6d4ef`: greedy, thinking off, output forced to 400 tokens, sent to the engine
endpoint. Each cell is the **median of three runs, with the range in brackets**;
sparkDash itself makes one pass per level. Every run is in
[results/2026-11-11.json](results/2026-11-11.json).

The measured build is the published image below, built with `docker build --no-cache`
from the pinned sources, serving the checkpoint that `download.sh` assembled on the
same host.

**Decode, code**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 343.5 (343.4–351.7) | 343.5 (343.4–351.7) | 89 ms |
| 2 | 608.5 (599.3–609.7) | 315.0 (312.2–317.4) | 146 ms |
| 4 | 868.3 (839.0–921.9) | 254.0 (245.2–254.2) | 176 ms |
| 8 | 1,233.8 (1,196.0–1,235.6) | 172.6 (170.6–174.9) | 279 ms |

**Decode, json**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 325.7 (325.7–330.4) | 325.7 (325.7–330.4) | 67 ms |
| 2 | 552.9 (534.6–558.7) | 282.8 (275.8–285.9) | 151 ms |
| 4 | 880.4 (864.2–882.3) | 229.5 (227.7–233.1) | 178 ms |
| 8 | 1,188.0 (1,168.6–1,208.8) | 157.9 (155.2–160.8) | 294 ms |

**Decode, prose**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 196.8 (196.2–201.1) | 196.8 (196.2–201.1) | 64 ms |
| 2 | 331.1 (322.9–344.2) | 172.6 (168.7–177.4) | 91 ms |
| 4 | 549.7 (537.8–590.9) | 143.6 (141.1–155.1) | 135 ms |
| 8 | 741.8 (732.2–764.4) | 100.8 (97.5–105.3) | 210 ms |

**Decode, structured**

| Concurrent requests | Aggregate tok/s | Per request tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 1 | 378.6 (378.6–386.8) | 378.6 (378.6–386.8) | 61 ms |
| 2 | 705.3 (700.1–720.0) | 357.5 (354.9–364.4) | 105 ms |
| 4 | 1,147.5 (1,126.9–1,182.4) | 292.2 (289.7–299.5) | 132 ms |
| 8 | 1,498.2 (1,485.4–1,546.9) | 194.7 (194.3–200.6) | 194 ms |

**Prefill** (one cold request)

| Prompt | Tokens | Prefill tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,135 | 4,209 (4,191–4,261) | 1.07 s |
| 8k | 8,229 | 4,268 (4,262–4,282) | 2.01 s |
| 16k | 16,421 | 4,023 (4,001–4,035) | 4.17 s |
| 32k | 32,807 | 3,629 (3,628–3,640) | 9.13 s |
| 64k | 65,575 | 3,060 (3,059–3,064) | 21.51 s |
| 128k | 131,111 | 2,335 (2,335–2,337) | 56.22 s |

**Decode at depth** (our addition, not part of sparkDash: one request, the prose
prompt after filler, 400 tokens)

| Context | Tokens | Decode tok/s | Time to first token |
| ---: | ---: | ---: | ---: |
| 4k | 4,125 | 180.9 (174.2–184.9) | 1.06 s |
| 8k | 8,221 | 183.3 (178.1–185.0) | 2.01 s |
| 16k | 16,414 | 170.3 (168.4–177.0) | 4.16 s |
| 32k | 32,798 | 159.6 (158.1–165.2) | 9.11 s |
| 64k | 65,566 | 144.3 (141.0–148.3) | 21.48 s |
| 128k | 131,101 | 118.2 (116.5–124.4) | 56.22 s |

### Quality

Sent to the engine directly. Greedy with thinking off, except GPQA Diamond. All on
the 40 GB card, except the 262,144-token window: the same image and checkpoint
started by `start.sh` on a 64 GB card.

| Check | Result |
|---|---|
| HumanEval, pass@1 (164 problems) | **155/164** (94.5%) |
| GSM8K (1,319 test questions) | **1,266/1,319** (96.0%) |
| GPQA Diamond (198 questions, thinking on) | **170/198** (85.9%) |
| Image and video input | 7/7: a text control, three single images (4000×3000 to 64×64), two images in one message, two short videos |
| Needle retrieval at 4k, 16k, 32k and 64k | 16/16 |
| 163,840-token window (40 GB card) | 163,839 prompt tokens + 1 output token accepted; 163,840 refused |
| 262,144-token window (64 GB card) | 262,143 prompt tokens + 1 output token accepted; 262,144 refused |

- **HumanEval** uses EvalPlus's instruct prompt as one chat message (1,024
  tokens) and OpenAI's `check_correctness` program (3 s limit). The Codex
  paper's raw-prompt completion mode scores 127/164.
- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in its chat mode:
  5-shot, 1,024 tokens, the last integer in the reply. The script's default
  completion mode scores 1,058/1,319.
- **GPQA Diamond** uses OpenAI simple-evals' prompt and answer pattern, with
  thinking on, temperature 1.0, top_p 0.95 and top_k 20 (the checkpoint's
  `generation_config.json` and Qwen's thinking-mode recommendation), up to
  65,536 tokens per reply, one pass. Eight replies hit the limit and count as
  wrong.

## What runs

| | |
|---|---|
| Engine | Three pins from [AlexBrandes/vllm](https://github.com/AlexBrandes/vllm), in `versions.env`: upstream vLLM's wheel for [`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d); the fork's vLLM branch at [`eb0f19d117`](https://github.com/AlexBrandes/vllm/commit/eb0f19d117e35e32631e41be1cced3d98368d776), Python files over the wheel; and the vllm-cmp170hx plugin at [`b03ecee908`](https://github.com/AlexBrandes/vllm/commit/b03ecee908aaf9abb26b0c97472faa871fd814d1) |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-27b@sha256:1c9a3796337e3f8ab2ad5d359465a8b99dbe5df32342a18ca5ea715748195920` (tag `engine-eb0f19d117-b03ecee908`): CUDA 13.0 Ubuntu 24.04 base pinned by digest, plus the three pins, each fetched by commit and checked by sha256. `docker/build.sh` rebuilds it (about 17 minutes, no GPU needed) |
| Checkpoint | [`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound/tree/1f05c441c4e64ae0549de44fa9ea5a6d43610314) at revision `1f05c441c4e64ae0549de44fa9ea5a6d43610314`, in [HyperQwen](https://github.com/syv-ai/HyperQwen)'s "fast" layout: an INT8 embedding table, plus the int4 LM head and MTP module from [`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant/tree/124c14e7e8c7d2f5402933b9af368e772a9fcf0c). `download.sh` assembles it and checks every file against `tools/model.sha256` |
| Drafter | [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16/tree/4d30ec736ffc6b8688dc2ae2b502d9b48bdec279) at revision `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`, 7 draft tokens per step |
| Scheduling | up to 8,192 tokens per step; a new prompt prefills beside running decodes in pieces of about 500 ms |
| Cache | BF16 KV, prefix caching, FP16 Mamba state, 262,144-token window on a 64 GB card and 163,840 on a 40 GB card, up to 8 sequences, GPU memory utilization 0.93 |
| Graphs | `FULL_AND_PIECEWISE` CUDA graphs |

`start.sh` holds every environment variable and engine argument.

The vLLM branch is upstream plus the fork's commits; the
[plugin](https://github.com/AlexBrandes/vllm/tree/plugin) registers the CMP 170HX
(sm_80) platform and carries the kernels. Among them, this model uses:

1. **INT8 activations for the linear layers** (`VLLM_MARLIN_INPUT_DTYPE=int8`):
   the int4 weights run against INT8 activations, with the kernel shape for
   each matrix picked from a table measured on this card.
2. **INT8 prefill** (`VLLM_CMP_I8MM`, `VLLM_CMP_PREFILL_INT8_ATTN`): long
   prompts run each layer as one INT8 matrix multiply and compute attention
   scores on INT8 tensor cores. Prefill on this card is limited by the 170 W
   power limit, and these paths do the same work on less energy.
3. **Fused INT8 steps** (`VLLM_CMP_INT8_ACT_FUSION`): normalization and
   activation functions hand the next layer INT8 inputs directly.
4. **Faster checking of draft tokens** (`VLLM_SPEC_DECODE_ATTN`,
   `VLLM_SPEC_ATTN_POLICY=ga100`): HyperQwen's attention kernel for
   speculative decoding, tuned for this card's 74 SMs.
5. **Prefill beside running decodes** (`VLLM_CMP_PIECE_BUDGET_MS`): a new long
   prompt keeps advancing while other requests decode, without stalling them
   for more than about half a second.

### Host settings (`host-tune.sh`)

`start.sh` runs it through `sudo`; `HOST_TUNE=0` skips it.

- **170 W power limit** on the card.

### Reasoning

Thinking is off by default (`--default-chat-template-kwargs
{"enable_thinking":false}`); send `"chat_template_kwargs": {"enable_thinking":
true}` to turn it on. The speed tables, HumanEval and GSM8K have thinking off.

## Caveats

- One host: CMP 170HX at PCIe Gen2 x4, Intel Core Ultra 5 225, 93 GB RAM,
  Ubuntu 24.04, driver 610.43.02. The speed tables come from a 40 GB card (PCI
  ID `10de:2082`); other hosts have not been measured.
- The first start compiles kernels and CUDA graphs (several minutes); later
  starts reuse `./cache`.
- The native kernels are compiled at build time, so an image rebuilt with
  `docker/build.sh` has its own identity and hashes.
- Greedy decoding at several requests in flight with speculative decoding is
  not bitwise repeatable; HumanEval's failing set can vary by a task between
  sessions.
- The decode-at-depth table and the three-run medians are additions to the
  sparkDash protocol; everything else follows it.

## Licence

MIT for this repository ([LICENSE](LICENSE)), except the two HyperQwen tools
(Apache-2.0). The engine is Apache-2.0 and the model has its own licence; see
[NOTICE](NOTICE) for every component and credit.
