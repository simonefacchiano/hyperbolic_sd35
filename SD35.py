#!/usr/bin/env python3

import argparse
import datetime
import re
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

from paths_config import ADAPTER_SD35_PATH, SD35_MODEL_PATH

SD35_LARGE_PATH = str(SD35_MODEL_PATH)
ADAPTER_SD35_PATH = str(ADAPTER_SD35_PATH)

DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE_BASELINE = DEFAULT_GUIDANCE_SCALE_STEER = 4.5
DEFAULT_STEER_SCALE = 1.0
DEFAULT_PROMPT_SCALE = 0.25
DEFAULT_OPENCLIP_SCALE = 0.0


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def read_single_prompt(prompt: str):
    yield {"image_id": "0", "caption": prompt}


def iter_batches(rows, batch_size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def slugify_upper(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return (slug or "STEER").upper()


def scale_to_name(scale_val: float) -> str:
    return str(scale_val).replace("-", "m").replace(".", "p")


def prompt_stub(prompt: str, max_words: int = 8) -> str:
    words = prompt.strip().split()
    words = words[:max_words]
    cleaned = []
    for w in words:
        token = re.sub(r"[^a-zA-Z0-9]+", "", w).lower()
        if token:
            cleaned.append(token)
    return "_".join(cleaned) if cleaned else "prompt"


def output_filename_from_prompt(prompt: str, steer: bool) -> str:
    prefix = "adapter" if steer else "vanilla"
    return f"{prefix}_{prompt_stub(prompt)}.png"


def parse_steer_scales(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            for chunk in str(item).split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                parts.append(float(chunk))
        return parts
    if value is None:
        return []
    text = str(value)
    parts = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(float(chunk))
    return parts

def concat_images_horiz(images: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    combined = Image.new("RGB", (total_width, max_height))
    x = 0
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        combined.paste(img, (x, 0))
        x += img.size[0]
    return combined


def build_output_dir(base_dir: Path, steer: bool, steer_scale: float) -> Path:
    _ = steer
    _ = steer_scale
    return base_dir


def sync_cuda_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_sd35(model_path: str, dtype: torch.dtype, device: str):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        add_watermarker=False,
        local_files_only=True,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SD3.5 baseline or steered generation from a single prompt."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Single prompt text.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory (suffix added automatically).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts per batch (default: 1).",
    )
    parser.add_argument(
        "--steer",
        type=parse_bool,
        default=False,
        help="Enable hyperbolic steering (default: False).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE_BASELINE,
        help="Override guidance scale (defaults depend on --steer).",
    )
    parser.add_argument(
        "--steer_scale",
        nargs="+",
        default=DEFAULT_STEER_SCALE,
        help="Scale multiplier for adapter-conditioned pooled embeddings.",
    )
    parser.add_argument(
        "--prompt_scale",
        type=float,
        default=DEFAULT_PROMPT_SCALE,
        help="Scale for T5 prompt embeddings in steer mode.",
    )
    parser.add_argument(
        "--openclip_scale",
        type=float,
        default=DEFAULT_OPENCLIP_SCALE,
        help="Scale for OpenCLIP pooled embeddings in steer mode.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Override SD3.5 model path.",
    )
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="Override adapter path (steer mode only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used for per-image seeding (final seed = base + image_id).",
    )
    parser.add_argument(
        "--time_breakdown",
        type=parse_bool,
        default=False,
        help="Measure steering-compute vs inference time separately.",
    )
    parser.add_argument(
        "--timing_max_generations",
        type=int,
        default=10,
        help="Number of generations to average in --time_breakdown mode.",
    )
    parser.add_argument(
        "--timing_save_images",
        type=parse_bool,
        default=False,
        help="Save images in --time_breakdown mode (default: False).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.time_breakdown and args.timing_max_generations < 1:
        raise SystemExit("--timing_max_generations must be >= 1")
    steer_scales = parse_steer_scales(args.steer_scale)
    if args.steer and not steer_scales:
        raise SystemExit("--steer_scale must have at least one value")

    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = (
            DEFAULT_GUIDANCE_SCALE_STEER if args.steer else DEFAULT_GUIDANCE_SCALE_BASELINE
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_path = args.model_path or SD35_LARGE_PATH

    output_dir = build_output_dir(
        Path(args.output_dir),
        args.steer,
        steer_scales[0] if steer_scales else DEFAULT_STEER_SCALE,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in {output_dir}")

    start_ts = time.time()

    pipe = load_sd35(model_path, dtype, device)

    input_rows = read_single_prompt(args.prompt)

    total_images = 0
    steering_time_total = 0.0
    prompt_encode_time_total = 0.0
    inference_time_total = 0.0
    timed_generations = 0
    effective_batch_size = 1 if args.time_breakdown else args.batch_size

    if not args.steer:
        row_index = 0
        for batch in iter_batches(input_rows, effective_batch_size):
            prompts = []
            image_ids = []
            generators = []
            for row in batch:
                row_idx = row_index
                caption = None
                image_id = None
                if "image_id" in row and "caption" in row:
                    caption = row["caption"].strip()
                    image_id = int(row["image_id"])
                elif "adv_prompt" in row:
                    caption = row["adv_prompt"].strip()
                    if "row_id" in row and str(row["row_id"]).strip():
                        image_id = int(str(row["row_id"]).strip())
                    elif "image_id" in row and row["image_id"].strip():
                        image_id = int(row["image_id"])
                    else:
                        image_id = row_idx
                elif "sensitive prompt" in row:
                    caption = row["sensitive prompt"].strip()
                    if "row_id" in row and str(row["row_id"]).strip():
                        image_id = int(str(row["row_id"]).strip())
                    else:
                        image_id = row_idx
                else:
                    raise SystemExit(
                        "CSV must have image_id+caption, adv_prompt, or a sensitive prompt column."
                    )
                if not caption:
                    row_index += 1
                    continue
                output_path = output_dir / output_filename_from_prompt(caption, steer=False)
                if (not args.time_breakdown) and output_path.exists():
                    now = datetime.datetime.now().isoformat()
                    print(f"Skip existing {output_path}")
                    row_index += 1
                    continue
                prompts.append(caption)
                image_ids.append(image_id)
                seed = int(args.seed) + int(image_id)
                generators.append(torch.Generator(device=device).manual_seed(seed))
                row_index += 1

            if not prompts:
                continue

            if args.time_breakdown:
                result_images = []
                for prompt_text, generator in zip(prompts, generators):
                    sync_cuda_if_available()
                    pe_t0 = time.perf_counter()
                    encoded = pipe.encode_prompt(
                        prompt=prompt_text,
                        prompt_2=prompt_text,
                        prompt_3=prompt_text,
                        device=pipe._execution_device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    prompt_embeds = encoded[0].to(pipe._execution_device, dtype=pipe.dtype)
                    pooled_prompt_embeds = None
                    for item in encoded[1:]:
                        if isinstance(item, torch.Tensor) and item.ndim == 2:
                            pooled_prompt_embeds = item.to(pipe._execution_device, dtype=pipe.dtype)
                            break
                    if pooled_prompt_embeds is None:
                        raise RuntimeError("Could not locate pooled prompt embeds in baseline SD3.5 encode.")
                    sync_cuda_if_available()
                    prompt_encode_time_total += time.perf_counter() - pe_t0

                    sync_cuda_if_available()
                    infer_t0 = time.perf_counter()
                    with torch.inference_mode():
                        image = pipe(
                            prompt_embeds=prompt_embeds * args.prompt_scale,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            output_type="pil",
                        ).images[0]
                    sync_cuda_if_available()
                    inference_time_total += time.perf_counter() - infer_t0
                    timed_generations += 1
                    result_images.append(image)
            else:
                sync_cuda_if_available()
                pe_t0 = time.perf_counter()
                encoded = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    prompt_3=prompts,
                    device=pipe._execution_device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
                prompt_embeds = encoded[0].to(pipe._execution_device, dtype=pipe.dtype)
                pooled_prompt_embeds = None
                for item in encoded[1:]:
                    if isinstance(item, torch.Tensor) and item.ndim == 2:
                        pooled_prompt_embeds = item.to(pipe._execution_device, dtype=pipe.dtype)
                        break
                if pooled_prompt_embeds is None:
                    raise RuntimeError("Could not locate pooled prompt embeds in baseline SD3.5 encode.")
                sync_cuda_if_available()
                prompt_encode_time_total += time.perf_counter() - pe_t0
                sync_cuda_if_available()
                infer_t0 = time.perf_counter()
                with torch.inference_mode():
                    result = pipe(
                        prompt_embeds=prompt_embeds * args.prompt_scale,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generators,
                        output_type="pil",
                    )
                sync_cuda_if_available()
                result_images = result.images
            for image_id, image in zip(image_ids, result_images):
                output_path = output_dir / output_filename_from_prompt(args.prompt, steer=False)
                if (not args.time_breakdown) or args.timing_save_images:
                    image.save(output_path, format="PNG")
                total_images += 1
                now = datetime.datetime.now().isoformat()
                if (not args.time_breakdown) or args.timing_save_images:
                    print(f"Saved {output_path}")
                else:
                    print(f"Generated {image_id} (not saved)")
                if args.time_breakdown and timed_generations >= args.timing_max_generations:
                    break
            if args.time_breakdown and timed_generations >= args.timing_max_generations:
                break
    else:
        from utils import load_hycoCLIP  # noqa: E402
        from hycoclip.tokenizer import Tokenizer  # noqa: E402
        from simone_adapter_SD35 import load_adapter_sd35  # noqa: E402

        adapter_path = args.adapter_path or ADAPTER_SD35_PATH
        adapter = load_adapter_sd35(adapter_path)
        adapter_out_dim = None
        try:
            adapter_out_dim = adapter.backbone[-1].out_features
        except Exception:
            pass

        model = load_hycoCLIP()
        hycoclip_tokenizer = Tokenizer()

        def get_hyperbolic_embedding(prompt):
            tokens = hycoclip_tokenizer(
                [prompt] if isinstance(prompt, str) else prompt, max_length=25
            )
            with torch.no_grad():
                hyperbolic_text_features = model.encode_text(
                    tokens, project=True, return_activations=False
                ).cpu()
            return hyperbolic_text_features

        def get_adapter_input(prompt_text: str):
            hyperbolic_features = get_hyperbolic_embedding(prompt_text)
            with torch.no_grad():
                tangent_features = model.reverse_hyperbolic_projection(
                    hyperbolic_features.to("cuda")
                ).cpu()
            return tangent_features

        def split_pooled(pooled: torch.Tensor, clip_dim: int = 768, openclip_dim: int = 1280):
            if pooled is None or pooled.ndim != 2 or pooled.shape[-1] != clip_dim + openclip_dim:
                return None, None
            return pooled[..., :clip_dim], pooled[..., clip_dim : clip_dim + openclip_dim]

        def encode_sd35(prompt_text: str):
            encoded = pipe.encode_prompt(
                prompt=prompt_text,
                prompt_2=prompt_text,
                prompt_3=prompt_text,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            prompt_embeds = encoded[0]

            pooled = None
            for item in encoded[1:]:
                if isinstance(item, torch.Tensor) and item.ndim == 2:
                    pooled = item
                    break
            _, openclip_pool = split_pooled(pooled)
            return prompt_embeds, openclip_pool
        
        row_index = 0
        for batch in iter_batches(input_rows, effective_batch_size):
            for row in batch:
                row_idx = row_index
                caption = None
                image_id = None
                if "image_id" in row and "caption" in row:
                    caption = row["caption"].strip()
                    image_id = int(row["image_id"])
                elif "adv_prompt" in row:
                    caption = row["adv_prompt"].strip()
                    if "row_id" in row and str(row["row_id"]).strip():
                        image_id = int(str(row["row_id"]).strip())
                    elif "image_id" in row and row["image_id"].strip():
                        image_id = int(row["image_id"])
                    else:
                        image_id = row_idx
                elif "sensitive prompt" in row:
                    caption = row["sensitive prompt"].strip()
                    if "row_id" in row and str(row["row_id"]).strip():
                        image_id = int(str(row["row_id"]).strip())
                    else:
                        image_id = row_idx
                else:
                    raise SystemExit(
                        "CSV must have image_id+caption, adv_prompt, or a sensitive prompt column."
                    )
                if not caption:
                    row_index += 1
                    continue
                per_scale_paths = []
                for steer_scale in steer_scales:
                    _ = steer_scale
                    per_scale_paths.append(
                        output_dir / output_filename_from_prompt(caption, steer=True)
                    )
                if (not args.time_breakdown) and all(path.exists() for path in per_scale_paths):
                    now = datetime.datetime.now().isoformat()
                    for path in per_scale_paths:
                        print(f"Skip existing {path}")
                    row_index += 1
                    continue
                sync_cuda_if_available()
                steering_t0 = time.perf_counter()
                seed = int(args.seed) + int(image_id)
                base_adapter_input = get_adapter_input(caption)
                sync_cuda_if_available()
                steering_pre_scale_elapsed = time.perf_counter() - steering_t0

                sync_cuda_if_available()
                pe_t0 = time.perf_counter()
                prompt_embeds, openclip_pool = encode_sd35(caption)
                prompt_embeds = prompt_embeds.to(pipe._execution_device, dtype=pipe.dtype)
                sync_cuda_if_available()
                prompt_encode_elapsed = time.perf_counter() - pe_t0
                scales_for_avg = len(steer_scales) if args.time_breakdown else 1

                for steer_scale in steer_scales:
                    output_path = output_dir / output_filename_from_prompt(caption, steer=True)
                    if (not args.time_breakdown) and output_path.exists():
                        now = datetime.datetime.now().isoformat()
                        print(f"Skip existing {output_path}")
                        continue
                    sync_cuda_if_available()
                    steering_scale_t0 = time.perf_counter()
                    adapter_input = (steer_scale * base_adapter_input).to("cuda")
                    hyco_encodings_adapted = adapter(adapter_input)
                    hyco_encodings_adapted = hyco_encodings_adapted.to(
                        pipe._execution_device, dtype=pipe.dtype
                    )
                    if adapter_out_dim is None:
                        adapter_out_dim = hyco_encodings_adapted.shape[-1]
                    if adapter_out_dim >= 2048:
                        pooled_prompt_embeds = hyco_encodings_adapted
                    else:
                        if openclip_pool is None:
                            raise RuntimeError(
                                "Could not locate OpenCLIP pooled component in SD3.5 encoding."
                            )
                        openclip_pool = openclip_pool.to(
                            pipe._execution_device, dtype=pipe.dtype
                        )
                        pooled_prompt_embeds = torch.cat(
                            [hyco_encodings_adapted, args.openclip_scale * openclip_pool],
                            dim=-1,
                        )
                    pooled_prompt_embeds = pooled_prompt_embeds.to(
                        pipe._execution_device, dtype=pipe.dtype
                    )
                    sync_cuda_if_available()
                    steering_scale_elapsed = time.perf_counter() - steering_scale_t0

                    generator = torch.Generator("cuda").manual_seed(seed)
                    sync_cuda_if_available()
                    infer_t0 = time.perf_counter()
                    with torch.inference_mode():
                        image = pipe(
                            prompt_embeds=prompt_embeds * args.prompt_scale,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            output_type="pil",
                            generator=generator,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=guidance_scale,
                        ).images[0]
                    sync_cuda_if_available()
                    infer_elapsed = time.perf_counter() - infer_t0
                    image = image.point(lambda x: max(0, min(254, x)))
                    if (not args.time_breakdown) or args.timing_save_images:
                        image.save(output_path, format="PNG")
                    total_images += 1
                    if args.time_breakdown:
                        steering_time_total += (
                            steering_scale_elapsed
                            + (steering_pre_scale_elapsed / max(scales_for_avg, 1))
                        )
                        prompt_encode_time_total += (
                            prompt_encode_elapsed / max(scales_for_avg, 1)
                        )
                        inference_time_total += infer_elapsed
                        timed_generations += 1
                    now = datetime.datetime.now().isoformat()
                    if (not args.time_breakdown) or args.timing_save_images:
                        print(f"Saved {output_path}")
                    else:
                        print(f"Generated {output_path.name} (not saved)")
                    if args.time_breakdown and timed_generations >= args.timing_max_generations:
                        break
                if args.time_breakdown and timed_generations >= args.timing_max_generations:
                    row_index += 1
                    break
                row_index += 1
            if args.time_breakdown and timed_generations >= args.timing_max_generations:
                break

    end_dt = datetime.datetime.now()
    total_time = time.time() - start_ts
    avg_time = total_time / total_images if total_images else 0.0
    if args.time_breakdown:
        timed = max(timed_generations, 1)
        avg_steering = steering_time_total / timed
        avg_prompt_encode = prompt_encode_time_total / timed
        avg_inference = inference_time_total / timed
        print(f"Timed generations: {timed_generations}")
        print(f"Steering compute avg/gen (s): {avg_steering:.4f}")
        print(f"Prompt encode avg/gen (s): {avg_prompt_encode:.4f}")
        print(f"Inference avg/gen (s): {avg_inference:.4f}")
    print("Complete SD3.5")


if __name__ == "__main__":
    main()
