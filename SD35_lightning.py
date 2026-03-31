#!/usr/bin/env python3

import argparse
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    import lightning.pytorch as pl  # type: ignore
    _PL_AVAILABLE = True
except Exception:
    try:
        import pytorch_lightning as pl  # type: ignore
        _PL_AVAILABLE = True
    except Exception:
        _PL_AVAILABLE = False

        class _PLStub:
            class LightningModule(nn.Module):
                pass

            class LightningDataModule:
                pass

            class Trainer:
                pass

        pl = _PLStub()  # type: ignore

from paths_config import ADAPTER_SD35_PATH, SD35_MODEL_PATH
from simone_adapter_SD35 import (
    DEFAULT_SD35_CLIP_EMB_PATH,
    EmbeddingMLP,
    embed_sd35_clip,
    load_adapter_sd35,
)
from utils import (
    DEFAULT_CAPTIONS,
    DEFAULT_HYCO_CFG,
    DEFAULT_HYCO_CKPT,
    DEFAULT_HYCO_EMB_PATH,
    DEFAULT_IMAGES,
    embedd_hyco,
    load_hycoCLIP,
)


DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_STEER_SCALE = 1.0
DEFAULT_PROMPT_SCALE = 0.25
DEFAULT_OPENCLIP_SCALE = 0.0


def ensure_lightning() -> None:
    if not _PL_AVAILABLE:
        raise SystemExit(
            "PyTorch Lightning is not installed in this env. Install one of: "
            "`pip install lightning` or `pip install pytorch-lightning`."
        )


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def prompt_stub(prompt: str, max_words: int = 8) -> str:
    words = prompt.strip().split()[:max_words]
    cleaned = []
    for word in words:
        token = re.sub(r"[^a-zA-Z0-9]+", "", word).lower()
        if token:
            cleaned.append(token)
    return "_".join(cleaned) if cleaned else "prompt"


def output_filename_from_prompt(prompt: str, steer: bool) -> str:
    prefix = "adapter" if steer else "vanilla"
    return f"{prefix}_{prompt_stub(prompt)}_lightning.png"


def split_pooled(pooled: Optional[torch.Tensor], clip_dim: int = 768, openclip_dim: int = 1280):
    if pooled is None or pooled.ndim != 2 or pooled.shape[-1] != clip_dim + openclip_dim:
        return None, None
    return pooled[..., :clip_dim], pooled[..., clip_dim : clip_dim + openclip_dim]


class AdapterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        hyco_emb_path: str,
        sd35_emb_path: str,
        batch_size: int = 1024,
        val_size: int = 1000,
        num_workers: int = 4,
    ):
        super().__init__()
        self.hyco_emb_path = hyco_emb_path
        self.sd35_emb_path = sd35_emb_path
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        hyco = torch.load(self.hyco_emb_path, map_location="cpu").to(torch.float32)
        sd35 = torch.load(self.sd35_emb_path, map_location="cpu").to(torch.float32)

        if hyco.shape[0] != sd35.shape[0]:
            raise RuntimeError(f"Embedding size mismatch: {hyco.shape} vs {sd35.shape}")
        if hyco.shape[0] <= self.val_size:
            raise RuntimeError(
                f"Not enough samples ({hyco.shape[0]}) for val_size={self.val_size}."
            )

        self.train_ds = TensorDataset(hyco[:-self.val_size], sd35[:-self.val_size])
        self.val_ds = TensorDataset(hyco[-self.val_size :], sd35[-self.val_size :])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class AdapterLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 2048,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EmbeddingMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_prob=dropout,
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class PromptDataset(Dataset):
    def __init__(self, prompt: str):
        self.item = {"image_id": 0, "caption": prompt}

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.item


class SD35LightningGenerator(pl.LightningModule):
    def __init__(
        self,
        prompt: str,
        output_dir: str,
        steer: bool,
        model_path: str,
        adapter_path: str,
        steer_scale: float = DEFAULT_STEER_SCALE,
        prompt_scale: float = DEFAULT_PROMPT_SCALE,
        openclip_scale: float = DEFAULT_OPENCLIP_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        seed: int = 42,
    ):
        super().__init__()
        self.prompt = prompt
        self.output_dir = Path(output_dir)
        self.steer = steer
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.steer_scale = steer_scale
        self.prompt_scale = prompt_scale
        self.openclip_scale = openclip_scale
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed

        self.pipe = None
        self.hycoclip = None
        self.tokenizer = None
        self.adapter = None

    def setup(self, stage=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            safety_checker=None,
            add_watermarker=False,
            local_files_only=True,
        )
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        if self.steer:
            from hycoclip.tokenizer import Tokenizer

            self.hycoclip = load_hycoCLIP()
            self.tokenizer = Tokenizer()
            self.adapter = load_adapter_sd35(self.adapter_path)

    def _encode_sd35_prompt(self, prompt_text: str):
        encoded = self.pipe.encode_prompt(
            prompt=prompt_text,
            prompt_2=prompt_text,
            prompt_3=prompt_text,
            device=self.pipe._execution_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        prompt_embeds = encoded[0].to(self.pipe._execution_device, dtype=self.pipe.dtype)

        pooled = None
        for item in encoded[1:]:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                pooled = item
                break
        if pooled is not None:
            pooled = pooled.to(self.pipe._execution_device, dtype=self.pipe.dtype)
        _, openclip_pool = split_pooled(pooled)
        if openclip_pool is not None:
            openclip_pool = openclip_pool.to(self.pipe._execution_device, dtype=self.pipe.dtype)
        return prompt_embeds, pooled, openclip_pool

    def _adapter_pooled_embedding(self, prompt_text: str, openclip_pool: Optional[torch.Tensor]):
        tokens = self.tokenizer([prompt_text], max_length=25)
        with torch.no_grad():
            hyper = self.hycoclip.encode_text(tokens, project=True, return_activations=False)
            tangent = self.hycoclip.reverse_hyperbolic_projection(hyper)
            adapted = self.adapter(self.steer_scale * tangent).to(
                self.pipe._execution_device,
                dtype=self.pipe.dtype,
            )

        if adapted.shape[-1] >= 2048:
            return adapted

        if openclip_pool is None:
            raise RuntimeError("OpenCLIP pooled component missing in SD3.5 encode.")
        return torch.cat([adapted, self.openclip_scale * openclip_pool], dim=-1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, dict):
            caption = batch["caption"]
            if isinstance(caption, (list, tuple)):
                prompt_text = caption[0]
            else:
                prompt_text = caption
        else:
            row = batch[0] if isinstance(batch, (list, tuple)) else batch
            prompt_text = row["caption"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / output_filename_from_prompt(prompt_text, self.steer)

        prompt_embeds, pooled, openclip_pool = self._encode_sd35_prompt(prompt_text)
        if self.steer:
            pooled_prompt_embeds = self._adapter_pooled_embedding(prompt_text, openclip_pool)
        else:
            pooled_prompt_embeds = pooled
            if pooled_prompt_embeds is None:
                raise RuntimeError("Could not locate pooled prompt embeds in baseline SD3.5 encode.")

        generator = torch.Generator(device=self.pipe._execution_device).manual_seed(self.seed)
        with torch.inference_mode():
            image = self.pipe(
                prompt_embeds=prompt_embeds * self.prompt_scale,
                pooled_prompt_embeds=pooled_prompt_embeds,
                output_type="pil",
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]

        image = image.point(lambda x: max(0, min(254, x)))
        image.save(out_path, format="PNG")
        print(f"Saved {out_path}")
        return str(out_path)


def run_train(args):
    ensure_lightning()

    if parse_bool(args.build_hyco):
        embedd_hyco(
            save_path=args.hyco_emb,
            captions_path=args.captions,
            images_path=args.images,
            hyco_cfg_path=args.hyco_cfg,
            hyco_ckpt_path=args.hyco_ckpt,
        )

    if parse_bool(args.build_sd35):
        embed_sd35_clip(
            save_path=args.sd35_emb,
            captions_path=args.captions,
            images_path=args.images,
            sd35_path=args.sd35_path,
            batch_size=args.embed_batch_size,
            full_dim=parse_bool(args.full_dim),
        )

    output_dim = 2048 if parse_bool(args.full_dim) else 768
    dm = AdapterDataModule(
        hyco_emb_path=args.hyco_emb,
        sd35_emb_path=args.sd35_emb,
        batch_size=args.train_batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
    )
    module = AdapterLightningModule(
        input_dim=512,
        output_dim=output_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

    try:
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
    except Exception:
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore

    ckpt_cb = ModelCheckpoint(
        dirpath=str(Path(args.adapter_out).parent),
        filename=Path(args.adapter_out).stem,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    es_cb = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[ckpt_cb, es_cb],
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(module, datamodule=dm)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        state = torch.load(best_ckpt, map_location="cpu")
        state_dict = state.get("state_dict", state)
        cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
        if not cleaned:
            cleaned = state_dict
        Path(args.adapter_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cleaned, args.adapter_out)
        print(f"Saved adapter weights to {args.adapter_out}")


def run_generate(args):
    ensure_lightning()

    ds = PromptDataset(args.prompt)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    generator = SD35LightningGenerator(
        prompt=args.prompt,
        output_dir=args.output_dir,
        steer=parse_bool(args.steer),
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        steer_scale=args.steer_scale,
        prompt_scale=args.prompt_scale,
        openclip_scale=args.openclip_scale,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    trainer.predict(generator, dataloaders=dl)


def build_parser():
    parser = argparse.ArgumentParser(description="SD3.5 Lightning pipeline (train + generate).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train adapter with Lightning.")
    train.add_argument("--build-hyco", default="true")
    train.add_argument("--build-sd35", default="true")
    train.add_argument("--captions", default=DEFAULT_CAPTIONS)
    train.add_argument("--images", default=DEFAULT_IMAGES)
    train.add_argument("--hyco-cfg", default=DEFAULT_HYCO_CFG)
    train.add_argument("--hyco-ckpt", default=DEFAULT_HYCO_CKPT)
    train.add_argument("--hyco-emb", default=DEFAULT_HYCO_EMB_PATH)
    train.add_argument("--sd35-path", default=str(SD35_MODEL_PATH))
    train.add_argument("--sd35-emb", default=DEFAULT_SD35_CLIP_EMB_PATH)
    train.add_argument("--adapter-out", default=str(ADAPTER_SD35_PATH))
    train.add_argument("--full-dim", default="true")
    train.add_argument("--epochs", type=int, default=1000)
    train.add_argument("--patience", type=int, default=15)
    train.add_argument("--val-size", type=int, default=1000)
    train.add_argument("--train-batch-size", type=int, default=1024)
    train.add_argument("--embed-batch-size", type=int, default=256)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--dropout", type=float, default=0.2)

    gen = sub.add_parser("generate", help="Generate one image with Lightning predict.")
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--output_dir", required=True)
    gen.add_argument("--steer", type=parse_bool, default=False)
    gen.add_argument("--model_path", default=str(SD35_MODEL_PATH))
    gen.add_argument("--adapter_path", default=str(ADAPTER_SD35_PATH))
    gen.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    gen.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    gen.add_argument("--steer_scale", type=float, default=DEFAULT_STEER_SCALE)
    gen.add_argument("--prompt_scale", type=float, default=DEFAULT_PROMPT_SCALE)
    gen.add_argument("--openclip_scale", type=float, default=DEFAULT_OPENCLIP_SCALE)
    gen.add_argument("--seed", type=int, default=42)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "generate":
        run_generate(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
