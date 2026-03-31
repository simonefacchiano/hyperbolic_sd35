from pathlib import Path

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from paths_config import ADAPTER_SD35_PATH as CFG_ADAPTER_SD35_PATH, SD35_MODEL_PATH
from utils import (  # noqa: E402
    DEFAULT_CAPTIONS,
    DEFAULT_HYCO_EMB_PATH,
    DEFAULT_IMAGES,
    get_flickr_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SD35_PATH = os.environ.get("SD35_MODEL_PATH", str(SD35_MODEL_PATH))
DEFAULT_SD35_CLIP_EMB_PATH = os.environ.get("SD35_CLIP_EMB_PATH", str(PROJECT_ROOT / "sd35_clip_fulldim.pt"))
DEFAULT_ADAPTER_SD35_SAVE_PATH = os.environ.get("ADAPTER_SD35_PATH", str(CFG_ADAPTER_SD35_PATH))
FULL_DIM = True  # True -> 2048-d pooled embeddings, False -> 768-d


def encode_prompts_sd35(
    prompts,
    sd35_path: str = SD35_PATH,
    batch_size: int = 256,
    full_dim: bool = FULL_DIM,
):
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe = StableDiffusion3Pipeline.from_pretrained(
        sd35_path,
        torch_dtype=torch.float16,
        variant="fp16",
        add_watermarker=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)

    all_emb = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Embedding SD3.5 CLIP"):
            subset = prompts[i : i + batch_size]
            if full_dim:
                if not hasattr(pipe, "tokenizer_2") or not hasattr(pipe, "text_encoder_2"):
                    raise RuntimeError("SD3.5 pipeline missing tokenizer_2/text_encoder_2.")
                tokenizer = pipe.tokenizer
                tokenizer_2 = pipe.tokenizer_2
                text_encoder = pipe.text_encoder.to(device).eval()
                text_encoder_2 = pipe.text_encoder_2.to(device).eval()

                tokens_1 = tokenizer(
                    subset,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                tokens_2 = tokenizer_2(
                    subset,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                outputs_1 = text_encoder(**tokens_1, output_hidden_states=False, return_dict=True)
                outputs_2 = text_encoder_2(**tokens_2, output_hidden_states=False, return_dict=True)

                pooled_1 = outputs_1.text_embeds if hasattr(outputs_1, "text_embeds") else outputs_1[1]
                pooled_2 = outputs_2.text_embeds if hasattr(outputs_2, "text_embeds") else outputs_2[1]

                proj_1 = pipe.text_encoder_projection(pooled_1) if hasattr(pipe, "text_encoder_projection") else pooled_1
                proj_2 = pipe.text_encoder_2_projection(pooled_2) if hasattr(pipe, "text_encoder_2_projection") else pooled_2
                clip_pool = torch.cat([proj_1, proj_2], dim=-1)
            else:
                tokenizer = pipe.tokenizer
                text_encoder = pipe.text_encoder.to(device).eval()
                tokens = tokenizer(
                    subset,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                outputs = text_encoder(**tokens, output_hidden_states=False, return_dict=True)
                clip_pool = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[1]
                clip_pool = clip_pool[..., :768]

            all_emb.append(clip_pool.to(torch.float32).cpu())
            torch.cuda.empty_cache()

    return torch.cat(all_emb, dim=0)


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=768, dropout_prob=0.2, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2048, output_dim),
        )
        self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, dim=-1)
        main = self.backbone(x)
        out = main + self.skip(x)
        if self.normalize:
            out = F.normalize(out, dim=-1)
        return out


def embed_sd35_clip(
    save_path: str = DEFAULT_SD35_CLIP_EMB_PATH,
    captions_path: str = DEFAULT_CAPTIONS,
    images_path: str = DEFAULT_IMAGES,
    sd35_path: str = SD35_PATH,
    batch_size: int = 256,
    full_dim: bool = FULL_DIM,
):
    save_path = Path(save_path)
    if save_path.exists():
        print(f"[embed_sd35_clip] Found existing embeddings at {save_path}, skipping.")
        return

    dataloader = get_flickr_dataloader(captions_path=captions_path, images_path=images_path)
    all_prompts = []
    for batch in dataloader:
        for texts in batch.values():
            all_prompts.extend(texts)

    all_emb_tensor = encode_prompts_sd35(
        all_prompts, sd35_path=sd35_path, batch_size=batch_size, full_dim=full_dim
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_emb_tensor, save_path)
    print(f"[embed_sd35_clip] Saved to {save_path} | shape={all_emb_tensor.shape}")


def train_adapter_sd35(
    epochs: int = 1000,
    save_path: str = DEFAULT_ADAPTER_SD35_SAVE_PATH,
    hyco_emb_path: str = DEFAULT_HYCO_EMB_PATH,
    sd35_clip_emb_path: str = DEFAULT_SD35_CLIP_EMB_PATH,
    patience: int = 15,
    full_dim: bool = FULL_DIM,
):
    hyco = torch.load(hyco_emb_path).to("cuda", dtype=torch.float32)
    sd35_clip = torch.load(sd35_clip_emb_path).to("cuda", dtype=torch.float32)

    dataset_train = TensorDataset(hyco[:-1000], sd35_clip[:-1000])
    dataset_val = TensorDataset(hyco[-1000:], sd35_clip[-1000:])

    batch_size = 1024
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 2048 if full_dim else 768
    model = EmbeddingMLP(output_dim=output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for data, target in dataloader_train:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            predictions = model(data)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(dataloader_train)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for data, target in dataloader_val:
                data, target = data.to(device), target.to(device)
                predictions = model(data)
                loss = criterion(predictions, target)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(dataloader_val)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[train_adapter_sd35] Saved best model @ epoch {epoch+1}")
        elif epoch - best_epoch >= patience:
            print(f"[train_adapter_sd35] Early stopping @ epoch {epoch+1}")
            break


def load_adapter_sd35(adapter_path: str = DEFAULT_ADAPTER_SD35_SAVE_PATH) -> EmbeddingMLP:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(adapter_path, map_location=device)

    inferred_out = None
    if "backbone.8.weight" in state_dict:
        inferred_out = state_dict["backbone.8.weight"].shape[0]
    elif "skip.weight" in state_dict:
        inferred_out = state_dict["skip.weight"].shape[0]

    output_dim = inferred_out if inferred_out is not None else 2048
    adapter = EmbeddingMLP(output_dim=output_dim)
    try:
        adapter.load_state_dict(state_dict)
    except RuntimeError:
        adapter.load_state_dict(state_dict, strict=False)

    adapter.eval()
    return adapter.to(device)
