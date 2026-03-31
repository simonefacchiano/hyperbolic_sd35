import csv
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from paths_config import FLICKR30K_CAPTIONS, FLICKR30K_IMAGES


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CFG_DIR = PROJECT_ROOT / "hycoclip_configs"

# Paths can be overridden when exporting/running on another machine.
DEFAULT_HYCO_CFG = os.environ.get(
    "HYCOCLIP_TRAIN_CONFIG",
    str(LOCAL_CFG_DIR / "train_hycoclip_vit_s.py"),
)
DEFAULT_HYCO_CKPT = os.environ.get(
    "HYCOCLIP_CHECKPOINT",
    str(PROJECT_ROOT / "hycoclip_vit_s.pth"),
)
DEFAULT_HYCO_EMB_PATH = os.environ.get(
    "HYCOCLIP_EMB_PATH",
    str(PROJECT_ROOT / "hyco_embeddings.pt"),
)
DEFAULT_CAPTIONS = os.environ.get(
    "FLICKR_CAPTIONS",
    str(FLICKR30K_CAPTIONS),
)
DEFAULT_IMAGES = os.environ.get(
    "FLICKR_IMAGES",
    str(FLICKR30K_IMAGES),
)

train_config = DEFAULT_HYCO_CFG
checkpoint_path = DEFAULT_HYCO_CKPT
config = os.environ.get(
    "HYCOCLIP_EVAL_CONFIG",
    str(LOCAL_CFG_DIR / "eval_zero_shot_retrieval.py"),
)


class FlickrCaptionsDataset(Dataset):
    def __init__(self, captions_path):
        captions_path = Path(captions_path).expanduser().resolve()
        if not captions_path.exists():
            raise FileNotFoundError("captions file not found: %s" % captions_path)

        self.caption_dic = {}
        with captions_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row["image"].strip()
                caption = row["caption"].strip().strip('"')
                self.caption_dic.setdefault(image_id, []).append(caption)

        self.keys = list(self.caption_dic.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        return key, self.caption_dic[key]


def get_flickr_dataloader(
    captions_path=DEFAULT_CAPTIONS,
    images_path=DEFAULT_IMAGES,
    batch_size=1024,
    num_workers=4,
):
    # images_path kept for backward compatibility with caller signatures.
    _ = images_path
    dataset = FlickrCaptionsDataset(captions_path=Path(captions_path).expanduser())

    def create_dictionary(batch):
        batches_dic = {}
        for image_id, texts in batch:
            batches_dic[image_id] = texts
        return batches_dic

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=create_dictionary,
    )


def load_hycoCLIP(cfg_path=DEFAULT_HYCO_CFG, ckpt_path=DEFAULT_HYCO_CKPT):
    from hycoclip.utils.checkpointing import CheckpointManager
    from hycoclip.config import LazyConfig, LazyFactory

    cfg_path = Path(cfg_path).expanduser().resolve()
    ckpt_path = Path(ckpt_path).expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError("HyCoCLIP config not found: %s" % cfg_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            "HyCoCLIP checkpoint not found: %s (put it in pytorch_project root or set HYCOCLIP_CHECKPOINT)"
            % ckpt_path
        )

    cfg = LazyConfig.load(str(cfg_path))
    model = LazyFactory.build_model(cfg, device="cuda").eval()
    for param in model.parameters():
        param.requires_grad = False
    CheckpointManager(model=model).load(str(ckpt_path))
    return model


def tokenize_batch(batch):
    from hycoclip.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokenized_tensor = None
    for image_id in batch:
        texts = batch[image_id]
        tokens = tokenizer(
            texts,
            max_length=77,
            pad_to_max=True,
        )
        tokens = torch.stack(tokens, dim=0)
        if tokenized_tensor is None:
            tokenized_tensor = tokens[None, ...]
        else:
            tokenized_tensor = torch.cat((tokenized_tensor, tokens[None, ...]), dim=0)
    return tokenized_tensor


def embedd_hyco(
    save_path=DEFAULT_HYCO_EMB_PATH,
    captions_path=DEFAULT_CAPTIONS,
    images_path=DEFAULT_IMAGES,
    hyco_cfg_path=DEFAULT_HYCO_CFG,
    hyco_ckpt_path=DEFAULT_HYCO_CKPT,
):
    save_path = Path(save_path)
    if save_path.exists():
        print("[embedd_hyco] Found existing embeddings at %s, skipping." % save_path)
        return

    dataloader = get_flickr_dataloader(captions_path=captions_path, images_path=images_path)
    model = load_hycoCLIP(cfg_path=hyco_cfg_path, ckpt_path=hyco_ckpt_path)

    all_emb = None
    with torch.no_grad():
        for el in tqdm(dataloader, desc="Embedding HyCoCLIP"):
            tokens = tokenize_batch(el)
            bsz = tokens.shape[0]
            n_caps = tokens.shape[1]
            tokens = tokens.reshape(bsz * n_caps, 77).to("cuda")
            encodings = model.encode_text(tokens, project=False)

            if all_emb is None:
                all_emb = encodings
            else:
                all_emb = torch.cat((all_emb, encodings), dim=0)

            del encodings, tokens
            torch.cuda.empty_cache()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_emb, save_path)
    print("[embedd_hyco] Saved embeddings to %s | shape=%s" % (save_path, tuple(all_emb.shape)))


__all__ = [
    "DEFAULT_CAPTIONS",
    "DEFAULT_HYCO_CFG",
    "DEFAULT_HYCO_CKPT",
    "DEFAULT_HYCO_EMB_PATH",
    "DEFAULT_IMAGES",
    "checkpoint_path",
    "config",
    "embedd_hyco",
    "get_flickr_dataloader",
    "load_hycoCLIP",
    "tokenize_batch",
    "train_config",
]
