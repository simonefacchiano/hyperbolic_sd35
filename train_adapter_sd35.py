import argparse
from pathlib import Path

from simone_adapter_SD35 import (
    SD35_PATH,
    DEFAULT_ADAPTER_SD35_SAVE_PATH,
    DEFAULT_SD35_CLIP_EMB_PATH,
    FULL_DIM,
    embed_sd35_clip,
    train_adapter_sd35,
)
from utils import (
    DEFAULT_CAPTIONS,
    DEFAULT_HYCO_CFG,
    DEFAULT_HYCO_CKPT,
    DEFAULT_HYCO_EMB_PATH,
    DEFAULT_IMAGES,
    embedd_hyco,
)


def parse_bool(s):
    return str(s).lower() in {"1", "true", "yes", "y"}


def main():
    parser = argparse.ArgumentParser(description="End-to-end SD3.5 adapter training.")
    parser.add_argument("--build-hyco", default="true", help="Build HyCo embeddings (true/false)")
    parser.add_argument("--build-sd35", default="true", help="Build SD3.5 CLIP embeddings (true/false)")
    parser.add_argument("--train", default="true", help="Train adapter (true/false)")

    parser.add_argument("--captions", default=DEFAULT_CAPTIONS)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--hyco-cfg", default=DEFAULT_HYCO_CFG)
    parser.add_argument("--hyco-ckpt", default=DEFAULT_HYCO_CKPT)
    parser.add_argument("--hyco-emb", default=DEFAULT_HYCO_EMB_PATH)

    parser.add_argument("--sd35-path", default=SD35_PATH)
    parser.add_argument("--sd35-emb", default=DEFAULT_SD35_CLIP_EMB_PATH)
    parser.add_argument("--adapter-out", default=DEFAULT_ADAPTER_SD35_SAVE_PATH)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full-dim", default=str(FULL_DIM).lower())

    args = parser.parse_args()
    full_dim = parse_bool(args.full_dim)

    Path(args.hyco_emb).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sd35_emb).parent.mkdir(parents=True, exist_ok=True)
    Path(args.adapter_out).parent.mkdir(parents=True, exist_ok=True)

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
            batch_size=args.batch_size,
            full_dim=full_dim,
        )

    if parse_bool(args.train):
        train_adapter_sd35(
            epochs=args.epochs,
            save_path=args.adapter_out,
            hyco_emb_path=args.hyco_emb,
            sd35_clip_emb_path=args.sd35_emb,
            patience=args.patience,
            full_dim=full_dim,
        )


if __name__ == "__main__":
    main()
