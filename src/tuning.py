import torch
from train import main as train_main
from model import EncoderSmall    
from pathlib import Path

Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)

configs = [
    # {"name": "baseline",     "depth": 2, "out_ch": 128, "beam": 1},
    # {"name": "deep_cnn",     "depth": 3, "out_ch": 256, "beam": 1},
    # {"name": "beam_search",  "depth": 2, "out_ch": 128, "beam": 3},
    # {"name": "best_combo",   "depth": 3, "out_ch": 256, "beam": 3},
    {"name": "resnet50", "depth": 0, "out_ch": 512, "beam": 3},
]

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"START FULL TRAINING: {cfg['name']}")
    print(f"Config: depth={cfg['depth']}, out_ch={cfg['out_ch']}, beam={cfg['beam']}")
    print(f"{'='*60}")
    enc = EncoderSmall(out_ch=cfg["out_ch"], train_backbone=False).to("cuda")

    cider = train_main(
        enc=enc,
        dec=None,
        epochs=15,
        beam=cfg["beam"],
        save_prefix=cfg["name"]
    )

    results.append({
        "name": cfg["name"],
        "depth": cfg["depth"],
        "out_ch": cfg["out_ch"],
        "beam": cfg["beam"],
        "CIDEr": round(cider, 4)
    })

print("\n" + "="*60)
print("FINAL FULL TRAINING RESULTS")
print("="*60)
print("Config\t\tDepth\tout_ch\tBeam\tCIDEr")
for r in results:
    print(f"{r['name']:<15}\t{r['depth']}\t{r['out_ch']}\t{r['beam']}\t{r['CIDEr']}")
print("="*60)
