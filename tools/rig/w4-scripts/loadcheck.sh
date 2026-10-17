set -e
tail -2 /mnt/data/runs/mini-agi/author/download.log || true
ls /mnt/data/runs/mini-agi/author/hf/weights | head; du -sh /mnt/data/runs/mini-agi/author/hf/weights
mkdir -p /mnt/data/runs/mini-agi/w4-B
[ -d /mnt/data/runs/mini-agi/w4-B/weights ] || cp -a /mnt/data/runs/mini-agi/author/hf/weights /mnt/data/runs/mini-agi/w4-B/weights
cd ~/train/mini-AGI
CUDA_VISIBLE_DEVICES="" /home/train/train/mini-AGI/.venv/bin/python - <<'PY'
import json, torch
from train import load_any
m = json.load(open("/mnt/data/runs/mini-agi/author/hf/weights/manifest.json"))
print("manifest cfg keys", sorted(m.get("cfg", {}).keys())[:12])
out = load_any("/mnt/data/runs/mini-agi/author/hf/weights", torch.device("cpu"), read_only=True)
model = out[0] if isinstance(out, tuple) else out
print("loaded", type(model).__name__, sum(p.numel() for p in model.parameters()))
PY
