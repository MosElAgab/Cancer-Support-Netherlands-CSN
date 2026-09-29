"""Download pinned public models only; does not read or fetch any candidate URLs.

Run from new/notebooks: ../../.venv/bin/python ../scripts/download_ranking_models.py
"""
import json
import os
from pathlib import Path

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from huggingface_hub import snapshot_download

base = Path(__file__).resolve().parents[1]
models = json.loads((base / 'data/reference/ranking_models.json').read_text())
for name, spec in models.items():
    snapshot_download(spec['model_id'], revision=spec['revision'],
                      cache_dir=base / '.cache/ranking/models',
                      allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'],
                      ignore_patterns=['onnx/*', 'openvino/*'], max_workers=3)
    print(f"Downloaded {name}: {spec['model_id']} at {spec['revision']}")
