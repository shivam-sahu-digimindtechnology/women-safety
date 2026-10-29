# Models Directory

This directory stores deep learning weights and model files used by the Women Safety AI system.

## Required and Supported Models

| Model File | Task | Default Source | Size |
|---|---|---|---|
| `yolov8n.pt` | Person Detection / Tracking | Ultralytics (auto-downloads if missing) | ~6.5 MB |
| `yolov8n-pose.pt` | Pose & Fall Detection | Ultralytics (auto-downloads if missing) | ~6.8 MB |
| `rizvandwiki/gender-classification` | Optional Gender Classification | Hugging Face Hub (cached automatically) | ~340 MB |
| `ViT-B/32` | Optional CLIP Violence Detection | OpenAI CLIP (cached in `~/.cache/clip`) | ~350 MB |

## Notes on Model Locations
- By default, `config.py` looks for `yolov8n.pt` and `yolov8n-pose.pt` in the project root or in `models/`.
- If missing, Ultralytics YOLO will automatically download them on first run when connected to the internet.
- Pretrained Hugging Face and CLIP models are downloaded to the standard user cache directory (`~/.cache/huggingface` and `~/.cache/clip`).
