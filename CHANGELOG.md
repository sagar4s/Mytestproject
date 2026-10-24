# Changelog

## Unreleased

### Dependencies (requirements.txt)
- gradio 5.9.1 -> 6.30.0 (major version; see "Code" below)
- torch 2.5.1 -> 2.14.1, torchvision 0.20.1 -> 0.29.1
- PyTorch wheel index cu124 -> cu130 (CUDA 13; older Pascal/Volta GPUs need the cu126 index)
- onnxruntime / onnxruntime-gpu 1.20.2 -> >=1.23.0 (resolves to the newest your Python supports)
- onnxruntime-silicon removed; regular onnxruntime supports Apple Silicon
- opencv-python-headless 4.10.0.84 -> 4.11.0.86 (last line that supports numpy 1.x)
- onnx 1.16.1 -> >=1.21.0, pydantic/tqdm/psutil/albucore now use minimum versions
- Kept: numpy 1.26.4 and insightface 0.7.3 (insightface 0.7.3 needs numpy 1.x and Python 3.11 or older)

### Code
- ui/main.py: `theme` and `css` moved from `gr.Blocks()` to `launch()` (Gradio 6 requirement)

### Project
- Added README, .gitignore, .dockerignore; slimmer non-root Dockerfile
- Installers, Colab notebook and README point to github.com/sagar4s/Mytestproject
- Removed empty placeholder files
