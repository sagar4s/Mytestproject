# Mytestproject

A Gradio-based face-swap and image-enhancement tool, based on
[roop-unleashed](https://codeberg.org/roop-unleashed/roop-unleashed) (version 4.4.1).
Licensed under the terms in [LICENSE](LICENSE).

Repository: https://github.com/sagar4s/Mytestproject

> **Responsible use:** only process images and video of people who have
> given their consent, and follow the laws that apply where you live.

## Requirements

- Python 3.11
- [ffmpeg](https://ffmpeg.org/) on your PATH
- An NVIDIA GPU with CUDA 12.4 is recommended (CPU and Apple Silicon also work, but slower)

## Quick start

```bash
git clone https://github.com/sagar4s/Mytestproject.git
cd Mytestproject
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Open the URL printed in the terminal (default port 7860).
Model files are downloaded automatically on first run, so the first start takes a while.

On macOS you can use `./runMacOS.sh`; on Windows see `installer/windows_run.bat`.

## Docker

```bash
docker build -t mytestproject .
docker run --rm -p 7860:7860 mytestproject
```

The image has no GPU support by default. For NVIDIA GPUs, use the NVIDIA Container Toolkit
and add `--gpus all`.

## Options

| Flag | Description |
| --- | --- |
| `--server_share` | Create a public Gradio share link |
| `--cuda_device_id N` | Choose which CUDA GPU to use |

Other settings (output format, video codec, theme, threads) live in `config.yaml`,
or in the Settings tab of the UI. `config_colab.yaml` is a preset for Google Colab.

## Project layout

```
run.py          entry point
roop/           core pipeline, face detection, processors (enhancers, masks, upscalers)
clip/           CLIP / CLIPSeg text-prompt masking
ui/             Gradio interface and tabs
installer/      one-click installers
```
