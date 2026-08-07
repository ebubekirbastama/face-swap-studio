# Face Swap Studio

Modern, local and CPU-focused desktop face-swap application built with Python, Tkinter, OpenCV and InsightFace.

> **Important:** This repository contains application code only. InsightFace pretrained models such as `buffalo_l` and `inswapper_128.onnx` are governed by InsightFace's own model licensing terms and are **not** covered by this repository's MIT License.

## Features

- Local desktop GUI; images are processed on the user's computer.
- CPU inference via ONNX Runtime.
- Source-face and target-image previews.
- Swap one face or all detected faces in a target image.
- Responsive desktop layout.
- Unicode-safe image loading/saving on Windows paths.
- Original, Full HD and 4K export presets.
- PNG, JPEG and WebP export.
- Background model loading so the UI stays responsive.
- Application log file under `~/.face_swap_studio/app.log`.
- Atomic model download using a temporary file before installation.

## Screenshot

![Main Interface](assets/screenshot.png)

## Requirements

- Python 3.10+
- Windows, Linux or macOS with Tk support
- Internet connection on the first run if required model files are not already installed

## Installation

```bash
git clone https://github.com/ebubekirbastama/face-swap-studio.git
cd face-swap-studio
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Linux / macOS

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

You can also run the package directly:

```bash
python -m src.face_swap_studio
```

## How to use

1. Select a clear source face.
2. Select the target photograph.
3. Choose whether the source identity should be applied to all detected target faces.
4. Click **Face Swap Uygula**.
5. Select an export resolution and save the result.

For better results, use a source photograph with a visible, front-facing face and adequate lighting.

## Output resolution

The 1080p and 4K options perform high-quality resizing. Upscaling a low-resolution input does not recreate real photographic detail; it only increases the output pixel dimensions.

## Models and licensing

This application uses the InsightFace ecosystem. The InsightFace project distinguishes between its source-code license and its pretrained model licenses. Their documentation states that pretrained models are intended for non-commercial research unless separate licensing is obtained. In particular, commercial licensing may be required for InSwapper and face-recognition model packages.

The model files are intentionally **not committed to this repository**. Review the upstream InsightFace license before using, redistributing, deploying, or commercializing any model.

Upstream project: `deepinsight/insightface` on GitHub.

## Responsible use

Face-swapping technology can be misused. Use this software only with images you are authorized to process and do not use it for impersonation, fraud, harassment, non-consensual intimate imagery, deceptive political/media content, or other unlawful or harmful purposes.

Outputs should be clearly disclosed as edited or synthetic when context could otherwise mislead viewers.

## Project structure

```text
face-swap-studio/
├── .github/
│   └── ISSUE_TEMPLATE/
├── assets/
├── src/
│   └── face_swap_studio/
│       ├── __init__.py
│       ├── __main__.py
│       └── app.py
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── main.py
├── pyproject.toml
└── requirements.txt
```

## Development

Install development tools:

```bash
pip install -r requirements-dev.txt
```

Run static checks:

```bash
ruff check .
python -m compileall src main.py
```

Format code:

```bash
ruff format .
```

## Contributing

Pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

Application code in this repository is released under the MIT License. See [LICENSE](LICENSE).

Third-party packages and model files retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
