# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Linly-Dubbing 是一个智能视频多语言 AI 配音和翻译工具。它通过集成多种 AI 模型实现视频下载、人声分离、语音识别、字幕翻译、语音合成和视频合成的完整工作流。

## 核心工作流程

项目采用模块化的步骤式处理流程，每个步骤对应一个 `tools/stepXXX_*.py` 文件：

1. **Step 000**: 视频下载 (`step000_video_downloader.py`) - 使用 yt-dlp 从 YouTube 等平台下载视频
2. **Step 010**: 人声分离 (`step010_demucs_vr.py`) - 使用 Demucs 分离人声和背景音乐
3. **Step 020**: 语音识别 (`step020_asr.py`) - 通过 WhisperX 或 FunASR 进行语音转文字
4. **Step 030**: 字幕翻译 (`step030_translation.py`) - 使用 LLM（GPT/Qwen）或 Google Translate 翻译字幕
5. **Step 040**: 语音合成 (`step040_tts.py`) - 通过 XTTS、CosyVoice、Edge TTS、F5-TTS 等生成配音
6. **Step 050**: 视频合成 (`step050_synthesize_video.py`) - 将新配音和字幕合成到视频中

完整的自动化流程由 `tools/do_everything.py` 协调执行。

## 主要命令

### 环境配置

```bash
# 创建 conda 环境
conda create -n linly_dubbing python=3.10 -y
conda activate linly_dubbing

# 安装 ffmpeg
conda install ffmpeg==7.0.2 -c conda-forge

# 安装 PyTorch (选择对应的 CUDA 版本)
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
conda install -y pynini==2.1.5 -c conda-forge
pip install -r requirements.txt
pip install -r requirements_module.txt
```

### 下载模型

```bash
# Linux
bash scripts/download_models.sh

# Windows
python scripts/modelscope_download.py

# 下载 wav2vec2 模型
wget -nc https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth \
    -O models/ASR/whisper/wav2vec2_fairseq_base_ls960_asr_ls960.pth
```

下载的模型包括：
- XTTS-v2 (TTS)
- Qwen1.5-4B-Chat (LLM 翻译)
- faster-whisper-large-v3 (ASR)

### 运行应用

```bash
# 启动 WebUI (默认端口 6006)
python webui.py

# 访问 http://127.0.0.1:6006
```

### 可选：安装 F5-TTS

```bash
python scripts/install_f5tts.py
pip install -r requirements_f5tts.txt
```

## 环境变量配置

复制 `env.example` 为 `.env` 并配置以下变量：

**必需配置**：
- `MODEL_NAME`: LLM 模型名称（默认 `qwen/Qwen1.5-4B-Chat`）
- `HF_TOKEN`: Hugging Face token（从 https://huggingface.co/settings/tokens 获取）

**可选配置**：
- `OPENAI_API_KEY`: OpenAI API 密钥（用于 GPT 翻译）
- `OPENAI_API_BASE`: 自部署 OpenAI 端点
- `GEMINI_API_KEY`: Google Gemini API 密钥（从 https://aistudio.google.com/app/apikey 获取，用于 Gemini 翻译）
- `GEMINI_MODEL_NAME`: Gemini 模型名称（默认 `gemini-3.5-flash`，可选 `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-2.0-flash`）
- `BAIDU_API_KEY` / `BAIDU_SECRET_KEY`: 百度文心一言 API（免费）
- `BYTEDANCE_APPID` / `BYTEDANCE_ACCESS_TOKEN`: 火山引擎 TTS
- `HF_ENDPOINT`: Hugging Face 镜像端点（如 `https://hf-mirror.com`）

**Speaker Diarization**：如需使用说话人分离功能，需在 https://huggingface.co/pyannote/speaker-diarization-3.1 申请访问权限。

## 代码架构

### 目录结构

```
Linly-Dubbing/
├── tools/               # 核心处理模块
│   ├── step000_*.py    # 视频下载
│   ├── step01*_*.py    # 人声分离
│   ├── step02*_*.py    # 语音识别 (ASR)
│   ├── step03*_*.py    # 字幕翻译
│   ├── step04*_*.py    # 语音合成 (TTS)
│   ├── step050_*.py    # 视频合成
│   ├── do_everything.py # 完整流程协调器
│   └── utils.py        # 工具函数
├── tabs/               # Gradio WebUI 标签页
│   ├── full_auto_tab.py    # 全自动处理
│   ├── download_tab.py     # 视频下载
│   ├── demucs_tab.py       # 人声分离
│   ├── asr_tab.py          # 语音识别
│   ├── translation_tab.py  # 字幕翻译
│   ├── tts_tab.py          # 语音合成
│   ├── video_tab.py        # 视频处理
│   └── settings_tab.py     # 设置
├── scripts/            # 辅助脚本
├── submodules/         # Git 子模块
│   ├── TTS/           # Coqui TTS (XTTS)
│   ├── demucs/        # 人声分离
│   ├── whisper/       # Whisper ASR
│   └── whisperX/      # WhisperX ASR
├── models/            # 模型存储目录
├── webui.py          # WebUI 入口
└── gui.py            # 替代 GUI 入口
```

### 模块化设计

每个处理步骤都被设计为独立模块，具有以下特点：

1. **stepXXX 系列模块**：每个步骤有主模块 (`step0X0_*.py`) 和实现变体 (`step0X1_*.py`, `step0X2_*.py` 等)
2. **命名规范**：
   - `step0X0_*.py`: 主接口/协调器
   - `step0X1_*.py`, `step0X2_*.py`: 具体实现（不同的 AI 模型/方法）
3. **模型初始化**：大型模型在 `do_everything.py` 中通过 `initialize_models()` 预加载，避免重复初始化

### TTS 引擎

支持的 TTS 方法（按质量排序）：
- `xtts`: Coqui XTTS-v2 (高质量语音克隆)
- `cosyvoice`: CosyVoice (多语言，阿里通义)
- `f5tts`: F5-TTS (Flow Matching，17+ 语言)
- `EdgeTTS`: Microsoft Edge TTS (快速，多语言)
- `minimax`: Minimax TTS (商业 API)
- `bytedance`: 字节跳动火山引擎

### ASR 引擎

- `WhisperX`: OpenAI Whisper 扩展，支持时间戳对齐和说话人分离
- `FunASR`: 阿里达摩院，优化中文识别

### 翻译引擎

- OpenAI API (GPT-4, GPT-3.5-turbo)
- Google Gemini (gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash)
- Qwen (本地 LLM)
- Ollama (本地 LLM 服务)
- Ernie Bot (百度文心一言)
- Google Translate

## YouTube Cookie 处理

当遇到 YouTube 登录验证或"确认您不是机器人"提示时：

```bash
# 从浏览器导出 cookies
yt-dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.youtube.com/watch?v=xxxx"
```

将 `cookies.txt` 放在项目根目录，或通过 WebUI 上传。也可以设置环境变量：

```bash
export YTDLP_COOKIES_BROWSER=chrome        # 或 firefox/edge
export YTDLP_COOKIES_PROFILE=Default       # 浏览器配置文件名
export YTDLP_COOKIES_KEYRING=None          # macOS keyring
export YTDLP_COOKIES_CONTAINER=None        # Firefox container
```

## 测试环境

- Python 3.10
- PyTorch 2.3.1
- CUDA 11.8 或 12.1
- ffmpeg 7.0.2

## 常见故障排查

### cuDNN 库加载错误

```bash
export LD_LIBRARY_PATH=`python3 -c 'import os; import torch; print(os.path.dirname(os.path.dirname(torch.__file__)) +"/nvidia/cudnn/lib")'`:$LD_LIBRARY_PATH
```

### WhisperX VAD 模型问题

运行修复脚本：

```bash
bash scripts/fix_whisperx_vad.sh
# 或
python scripts/fix_whisperx_vad.py
```

### 模型下载问题

1. 配置 HuggingFace 镜像：在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com`
2. 使用 ModelScope 镜像：`python scripts/modelscope_download.py`

## 子模块说明

项目使用 Git 子模块管理外部依赖：

```bash
# 初始化子模块
git submodule update --init --recursive
```

子模块包括：
- `submodules/TTS`: Coqui TTS (XTTS 模型)
- `submodules/demucs`: Facebook Demucs (人声分离)
- `submodules/whisper`: OpenAI Whisper
- `submodules/whisperX`: WhisperX (时间戳对齐)

## 数字人对口型技术

项目集成了 Linly-Talker 的数字人对口型技术（开发中），可实现配音与视频人物口型的精确同步。相关代码在 `tabs/linly_talker_tab.py`。

参考：https://github.com/Kedreamix/Linly-Talker

## Colab 在线体验

提供 Google Colab 脚本用于在线体验：
https://colab.research.google.com/github/Kedreamix/Linly-Dubbing/blob/main/colab_webui.ipynb
