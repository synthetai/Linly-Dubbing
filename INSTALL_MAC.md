# Linly-Dubbing Mac 安装指南（Apple Silicon）

> 适用平台：Apple M1 / M2 / M3，macOS 13+  
> 本指南以 **VolcEngine ASR + EdgeTTS + 云端翻译** 为主线，完全绕开 CUDA 依赖。

---

## 前置条件

### 1. 安装 Homebrew（如未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装 Conda（如未安装）

```bash
brew install --cask miniconda
conda init zsh   # 或 conda init bash
# 重新打开终端使其生效
```

---

## 第一步：创建 Python 环境

项目要求 Python 3.10（系统自带的版本不兼容）：

```bash
conda create -n linly_dubbing python=3.10 -y
conda activate linly_dubbing
```

---

## 第二步：安装 FFmpeg

```bash
conda install ffmpeg==7.0.2 -c conda-forge
```

> ⚠️ 不要用 `brew install ffmpeg`，conda-forge 版本与项目依赖更兼容。

---

## 第三步：安装 PyTorch（Apple MPS 版本）

Mac 上**不要使用** README 里的 CUDA 安装命令，直接用标准版即可（会自动带 MPS 支持）：

```bash
pip install torch torchaudio
```

验证安装：

```bash
python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

正常输出示例：`2.x.x` 和 `MPS: True`

---

## 第四步：安装核心依赖

```bash
pip install \
    numpy==1.26.3 \
    transformers==4.39.3 \
    edge-tts \
    gradio \
    loguru \
    yt-dlp \
    scipy \
    python-dotenv \
    openai \
    google-genai \
    audiostretchy \
    librosa==0.10.2 \
    moviepy \
    requests \
    accelerate \
    modelscope \
    diffusers==0.27.2 \
    gdown==5.1.0 \
    pyarrow \
    wget==3.2 \
    translators \
    minio
```

> `minio` 是火山引擎 ASR 集成所需的对象存储客户端。

---

## 第五步：安装 Demucs（人声分离）

```bash
pip install -e submodules/demucs
```

> 项目里 Demucs 是通过 git 子模块管理的，但 pip 版本完全等效，在 Mac 上更易安装。

---

## 第六步：（可选）安装翻译库

如果需要使用 Google Translate / Bing Translate：

```bash
conda install -c conda-forge nodejs   # translators 包依赖 Node.js
pip install translators
```

---

## 第七步：克隆项目

```bash
git clone https://github.com/Kedreamix/Linly-Dubbing.git
cd Linly-Dubbing
```

> 无需执行 `git submodule update`，本指南不使用 whisperX / XTTS / CosyVoice 子模块。

---

## 第八步：配置环境变量

```bash
cp env.example .env
```

用编辑器打开 `.env`，填入以下必要项：

```dotenv
# ===== 火山引擎 ASR =====
# 新版控制台只需填 API Key
VOLCENGINE_ASR_API_KEY=
VOLCENGINE_ASR_RESOURCE_ID=volc.seedasr.auc

# ===== MinIO 存储（ASR 需要公网可访问的音频 URL）=====
MINIO_ENDPOINT=192.168.x.x:9000    # 不含 http://
MINIO_ACCESS_KEY=你的AccessKey
MINIO_SECRET_KEY=你的SecretKey
MINIO_BUCKET=linly-dubbing
MINIO_SECURE=false

# ===== 翻译 API（二选一或多选）=====
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1

GEMINI_API_KEY=你的GeminiKey
GEMINI_MODEL_NAME=gemini-2.5-flash

# ===== MiniMax TTS（可选）=====
MINIMAX_API_KEY=你的MinimaxKey
MINIMAX_GROUP_ID=你的GroupId
```

---

## 第九步：启动 WebUI

```bash
python webui.py
```

浏览器访问：http://127.0.0.1:6006

---

## 推荐工作流配置

在 WebUI 的各步骤中选择以下选项，完全兼容 Mac：

| 步骤 | 推荐选项 | 说明 |
|------|---------|------|
| 人声分离 | Demucs（默认） | MPS 加速，M3 约 1-3 分钟/视频 |
| **语音识别** | **VolcEngine** | 云端 API，不依赖 GPU |
| 字幕翻译 | OpenAI / Gemini | 云端 API |
| 语音合成 | **EdgeTTS** | 微软云端 TTS，速度快质量好 |
| 视频合成 | 默认 | 纯 ffmpeg，Mac 原生支持 |

---

## 不支持 / 不推荐在 Mac 使用的功能

| 功能 | 原因 |
|------|------|
| WhisperX ASR | ctranslate2 在 M 芯片上性能差，large-v3 模型极慢 |
| FunASR | WeTextProcessing 在 ARM Mac 上编译困难 |
| XTTS | 依赖 Coqui TTS 子模块，ARM 兼容性差 |
| CosyVoice | WeTextProcessing / conformer 在 ARM 上无法安装 |
| F5-TTS | 额外依赖复杂，未验证 |

---

## 常见问题

### Q: `import demucs` 报错找不到模块

```bash
pip install demucs
```

### Q: `librosa` 安装报错

```bash
conda install -c conda-forge librosa
```

### Q: `moviepy` 运行时报 ffmpeg 路径错误

```bash
# 确认 ffmpeg 来自 conda，不是系统路径
which ffmpeg   # 应该显示 conda 环境路径
```

### Q: MinIO 预签名 URL 火山引擎无法访问

MinIO 服务需要对外网络可达（即火山引擎的服务器能 HTTP 访问你的 MinIO）。  
如果 MinIO 部署在内网，需要通过内网穿透（如 frp、ngrok）或将 MinIO 部署在有公网 IP 的服务器上。

### Q: MPS 报错 fallback to CPU

部分算子 MPS 不支持，PyTorch 会自动降回 CPU，属正常现象，不影响结果。
