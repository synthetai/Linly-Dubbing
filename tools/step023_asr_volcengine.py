# -*- coding: utf-8 -*-
import os
import uuid
import time
import requests
from datetime import timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("minio package not installed. Run: pip install minio")


def _get_minio_client():
    endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY')
    secret_key = os.getenv('MINIO_SECRET_KEY')
    secure = os.getenv('MINIO_SECURE', 'false').lower() == 'true'

    if not access_key or not secret_key:
        raise ValueError("MINIO_ACCESS_KEY 和 MINIO_SECRET_KEY 未配置")

    # 去掉 endpoint 中可能携带的协议前缀
    endpoint = endpoint.replace('http://', '').replace('https://', '').rstrip('/')

    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _upload_audio(local_path: str) -> tuple[str, str, str]:
    """上传本地音频到 MinIO，返回 (presigned_url, bucket, object_name)"""
    if not MINIO_AVAILABLE:
        raise ImportError("minio 未安装，请执行 pip install minio")

    client = _get_minio_client()
    bucket = os.getenv('MINIO_BUCKET', 'linly-dubbing')

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"创建 MinIO bucket: {bucket}")

    ext = os.path.splitext(local_path)[1].lower()
    object_name = f"asr-tmp/{uuid.uuid4()}{ext}"

    client.fput_object(bucket, object_name, local_path)
    logger.info(f"音频已上传到 MinIO: {bucket}/{object_name}")

    url = client.presigned_get_object(bucket, object_name, expires=timedelta(hours=2))
    return url, bucket, object_name


def _delete_audio(bucket: str, object_name: str):
    try:
        client = _get_minio_client()
        client.remove_object(bucket, object_name)
        logger.info(f"已清理 MinIO 临时文件: {bucket}/{object_name}")
    except Exception as e:
        logger.warning(f"清理 MinIO 文件失败（可忽略）: {e}")


def _build_headers(task_id: str) -> dict:
    """根据配置构建请求头，兼容新旧版控制台"""
    resource_id = os.getenv('VOLCENGINE_ASR_RESOURCE_ID', 'volc.seedasr.auc')

    api_key = os.getenv('VOLCENGINE_ASR_API_KEY')
    if api_key:
        return {
            'X-Api-Key': api_key,
            'X-Api-Resource-Id': resource_id,
            'X-Api-Request-Id': task_id,
            'X-Api-Sequence': '-1',
            'Content-Type': 'application/json',
        }

    app_id = os.getenv('VOLCENGINE_ASR_APP_ID') or os.getenv('BYTEDANCE_APPID')
    access_token = os.getenv('VOLCENGINE_ASR_ACCESS_TOKEN') or os.getenv('BYTEDANCE_ACCESS_TOKEN')
    if not app_id or not access_token:
        raise ValueError("未配置火山引擎 ASR 凭证，请设置 VOLCENGINE_ASR_API_KEY 或 BYTEDANCE_APPID + BYTEDANCE_ACCESS_TOKEN")

    return {
        'X-Api-App-Key': app_id,
        'X-Api-Access-Key': access_token,
        'X-Api-Resource-Id': resource_id,
        'X-Api-Request-Id': task_id,
        'X-Api-Sequence': '-1',
        'Content-Type': 'application/json',
    }


def _submit_task(audio_url: str, task_id: str, audio_format: str,
                 language: str | None, diarization: bool) -> None:
    headers = _build_headers(task_id)

    payload = {
        'user': {'uid': 'linly-dubbing'},
        'audio': {
            'format': audio_format,
            'url': audio_url,
        },
        'request': {
            'model_name': 'bigmodel',
            'enable_itn': True,
            'enable_punc': True,
            'show_utterances': True,
            'enable_speaker_info': diarization,
        },
    }
    if language:
        payload['audio']['language'] = language
    if diarization:
        payload['request']['ssd_version'] = '200'

    resp = requests.post(
        'https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit',
        headers=headers,
        json=payload,
        timeout=30,
    )

    status = resp.headers.get('X-Api-Status-Code', '')
    message = resp.headers.get('X-Api-Message', '')
    if status != '20000000':
        raise RuntimeError(f"任务提交失败: {message} (code={status})")

    logger.info(f"ASR 任务已提交，task_id={task_id}")


def _poll_result(task_id: str, timeout: int = 300) -> dict:
    """轮询查询结果，返回完整 response body"""
    # 查询头不带 X-Api-Sequence
    headers = {k: v for k, v in _build_headers(task_id).items() if k != 'X-Api-Sequence'}

    interval = 3
    elapsed = 0

    while elapsed < timeout:
        time.sleep(interval)
        elapsed += interval

        resp = requests.post(
            'https://openspeech.bytedance.com/api/v3/auc/bigmodel/query',
            headers=headers,
            json={},
            timeout=30,
        )
        status = resp.headers.get('X-Api-Status-Code', '')
        message = resp.headers.get('X-Api-Message', '')

        if status == '20000000':
            logger.info(f"ASR 识别完成（{elapsed}s）")
            return resp.json()
        elif status in ('20000001', '20000002'):
            logger.info(f"识别中... ({elapsed}s / {timeout}s)")
            continue
        else:
            raise RuntimeError(f"查询失败: {message} (code={status})")

    raise TimeoutError(f"ASR 识别超时（>{timeout}s），task_id={task_id}")


def _parse_utterances(result: dict, diarization: bool) -> list[dict]:
    utterances = result.get('result', {}).get('utterances', [])
    transcript = []
    for utt in utterances:
        text = utt.get('text', '').strip()
        if not text:
            continue
        speaker_id = utt.get('speaker_id', 0) if diarization else 0
        transcript.append({
            'start': utt['start_time'] / 1000.0,
            'end': utt['end_time'] / 1000.0,
            'text': text,
            'speaker': f"SPEAKER_{int(speaker_id):02d}",
        })
    return transcript


def volcengine_transcribe_audio(wav_path: str, language: str | None = None,
                                diarization: bool = True) -> list[dict]:
    """
    使用火山引擎大模型录音文件识别 API 转写音频。

    :param wav_path: 本地音频文件路径（wav/mp3）
    :param language: 指定语言代码，如 'zh-CN'、'en-US'，None 表示自动检测
    :param diarization: 是否启用说话人分离
    :return: transcript 列表，格式同 WhisperX/FunASR
    """
    ext = os.path.splitext(wav_path)[1].lower().lstrip('.')
    audio_format = ext if ext in ('wav', 'mp3', 'ogg') else 'wav'

    task_id = str(uuid.uuid4())
    bucket = object_name = None

    try:
        audio_url, bucket, object_name = _upload_audio(wav_path)
        _submit_task(audio_url, task_id, audio_format, language, diarization)
        result = _poll_result(task_id)
        return _parse_utterances(result, diarization)
    finally:
        if bucket and object_name:
            _delete_audio(bucket, object_name)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'videos/test/audio_vocals.wav'
    transcript = volcengine_transcribe_audio(path)
    for seg in transcript:
        print(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
