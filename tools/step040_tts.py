import json
import os
import re
import librosa

from loguru import logger
import numpy as np

from .utils import save_wav, save_wav_norm

# TTS 引擎全部懒加载，避免启动时因缺少依赖而崩溃
_xtts_tts = None
_cosyvoice_tts = None
_edge_tts = None
_minimax_tts = None
_f5tts_tts = None

def _load_tts(method):
    global _xtts_tts, _cosyvoice_tts, _edge_tts, _minimax_tts, _f5tts_tts
    if method == 'xtts' and _xtts_tts is None:
        from .step042_tts_xtts import tts as xtts_tts
        _xtts_tts = xtts_tts
    elif method == 'cosyvoice' and _cosyvoice_tts is None:
        from .step043_tts_cosyvoice import tts as cosyvoice_tts
        _cosyvoice_tts = cosyvoice_tts
    elif method == 'EdgeTTS' and _edge_tts is None:
        from .step044_tts_edge_tts import tts as edge_tts
        _edge_tts = edge_tts
    elif method == 'minimax' and _minimax_tts is None:
        from .step046_tts_minimax import tts as minimax_tts
        _minimax_tts = minimax_tts
    elif method == 'f5tts' and _f5tts_tts is None:
        from .step045_tts_f5tts import tts as f5tts_tts
        _f5tts_tts = f5tts_tts

def _is_available(method):
    try:
        _load_tts(method)
        return True
    except ImportError:
        return False

MINIMAX_AVAILABLE = _is_available('minimax')
F5TTS_AVAILABLE = _is_available('f5tts')
from .cn_tx import TextNorm
from audiostretchy.stretch import stretch_audio
normalizer = TextNorm()
def preprocess_text(text):
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text
    
    
def adjust_audio_length(wav_path, desired_length, sample_rate = 24000, min_speed_factor = 0.6, max_speed_factor = 1.1):
    # 检查文件是否存在
    if not os.path.exists(wav_path):
        logger.error(f'音频文件不存在: {wav_path}')
        # 检查是否有MP3版本
        mp3_path = wav_path.replace('.wav', '.mp3')
        if os.path.exists(mp3_path):
            wav_path = mp3_path
            logger.info(f'使用MP3文件: {wav_path}')
        else:
            logger.error(f'音频文件和MP3文件都不存在: {wav_path}')
            # 创建一个短的静音音频作为后备
            empty_audio = np.zeros(int(desired_length * sample_rate))
            return empty_audio, desired_length
    
    try:
        wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    except Exception as e:
        logger.error(f'加载音频文件失败: {wav_path}, 错误: {e}')
        # 尝试加载MP3文件
        if wav_path.endswith('.wav'):
            wav_path = wav_path.replace('.wav', '.mp3')
            try:
                wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
                logger.info(f'成功加载MP3文件: {wav_path}')
            except Exception as e2:
                logger.error(f'加载MP3文件也失败: {wav_path}, 错误: {e2}')
                # 创建静音音频作为后备
                empty_audio = np.zeros(int(desired_length * sample_rate))
                return empty_audio, desired_length
        else:
            # 创建静音音频作为后备
            empty_audio = np.zeros(int(desired_length * sample_rate))
            return empty_audio, desired_length
    current_length = len(wav)/sample_rate
    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    logger.info(f"Speed Factor {speed_factor}")
    desired_length = current_length * speed_factor
    if wav_path.endswith('.wav'):
        target_path = wav_path.replace('.wav', f'_adjusted.wav')
    elif wav_path.endswith('.mp3'):
        target_path = wav_path.replace('.mp3', f'_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    return wav[:int(desired_length*sample_rate)], desired_length

# 基础TTS支持语言
tts_support_languages = {
    # XTTS-v2 supports 17 languages: English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko) Hindi (hi).
    'xtts': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    'bytedance': [],
    'GPTSoVits': [],
    'EdgeTTS': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Polish', 'Spanish'],
    # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
    'cosyvoice': ['中文', '粤语', 'English', 'Japanese', 'Korean', 'French'],
    # Minimax commercial TTS API
    'minimax': ['中文', 'English', 'Japanese', 'Korean', 'French', 'Spanish', 'German', 'Italian', 'Portuguese'],
}

# 如果F5-TTS可用，添加支持
if F5TTS_AVAILABLE:
    tts_support_languages['f5tts'] = ['中文', 'English', 'Japanese', 'Korean', 'French', 'Spanish', 'German', 'Italian', 'Portuguese', 'Polish', 'Turkish', 'Russian', 'Dutch', 'Czech', 'Arabic', 'Hungarian', 'Hindi']

# 如果Minimax可用，已在上面添加支持

def generate_wavs(method, folder, target_language='中文', voice='zh-CN-XiaoxiaoNeural', **kwargs):
    assert method in ['xtts', 'bytedance', 'cosyvoice', 'EdgeTTS', 'f5tts', 'minimax']
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    speakers = set()
    
    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')

    if target_language not in tts_support_languages[method]:
        logger.error(f'{method} does not support {target_language}')
        return f'{method} does not support {target_language}'
        
    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        # if num_speakers == 1:
            # bytedance_tts(text, output_path, speaker_wav, voice_type='BV701_streaming')
        
        if method == 'bytedance':
            logger.warning("Bytedance TTS not implemented yet")
        elif method == 'xtts':
            _load_tts('xtts')
            _xtts_tts(text, output_path, speaker_wav, target_language = target_language)
        elif method == 'cosyvoice':
            _load_tts('cosyvoice')
            _cosyvoice_tts(text, output_path, speaker_wav, target_language = target_language)
        elif method == 'f5tts':
            if F5TTS_AVAILABLE:
                _load_tts('f5tts')
                success = _f5tts_tts(text, output_path, speaker_wav, target_language = target_language)
                if not success:
                    logger.error(f'F5-TTS生成失败: {text}')
                    empty_audio = np.zeros(int(0.5 * 24000))
                    save_wav(empty_audio, output_path)
                    logger.info(f'已创建静音文件: {output_path}')
            else:
                logger.error(f'F5-TTS not available. Please install with: pip install f5-tts')
                empty_audio = np.zeros(int(0.5 * 24000))
                save_wav(empty_audio, output_path)
                logger.info(f'已创建静音文件: {output_path}')
        elif method == 'EdgeTTS':
            _load_tts('EdgeTTS')
            success = _edge_tts(text, output_path, target_language = target_language, voice = voice)
            if not success:
                logger.error(f'EdgeTTS生成失败: {text}')
                empty_audio = np.zeros(int(0.5 * 24000))
                save_wav(empty_audio, output_path)
                logger.info(f'已创建静音文件: {output_path}')
        elif method == 'minimax':
            if MINIMAX_AVAILABLE:
                _load_tts('minimax')
                # 从kwargs中获取voice_id参数
                voice_id = kwargs.get('voice_id', None)
                success = _minimax_tts(text, output_path, speaker_wav, target_language=target_language, voice_id=voice_id, **kwargs)
                if not success:
                    logger.error(f'Minimax TTS生成失败: {text}')
                    # 创建一个空的音频文件以避免后续错误
                    empty_audio = np.zeros(int(0.5 * 24000))  # 0.5秒的静音
                    save_wav(empty_audio, output_path)
                    logger.info(f'已创建静音文件: {output_path}')
            else:
                logger.error(f'Minimax TTS not available. Please set MINIMAX_API_KEY in .env file')
                # 创建静音文件
                empty_audio = np.zeros(int(0.5 * 24000))  # 0.5秒的静音
                save_wav(empty_audio, output_path)
                logger.info(f'已创建静音文件: {output_path}')
        start = line['start']
        end = line['end']
        length = end-start
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        start = len(full_wav)/24000
        line['start'] = start
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            next_end = next_line['end']
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end-start)

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = start + length
        
    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    
    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)
    
    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    combined_wav = full_wav + instruments_wav
    # combined_wav /= np.max(np.abs(combined_wav))
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
    return os.path.join(folder, 'audio_combined.wav'), os.path.join(folder, 'audio.wav')

def generate_all_wavs_under_folder(root_folder, method, target_language='中文', voice='zh-CN-XiaoxiaoNeural', **kwargs):
    wav_combined, wav_ori = None, None
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            wav_combined, wav_ori = generate_wavs(method, root, target_language, voice, **kwargs)
        elif 'audio_combined.wav' in files:
            wav_combined, wav_ori = os.path.join(root, 'audio_combined.wav'), os.path.join(root, 'audio.wav')
            logger.info(f'Wavs already generated in {root}')
    return f'Generated all wavs under {root_folder}', wav_combined, wav_ori

if __name__ == '__main__':
    folder = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候'
    generate_wavs('xtts', folder)
