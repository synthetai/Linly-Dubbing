#!/usr/bin/env python3
"""
Minimax TTS 集成模块
提供Minimax语音合成API的集成支持
基于ComfyUI节点代码改进
"""

import os
import json
import time
import binascii
import requests
from loguru import logger
from .utils import save_wav
import tempfile

# Minimax TTS 全局配置
minimax_api_key = None
minimax_api_base = "https://api.minimax.io/v1/t2a_v2"

def init_minimax_api():
    """初始化Minimax API配置"""
    global minimax_api_key

    minimax_api_key = (os.getenv('MINIMAX_API_KEY') or '').strip()

    if not minimax_api_key:
        logger.error("Minimax API Key未设置，请在.env文件中添加 MINIMAX_API_KEY")
        return False

    logger.info("Minimax API初始化成功")
    return True

# 语言映射配置（参考ComfyUI节点）
language_map = {
    '中文': 'Chinese',
    'English': 'English', 
    'Japanese': 'Japanese',
    'Korean': 'Korean',
    'French': 'French',
    'Spanish': 'Spanish',
    'German': 'German',
    'Italian': 'Italian',
    'Portuguese': 'Portuguese',
    '粤语': 'Chinese,Yue'
}

# Minimax TTS 模型配置 - 仅支持最新模型
TTS_MODEL = 'speech-2.6-hd'

# 默认音色ID（Minimax官方提供）
DEFAULT_VOICE_ID = 'cobra_design_20250717_162427_683071'  # 都市白领

def tts_with_minimax(text, voice_id, target_language='中文', speed=1.0, volume=1.0, pitch=0, emotion=""):
    """
    使用Minimax API进行文本转语音
    
    Args:
        text: 要合成的文本
        voice_id: 使用的声音ID
        target_language: 目标语言
        speed: 语速 (0.5-2.0)
        volume: 音量 (0.1-10.0) 
        pitch: 音高 (-12 to 12)
        emotion: 情感（可选）
    
    Returns:
        audio_data: 音频二进制数据
    """
    if not minimax_api_key:
        if not init_minimax_api():
            return None
    
    try:
        url = minimax_api_base
        headers = {
            "Authorization": f"Bearer {minimax_api_key}",
            "Content-Type": "application/json",
            "accept": "application/json, text/plain, */*"
        }
        
        # 构建voice_setting
        voice_setting = {
            "voice_id": voice_id,
            "speed": max(0.5, min(2.0, speed)),
            "vol": max(0.1, min(10.0, volume)),
            "pitch": max(-12, min(12, int(pitch)))
        }
        
        # 添加情感参数（如果提供）
        if emotion and emotion.strip():
            voice_setting["emotion"] = emotion
            logger.info(f"添加情感参数: {emotion}")
        
        payload = {
            "model": TTS_MODEL,  # 使用固定的模型
            "text": text,
            "voice_setting": voice_setting,
            "language_boost": "auto",
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            },
            "subtitle_enable": False,
            "output_format": "hex"  # 使用hex格式获取音频数据
        }
        
        logger.info(f"发送请求到: {url}")
        logger.info(f"使用音色ID: {voice_id}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        logger.info(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            try:
                resp_data = response.json()
            except json.JSONDecodeError:
                logger.error("无法解析JSON响应")
                return None
            
            # 检查API错误响应
            base_resp = resp_data.get("base_resp", {})
            status_code = base_resp.get("status_code")
            status_msg = base_resp.get("status_msg", "Unknown error")
            
            if status_code is not None and status_code != 0:
                logger.error(f"API错误 {status_code}: {status_msg}")
                return None
            
            data = resp_data.get("data", {})
            if not data:
                logger.error("API未返回数据")
                return None
            
            # 处理hex格式的音频数据
            audio_hex = data.get("audio", "")
            if not audio_hex:
                logger.error("未获取到音频数据")
                return None
            
            logger.info(f"收到音频hex数据长度: {len(audio_hex)}")
            audio_data = binascii.unhexlify(audio_hex)
            logger.info(f"解码后音频数据长度: {len(audio_data)}")
            
            return audio_data
        else:
            logger.error(f"TTS请求失败，状态码: {response.status_code}, 响应: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"请求错误: {str(e)}")
        return None
    except (ValueError, binascii.Error) as e:
        logger.error(f"数据处理错误: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Minimax TTS未知错误: {e}")
        return None

def tts(text, output_path, speaker_wav=None, target_language='中文', voice_id=None, **kwargs):
    """
    Minimax TTS主函数，兼容项目现有接口
    
    Args:
        text: 要合成的文本
        output_path: 输出音频文件路径
        speaker_wav: 参考音频文件路径（暂不支持自动语音克隆）
        target_language: 目标语言
        voice_id: 用户指定的音色ID
        **kwargs: 其他参数，如model, speed, volume, pitch, emotion
    """
    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return True
    
    if not init_minimax_api():
        logger.error("Minimax API初始化失败")
        return False
    
    # 确定使用的音色ID
    if not voice_id:
        # 如果未指定voice_id，使用默认音色
        voice_id = DEFAULT_VOICE_ID
        logger.info(f"使用默认音色ID: {voice_id} (都市白领)")
    else:
        logger.info(f"使用指定音色ID: {voice_id}")
    
    # 重试机制
    for retry in range(3):
        try:
            # 执行TTS
            audio_data = tts_with_minimax(
                text=text,
                voice_id=voice_id,
                target_language=target_language,
                speed=kwargs.get('speed', 1.0),
                volume=kwargs.get('volume', 1.0),
                pitch=kwargs.get('pitch', 0),
                emotion=kwargs.get('emotion', '')
            )
            
            if audio_data:
                # 直接保存MP3格式的音频数据
                # Minimax返回的是MP3格式，但我们需要转换为WAV
                try:
                    # 先保存为临时MP3文件
                    temp_mp3_path = output_path.replace('.wav', '_temp.mp3')
                    with open(temp_mp3_path, 'wb') as f:
                        f.write(audio_data)
                    
                    # 使用librosa转换为WAV格式
                    import librosa
                    import soundfile as sf
                    audio, sr = librosa.load(temp_mp3_path, sr=24000)
                    sf.write(output_path, audio, sr)
                    
                    # 清理临时文件
                    if os.path.exists(temp_mp3_path):
                        os.unlink(temp_mp3_path)
                    
                    logger.info(f'Minimax TTS生成成功: {text}')
                    return True
                    
                except Exception as e:
                    logger.warning(f"音频格式转换失败，保存为MP3格式: {e}")
                    # 如果转换失败，保存为MP3格式
                    mp3_output_path = output_path.replace('.wav', '.mp3')
                    with open(mp3_output_path, 'wb') as f:
                        f.write(audio_data)
                    logger.info(f'Minimax TTS生成成功(MP3格式): {text}')
                    return True
            else:
                logger.warning(f'Minimax TTS生成失败: {text} (第{retry+1}次尝试)')
                
        except Exception as e:
            logger.warning(f'Minimax TTS出错: {text} (第{retry+1}次尝试): {e}')
        
        if retry < 2:  # 不是最后一次重试
            logger.info(f'等待2秒后重试...')
            time.sleep(2)
    
    logger.error(f'Minimax TTS最终失败: {text}')
    return False

def get_default_voice_info():
    """
    获取默认音色信息
    
    Returns:
        voice_info: 默认音色信息
    """
    return {
        'voice_id': DEFAULT_VOICE_ID,
        'voice_name': '都市白领',
        'description': 'Minimax官方默认音色，适合大多数场景使用',
        'model': TTS_MODEL
    }

def test_voice_id(voice_id, test_text="这是一个测试"):
    """
    测试指定音色ID是否可用
    
    Args:
        voice_id: 要测试的音色ID
        test_text: 测试文本
        
    Returns:
        bool: 是否可用
    """
    try:
        audio_data = tts_with_minimax(test_text, voice_id, '中文')
        return audio_data is not None
    except Exception as e:
        logger.error(f"测试音色ID {voice_id} 失败: {e}")
        return False

if __name__ == '__main__':
    # 测试代码
    test_text = "这是一个测试文本，用于验证Minimax TTS功能。"
    test_output = "test_minimax_output.wav"
    
    # 确保设置了API Key
    if not os.getenv('MINIMAX_API_KEY'):
        print("请设置环境变量 MINIMAX_API_KEY 和 MINIMAX_GROUP_ID")
        print("示例:")
        print("export MINIMAX_API_KEY='your_api_key_here'")
        print("export MINIMAX_GROUP_ID='your_group_id_here'")
    else:
        # 测试默认音色
        print("测试默认音色...")
        success = tts(test_text, test_output, target_language='中文')
        if success:
            print(f"测试成功，输出文件: {test_output}")
        else:
            print("测试失败")
        
        # 显示默认音色信息
        print("\n默认音色信息:")
        voice_info = get_default_voice_info()
        print(f"  音色ID: {voice_info['voice_id']}")
        print(f"  音色名称: {voice_info['voice_name']}")
        print(f"  模型版本: {voice_info['model']}")
        print(f"  说明: {voice_info['description']}")
        
        print("\n注意: 您也可以使用自己通过Minimax语音克隆功能创建的音色ID")
