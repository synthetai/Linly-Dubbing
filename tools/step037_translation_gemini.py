# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from loguru import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    logger.error("请安装 google-genai 库: pip install google-genai")
    raise

# 加载环境变量
load_dotenv()

def gemini_response(messages):
    """
    使用 Google Gemini API 进行翻译

    Args:
        messages: 对话消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]

    Returns:
        str: Gemini 模型的响应内容
    """
    try:
        # 获取 API Key
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("未设置 GEMINI_API_KEY 环境变量")

        # 创建客户端
        client = genai.Client(api_key=api_key)

        # 获取模型名称，默认使用 gemini-3.5-flash
        model_name = os.getenv('GEMINI_MODEL_NAME', 'gemini-3.5-flash')
        logger.info(f"使用 Gemini 模型: {model_name}")

        # 转换消息格式
        # Gemini API 需要提取 system instruction 和 contents
        system_instruction = None
        conversation_contents = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                # Gemini 使用 system_instruction 参数
                system_instruction = content
            elif role == 'user':
                conversation_contents.append({
                    'role': 'user',
                    'parts': [{'text': content}]
                })
            elif role == 'assistant':
                conversation_contents.append({
                    'role': 'model',  # Gemini 使用 'model' 而不是 'assistant'
                    'parts': [{'text': content}]
                })

        # 准备配置
        config_params = {}
        if system_instruction:
            config_params['system_instruction'] = system_instruction

        config = types.GenerateContentConfig(**config_params)

        # 如果只有一条用户消息（最常见的情况）
        if len(conversation_contents) == 1 and conversation_contents[0]['role'] == 'user':
            # 直接发送内容
            contents = conversation_contents[0]['parts'][0]['text']
        else:
            # 多轮对话，需要传递完整的对话历史
            contents = conversation_contents

        # 调用 API
        response = client.models.generate_content(
            model=model_name,
            config=config,
            contents=contents
        )

        return response.text

    except Exception as e:
        logger.error(f"Gemini API 调用失败: {str(e)}")
        raise

if __name__ == '__main__':
    # 测试代码
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = gemini_response(test_message)
    print(response)
