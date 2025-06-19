# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# 加载环境变量
load_dotenv()

extra_body = {
    'repetition_penalty': 1.1,
}

def openai_response(messages):
    """
    使用OpenAI API进行翻译
    """
    try:
        # 获取模型名称，确保安全默认值
        model_name = os.getenv('MODEL_NAME', 'gpt-4o-mini')
        logger.info(f"使用OpenAI模型: {model_name}")
        
        # 验证模型名称，如果不是GPT模型则使用默认值
        if not model_name or 'gpt' not in model_name.lower():
            model_name = 'gpt-4o-mini'
            logger.warning(f"模型名称无效，使用默认模型: {model_name}")
        
        # 获取API配置
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("未设置OPENAI_API_KEY环境变量")
        
        base_url = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # 调用API
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            timeout=240,
            extra_body=extra_body
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API调用失败: {str(e)}")
        raise

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = openai_response(test_message)
    print(response)