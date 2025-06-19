# -*- coding: utf-8 -*-
import json
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# 尝试导入translators，如果失败则提供替代方案
try:
    import translators as ts
    TRANSLATORS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"translators package not available: {e}")
    logger.warning("Google Translate and Bing Translate will not work. Please install Node.js or use other translation methods.")
    TRANSLATORS_AVAILABLE = False

def translator_response(messages, to_language = 'zh-CN', translator_server = 'bing'):
    if not TRANSLATORS_AVAILABLE:
        logger.error("translators package is not available. Please install Node.js first:")
        logger.error("Ubuntu/Debian: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt-get install -y nodejs")
        logger.error("Or use conda: conda install -c conda-forge nodejs")
        return f"Translation failed: Node.js required for {translator_server} translation."
    
    if '中文' in to_language:
        to_language = 'zh-CN'
    elif 'English' in to_language:
        to_language = 'en'
    translation = ''
    for retry in range(3):
        try:
            translation = ts.translate_text(query_text=messages, translator=translator_server, from_language='auto', to_language=to_language)
            break
        except Exception as e:
            logger.info(f'translate failed! {e}')
            print('tranlate failed!')
    return translation

if __name__ == '__main__':
    response = translator_response('Hello, how are you?', '中文', 'bing')
    print(response)
    response = translator_response('你好，最近怎么样？ ', 'en', 'google')
    print(response)