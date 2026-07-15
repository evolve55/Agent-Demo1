import os
from openai import OpenAI

# 注意: 不同地域的base_url不通用（下方示例使用北京地域的 base_url）
# - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 DASHSCOPE_API_KEY，请先配置")

#openai-python SDK初始化标准
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
         {'role': 'system', 'content': '你是一个话不多的数学专家，喜欢用简洁的语言回答问题。'},
         
        {'role': 'assistant', 'content': '你好！有什么数学问题需要我帮忙解答吗？'},
        {'role': 'user', 'content': '请问，什么是微积分？'}
        ],
        stream=True
)
for chunk in completion:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)