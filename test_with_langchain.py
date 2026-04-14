import os
import dashscope
from http import HTTPStatus

# ================= 配置区域 =================
API_KEY = "sk-ac15eeac6afd436fab401c838478b3fc"
FILE_PATH = r"D:\学习资料\大模型1\实践报告.txt"  # 请确认路径正确

# 设置密钥
os.environ["DASHSCOPE_API_KEY"] = API_KEY
dashscope.api_key = API_KEY

# ================= 主程序逻辑 =================

try:
    # 导入另一个文件的函数
    from search_with_langchain import init_and_store, search

    # 1. 检查并读取文件
    if not os.path.exists(FILE_PATH):
        print(f"错误：找不到文件 '{FILE_PATH}'")
        exit()

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 2. 初始化数据库
    init_and_store(content)

    # 3. 开始问答循环
    print("\n" + "="*40)
    print("输入问题开始提问 (输入 q 退出)")
    print("="*40)

    while True:
        query = input("\n🙋 你的问题：").strip()
        if query.lower() in ['q', 'quit', 'exit']:
            break
        if not query:
            continue

        # A. 检索上下文
        context = search(query, k=3)
        if "未找到" in context or "出错" in context:
            print(f"💡 {context}")
            continue

        # B. 构造提示词
        prompt = f"""
        请严格根据以下【参考资料】回答问题。如果资料里没有答案，请直接说“不知道”。
        
        【参考资料】：
        {context}
        【问题】：
        {query}
        【回答】：
        """

        # C. 直接调用阿里云接口 (简化版，无需自定义类)
        print("思考中...", end="\r")
        response = dashscope.Generation.call(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            result_format='message'
        )

        # 清除"思考中"并输出
        print(" " * 15, end="\r") 
        
        if response.status_code == HTTPStatus.OK:
            answer = response.output.choices[0]['message']['content']
            print("-" * 30)
            print(answer)
            print("-" * 30)
        else:
            print(f"❌ 生成失败：{response.code} - {response.message}")

except ImportError as e:
    print(f"❌ 导入失败：{e}")
    print("💡 请确保当前目录下存在 'search_with_langchain.py' 文件！")
except Exception as e:
    print(f"❌ 运行出错：{e}")







    
except ImportError as e:
    print(f"❌ 导入其他模块失败：{e}")
except Exception as e:
    print(f"❌ 运行出错：{e}")