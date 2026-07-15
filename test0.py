"""
mport chromadb
import langchain
import pydantic_settings
print("ChromaDB版本：", chromadb.__version__)
print("LangChain版本：", langchain.__version__)
print("pydantic-settings版本：", pydantic_settings.__version__)
"""
import langchain
import langchain_core
import langsmith

print(f"LangChain: {langchain.__version__}")
print(f"LangChain Core: {langchain_core.__version__}")
print(f"LangSmith: {langsmith.__version__}")

# 简单测试导入
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
print("✅ 所有组件导入成功！版本兼容。")