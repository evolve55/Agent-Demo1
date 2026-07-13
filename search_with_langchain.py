import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 配置数据库路径和集合名
DB_PATH = "./chroma_db"
COLLECTION_NAME = "customer_policy_collection"

# 初始化嵌入模型 (使用阿里云 API，避免本地下载模型导致的联网报错)
# 确保环境变量 DASHSCOPE_API_KEY 已设置
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

#文本切分，向量化，数据库的初始化以及存储的函数封装
def init_and_store(text_content: str):
    """
    初始化数据库并将文本存入。
    如果数据库已有数据，则跳过存入步骤。
    """
    # 1. 文本切片 (封装了langchain的文本切分逻辑)
    #text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # 递归切分器，适合层次化文本
    texts = text_splitter.split_text(text_content)
    documents = [Document(page_content=t) for t in texts]

    # 2. 连接/创建数据库（调用langchain的Chroma封装，底层自动处理向量化和存储）
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # 3. 判断是否为空，为空则写入，否则跳过
    if vector_store._collection.count() == 0:
        print("💾 数据库为空，正在构建索引...")
        vector_store.add_documents(documents)
        print(f"✅ 索引构建完成，共存入 {len(documents)} 个片段。")
    else:
        count = vector_store._collection.count()
        print(f"✅ 数据库已加载，现有 {count} 个片段 (跳过重复写入)。")

    return vector_store

def search(query: str, k: int = 3) -> str:
    """
    执行向量检索，返回拼接好的上下文文本。
    :param query: 用户问题
    :param k: 返回最相关的 k 条内容
    :return: 拼接后的字符串
    """
    try:
        # 重新连接数据库 (确保能读到最新数据)
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        # 执行检索
        #docs = vector_store.similarity_search(query, k=k)

        

        # 这个方法可以拿到每条结果的相似度分数，方便调试和过滤低相关内容
        docs_with_score = vector_store.similarity_search_with_score(query, k=k)
        valid_docs = [doc for doc, score in docs_with_score if score >= 0.7]
        
        

        
        if not valid_docs:
            return "未找到相关参考内容。"
        
        # 将检索到的内容拼接成字符串
        #context = "\n\n---\n\n".join([d.page_content for d in docs])
        context = "\n".join([d.page_content for d in valid_docs]) #如果原文本是按行切的，可以用换行拼接
        return context
        
    except Exception as e:
        return f"检索出错：{str(e)}"
