import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
from sentence_transformers import SentenceTransformer#向量模型
import numpy as np

# 初始化模型（只加载一次）
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(text: str, question: str, top_k=3):
    """
    无向量库的向量检索：向量只存在内存，纯计算相似度
    """
    # 1. 切分文本（不变）
    module_anchors = ["实践体会", "校友寻访", "前期筹备"]
    for anchor in module_anchors:
        if anchor in text:
            text = text.replace(anchor, f"\n\n{anchor}")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip() and len(c) > 20]
    if not chunks:
        return text
    
    # 2. 文本/问题转向量（存在内存）
    chunk_vectors = model.encode(chunks)  # 数组形式存在内存
    question_vector = model.encode([question])[0]
    
    # 3. 纯内存计算余弦相似度（核心：不用向量库，手动算）
    # 归一化向量（保证相似度计算准确）
    chunk_vectors = chunk_vectors / np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    question_vector = question_vector / np.linalg.norm(question_vector)
    # 计算每个段落向量与问题向量的相似度（值越大越相似）
    similarities = np.dot(chunk_vectors, question_vector)
    
    # 4. 取相似度最高的top_k个段落
    top_indices = similarities.argsort()[-top_k:][::-1]  # 按相似度降序排序
    relevant_chunks = [chunks[i] for i in top_indices if i < len(chunks)]
    
    return "\n\n".join(relevant_chunks)