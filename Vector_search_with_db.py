import os
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
from sentence_transformers import SentenceTransformer#向量模型
import chromadb
from chromadb.config import Settings

# 国内镜像（避免模型下载慢）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化ChromaDB（持久化，向量存本地）
client = chromadb.PersistentClient(path="./chroma_practice_db")
collection = client.get_or_create_collection(name="practice_report")

def init_chroma_db(text):
    """初始化向量库：只运行1次，把文本转向量存入Chroma"""
    # 切分文本（复用你现有的逻辑）
    module_anchors = ["实践体会", "校友寻访", "前期筹备"]
    for anchor in module_anchors:
        if anchor in text:
            text = text.replace(anchor, f"\n\n{anchor}")
    chunks = [c.strip() for c in text.split("\n\n") if c.strip() and len(c) > 20]
    if not chunks:
        return text
    # 转向量
    vectors = model.encode(chunks).tolist()

    # 存入向量库（ID自定义）
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(embeddings=vectors, documents=chunks, ids=ids)

def search(question, top_k=3):
    """检索：直接从Chroma查，无需重复转向量"""
    q_vector = model.encode([question]).tolist()
    results = collection.query(query_embeddings=q_vector, n_results=top_k)
    return "\n\n".join(results["documents"][0])