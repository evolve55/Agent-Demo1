# doc_loader.py
import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

DATA_FOLDER = "D:/学习资料/大模型1/data_docs"

# ============================
# 工具1：加载所有 CSV（自动）
# ============================
def load_customers():
    docs = []
    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)
        if filename.endswith(".csv"):
            encodings = ["utf-8", "gbk", "gb2312", "latin-1", "utf-8-sig"]
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except:
                    continue
            if df is not None:
                for _, row in df.iterrows():
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                    docs.append(Document(page_content=content))
    return docs

# ============================
# 工具2：加载所有 政策文档（自动）
# ============================
def load_policy():
    docs = []
    for filename in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, filename)
        if filename.endswith((".txt", ".md", ".pdf", ".docx")):
            ext = os.path.splitext(path)[-1].lower()
            try:
                if ext in [".txt", ".md"]:
                    loader = TextLoader(path, autodetect_encoding=True)
                elif ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(path)
                else:
                    continue

                loaded = loader.load()
                docs.extend(loaded)
            except:
                continue
    return docs

# ============================
# 兼容旧函数
# ============================
def load_all_documents():
    docs = []
    docs.extend(load_customers())
    docs.extend(load_policy())
    return docs

