"""极简关键词检索：适配所有文本类型（知识点/文章）"""
from collections import Counter
import re

# 1. 自动提取文本核心关键词（不用手动写映射）
def extract_keywords(text: str) -> list:
    # 提取中文+英文关键词（过滤无用词）
    # 匹配中文+英文单词+数字
    words = re.findall(r"[\u4e00-\u9fa5]+|[\w]+", text)
    # 过滤停用词（可自己加）
    stop_words = ["的", "是", "在", "有", "和", "为", "我", "你", "它", "这", "那"]
    # 只保留长度>1的词
    words = [w for w in words if w not in stop_words and len(w) > 1]
    # 取出现次数最多的10个词（核心关键词）
    return [w for w, _ in Counter(words).most_common(10)]

# 2. 智能切分文本（自动适配行/段落）
def split_text(text: str) -> list:
    # 优先按空行切（文章），再按换行切（知识点）
    if "\n\n" in text:
        chunks = text.split("\n\n")  # 文章按段落切
    else:
        chunks = text.split("\n")    # 知识点按行切
    # 过滤空内容
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# 3. 核心检索函数（通用，不管什么文本都能用）
def search(text: str, question: str) -> str:
    """
    通用检索：输入完整文本+用户问题，返回相关内容
    :param text: 你的知识库文本（知识点/文章）
    :param question: 用户提问
    :return: 匹配的相关内容
    """
    # 步骤1：切分文本成小片段
    chunks = split_text(text)
    if not chunks:
        return text
    
    # 步骤2：提取文本的核心关键词（自动生成映射）
    text_keywords = extract_keywords(text)
    
    # 步骤3：找问题里和文本关键词匹配的词
    matched_kw = []
    question_lower = question.lower()
    for kw in text_keywords:
        if kw.lower() in question_lower:
            matched_kw.append(kw)
    
    # 步骤4：匹配包含关键词的片段
    relevant_chunks = []
    if matched_kw:
        # 只保留包含任意一个匹配关键词的片段
        for chunk in chunks:
            if any(kw in chunk for kw in matched_kw):
                relevant_chunks.append(chunk)
    else:
        # 没匹配到就返回全文
        relevant_chunks = chunks
    
    # 拼接结果（按原格式）
    return "\n\n".join(relevant_chunks) if "\n\n" in text else "\n".join(relevant_chunks)