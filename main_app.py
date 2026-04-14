import os
from openai import OpenAI
from search_with_langchain import init_and_store, search
from doc_loader import load_policy, load_customers

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 DASHSCOPE_API_KEY，请先配置")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = """
你是专业的智能销售助理，严格按照以下规则回答：
1. 先判断问题类型：商品/订单/消费使用交易数据；运费/折扣/售后/政策使用政策文档。
2. 问题模糊则主动追问。
3. 只使用提供的资料，不编造、不猜测。
4. 回答自然、口语化、简洁专业。
"""

policy_content = load_policy()
csv_content = load_customers()

policy_str = "\n\n".join([doc.page_content for doc in policy_content])
csv_str = "\n\n".join([doc.page_content for doc in csv_content])



init_and_store(policy_str)
init_and_store(csv_str)

#工具选择模块（根据问题选用什么工具）
def get_relevant_content(question):
    question = question.lower()
    policy_keywords = ["包邮", "运费", "折扣", "售后", "发货", "政策", "退换", "退款", "保障", "多久", "规则"]
    csv_keywords = ["卫衣", "商品", "订单", "买过", "客户", "消费", "价格", "推荐", "多少钱", "卖"]

    for kw in policy_keywords:
        if kw in question:
            return policy_content

    for kw in csv_keywords:
        if kw in question:
            return csv_content

    return ""

while True:
    user_question = input("\n你的问题（输入退出结束）：")
    if user_question == "退出":
        break

    context = get_relevant_content(user_question)
    relevant_content = search(user_question, 5)

    completion = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": system_prompt + f"\n参考资料：{relevant_content}"},
            {"role": "user", "content": user_question}
        ],
        stream=True
    )

    print("智能助手：", end="")
    for chunk in completion:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            print(delta.content, end="", flush=True)

