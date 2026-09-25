# -*- coding: utf-8 -*-
"""
智能知识库 Agent（升级版）
能力：RAG知识库问答 + 查天气 + 查时间 + 计算器 + 多轮对话记忆
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
from datetime import datetime, timedelta, timezone
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能客服助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# ===== 自定义样式 =====
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    
    .feature-tags {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .tag {
        background: #f0f7ff;
        color: #1f77b4;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ===== 标题 =====
st.markdown('<h1 class="main-title">🤖 智能客服助手</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">会查资料 · 会写代码 · 会查天气 · 会记事</p>', unsafe_allow_html=True)

# 功能标签
st.markdown("""
<div class="feature-tags">
    <span class="tag">📚 知识库问答</span>
    <span class="tag">🌤️ 实时天气</span>
    <span class="tag">⏰ 时间查询</span>
    <span class="tag">🧮 计算器</span>
    <span class="tag">💬 多轮对话</span>
</div>
""", unsafe_allow_html=True)

# ===== 获取 API Key =====
def get_api_key():
    try:
        if "DEEPSEEK_API_KEY" in st.secrets:
            return st.secrets["DEEPSEEK_API_KEY"]
    except:
        pass
    return "sk-92ded79649df4621b901234f05381d60"

# ===== 不依赖知识库的工具（全局定义）=====
@tool
def get_current_time() -> str:
    """获取当前的日期和时间，当用户问"现在几点了"、"今天几号"时使用。"""
    # 东八区时间（北京时间）
    bj_tz = timezone(timedelta(hours=8))
    now = datetime.now(bj_tz)
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return f"现在是 {now.strftime('%Y年%m月%d日 %H:%M')}，星期{weekdays[now.weekday()]}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，当用户问"XX天气怎么样"、"XX今天多少度"时使用。"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        humidity = current["humidity"]
        desc_en = current["weatherDesc"][0]["value"]
        desc_map = {
            "Sunny": "晴", "Clear": "晴", "Partly cloudy": "多云",
            "Cloudy": "阴", "Overcast": "阴", "Mist": "薄雾",
            "Fog": "雾", "Light rain": "小雨", "Moderate rain": "中雨",
            "Heavy rain": "大雨", "Patchy rain possible": "局部有雨",
            "Smoky haze": "烟霾", "Light snow": "小雪",
        }
        desc = desc_map.get(desc_en, desc_en)
        return f"{city}当前天气：{desc}，温度 {temp}°C，湿度 {humidity}%"
    except Exception as e:
        return f"查询{city}天气失败：{str(e)}"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，比如 "3+5*2"、"100/4"，当用户问数学题时使用。"""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式只能包含数字和 +-*/() 符号"
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

# ===== 缓存：加载知识库和 Agent =====
@st.cache_resource(show_spinner="⏳ 正在初始化智能 Agent，请稍候...")
def load_agent():
    # 读文档
    with open("产品手册.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 切块
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_text(text)

    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"source": "产品手册", "chunk_id": i+1}
        )
        documents.append(doc)

    # embedding + 向量库（持久化）
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    else:
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_dir
        )

    # ===== 知识库工具（闭包访问 db）=====
    @tool
    def search_knowledge(question: str) -> str:
        """从产品知识库中搜索相关内容，当用户问关于产品功能、价格、售后、技术参数、部署方式等问题时使用。"""
        results = db.similarity_search(question, k=3)
        if not results:
            return "知识库中没有找到相关内容。"
        context = "\n\n".join([f"[资料{doc.metadata['chunk_id']}] {doc.page_content}" for doc in results])
        return f"从知识库检索到以下内容：\n{context}"

    # 初始化 LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=get_api_key(),
        base_url="https://api.deepseek.com"
    )

    # 创建 Agent
    tools = [search_knowledge, get_current_time, get_weather, calculator]
    agent = create_react_agent(llm, tools=tools)

    return agent

# ===== 加载 Agent =====
agent = load_agent()

# ===== 初始化聊天历史 =====
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的智能助手 🤖\n\n我可以帮你：\n- 📚 查询产品手册里的内容\n- 💻 写代码、解释代码、前后端设计建议\n- 🌤️ 查询任意城市的天气\n- ⏰ 查现在几点了\n- 🧮 算数学题\n- 💬 陪你聊天，回答各种问题\n\n有什么可以帮你的？"}
    ]

# ===== 显示历史消息 =====
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===== 用户输入 =====
if prompt := st.chat_input("💬 输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考..."):
            # 用 Agent 处理（传入完整对话历史，让它有记忆）
            result = agent.invoke({"messages": st.session_state.messages})
            answer = result["messages"][-1].content
            st.markdown(answer)

    # 保存到历史
    st.session_state.messages.append({"role": "assistant", "content": answer})
