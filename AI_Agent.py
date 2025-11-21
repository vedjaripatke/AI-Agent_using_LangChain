import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from langchain import hub

# 1. Page Config
st.set_page_config(page_title="LangChain AI Agent", page_icon="🤖")
st.title("🤖 AI Agent with Search & Math")

# 2. Handle API Key (Securely for Streamlit Cloud)
# Try loading from Streamlit secrets first, otherwise ask user
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API Key to continue.")
    st.stop()

# 3. Initialize the LLM
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

    # 4. Define Tools
    # DuckDuckGo for search
    search_tool = DuckDuckGoSearchRun()
    
    # Math tool (requires llm-math from langchain-community)
    # We use the updated import path for load_tools
    math_tool = load_tools(["llm-math"], llm=llm)[0]
    
    tools = [search_tool, math_tool]

    # 5. Pull the Prompt Template
    # This pulls the standard "ReAct" prompt from LangChain Hub (standard v0.1+ method)
    prompt = hub.pull("hwchase17/react")

    # 6. Create the Agent
    agent = create_react_agent(llm, tools, prompt)

    # 7. Create the Agent Executor (The runtime)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 8. Chat Interface
    user_input = st.text_input("Ask a question (e.g., 'Price of AAPL * 5?')")

    if st.button("Run Agent") and user_input:
        with st.spinner("Thinking..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                st.success(response["output"])
            except Exception as e:
                st.error(f"An error occurred: {e}")

except Exception as e:
    st.error(f"Setup Error: {e}")
