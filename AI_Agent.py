import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools, initialize_agent, AgentType

# 1. Load Environment Variables
# This looks for the .env file and loads the keys into the script.
load_dotenv()

# Check if key is loaded (Best practice for debugging)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

def run_agent():
    # 2. Initialize the LLM
    # Temperature=0 keeps the agent focused and less random.
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0
    )

    # 3. Define Tools
    # We combine DuckDuckGo (Search) with standard tools like Math.
    search_tool = DuckDuckGoSearchRun()
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(search_tool)

    # 4. Initialize the ReAct Agent
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True,
        handle_parsing_errors=True
    )

    # 5. Run a Test Query
    # This complex query forces the agent to use both Search and Math.
    question = "What is the current stock price of Apple (AAPL)? If I buy 15 shares, how much will it cost?"
    
    print(f"--- Agent Starting ---\nQuestion: {question}\n")
    response = agent.run(question)
    print(f"\n--- Agent Finished ---\nFinal Answer: {response}")

if __name__ == "__main__":
    run_agent()