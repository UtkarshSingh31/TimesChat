import aiosqlite
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from state import AgentState

load_dotenv()

search_tool = TavilySearch(k=3)
tools = [search_tool]
tool_node = ToolNode(tools)

model = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    streaming=True 
).bind_tools(tools)

# print(model)

def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
