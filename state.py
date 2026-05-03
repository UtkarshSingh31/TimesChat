from pydantic import BaseModel
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of our graph.
    
    messages: 
        A list of messages (System, Human, AI, Tool). 
        The 'Annotated[..., add_messages]' part is CRITICAL. It tells 
        LangGraph that every time a node returns a 'messages' key, 
        it should be appended to the existing list in the database.
    """
    messages: Annotated[list[BaseMessage], add_messages]


class ChatRequest(BaseModel):
    """The structure of the JSON sent from your frontend."""
    message: str
    thread_id: str