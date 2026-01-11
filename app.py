import json
import os
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

os.environ["GOOGLE_API_KEY"] = "AIzaSyDDrcVcpV-pjDUV_MO8wJMpCFVTrBA_i4A"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The history of messages"]
    lead_data: dict  


def mock_lead_capture(name, email, platform):
    print(f"\n[SYSTEM] Lead captured successfully: {name}, {email}, {platform}")
    return f"Success! Lead for {name} has been recorded."

def get_knowledge_base():
    with open("knowledge.json", "r") as f:
        return json.load(f)

def call_model(state: AgentState):
    knowledge = get_knowledge_base()
    prompt = f"""
    You are an AI assistant for AutoStream.
    Knowledge Base: {knowledge}
    
    Rules:
    1. Identify intent: Greeting, Inquiry, or High-intent [cite: 20-23].
    2. If High-intent, you MUST collect: Name, Email, and Creator Platform [cite: 45-50].
    3. Do not call the lead capture tool until you have all 3 pieces of info[cite: 54].
    """
    messages = [{"role": "system", "content": prompt}] + state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
app = workflow.compile()

if __name__ == "__main__":
    state = {"messages": [], "lead_data": {}}
    print("AutoStream Agent Active (Type 'exit' to quit)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit": break
        
        state["messages"].append(HumanMessage(content=user_input))
        output = app.invoke(state)
        state["messages"] = output["messages"]
        print(f"Agent: {output['messages'][-1].content}")