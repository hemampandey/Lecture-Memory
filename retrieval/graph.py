from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from retrieval.retriever import TemporalRetriever, RetrievalConfig
from retrieval.contradiction import check_contradiction
from openai import OpenAI
from config import config
from logger import logger

client = OpenAI(
    api_key=config.OPENAI_API_KEY,
    base_url=config.OPENAI_BASE_URL
)
retriever = TemporalRetriever()

class AgentState(TypedDict):
    query: str
    session_from: int
    session_to: int
    chunks: List[dict]
    contradiction: dict
    answer: str

def retrieve_node(state: AgentState):   #Fetch relevant lecture chunks
    config = RetrievalConfig(
        session_from=state.get("session_from"),
        session_to=state.get("session_to")
    )
    chunks = retriever.retrieve(state["query"], config)
    return {"chunks": chunks}

def analyze_node(state: AgentState):
    #check only if there are enough chunks and no specific session filter
    if len(state["chunks"]) > 1 and not state.get("session_from"):
        result = check_contradiction(state["query"])
        return {"contradiction": result}
    return {"contradiction": {"found": False}}

def synthesize_node(state: AgentState):
    context = ""
    for c in state["chunks"]:
        p = c["payload"]
        context += f"\n[Session {p['session_number']} | {p['timestamp_label']}]: {p['text'][:600]}"
    
    system_msg = "You are a PhD-level Teaching Assistant. Answer based ONLY on the excerpts provided. Cite session numbers."
    user_msg = f"Student Question: {state['query']}\n\nExcerpts:\n{context}"
    
    if state["contradiction"]["found"]:
        user_msg += f"\n\nCRITICAL: Mention this evolution/contradiction: {state['contradiction']['explanation']}"

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.3
    )
    return {"answer": response.choices[0].message.content}

workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("synthesize", synthesize_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", "synthesize")
workflow.add_edge("synthesize", END)

graph = workflow.compile()
