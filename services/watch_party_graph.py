from langgraph.graph import StateGraph, END
from typing import TypedDict

class WatchPartyState(TypedDict):
    roomId: str
    question: str
    context: str
    answer: str
    spoiler_free: bool

class WatchPartyGraph:
    def __init__(self):
        self.llm = LLMService()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(WatchPartyState)
        workflow.add_node("build_context", self._build_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("spoiler_check", self._spoiler_check)
        
        workflow.set_entry_point("build_context")
        workflow.add_edge("build_context", "generate_answer")
        workflow.add_edge("generate_answer", "spoiler_check")
        workflow.add_edge("spoiler_check", END)
        return workflow.compile()

    async def run(self, request):
        return await self.graph.ainvoke({
            "roomId": request.roomId,
            "question": request.userQuestion,
            "spoiler_free": request.spoilerProtection
        })