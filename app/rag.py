from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.retrieval import HybridRetriever
from app.utils import citations_are_grounded


class RAGState(TypedDict, total=False):
    question: str
    chunks: list[dict]
    answer: str
    grounded: bool


SYSTEM_PROMPT = """You are a grounded document QA assistant.
Answer ONLY from the supplied evidence.
Every factual claim must include a page citation like [p.12].
Never cite a page that is not present in the evidence.
If the evidence is insufficient, say that the documents do not support an answer.
Be concise and do not invent facts."""


class RAGWorkflow:
    def __init__(self, retriever: HybridRetriever, settings: Settings):
        self.retriever = retriever
        self.settings = settings
        self.llm = self._build_llm()
        self.graph = self._build_graph()

    def _build_llm(self):
        if self.settings.model_provider.lower() == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
            return ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                temperature=0,
            )
        return ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=0,
        )

    def _retrieve(self, state: RAGState) -> RAGState:
        chunks = [c.to_dict() for c in self.retriever.search(state["question"])]
        return {"chunks": chunks}

    def _answer(self, state: RAGState) -> RAGState:
        chunks = state.get("chunks", [])
        if not chunks:
            return {"answer": "The indexed documents do not contain enough evidence to answer."}

        evidence = "\n\n".join(
            f'[p.{c["page"]} | {c["source"]}]\n{c["content"]}' for c in chunks
        )
        prompt = f"QUESTION:\n{state['question']}\n\nEVIDENCE:\n{evidence}"
        response = self.llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return {"answer": str(response.content)}

    def _grounding_guard(self, state: RAGState) -> RAGState:
        pages = [int(c["page"]) for c in state.get("chunks", [])]
        grounded = citations_are_grounded(state.get("answer", ""), pages)
        if not grounded and state.get("chunks"):
            available = sorted(set(pages))
            answer = (
                "I could not produce an answer that satisfied the citation grounding "
                f"contract. Retrieved evidence was available on pages {available}."
            )
            return {"answer": answer, "grounded": False}
        return {"grounded": grounded}

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.add_node("grounding_guard", self._grounding_guard)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", "grounding_guard")
        graph.add_edge("grounding_guard", END)
        return graph.compile()

    def ask(self, question: str) -> RAGState:
        return self.graph.invoke({"question": question})
