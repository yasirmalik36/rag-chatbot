"""
LLM Service — GPT-4o via LangChain with conversation memory.
"""

import logging
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant.

Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer fully, say so clearly.
Do NOT make up information or use knowledge outside the provided context.

Guidelines:
- Be direct and concise
- If quoting from context, mention the source
- If unsure, say "Based on the available information..."
- Always be helpful and professional

Context:
{context}
"""


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        logger.info(f"LLMService initialised — model={settings.OPENAI_MODEL}")

    def generate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer grounded in retrieved context.
        Returns {answer, tokens_used, model}.
        """
        context_text = self._format_context(context_chunks)
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context_text))

        messages = [system_msg]

        # Inject conversation history (last 6 turns max to stay within context window)
        for turn in (history or [])[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=question))

        logger.debug(f"Calling {settings.OPENAI_MODEL} with {len(messages)} messages")
        response = self.llm.invoke(messages)

        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)

        return {
            "answer": response.content,
            "tokens_used": tokens,
            "model": settings.OPENAI_MODEL,
        }

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['source']} | Score: {chunk['score']:.2f}]\n{chunk['content']}"
            )
        return "\n\n---\n\n".join(parts)
