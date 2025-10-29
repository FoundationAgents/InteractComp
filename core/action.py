from abc import abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from core.engine.logs import logger, LogLevel


class Action(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = None

    @abstractmethod
    async def __call__(self, **kwargs) -> str:
        """Execute the action with given parameters."""

    def to_param(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_text(self) -> str:
        """
        Return a human-readable text description of this action and its parameters.
        The output is concise and suitable for prompts or logs.
        """
        lines = [f"Action: {self.name}"]
        if self.description:
            lines.append(f"Description: {self.description}")

        schema = self.parameters or {}
        if isinstance(schema, dict) and schema.get("type") == "object":
            props: Dict[str, Any] = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            if props:
                lines.append("Parameters:")
                for pname, meta in props.items():
                    if not isinstance(meta, dict):
                        lines.append(f"- {pname}: {meta}")
                        continue
                    ptype = meta.get("type", "any")
                    pdesc = meta.get("description", "")
                    req = "required" if pname in required else "optional"
                    if pdesc:
                        lines.append(f"- {pname} ({ptype}, {req}): {pdesc}")
                    else:
                        lines.append(f"- {pname} ({ptype}, {req})")
            else:
                lines.append("Parameters: none")
        else:
            # Fallback string for non-object schemas or unexpected shapes
            lines.append("Parameters: none" if not schema else f"Parameters: {schema}")

        return "\n".join(lines)


class AskAction(Action):
    """Ask the user a closed-ended question (yes/no/i don't know)."""

    name: str = Field(default="ask")
    description: str = Field(default="Ask a clarifying question to the user.")
    responder: Optional[Any] = Field(default=None)
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": """
You can ask user to get more information. 
When asking, you can only get replies of yes, no, or i don't know, so please pay attention to how you phrase the question.
Example:
✅ Correct: "Is this question about technical matters?"
❌ Wrong: "What specifically do you want to know?"
"""
                }
            },
            "required": ["question"]
        }
    )

    async def __call__(self, *, question: str, query_id: str) -> str:  # type: ignore[override]
        logger.action(f"AskAction: {question}")
        if self.responder is not None:
            try:
                reply = await self.responder(question, query_id)
            except Exception as e:
                logger.error(f"AskAction responder failed: {e}")
                reply = "i don't know"
        logger.action(f"AskReply: {reply}")
        return reply


class SearchAction(Action):
    """Search externally using an injected search engine."""

    name: str = Field(default="search")
    description: str = Field(default="Search for information relevant to a query.")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web"
                }
            },
            "required": ["query"]
        }
    )

    search_engine: Optional[Any] = Field(default=None)

    async def __call__(self, *, query: str) -> str:  # type: ignore[override]
        logger.action(f"SearchAction: {query}")
        if self.search_engine is None:
            return "No search engine configured"
        try:
            results = await self.search_engine.search(query)
            lines = []
            for i, item in enumerate(results[:5], 1):
                title = item.get("title", "Untitled")
                snippet = item.get("snippet", "")
                source = item.get("source", "")
                lines.append(f"{i}. {title} — {snippet} (src: {source})")
            result = "\n".join(lines) if lines else "No results"
            logger.log_to_file(LogLevel.INFO, result)
            return result
        except Exception as e:
            logger.error(f"Search engine error: {e}")
            return "Search failed"


class AnswerAction(Action):
    """Return the final answer string."""

    name: str = Field(default="answer")
    description: str = Field(default="Provide the final answer when confident.")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer text to return"
                },
                "confidence":{
                    "type": "string",
                    "description": "The confidence level of the answer, from 0 to 100"
                }
            },
            "required": ["answer", "confidence"]
        }
    )

    async def __call__(self, *, answer: str, confidence: str, **kwargs) -> str:  # type: ignore[override]
        logger.action(f"Answer:{answer}, Confidence:{confidence}")

        return {"answer": answer.strip(), "confidence": str(confidence)}
