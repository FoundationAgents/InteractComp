import json
import re
from typing import Optional, Tuple

from core.engine.async_llm import AsyncLLM
from core.engine.logs import logger
from core.prompt import ASK_NL_GUARD_PROMPT


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


class AskNLValidator:
    def __init__(self, llm: Optional[AsyncLLM] = None):
        self.llm = llm

    async def check_question(self, question: str) -> Tuple[bool, str]:
        if not question or not question.strip():
            return False, "empty question"
        if self.llm is None:
            return False, "validator unavailable"
        prompt = ASK_NL_GUARD_PROMPT.format(question=question.strip())
        try:
            raw = await self.llm(prompt)
        except Exception as e:
            logger.error(f"AskNLValidator question check failed: {e}")
            return False, "validator error"
        data = _extract_json(raw)
        if not data:
            return False, "invalid validator output"
        ok = bool(data.get("ok"))
        reason = str(data.get("reason") or "").strip()
        if not reason:
            reason = "non-compliant question"
        return ok, reason

    def check_answer(self, answer: str) -> Tuple[bool, str]:
        if not answer or not answer.strip():
            return False, "empty answer"
        return True, "ok"
