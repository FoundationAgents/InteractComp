from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from core.engine.async_llm import AsyncLLM
from core.engine.logs import logger
from core.prompt import RESPONDER_PROMPT, RESPONDER_PROMPT_NL
from core.ask_nl_validator import AskNLValidator


class Responder(BaseModel):
    dataset_file: str = Field(default="data/InteractComp210.decrypted.jsonl")
    llm: Optional[AsyncLLM] = Field(default=None)
    response_mode: str = Field(default="closed")
    max_nl_retries: int = Field(default=0)
    validator: Optional[AskNLValidator] = Field(default=None)

    # Internal cache for id -> context
    context_map: Dict[int, str] = Field(default_factory=dict, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _ensure_loaded(self) -> None:
        if self.context_map:
            return
        path = Path(self.dataset_file)
        if not path.exists():
            logger.warning(f"Responder dataset not found: {path}")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        qid = item.get("id")
                        ctx = item.get("context", "")
                        if isinstance(qid, int) and isinstance(ctx, str):
                            self.context_map[qid] = ctx
                    except Exception:
                        continue
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")

    def _normalize(self, text: str) -> str:
        if not text:
            return "i don't know"
        t = text.strip().lower()
        t = t.splitlines()[0].strip()
        # Simple keyword mapping (EN/CN)
        if t in {"yes", "y", "sure", "correct", "true", "是", "是的", "对", "对的", "正确"}:
            return "yes"
        if t in {"no", "n", "false", "incorrect", "不是", "不对", "否", "错误"}:
            return "no"
        if t in {"idk", "i don't know", "i dont know", "不知道", "不清楚", "无法确定", "信息不足"}:
            return "i don't know"
        if "yes" in t:
            return "yes"
        if "no" in t:
            return "no"
        if "don't know" in t or "dont know" in t:
            return "i don't know"
        return "i don't know"

    def _build_prompt(self, context: str, question: str, use_nl: bool, retry_hint: str = "") -> str:
        if use_nl:
            hint = retry_hint.strip()
            if hint:
                hint = f"{hint}\n"
            return RESPONDER_PROMPT_NL.format(
                context=context,
                question=question,
                retry_hint=hint,
            )
        return RESPONDER_PROMPT.format(context=context, question=question)

    async def __call__(self, question: str, query_id: Any, response_mode: Optional[str] = None) -> str:
        # Ensure LLM is present
        if self.llm is None:
            logger.warning("Responder has no LLM injected; returning idk")
            return "i don't know"

        self._ensure_loaded()

        # parse query_id
        qid: Optional[int]
        if isinstance(query_id, int):
            qid = query_id
        else:
            try:
                qid = int(str(query_id).strip())
            except Exception:
                qid = None

        if qid is None or qid not in self.context_map:
            logger.warning(f"Responder: query_id not found: {query_id}")
            return "i don't know"

        context = self.context_map[qid]
        mode = (response_mode or self.response_mode or "closed").strip().lower()
        use_nl = mode in {"natural", "nl", "open"}
        prompt = self._build_prompt(context=context, question=question, use_nl=use_nl)

        try:
            if not use_nl:
                raw = await self.llm(prompt)
                return self._normalize(raw)

            retry_hint = ""
            last_raw = ""
            for attempt in range(self.max_nl_retries + 1):
                if retry_hint:
                    prompt = self._build_prompt(
                        context=context,
                        question=question,
                        use_nl=True,
                        retry_hint=retry_hint,
                    )
                raw = await self.llm(prompt)
                last_raw = raw
                if self.validator is None:
                    return raw.strip()
                ok, _reason = self.validator.check_answer(raw)
                if ok:
                    return raw.strip()
                retry_hint = "Previous reply was invalid. Reply again."

            return (last_raw or "").strip()
        except Exception as e:
            logger.error(f"Responder failed: {e}")
            return "i don't know"
