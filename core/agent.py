import json
import re
from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field

from core.engine.async_llm import AsyncLLM
from core.prompt import SYSTEM_PROMPT, ACT_PROMPT, FINAL_ROUND_ACT_PROMPT
from core.action import Action


class LinearMemory(BaseModel):
    """
    LinearMemory.contexts
    [
        {
            "observation": "",
            "action": ""       
        },
        {
            "observation": "",
            "action": ""
        }

    ]
    """
    contexts: List[Dict[str, Any]] = Field(default_factory=list)
    max_contexts: int = Field(default=100)

    def add_context(self, context: Dict[str, Any]) -> None:
        self.contexts.append(context)
        if len(self.contexts) > self.max_contexts:
            self.contexts = self.contexts[-self.max_contexts :]

    def add_contexts(self, contexts: List[Dict[str, Any]]) -> None:
        self.contexts.extend(contexts)
        if len(self.contexts) > self.max_contexts:
            self.contexts = self.contexts[-self.max_contexts :]

    def clear(self) -> None:
        self.contexts.clear()

    def get_recent_contexts(self, n: int) -> List[Dict[str, Any]]:
        return self.contexts[-n:]

    def get_all(self) -> List[Dict[str, Any]]:
        return self.contexts

    def to_text(self) -> str:
        lines = []
        for idx, ctx in enumerate(self.contexts):
            user = ctx.get("user_query")
            obs = ctx.get("observation", "")
            action = ctx.get("action", {})
            action_name = action.get("name", "")
            properties = action.get("properties", {})
            prop_str = ", ".join(f"{k}: {v}" for k, v in properties.items())

            lines.append(f"[Round {idx+1}]")
            if user:
                lines.append(f"User: {user}")
            lines.append(f"    Action: {action_name} ({prop_str})")
            lines.append(f"Observation: {obs}")
        return "\n".join(lines)

    @property
    def contexts_text(self) -> str:
        return self.to_text()


class ReActAgent(BaseModel):
    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Actions
    actions: List[Action] = Field(default=[], description="avaiable actions")

    # Dependencies
    llm: Optional[AsyncLLM] = Field(default=None, description="Language model instance (inject)")

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=1, description="Current step in execution")

    memory: LinearMemory = Field(default_factory=LinearMemory, description="Memory of the agent")
    last_invalid_feedback: Optional[str] = Field(default=None, description="Feedback for the last invalid action")
    invalid_action_total: int = Field(default=0, description="Count of invalid actions during the current run")
    max_invalid_retries: int = Field(default=3, description="Maximum consecutive invalid attempts before skipping")
    skip_due_to_invalid: bool = Field(default=False, description="Whether the current run should be skipped due to repeated invalid actions")
    # Enforcement for forced_ask mode: require at least N asks before answering (non-final rounds only)
    enforce_ask_min: Optional[int] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
    
    async def step(self, query_id: str = None) -> Tuple[bool, str]:
        """Perform a single ReAct step. Returns (is_answered, observation_or_answer)."""
        assert self.llm is not None, "AsyncLLM instance must be provided to ReActAgent"

        # Prepare utilities
        question = (self.memory.get_all()[0] or {}).get("user_query", "") if self.memory.get_all() else ""
        actions_map = {a.name: a for a in self.actions}
        answer_action = actions_map.get("answer")
        available_actions_text = ", ".join(actions_map.keys())
        # Build actions text
        all_actions_text = "\n\n".join([a.to_text() for a in self.actions]) if self.actions else ""
        answer_actions_text = answer_action.to_text() if answer_action else ""

        round_info = f"{min(self.current_step, self.max_steps)}/{self.max_steps}"

        # Final round uses only the answer action in system message
        is_final_round = self.current_step >= self.max_steps
        if is_final_round:
            self.llm.sys_msg = SYSTEM_PROMPT.format(actions=answer_actions_text, question=question)
            final_round_prompt = FINAL_ROUND_ACT_PROMPT.format(question=question, memory=self.memory.contexts_text, round_info=round_info)
            resp = await self.llm(final_round_prompt)
            parsed = await self.parse_actions(resp)
            if parsed.get("name") == "answer":
                params = parsed.get("params", {}) or {}
                if answer_action is not None:
                    final_answer = await answer_action(**params)
                else:
                    final_answer = params.get("answer", "")
                # Merge or append as round into memory
                if self.current_step == 1 and self.memory.contexts and "user_query" in self.memory.contexts[0] and "action" not in self.memory.contexts[0]:
                    self.memory.contexts[0].update({
                        "observation": final_answer,
                        "action": {"name": "answer", "properties": params},
                    })
                else:
                    self.memory.add_context({
                        "observation": final_answer,
                        "action": {"name": "answer", "properties": params},
                    })
                return True, final_answer
            content = parsed.get("params", {}).get("answer") or ""
            # Record even if parsing was imperfect
            if self.current_step == 1 and self.memory.contexts and "user_query" in self.memory.contexts[0] and "action" not in self.memory.contexts[0]:
                self.memory.contexts[0].update({
                    "observation": content or "No Answer",
                    "action": {"name": "answer", "properties": parsed.get("params", {})},
                })
            else:
                self.memory.add_context({
                    "observation": content or "No Answer",
                    "action": {"name": "answer", "properties": parsed.get("params", {})},
                })
            return True, content or "No Answer"

        invalid_retries = 0
        while True:
            round_info = f"{min(self.current_step, self.max_steps)}/{self.max_steps}"

            self.llm.sys_msg = SYSTEM_PROMPT.format(actions=all_actions_text, question=question)

            last_ctx = self.memory.get_recent_contexts(1)
            last_action_str = "" if not last_ctx else last_ctx[0].get("action", {}).get("name", "")
            last_observation_str = (
                self.last_invalid_feedback
                or ("" if not last_ctx else last_ctx[0].get("observation", ""))
            )
            self.last_invalid_feedback = None

            act_prompt = ACT_PROMPT.format(
                memory=self.memory.contexts_text,
                last_action=last_action_str,
                last_observation=last_observation_str,
                question=question,
                round_info=round_info,
            )
            resp = await self.llm(act_prompt)
            parsed = await self.parse_actions(resp)

            act_name = parsed.get("name")
            params = parsed.get("params", {}) or {}

            if (
                self.enforce_ask_min is not None
                and act_name == "answer"
                and self.current_step < self.max_steps
            ):
                ask_count = 0
                try:
                    for ctx in self.memory.get_all():
                        if (ctx or {}).get("action", {}).get("name") == "ask":
                            ask_count += 1
                except Exception:
                    ask_count = ask_count
                if ask_count < int(self.enforce_ask_min):
                    obs = (
                        f"Answer is not allowed before {self.enforce_ask_min} asks. "
                        f"Asks so far: {ask_count}/{self.enforce_ask_min}. Please use ask or search."
                    )
                    self.invalid_action_total += 1
                    invalid_retries += 1
                    self.last_invalid_feedback = obs
                    if invalid_retries > self.max_invalid_retries:
                        self.skip_due_to_invalid = True
                        return False, ""
                    continue

            if isinstance(act_name, str):
                act_name = act_name.strip()

            if not act_name or act_name not in actions_map:
                self.invalid_action_total += 1
                invalid_retries += 1
                feedback = "Invalid action."
                if available_actions_text:
                    feedback = f"Invalid action. Please choose from: {available_actions_text}."
                self.last_invalid_feedback = feedback
                if invalid_retries > self.max_invalid_retries:
                    self.skip_due_to_invalid = True
                    return False, ""
                continue
    
            action_obj = actions_map[act_name]
            if act_name == "ask":
                obs = await action_obj(**params, query_id=query_id)
            else:
                obs = await action_obj(**params)

            invalid_retries = 0
            self.last_invalid_feedback = None

            if act_name == "answer":
                # First round: merge into initial user_query context if present
                if (
                    self.current_step == 1
                    and self.memory.contexts
                    and "user_query" in self.memory.contexts[0]
                    and "action" not in self.memory.contexts[0]
                ):
                    self.memory.contexts[0].update({
                        "observation": obs,
                        "action": {"name": act_name, "properties": params},
                    })
                else:
                    self.memory.add_context({
                        "observation": obs,
                        "action": {"name": act_name, "properties": params},
                    })
                self.current_step += 1
                return True, obs

            if (
                self.current_step == 1
                and self.memory.contexts
                and "user_query" in self.memory.contexts[0]
                and "action" not in self.memory.contexts[0]
            ):
                # Merge first action + observation into the initial user_query context
                self.memory.contexts[0].update({
                    "observation": obs,
                    "action": {"name": act_name, "properties": params},
                })
            else:
                self.memory.add_context({
                    "observation": obs,
                    "action": {"name": act_name, "properties": params},
                })
            self.current_step += 1
            return False, obs

    async def run(self, query: str, query_id: str = None) -> str:
        """
        Run the agent with the given query.
        """
        self.current_step = 1
        self.skip_due_to_invalid = False
        self.invalid_action_total = 0
        self.last_invalid_feedback = None
        self.memory.clear()
        self.memory.add_context({"user_query": query})

        answered = False
        output = ""
        while (
            self.current_step <= self.max_steps
            and not answered
            and not self.skip_due_to_invalid
        ):
            answered, output = await self.step(query_id)

        if not answered and not self.skip_due_to_invalid:
            self.current_step = self.max_steps
            answered, output = await self.step(query_id)

        if self.skip_due_to_invalid:
            return ""

        return output if answered else "No Answer"
    
    async def parse_actions(self, llm_output: str) -> Dict[str, Any]:
        """Parse the LLM output into an action dict: {name, params}.

        The LLM is expected to return a JSON object:
        {
          "action": "<name>",
          "params": { ... }
        }
        This parser is defensive and extracts JSON from code blocks or text.
        """
        text = llm_output or ""

        def try_parse(obj_text: str) -> Optional[Dict[str, Any]]:
            try:
                data = json.loads(obj_text)
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
            return None

        data = try_parse(text)

        if data is None:
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                data = try_parse(m.group(1))

        if data is None:
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                data = try_parse(m.group(1))

        if not data:
            return {"name": "", "params": {}}

        action_name = data.get("action") or data.get("name") or ""
        params = data.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        return {"name": str(action_name), "params": params}

    async def __call__(self, **kwargs) -> Any:
        """Execute the agent with given parameters."""
        return await self.run(**kwargs)
