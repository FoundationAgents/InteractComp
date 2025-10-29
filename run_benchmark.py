import asyncio
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Tuple, Dict

import time
import yaml

from core.engine.async_llm import create_llm_instance, LLMsConfig, AsyncLLM
from core.engine.search import create_search_engine
from core.engine.logs import logger
from core.action import AnswerAction, SearchAction, AskAction
from core.responder import Responder
from core.agent import ReActAgent
from core.benchmark import InteractCompBenchmark
from enum import Enum

class BenchmarkMode(Enum):
    ANSWER_ONLY = "answer_only"
    SEARCH_ONLY = "search_only"
    FULL = "full"
    FULL_WITH_CONTEXT = "full_with_context"
    FORCED_ASK = "forced_ask"

class AgentCallable:
    """Wraps ReActAgent creation per task to be concurrency-safe.

    Shares a single AsyncLLM across tasks to accumulate total cost consistently.
    Returns per-task history directly to avoid shared state issues.
    """

    def __init__(self, *, llm: AsyncLLM, search_engine, max_steps: int, mode: str, benchmark: InteractCompBenchmark, responder_model: Any, enforce_ask_min: int = 5):
        self.llm = llm
        self.search_engine = search_engine
        self.max_steps = max_steps
        self.mode = mode
        self.benchmark = benchmark
        self.enforce_ask_min = enforce_ask_min
        responder_llm = create_llm_instance(responder_model)
        responder_llm.config.temperature = 1.0
        self.responder = Responder(dataset_file=self.benchmark.file_path, llm=responder_llm)

    def _build_actions(self):
        if self.mode == BenchmarkMode.ANSWER_ONLY.value:
            return [AnswerAction()]
        elif self.mode == BenchmarkMode.SEARCH_ONLY.value:
            return [SearchAction(search_engine=self.search_engine), AnswerAction()]
        elif self.mode == BenchmarkMode.FULL.value or self.mode == BenchmarkMode.FULL_WITH_CONTEXT.value or self.mode == BenchmarkMode.FORCED_ASK.value:
            return [AnswerAction(), SearchAction(search_engine=self.search_engine), AskAction(responder=self.responder)]
        # default to full
        return [AnswerAction(), SearchAction(search_engine=self.search_engine), AskAction(responder=self.responder)]


    async def __call__(self, task: dict) -> Tuple[str, str, str, float, Dict[str, int]]:
        question = (task or {}).get("question", "")
        context = (task or {}).get("context", "")
        query_id = (task or {}).get("id")

        actions = self._build_actions()
        # Inject enforce_ask_min only for forced_ask mode
        if self.mode == BenchmarkMode.FORCED_ASK.value:
            agent = ReActAgent(name="benchmark_agent", actions=actions, llm=self.llm, max_steps=self.max_steps, enforce_ask_min=self.enforce_ask_min)
        else:
            agent = ReActAgent(name="benchmark_agent", actions=actions, llm=self.llm, max_steps=self.max_steps)

        # Merge context into the question when using full_with_context mode
        if self.mode == BenchmarkMode.FULL_WITH_CONTEXT.value and context:
            base_question = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        else:
            base_question = question

        # Prepend constraints hint for forced_ask mode
        if self.mode == BenchmarkMode.FORCED_ASK.value:
            constraints = (
                f"Constraints: You must perform at least {self.enforce_ask_min} ask actions before answering. "
                f"Early answers will be rejected. The final round must answer."
            )
            final_question = f"{constraints}\n\nQUESTION:\n{base_question}"
        else:
            final_question = base_question

        predicted = await agent.run(final_question, query_id=query_id)

        # normalize answer/confidence
        confidence = 0
        if agent.skip_due_to_invalid:
            predicted = ""
        elif isinstance(predicted, dict):
            try:
                confidence = str(predicted.get("confidence", 0) or 0)
            except Exception:
                confidence = 0
            predicted = str(predicted.get("answer", ""))
        elif isinstance(predicted, str):
            predicted = predicted
        else:
            predicted = str(predicted)
        # Use cumulative total cost so benchmark can pick max() as final total
        usage = self.llm.get_usage_summary() 
        total_cost = float(usage.get("total_cost", 0.0))

        history_text = agent.memory.contexts_text

        # Build dynamic action counts based on available action space and actual memory
        action_space = [a.name for a in actions] if actions else []
        action_counts: Dict[str, int] = {name: 0 for name in action_space}
        action_counts["invalid"] = 0
        try:
            for ctx in agent.memory.get_all():
                act = (ctx or {}).get("action", {}) or {}
                name = act.get("name")
                if not name:
                    continue
                if name in action_counts:
                    action_counts[name] += 1
                else:
                    action_counts["invalid"] += 1
        except Exception:
            # In case of any unexpected structure, keep counts as-in
            pass

        action_counts["invalid"] += getattr(agent, "invalid_action_total", 0)

        if agent.skip_due_to_invalid:
            action_counts = {"valid": True}

        return predicted, confidence, history_text, total_cost, action_counts

async def main():
    parser = argparse.ArgumentParser(description="Run InteractComp benchmark")
    parser.add_argument("--config", default="config/exp_config.yaml", help="Path to experiment config YAML")
    args = parser.parse_args()

    # Load experiment config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    llm_cfg = cfg.get("llms", {})
    experiment = cfg.get("experiment_setting", {})
    paths = cfg.get("file_paths", {})

    data_path = paths.get("data_path", "data/interactcomp-1.jsonl")

    timestamp = time.strftime("%Y%m%d%H%M")
    result_root = Path(paths.get("result_folder_path", "data/results")) / timestamp
    result_root.mkdir(parents=True, exist_ok=True)

    test_models = llm_cfg.get("test_llm", []) or []
    responder_model_config = LLMsConfig.default().get(llm_cfg.get("user_llm", "gpt-4o-mini"))
    grader_model_config = LLMsConfig.default().get(llm_cfg.get("user_llm", "gpt-4o-mini"))
    temperature = llm_cfg.get("temperature", 0.6)
    top_p = llm_cfg.get("top_p", 0.95)

    max_concurrency = int(experiment.get("max_concurrency", 10))
    mode = str(experiment.get("mode", "full")).strip().lower()
    enforce_ask_min = int(experiment.get("enforce_ask_min", 5))
    if mode == BenchmarkMode.ANSWER_ONLY.value:
        max_rounds= 1
    else:
        max_rounds = int(experiment.get("max_rounds", 5))

    # Strict validation for forced_ask: require at least enforce_ask_min asks + 1 answer round
    if mode == BenchmarkMode.FORCED_ASK.value and max_rounds < (enforce_ask_min + 1):
        raise ValueError(
            f"forced_ask mode requires at least {enforce_ask_min + 1} rounds "
            f"({enforce_ask_min} asks + 1 answer). Got max_rounds={max_rounds}."
        )

    # Dataset name for logging
    dataset_name = Path(data_path).stem

    async def run_single_model(model_name: str):
        logger.info(f"Starting benchmark on model: {model_name}")

        # Create benchmark with a dedicated grader LLM instance (avoid shared state across models)
        model_result_dir = result_root / model_name.replace("/", "_")
        model_result_dir.mkdir(parents=True, exist_ok=True)

        # Create a fresh grader per model to avoid shared usage state/rate interactions
        _grader = create_llm_instance(grader_model_config)

        benchmark = InteractCompBenchmark(
            name=f"{dataset_name}-{model_name}",
            file_path=data_path,
            log_path=str(model_result_dir),
            grader_llm=_grader,
        )

        # Create LLM for the test model
        try:
            llm_config = LLMsConfig.default().get(model_name)
            llm = create_llm_instance(llm_config)
            # Override sampling params from experiment config
            llm.config.temperature = temperature
            llm.config.top_p = top_p
        except Exception as e:
            logger.error(f"Failed to initialize LLM {model_name}: {e}")
            return None

        search_engine = create_search_engine()

        agent_callable = AgentCallable(
            llm=llm,
            search_engine=search_engine,
            max_steps=max_rounds,
            mode=mode,
            benchmark=benchmark,
            responder_model=responder_model_config,
            enforce_ask_min=enforce_ask_min,
        )

        return await benchmark.run_evaluation(agent_callable, max_concurrent_tasks=max_concurrency)

    # Run different models concurrently
    model_concurrency = int(experiment.get("model_concurrency", len(test_models) if test_models else 1))

    semaphore = asyncio.Semaphore(model_concurrency)

    async def sem_task(model_name: str):
        async with semaphore:
            return await run_single_model(model_name)

    results = await asyncio.gather(*(sem_task(m) for m in test_models), return_exceptions=True)

    # Aggregate and log per-model cost and performance
    summary_rows = []
    for model_name, res in zip(test_models, results):
        if isinstance(res, Exception) or res is None:
            logger.error(f"Model {model_name} failed or returned no result: {res}")
            continue
        try:
            avg_score, avg_cost, total_cost = res
        except Exception as e:
            logger.error(f"Unexpected result shape for {model_name}: {res} ({e})")
            continue
        summary_rows.append({
            "model": model_name,
            "avg_score": float(avg_score),
            "avg_cost": float(avg_cost),
            "total_cost": float(total_cost),
        })

    if summary_rows:
        logger.info("==== Models Summary (performance & cost) ====")
        for row in summary_rows:
            logger.info(
                f"{row['model']}: avg_score={row['avg_score']:.5f}, total_cost=${row['total_cost']:.4f}, avg_cost=${row['avg_cost']:.4f}"
            )

        # Persist summary files in the root result folder for this run
        summary_csv = result_root / "models_summary.csv"
        summary_json = result_root / "models_summary.json"
        try:
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["model", "avg_score", "avg_cost", "total_cost"])
                writer.writeheader()
                for row in summary_rows:
                    writer.writerow(row)
            with summary_json.open("w", encoding="utf-8") as f:
                json.dump(summary_rows, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved summary to {summary_csv} and {summary_json}")
        except Exception as e:
            logger.error(f"Failed to write summary files: {e}")


if __name__ == "__main__":
    asyncio.run(main())
