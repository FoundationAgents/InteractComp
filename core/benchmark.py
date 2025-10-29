# InteractCompBenchmark, Only one class here.

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Tuple

import aiofiles
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from core.engine.logs import logger
from core.engine.utils import write_json_file
from core.engine.async_llm import AsyncLLM

from core.prompt import GRADING_PROMPT
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

class InteractCompBenchmark:
    
    def __init__(self, name: str, file_path: str, log_path: str, grader_llm):
        # basic identifiers and paths
        self.name = name
        self.file_path = file_path
        self.log_path = log_path

        # simple status constants (kept for compatibility)
        self.PASS = "PASS"
        self.FAIL = "FAIL"

        self.grader_llm = grader_llm
        

    async def load_data(self, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(self.file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data

    def save_results_to_csv(self, results: List[Tuple[Any, ...]], columns: List[str]):
        df = pd.DataFrame(results, columns=columns)
        # Normalize action_counts column to JSON string for readability and parsing convenience
        if "action_counts" in df.columns:
            def _to_json_str(v: Any) -> str:
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    try:
                        return repr(v)
                    except Exception:
                        return "{}"
                return repr(v) if v is not None else "{}"
            df["action_counts"] = df["action_counts"].apply(_to_json_str)
        avg_score = df["score"].mean()
        t_cost = df["cost"].max()
        a_cost = t_cost / len(df) if len(df) > 0 else 0
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{avg_score:.5f}_{current_time}.csv"
        output_file = os.path.join(self.log_path, filename)
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        return avg_score, a_cost, t_cost

    def log_mismatch(
        self,
        problem: str,
        expected_output: Any,
        prediction: str,
        extracted_output: Any,
        extract_answer_code: str = "None",
    ):
        log_data = {
            "question": problem,
            "right_answer": expected_output,
            "model_output": prediction,
            "extracted_output": extracted_output,
            "extract_answer_code": extract_answer_code,
        }
        log_file = Path(self.log_path) / "log.json"
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        data.append(log_data)
        write_json_file(log_file, data, encoding="utf-8", indent=4)

    async def evaluate_all_problems(self, data: List[dict], agent: Callable, max_concurrent_tasks: int = 50):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        async def sem_evaluate(problem):
            async with semaphore:
                return await self.evaluate_problem(problem, agent)

        tasks = [sem_evaluate(problem) for problem in data]
        return await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {self.name} problems", total=len(data))
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, agent, task: dict):
        return await agent(task)

    async def evaluate_problem(self, problem: dict, agent: Callable) -> Tuple[Any, ...]:
        
        question = problem["question"]
        correct_answer = problem.get("answer", "")
        
        logger.info(f"\n🎯 EVALUATING: {question}")
        
        try:
            question = problem["question"]
            correct_answer = problem.get("answer", "")
            
            predicted_answer, confidence, history, cost, action_counts = await self._generate_output(agent, problem)
            score = await self.calculate_score(question, correct_answer, predicted_answer)

            return question, correct_answer, predicted_answer, confidence, history, score, cost, action_counts
                
        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            print(f"❌ Evaluation Error: {e}")
            
            return question, correct_answer, "Error", 0, "Error", 0.0, 0.0, {}

    async def calculate_score(self, question: str, correct_answer: str, predicted_answer: str) -> float:
        """评估单个答案的正确性"""
        grading_prompt = GRADING_PROMPT.format(
            question=question,
            predicted_answer=predicted_answer,
            correct_answer=correct_answer
        )

        try:
            response = await self.grader_llm(grading_prompt)
            
            if "yes" in response.strip().lower():
                return 1.0
            elif "no" in response.strip().lower():
                return 0.0
            else:
                return 0.0
                    
        except Exception as e:
            logger.error(f"LLM grading failed: {e}")
            return 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "correct_answer", "predicted_answer", "confidence", "history", "score", "cost", "action_counts"]


    async def run_evaluation(self, agent: Callable, max_concurrent_tasks: int = 50):
        data = await self.load_data()
        results = await self.evaluate_all_problems(data, agent, max_concurrent_tasks)
        columns = self.get_result_columns()
        average_score, average_cost, total_cost = self.save_results_to_csv(results, columns)
        logger.info(f"Average score on {self.name} dataset: {average_score:.5f}")
        logger.info(f"Total Cost: {total_cost:.5f}")
        logger.info(f"Avg Cost:{average_cost:.5f}")
        return average_score, average_cost, total_cost
