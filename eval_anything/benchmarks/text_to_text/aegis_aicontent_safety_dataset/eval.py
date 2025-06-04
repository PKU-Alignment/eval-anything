import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from eval_anything.benchmarks.text_to_text.aegis_aicontent_safety_dataset.utils import (
    EVALUATE_PROMPT,
)
from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.cached_requests import cached_requests
from eval_anything.utils.data_type import EvaluationResult, InferenceInput, InferenceOutput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


def gpt_evaluate(
    inference_inputs: list[InferenceInput],
    inference_outputs: list[InferenceOutput],
    model: str = 'gpt-4o-mini',
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Extract an answer from a model response for a specific item.

    Args:
        question: Question(jailbreak question)
        response: Model's response
        model: Name of the extractor model (e.g., "gpt-4o-mini")
        api_key: API key for the extractor model
        api_base: Base URL for the extractor model API
        cache_dir: Cache directory for the extractor model
    Returns:
        Extracted answer
    """
    api_key = os.getenv('API_KEY')
    api_base = os.getenv('API_BASE')
    num_workers = int(os.getenv('NUM_WORKERS', 32))

    def _single_request(inference_input: InferenceInput, inference_output: InferenceOutput) -> str:
        inference_input.metadata['prompt']
        response = inference_output.response
        user_prompt = EVALUATE_PROMPT.format(response=response)

        messages = [{'role': 'user', 'content': user_prompt}]

        extraction = cached_requests(
            messages=messages,
            model=model,
            max_completion_tokens=1024,
            temperature=0.0,
            api_key=api_key,
            api_base=api_base,
            cache_dir=cache_dir,
            timeout=300,
        )
        return extraction

    results = {}
    max_workers = min(len(inference_inputs), num_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_single_request, inference_input, inference_output): idx
            for idx, (inference_input, inference_output) in enumerate(
                zip(inference_inputs, inference_outputs)
            )
        }

        for future in tqdm(
            as_completed(future_to_index), total=len(inference_inputs), desc='Evaluating responses'
        ):
            idx = future_to_index[future]
            result = future.result()
            results[idx] = result

    return [results[i] for i in range(len(inference_inputs))]


@BenchmarkRegistry.register('aegis_aicontent_safety_dataset')
class aegis_aicontent_safety_dataset(T2TBenchmark):
    def __init__(
        self,
        model: BaseModel,
        eval_cfgs: namedtuple,
        model_cfgs: namedtuple,
        infer_cfgs: namedtuple,
        output_path: str,
        cache_manager: CacheManager,
        logger: EvalLogger,
    ):
        super().__init__(
            model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger
        )
        self.benchmark_name = 'aegis_aicontent_safety_dataset'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a list of InferenceInput dict instances"""
        # assert data_path == 'nvidia/Aegis-AI-Content-Safety-Dataset-2.0'
        # assert split in ["test"], "Invalid split name"
        dataset = load_dataset('nvidia/Aegis-AI-Content-Safety-Dataset-2.0', split='test')
        """
         ref: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
         This will say "REDACTED" if an ID exists in the reconstruction_id_if_redacted column for redacted samples from the Suicide Detection dataset.
         In this case The ID specified in the reconstruction_id_if_redacted column matches the sample in the Suicide dataset(https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?select=Suicide_Detection.csv).
         You need to separately download those samples.
        """
        dataset = dataset.filter(lambda x: x['prompt'] != 'REDACTED')
        # dataset = dataset.shuffle(seed=42).select(range(100))  # Limit to 1000 samples for testing

        inference_inputs = []
        for item in dataset:
            inference_inputs.append(
                InferenceInput(
                    task='aegis_aicontent_safety_dataset',
                    conversation=[{'role': 'user', 'content': item['prompt']}],
                    ref_answer=None,
                    metadata={
                        'prompt': item['prompt'],
                    },
                )
            )

        return {'aegis_aicontent_safety_dataset': inference_inputs}

    def to_InferenceOutput(
        self, inference_inputs: dict[str, list[InferenceInput]]
    ) -> dict[str, list[InferenceOutput]]:
        """Convert a InferenceInput dict instances to a InferenceOutput dict instances"""
        inference_outputs = super().to_InferenceOutput(inference_inputs)

        jailbreak_question = inference_inputs['aegis_aicontent_safety_dataset']
        llm_response = inference_outputs['aegis_aicontent_safety_dataset']
        # eval_response = gpt_evaluate(jailbreak_question, llm_response, cache_dir=self.eval_cfgs.cache_dir)

        """Evaluate the jailbreak question and llm response using llama model"""
        from .utils import LlamaNemoGuardJudge

        safety_judger = LlamaNemoGuardJudge()
        questions = [input.metadata['prompt'] for input in jailbreak_question]
        responses = [output.response[0] for output in llm_response]
        eval_response = safety_judger(questions, responses)

        for output_item, eval_item in zip(
            inference_outputs['aegis_aicontent_safety_dataset'], eval_response
        ):
            try:
                output_item.user_safety = eval_item['User Safety'].lower() == 'safe'
                output_item.response_safety = eval_item['Response Safety'].lower() == 'safe'
                output_item.safety_categories = eval_item['Safety Categories']
            except Exception:
                output_item.user_safety = None
                output_item.response_safety = None
                output_item.safety_categories = None
        return inference_outputs

    def to_EvaluationResult(
        self, task_list: list[str], inference_outputs: dict[str, list[InferenceOutput]]
    ) -> tuple[
        dict[str, list[EvaluationResult]], dict[str, dict[str, float]], dict[str, dict[str, float]]
    ]:
        """
        Convert a InferenceOutput dict instances to evaluation details, evaluation results, and
        overall results
        """
        user_safety_results = []
        response_safety_results = []
        safety_categories_results = []

        for output_item in inference_outputs['aegis_aicontent_safety_dataset']:
            if not output_item:
                continue
            if output_item.user_safety is not None:
                user_safety_results.append(output_item.user_safety)
            if output_item.response_safety is not None:
                response_safety_results.append(output_item.response_safety)
            if output_item.safety_categories is not None:
                safety_categories_results.append(output_item.safety_categories)

        user_safety_rate = sum(user_safety_results) / len(user_safety_results)
        response_safety_rate = sum(response_safety_results) / len(response_safety_results)

        self.display_benchmark_results(
            self.benchmark_name,
            {
                'aegis_aicontent_safety_dataset': {
                    'user_safety_rate': {'default': user_safety_rate},
                    'response_safety_rate': {'default': response_safety_rate},
                }
            },
        )

        return (
            inference_outputs,
            {
                'aegis_aicontent_safety_dataset': {
                    'user_safety_rate': {'default': user_safety_rate},
                    'response_safety_rate': {'default': response_safety_rate},
                }
            },
            {},
        )
