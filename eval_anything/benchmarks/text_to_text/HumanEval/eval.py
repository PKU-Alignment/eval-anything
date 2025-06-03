"""
继承自eval-anything.pipeline.t2t_task
"""

from collections import namedtuple

from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('HumanEval')
class HumanEvalBenchmark(T2TBenchmark):
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
        self.benchmark_name = 'HumanEval'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a InferenceInput dict instances"""
        input_data = super().to_InferenceInput(task_list)

        for task, inference_input in input_data.items():
            metadata = [item.metadata for item in inference_input]
        for inference_input_item, metadata_item in zip(inference_input, metadata):
            inference_input_item.ref_answer = metadata_item

        return input_data
