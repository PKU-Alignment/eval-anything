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


@BenchmarkRegistry.register('ARC')
class ARCBenchmark(T2TBenchmark):
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
        self.benchmark_name = 'ARC'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        """Convert a task list to a InferenceInput dict instances"""
        input_data = super().to_InferenceInput(task_list)

        num_to_char = {
            '1': 'A',
            '2': 'B',
            '3': 'C',
            '4': 'D',
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'D',
            'E': 'E',
        }
        for task, inference_input in input_data.items():
            for item in inference_input:
                item.ref_answer = num_to_char[item.ref_answer]

        return input_data
