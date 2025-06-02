"""
继承自eval-anything.pipeline.mm_und_benchmark
"""

from collections import namedtuple

from eval_anything.models.base_model import BaseModel
from eval_anything.pipeline.mm_und_benchmark import MMUndBenchmark
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry


@BenchmarkRegistry.register('mathvision')
class mathvisionBenchmark(MMUndBenchmark):
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
        self.benchmark_name = 'mathvision'
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)
