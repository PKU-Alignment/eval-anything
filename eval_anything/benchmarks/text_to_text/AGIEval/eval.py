"""
继承自eval-anything.pipeline.t2t_task
"""
from eval_anything.pipeline.t2t_benchmark import T2TBenchmark
from collections import namedtuple
from eval_anything.utils.data_type import InferenceInput
from eval_anything.models.base_model import BaseModel
from eval_anything.utils.logger import EvalLogger
from eval_anything.utils.register import BenchmarkRegistry
from eval_anything.utils.cache_manager import CacheManager
from eval_anything.utils.utils import _strip_string
from eval_anything.benchmarks.text_to_text.AGIEval.tools import RegexMatchMultiLetter

math_output_datasets = {}

@BenchmarkRegistry.register('AGIEval')
class AGIEvalBenchmark(T2TBenchmark):
    def __init__(self, 
                 model: BaseModel, 
                 eval_cfgs: namedtuple, 
                 model_cfgs: namedtuple, 
                 infer_cfgs: namedtuple, 
                 output_path: str,
                 cache_manager: CacheManager,
                 logger: EvalLogger):
        super().__init__(model, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger)
        self.benchmark_name = "AGIEval"
        self.benchmark_cfgs = self.get_benchmark_cfgs(self.benchmark_name)

    def to_InferenceInput(self, task_list: list[str]) -> dict[str, list[InferenceInput]]:
        dataloader = self.init_dataloader(self.eval_cfgs, self.benchmark_cfgs)
        
        special_task_list = []
        if 'gaokao-mathcloze' in task_list:
            special_task_list.append('gaokao-mathcloze')
        if 'math' in task_list:
            special_task_list.append('math')
        special_input_data = dataloader.load_dataset(special_task_list)
        for task, inference_input in special_input_data.items():
            for item in inference_input:
                item.ref_answer = _strip_string(item.ref_answer)
        other_task_list = [task for task in task_list if task not in special_task_list]
        other_input_data = dataloader.load_dataset(other_task_list)

        return other_input_data.update(special_input_data)