from typing import Any, Dict, List
from eval_anything.utils.data_type import InferenceInput, InferenceOutput
from eval_anything.utils.template import *

MODEL_MAP = {
    "vllm_LM": "vllm_lm",
    "vllm_MM": "vllm_mm",
    "hf_LM": "hf_lm",
    "hf_MM": "hf_mm",
    "hf_VLA": "hf_vla",
    "api_LM": "api_lm",
    "api_MM": "api_mm",
}
CLASS_MAP = {
    "vllm_LM": "vllmLM",
    "vllm_MM": "vllmMM",
    "hf_LM": "HFLM",
    "hf_MM": "HFMM",
    "hf_VLA": "HFVLA",
    "api_LM": "APILM",
    "api_MM": "APIMM",
}
class BaseModel:
    def __init__(self, model_cfgs: Dict[str, Any]):
        self.model_cfgs = model_cfgs
        self.init_model()

    def init_model(self):
        pass
    
    def generation(self, inputs: Dict[str, List[InferenceInput]]) -> Dict[str, List[InferenceOutput]]:
        pass
    
    def _generation(self, inputs: Dict[str, List[InferenceInput]]) -> Dict[str, List[InferenceOutput]]:
        pass

    def shutdown_model(self):
        pass
