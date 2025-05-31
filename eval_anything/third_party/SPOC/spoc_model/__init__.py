from transformers import AutoConfig, AutoModel
from .early_fusion_tsfm_models import EarlyFusionCnnTransformer
from .early_fusion_tsfm_models import EarlyFusionCnnTransformerConfig
from .early_fusion_tsfm_models import EarlyFusionCnnTransformerAgent
globals()["EarlyFusionCnnTransformer"] = EarlyFusionCnnTransformer
globals()["EarlyFusionCnnTransformerAgent"] = EarlyFusionCnnTransformerAgent

AutoConfig.register("MM", EarlyFusionCnnTransformerConfig)
# AutoModel.register("MM", EarlyFusionCnnTransformer)