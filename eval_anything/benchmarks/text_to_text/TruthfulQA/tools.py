from eval_anything.utils.register import JudgeRegistry
from eval_anything.evaluate_tools.base_tools import BaseTool
import numpy as np

@JudgeRegistry.register("judge_mc1")
class JudgeMC1(BaseTool):
    def __init__(self):
        super().__init__()
    
    def apply(self, scores, best_answer_index) -> bool:
        return True if scores['scores_true'][best_answer_index] > max(scores['scores_false']) else False
    
    def __call__(self, scores, best_answer_index) -> bool:
        return self.apply(scores, best_answer_index)
    
@JudgeRegistry.register("judge_mc2")
class JudgeMC2(BaseTool):
    def __init__(self):
        super().__init__()

    def apply(self, scores):
        scores_true = scores['scores_true']
        scores_false = scores['scores_false']

        probs_true = np.exp(scores_true)
        probs_false = np.exp(scores_false)
        probs_true = probs_true / (sum(probs_true) + sum(probs_false))
        
        return sum(probs_true)

    def __call__(self, scores):
        return self.apply(scores)