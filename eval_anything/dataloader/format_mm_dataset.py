import re
from eval_anything.utils.register import MMDatasetRegistry
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import MultiChoicePromptBuilder, DialoguePromptBuilder
from eval_anything.dataloader.base_dataloader import TASK_TYPE_MAP
from eval_anything.utils.mm_data_manager import ImageManager
from datasets import Dataset
from typing import List
from collections import namedtuple

class BaseMMDataset:
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        self.bench_cfgs = bench_cfgs
        self.task = task
        self.enable_cot = enable_cot
        self.num_shot = num_shot
        self.few_shot_examples = []
        self.few_shot_mm_examples = []

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        for item in few_shot_dataset[:self.num_shot]:
            self.few_shot_examples.append({
                "question": item[self.task.question_key],
                "candidate_answers": item[self.task.answer_key],
                "ground_truth": item[self.task.ground_truth_key]
            })        

    def build_multi_choice_prompt(self, item: dict):
        self.prompt_builder = MultiChoicePromptBuilder(
            candidate_labels=self.task.candidate_labels,
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt

    def build_dialogue_prompt(self, item: dict):
        self.prompt_builder = DialoguePromptBuilder(
            few_shot_examples=self.few_shot_examples,
            cot=self.enable_cot
        )
        prompt = self.prompt_builder.build_prompt(item[self.task.question_key], item[self.task.answer_key])
        return prompt
    
    def _to_InferenceInput(self, dataset: Dataset):
        pass

    def __call__(self, dataset: Dataset):
        return self._to_InferenceInput(dataset)

@MMDatasetRegistry.register("mmmu")
class MMMUDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def get_image_indice(self, text: str)->List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError("MMMU does not support few-shot learning.")

    # refer: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py#L136
    def _to_InferenceInput(self, dataset: Dataset) -> List["InferenceInput"]:
        """
        Convert a dataset to a list of InferenceInput objects.
        
        Args:
            dataset: Dataset object containing questions, options, and images
            
        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []
        
        for item in dataset:
            question = item['question']
            if item['question_type'] == 'multiple-choice':
                options = eval(item['options'])
                example = ""
                letter_to_option = {}
                
                for idx, option in enumerate(options):
                    option_letter = chr(ord('A') + idx)
                    example += f"({option_letter}) {option}\n"
                    letter_to_option[option_letter] = option
                
                formatted_prompt = f"{question}\n\n{example}\n\nAnswer with the option's letter from the given choices directly."
            else:
                formatted_prompt = f"{question}\n\nAnswer the question using a single word or phrase."
            
            image_ids = self.get_image_indice(formatted_prompt)
            images = [item[f'image_{id}'] for id in image_ids]
            conversation = ImageManager.prompt_to_conversation(user_prompt=formatted_prompt, images=images)

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer']
                )
            )
        return inference_inputs
    
@MMDatasetRegistry.register("mathvision")
class mathvisionDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)


    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError("mathvision does not support few-shot learning.")

    # refer: https://github.com/mathllm/MATH-V/blob/main/models/Qwen-VL.py#L19
    def _to_InferenceInput(self, dataset: Dataset) -> List["InferenceInput"]:
        """
        Convert a dataset to a list of InferenceInput objects.
        
        Args:
            dataset: Dataset object containing questions, options, and images
            
        Returns:
            List of InferenceInput objects ready for model inference
        """
        inference_inputs = []
        
        
        for item in dataset:
            question = item['question']
            options = ''
            if len(item['options']) > 0:
                assert len(item['options']) == 5, item
                if ''.join(item['options']) != 'ABCDE':
                    options = f"(A) {item['options'][0]}\n(B) {item['options'][1]}\n(C) {item['options'][2]}\n(D) {item['options'][3]}\n(E) {item['options'][4]}\n"
            # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
            formatted_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
            formatted_prompt = formatted_prompt + "<image 1>" # to fullfill the requirement of the function  prompt_to_conversation
            images = [item['decoded_image']] # this benchmark will only use one image for one question
            
            conversation = ImageManager.prompt_to_conversation(user_prompt=formatted_prompt, images=images)

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=item['answer']
                )
            )
        return inference_inputs

@MMDatasetRegistry.register("olympiadbench")
class olympiadbenchDataset(BaseMMDataset):
    def __init__(self, bench_cfgs: namedtuple, task: namedtuple, enable_cot: bool, num_shot: int):
        super().__init__(bench_cfgs, task, enable_cot, num_shot)

    def set_few_shot_examples(self, few_shot_dataset: Dataset | None):
        raise NotImplementedError("olympiadbench does not support few-shot learning.")

    def get_image_indice(self, text: str)->List[int]:
        pattern = r'<image (\d+)>'
        matches = re.findall(pattern, text)
        return [int(num) for num in matches]


    # refer: https://github.com/mathllm/MATH-V/blob/main/models/Qwen-VL.py#L19
    def _to_InferenceInput(self, dataset: Dataset) -> List["InferenceInput"]:
        """
        Convert a dataset to a list of InferenceInput objects.
        
        Args:
            dataset: Dataset object containing questions, options, and images
            
        Returns:
            List of InferenceInput objects ready for model inference
        """
        
        # refer: https://github.com/OpenBMB/OlympiadBench/blob/main/inference/code/evaluators/evaluator.py
        chinese_answer_type_dict = {
            'Numerical': '数值',
            'Expression': '表达式',
            'Equation': '方程',
            'Interval': '区间'
        }
        english_answer_type_dict = {
            'Numerical': 'a numerical value',
            'Expression': 'an expression',
            'Equation': 'an equation',
            'Interval': 'an interval'
        }

        def get_single_answer_type_text(answer_type, is_chinese):
            if '-' in answer_type:    # No need now
                answer_type = answer_type[:answer_type.find('-')]
            for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
                if t in answer_type:
                    if is_chinese:
                        return chinese_answer_type_dict[t]
                    else:
                        return english_answer_type_dict[t]
            exit(f'Error parsing answer type {answer_type}!')
        
        def get_answer_type_text(answer_type, is_chinese, multiple_answer):
            if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
                full_answer_text = ''
            else:
                if not multiple_answer:
                    answer_text = get_single_answer_type_text(answer_type, is_chinese)
                    if is_chinese:
                        full_answer_text = f'，答案类型为{answer_text}'
                    else:
                        full_answer_text = f"The answer of The problem should be {answer_text}. "
                else:
                    if ',' not in answer_type:    # Same answer type for all answers
                        answer_text = get_single_answer_type_text(answer_type, is_chinese)
                        if is_chinese:
                            full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                        else:
                            full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                    else:
                        answer_types = answer_type.split(',')
                        answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                        if len(set(answer_types)) == 1:
                            answer_text = answer_types[0]
                            if is_chinese:
                                full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                            else:
                                full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                        else:
                            if is_chinese:
                                answer_text = '、'.join(answer_types)
                                full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
                            else:
                                answer_text = ', '.join(answer_types)
                                full_answer_text = f'The problem has multiple answers, with the answers in order being {answer_text}. '
            return full_answer_text
        
        def make_prompt(sample, is_chinese, is_math, is_theorem_proving):
            if is_chinese:
                subject_content = '数学' if is_math else '物理'
                if is_theorem_proving:
                    prompt = f'以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。证明过程中使用的变量和公式请使用LaTeX格式表示。'
                else:
                    answer_type_text = get_answer_type_text(sample['answer_type'], is_chinese=True, multiple_answer=sample['is_multiple_answer'])
                    if sample['is_multiple_answer']:
                        multiple_answer_text = '\\boxed{用英文逗号连接的多个答案}'
                    else:
                        multiple_answer_text = '\\boxed{答案}'
                    unit_text = ''
                    if sample['unit']:
                        multiple_answer_text += '(单位)'
                        unit_text = '，注意答案的单位不要放在\\boxed{}中'
                    prompt = f'以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以"所以最终答案是{multiple_answer_text}。"显式给出结果{unit_text}。'
            else:
                subject_content = 'Math' if is_math else 'Physics'
                if is_theorem_proving:
                    prompt = f'The following is a theorem proving problem from an International {subject_content} competition. Please use logical reasoning and common theorems to prove the proposition in the problem according to the given requirements. Please use LaTeX format to represent the variables and formulas used in the proof.'
                else:
                    if sample['is_multiple_answer']:
                        multiple_answer_text = '\\boxed{multiple answers connected with commas}'
                    else:
                        multiple_answer_text = '\\boxed{answer}'
                    unit_text = ''
                    if sample['unit']:
                        multiple_answer_text += '(unit)'
                        unit_text = ', note that the unit of the answer should not be included in \\boxed{}'
                    answer_type_text = get_answer_type_text(sample['answer_type'], is_chinese=False, multiple_answer=sample['is_multiple_answer'])
                    prompt = f'The following is an open-ended problem from an International {subject_content} competition. {answer_type_text}Please calculate the answer according to the given requirements and the information provided. Please use LaTeX format to represent the variables and formulas used in the solution process and results. Please end your solution with "So the final answer is {multiple_answer_text}." and give the result explicitly{unit_text}.'
            return prompt
        
        inference_inputs = []
        for item in dataset:
            is_chinese = item['language'] != 'English'
            is_math = item['subject'] == 'Math'
            is_theorem_proving = False # we won't test theorem proving in olympiadbench
            if is_chinese:
                subject = '数学' if is_math else '物理'
                system_prompt = f"你是一个中文人工智能助手。请根据要求，完成下面的{subject}竞赛题目。"
            else:
                subject = 'Math' if is_math else 'Physics'
                system_prompt = f'You are an AI assistant. Please answer the following {subject} competition problems as required.'
                
            formatted_prompt = make_prompt(item, is_chinese, is_math, is_theorem_proving)  + item['question']
            # original image placeholder is <image_1> , we need to transform it to <image 1> for the function prompt_to_conversation
            formatted_prompt = re.sub(r'<image_(\d+)>', r'<image \1>', formatted_prompt)
            
            image_ids = self.get_image_indice(formatted_prompt)
            images = [item[f'image_{id}'] for id in image_ids]
            conversation = ImageManager.prompt_to_conversation(user_prompt=formatted_prompt, system_prompt=system_prompt, images=images)

            inference_inputs.append(
                InferenceInput(
                    task=self.task.name,
                    conversation=conversation,
                    ref_answer=str(item['final_answer'][0])
                )
            )
        return inference_inputs