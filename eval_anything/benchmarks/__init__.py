import importlib
import os
import sys


benchmark_modules = [
    'benchmarks.text_to_text.gsm8k.eval',
    'benchmarks.text_to_text.mmlu.eval',
    'benchmarks.text_to_text.HumanEval.eval',
    'benchmarks.text_to_text.AGIEval.eval',
    'benchmarks.text_to_text.TruthfulQA.eval',
    'benchmarks.text_to_text.ARC.eval',
    'benchmarks.text_to_text.CMMLU.eval',
    'benchmarks.text_to_text.MMLUPRO.eval',
    'benchmarks.text_to_text.CEval.eval',
    'benchmarks.text_to_text.DoAnythingNow.eval',
    # 来自 my-dev 的模块
    'benchmarks.text_to_text.AdvBench.eval',
    'benchmarks.text_to_text.Anthropics.eval',
    'benchmarks.text_to_text.BBQ.eval',
    'benchmarks.text_to_text.BeaverTails.eval',
    'benchmarks.text_to_text.CDialBias.eval',
    'benchmarks.text_to_text.Cona.eval',
    'benchmarks.text_to_text.CyberAttackAssistance.eval',
    # 来自 main 的模块
    'benchmarks.text_to_text.DoNotAnswer.eval',
    'benchmarks.text_to_text.HarmBench.eval',
    'benchmarks.text_to_text.RedEval.eval',
    'benchmarks.text_to_text.HExPHI.eval',
    'benchmarks.text_to_text.LatentJailbreak.eval',
    'benchmarks.text_to_text.gptfuzzer.eval',
    'benchmarks.text_to_text.llm_jailbreak_study.eval',
    'benchmarks.text_to_text.jbb_behaviors.eval',
    'benchmarks.text_to_text.salad_bench.eval',
    'benchmarks.text_to_text.air_bench_2024.eval',
    # 共同模块
    'benchmarks.text_image_to_text.mmmu.eval',
    'benchmarks.text_image_to_text.mathvision.eval',
    'benchmarks.text_audio_to_text.mmau.eval',
    'benchmarks.text_video_to_text.mmvu.eval',
    'benchmarks.text_vision_to_action.chores.eval',
]
script_name = os.path.basename(sys.argv[0])
if script_name == 'eval_vla.sh':
    benchmark_modules = [
        'benchmarks.text_vision_to_action.chores.eval',
    ]

for module in benchmark_modules:
    importlib.import_module(module)
