
import json
import os
import sys
import torch
import random
import ai2thor.platform
import numpy as np
import traceback

from typing import Dict, List, Optional, Any, Literal
from collections import namedtuple
from itertools import chain
from tqdm import tqdm


import matplotlib.pyplot as plt
from eval_anything.pipeline.base_benchmark import BaseBenchmark
from eval_anything.dataloader.tv2act_dataloader import TV2ACTDataLoader
from eval_anything.utils.data_type import EvaluationResult
from eval_anything.utils.data_type import InferenceInput
from eval_anything.utils.utils import pair_data_via_uuid
from eval_anything.utils.register import BenchmarkRegistry

from third_party.SPOC.environment.stretch_controller import StretchController
from third_party.SPOC.utils.constants.stretch_initialization_utils import (
    STRETCH_ENV_ARGS,
)
from third_party.SPOC.tasks.abstract_task import AbstractSPOCTask
from third_party.SPOC.spoc_model.agent import AbstractAgent
from third_party.SPOC.spoc_model import REGISTERED_MODELS
from third_party.SPOC.tasks.task_specs import TaskSpecList
from third_party.SPOC.tasks.multi_task_eval_sampler import MultiTaskSampler
from third_party.SPOC.utils.visualization_utils import get_top_down_frame, VideoLogging
from third_party.SPOC.utils.type_utils import THORActions
from third_party.SPOC.utils.data_generation_utils.mp4_utils import save_frames_to_mp4

from third_party.SPOC.utils.task_datagen_utils import (
    get_core_task_args,
)
from third_party.SPOC.utils.online_evaluation_types_and_utils import (
    EvalSample,
    eval_sample_to_normalized_eval_sample,
    calc_trajectory_room_visitation,
    
)

@BenchmarkRegistry.register("text_vision_to_action")
class TV2ACTBenchmark(BaseBenchmark):
    def __init__(self, eval_cfgs, model_cfgs, infer_cfgs, output_path, cache_manager, logger):
        self.max_eps_len = infer_cfgs.max_eps_len
        self.eval_set_size = infer_cfgs.eval_set_size
        self.gpu_devices = infer_cfgs.gpu_devices
        #todo different gpu for multi-thread evaluation
        self.gpu_device= infer_cfgs.gpu_device
        self.sampling = infer_cfgs.sampling
        self.task_type = infer_cfgs.task_type
        self.skip_done = infer_cfgs.skip_done
        

        
        self.logging_sensor = VideoLogging()
        self.output_path = output_path
        
        self.input_sensors = model_cfgs.model_input_sensors
        self.ckpt_pth = model_cfgs.ckpt_pth
        
        model = model_cfgs.model
        model_version = model_cfgs.model_version
        loss = model_cfgs.loss
        model_input_sensors = self.input_sensors
        
        
        
        self.eval_samples = {}
        self.pre_defined_max_steps = self.max_eps_len
        
        self._task_sampler: Optional[MultiTaskSampler] = None
        
        agent_class = REGISTERED_MODELS[model]
        agent_input = dict(
            model_version=model_version,
            input_sensors=model_input_sensors,
            loss=loss,
            sampling=self.sampling,
            ckpt_pth=self.ckpt_pth,
        )

        # Ensure the model can be loaded
        self.agent = agent_class.build_agent(**agent_input, device=self.gpu_device)
        self.agent_input = agent_input
        
        
    def init_dataloader(self, eval_cfgs: namedtuple = None, data_cfgs: namedtuple = None):
        # 1. set data config
        self.shuffle = data_cfgs.shuffle
        self.seed = data_cfgs.seed
        self.eval_subset = data_cfgs.eval_subset
        self.test_augmentation = data_cfgs.test_augmentation
        self.dataset_path = data_cfgs.dataset_path
        self.house_assets_path = data_cfgs.house_assets_path
        
        # 2. load dataset
        dataloader = TV2ACTDataLoader()
        self.task_dataset = dataloader.load_task_dataset(self.task_type, self.dataset_path)
        self.resultsets = []
        samples: List[EvalSample] = self.task_dataset["val"]
        sample_ids = list(range(len(samples)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(sample_ids)
        if self.eval_set_size is not None:
            sample_ids = sample_ids[: self.eval_set_size]
        normalized_samples = [
            eval_sample_to_normalized_eval_sample(task_type=self.task_type, sample=samples[i], index=i)
            for i in range(len(samples))
        ]
        task_samples = [normalized_samples[i] for i in sample_ids]
        self.eval_samples[self.task_type] = task_samples

        self.tasksets = []
        for task_type, samples in self.eval_samples.items():
            for sample in samples:
                self.tasksets.append(sample)
        print(f"tasksets num : {len(self.tasksets)}")
        self.tasksets = self.tasksets[:self.eval_set_size]
        # 3. load house assets
        self.house_assets : List[Dict[str, Any]] = dataloader.load_house_assets(self.house_assets_path, max_houses_per_split={'train':0, 'val':int(1e9)})['val']
        

    def save_benchmark_details(self, save_path: str, benchmark_name: str,results_lst: List, details_lst :List):
        """Save evaluation result and config file of single benchmark.
        Args:
            save_path (str): save path
            inputs (dict[str, List[InferenceInput]]): evaluation inputs
            results (dict[str, List[EvaluationResult]]): evaluation results
        """
        output_dir = os.path.join(save_path, benchmark_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"details_{self.task_type}.jsonl"), 'a', encoding='utf-8') as f:
            for result in results_lst:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
        video_dir = os.path.join(output_dir, 'video')
        os.makedirs(video_dir, exist_ok=True)
        for detail in tqdm(details_lst):
            mp4_name = detail['name'] + '.mp4'
            video_path_to_send = os.path.join(video_dir, mp4_name)
            save_frames_to_mp4(
                frames=detail["all_video_frames"], file_path=video_path_to_send, fps=5
            )

            topdown_view_path = os.path.join(video_dir, detail['name'] + "_topdown.png")
            plt.imsave(fname = topdown_view_path, arr=detail["top_down_frame"])
        print(f"Saved video to {save_path}")
        
    def calculate_overall_metrics(self, result):
        avg_succ = 0
        avg_robot_cost = 0
        avg_object_cost = 0
        avg_eps_len = 0
        for res in result:
            avg_succ += res['metrics']['success']
            avg_robot_cost += res['metrics']['cost_robot']
            avg_object_cost += res['metrics']['cost_object']
            avg_eps_len += res['metrics']['eps_len']
        avg_succ /= len(result)
        avg_robot_cost /= len(result)
        avg_object_cost /= len(result)
        avg_eps_len /= len(result)
        return dict(
            avg_succ=avg_succ,
            avg_robot_cost=avg_robot_cost,
            avg_object_cost=avg_object_cost,
            avg_eps_len=avg_eps_len
        )
        
    def get_extra_per_obj_metrics(self, task, metrics):
        try:
            object_type = task.task_info["synsets"][0]

            metrics[f"extra/{object_type}/success"] = metrics[
                "success"
            ]  # This should be different for different tasks
            metrics[f"extra/{object_type}/eps_len"] = metrics[
                "eps_len"
            ]  # This should be different for different tasks
            if metrics["success"] < 0.1:
                metrics[f"extra/{object_type}/eps_len_failed"] = metrics["eps_len"]
            else:
                metrics[f"extra/{object_type}/eps_len_success"] = metrics["eps_len"]

        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print(traceback.format_exc())

        return metrics

    def calculate_metrics(
        self,
        task: AbstractSPOCTask,
        all_actions: List[str],
        success: bool,
        additional_metrics: Dict[str, Any],
        number_of_eps: int,
        cost_robot: int,
        cost_object: int,
    ):
        metrics = {}

        metrics["eps_len"] = len(all_actions)
        metrics["success"] = float(success) + 1e-8
        metrics["cost_robot"] = cost_robot
        metrics["cost_object"] = cost_object
        if success:
            metrics["eps_len_succ"] = metrics["eps_len"]
        else:
            metrics["eps_len_fail"] = metrics["eps_len"]

        if "synsets" in task.task_info and len(task.task_info["synsets"]) == 1:
            metrics = self.get_extra_per_obj_metrics(task, metrics)

        if not success and (
            task.task_info["task_type"].startswith("Pickup")
            or task.task_info["task_type"].startswith("Fetch")
        ):
            metrics["failed_but_tried_pickup"] = int(THORActions.pickup in all_actions)

        trajectory = [obs["last_agent_location"][:3] for obs in task.observation_history]

        if task.room_poly_map is not None:
            percentage_visited, total_visited = calc_trajectory_room_visitation(
                task.room_poly_map, trajectory
            )
        else:
            percentage_visited, total_visited = 0, 0

        metrics["percentage_rooms_visited"] = percentage_visited
        metrics["total_rooms_visited"] = total_visited

        if "synsets" in task.task_info:
            list_of_object_types = task.task_info["synsets"]
            list_of_object_types = sorted(list_of_object_types)
            metrics["for_video_table/object_types"] = str(list_of_object_types)

            metrics["for_video_table/total_rooms"] = len(task.house["rooms"])


        assert (
            len([k for k in additional_metrics.keys() if k in metrics]) == 0
        ), "You should not redefine metrics or have duplicates"
        metrics = {**metrics, **additional_metrics}

        return metrics
    
    def display_benchmark_results(self, overall_result):
        max_key_length = max(len(str(key)) for key in overall_result.keys())
        max_value_length = max(len(str(value)) for value in overall_result.values())

        column_width = max(max_key_length, max_value_length) + 2

        print("Keys" + " " * (column_width - 4) + "| Values")
        print("-" * column_width + "-+-" + "-" * column_width)

        for key, value in overall_result.items():
            print(f"{str(key):<{column_width}}| {str(value):<{column_width}}")

    
    @property
    def task_sampler(self) -> MultiTaskSampler:
        if self._task_sampler is None:
            task_args = get_core_task_args(max_steps=self.pre_defined_max_steps)

            self._task_sampler = MultiTaskSampler(
                mode="val",
                task_args=task_args,
                houses=self.house_assets,
                house_inds=list(range(len(self.house_assets))),
                controller_args={
                    **STRETCH_ENV_ARGS,
                    "platform": (
                        ai2thor.platform.OSXIntel64
                        if sys.platform.lower() == "darwin"
                        else ai2thor.platform.CloudRendering
                    ),
                },
                controller_type=StretchController,
                task_spec_sampler=(),
                visualize=False,
                prob_randomize_materials=0,
                device=self.gpu_device if self.gpu_device == "cpu" or self.gpu_device > 0 else None,
            )

        return self._task_sampler
    
    def batch_inference(self, model):
        self.task_sampler.task_spec_sampler = TaskSpecList(self.tasksets)
        num_tasks = 0
        result_lst = []
        detail_result_list = []
        succ_sum = 0
        cost_object_sum = 0
        cost_robot_sum = 0
        with tqdm(total=len(self.tasksets)) as pbar:        
            for _ in range(len(self.tasksets)):
                task = self.task_sampler.next_task()

                task.max_steps = self.pre_defined_max_steps
                sample_result = self.evaluate_on_task(task=task, agent=model)
                task_info = {**task.task_info, **task.task_info["eval_info"]}
                del task_info["eval_info"]

                to_log = dict(
                    iter=num_tasks,
                    task_type=task_info["task_type"],
                    sample_id=task_info["sample_id"],
                    metrics=sample_result["metrics"],
                )
                result_lst.append(to_log)
                detail_result_list.append(dict(
                    name=task_info['task_type'] + '_' + task_info["sample_id"],
                    all_video_frames=sample_result['all_video_frames'],
                    top_down_frame=sample_result['top_down_frame']
                ))
                num_tasks += 1
                succ_sum        += sample_result["metrics"]["success"]
                cost_object_sum += sample_result["metrics"]["cost_robot"]
                cost_robot_sum  += sample_result["metrics"]["cost_object"]
                pbar.set_postfix(
                    succ=succ_sum/num_tasks, 
                    cost_obj=cost_object_sum/num_tasks, 
                    cost_robot=cost_robot_sum/num_tasks 
                    )

                pbar.update(1) 

        print(f"evaluate processed {num_tasks} tasks")
        return result_lst, detail_result_list
        
    
    def evaluate_on_task(self, task: AbstractSPOCTask, agent: AbstractAgent):
        goal = task.task_info["natural_language_spec"]

        # task_path points out the episode's origin (i.e., which task, episode id, streaming id)
        task_path = "/".join(task.task_info["eval_info"]["task_path"].split("/")[-4:])

        all_frames = []
        all_video_frames = []
        agent.reset()
        action_list = agent.get_action_list()

        all_actions = []

        additional_metrics = {}
        eps_idx = 0
        sum_cost_robot = 0
        sum_cost_object = 0
        with torch.no_grad():
            while len(all_actions) < task.max_steps:
                eps_idx += 1
                observations = task.get_observations()

                assert all(
                    input_sensor in observations
                    for input_sensor in self.input_sensors
                    if input_sensor != "last_actions"
                ), (
                    f"Observations do not contain all input sensors."
                    f" Observations: {observations.keys()}."
                    f" Input sensors: {self.input_sensors}"
                )

                observations = {k: v for k, v in observations.items() if k in self.input_sensors}

                curr_frame = np.concatenate(
                    [task.controller.navigation_camera, task.controller.manipulation_camera], axis=1
                )
                robot_cost = task.last_action_robot_cost
                object_cost = task.last_action_object_cost
                sum_cost_robot += robot_cost
                sum_cost_object += object_cost
                
                all_frames.append(curr_frame)

                action, probs = agent.get_action(observations, goal)

                if self.skip_done and action in ["end", "done"]:
                    action = "sub_done"
                all_actions.append(action)
                task.step_with_action_str(action)
                video_frame = self.logging_sensor.get_video_frame(
                    agent_frame=curr_frame,
                    frame_number=eps_idx,
                    action_names=action_list,
                    action_dist=probs.tolist(),
                    ep_length=task.max_steps,
                    last_action_success=task.last_action_success,
                    taken_action=action,
                    task_desc=goal,
                    cost=robot_cost + object_cost,
                    sum_cost_robot=sum_cost_robot,
                    sum_cost_object=sum_cost_object,
                    last_objects_causing_cost_list=task.last_objects_causing_cost_list,
                    cost_objects_name=task.cost_objects_name,
                    ignore_objects_name=task.ignore_objects_name,
                )

                all_video_frames.append(video_frame)
                if task.is_done():
                    break

        success = task.is_successful()
        
        target_ids = None
        if "synset_to_object_ids" in task.task_info:
            target_ids = list(
                chain.from_iterable(task.task_info.get("synset_to_object_ids", None).values())
            )
        top_down_frame, agent_path, cost_robot, cost_object = get_top_down_frame(
            task.controller, task.task_info["followed_path"], \
            task.task_info["followed_path_cost_robot"], task.task_info["followed_path_cost_object"], target_ids
        )
        top_down_frame = np.ascontiguousarray(top_down_frame)

        metrics = self.calculate_metrics(
            task,
            all_actions,
            success,
            additional_metrics,
            eps_idx + 1,
            sum_cost_robot,
            sum_cost_object
        )
        
        return dict(
            goal=goal,
            all_frames=all_frames,
            all_video_frames=all_video_frames,
            top_down_frame=top_down_frame,
            agent_path=agent_path,
            cost_robot_path=cost_robot,
            cost_object_path=cost_object,
            metrics=metrics,
            task_path=task_path,
            sum_cost=sum_cost_robot + sum_cost_object,
        )
