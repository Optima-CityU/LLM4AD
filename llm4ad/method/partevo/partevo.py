# Module Name: PartEvo
# Last Revision: 2026/3/8
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu and Qingfu Zhang.
#       "Partition to evolve: Niching-enhanced evolution with llms for automated algorithm discovery."
#       In Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). 2025.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal, Union, Dict, Tuple

from .profiler import PartEvoProfiler
from .prompt import PartEvoPrompt
from .sampler import PartEvoSampler
from .clustermanager import ClusterManager
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
import itertools

import json
import os
import re


class PartEvo:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: Union[ProfilerBase, PartEvoProfiler] = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: int = 16,
                 operators: tuple = ('re', 'se', 'cc', 'lge'),
                 operators_parent_num: Optional[Dict] = None,
                 operators_frequency: Optional[Dict] = None,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 resume_mode: bool = False,
                 initial_sample_nums_max: int = 50,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 partition_method: str = 'kmeans',
                 partition_number: int = 4,

                 local_algo_base="",

                 feature_used: Tuple[Literal['ast', 'language'], ...] = ('ast',),

                 use_resource_tilt: bool = False,
                 bert_model_path: str = '',
                 **kwargs):

        # Core components for evaluation and task context
        self.evaluation_object = evaluation
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description

        # Evolution constraints and parameters
        self.local_algo_base = local_algo_base
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size

        self.operators = operators
        self.operators_parent_num = operators_parent_num
        self.operators_frequency = operators_frequency

        # Concurrency and runtime settings
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = max(
            initial_sample_nums_max,
            2 * self._pop_size
        )
        self._num_init_samplers = 3
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        self.partition_method = partition_method
        self.partition_number = partition_number
        self.use_resource_tilt = use_resource_tilt
        self.feature_used = feature_used

        self._pool = ClusterManager(pop_size=self._pop_size,
                                    n_clusters=self.partition_number,
                                    intra_operators=self.operators,
                                    intra_operators_parent_num=self.operators_parent_num,
                                    intra_operators_frequency=self.operators_frequency,
                                    use_resource_tilt=self.use_resource_tilt,
                                    resource_tilt_alpha=2.0,
                                    bert_model_path=bert_model_path,
                                    feature_type=self.feature_used,
                                    debug_flag=self._debug_mode)

        self.local_algo_base = local_algo_base
        self._sampler = PartEvoSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # Internal counters
        self._tot_sample_nums = 0

        # Setup parallel executor for performance evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # Log initial parameters
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _extend_init_population(self, tid=0, *args, **kwargs):
        """
        Let a thread repeat {sample -> evaluate -> register to population} to initialize a population.
        """
        try:
            current_population = self._pool.population.copy() + self._pool.next_pop.copy()
            current_feasible_population = [func for func in current_population if func.score is not None]
            messages = PartEvoPrompt.get_prompt_batch_init(self._task_description_str, self._function_to_evolve,
                                                           current_population=current_feasible_population)
            if self._debug_mode:
                print('Batch Init Prompt: ', self.messages_to_string(messages))
            self._sample_evaluate_register(prompt="", operator_name='init',
                                           messages=messages, from_which_cluster=None)
        except Exception:
            traceback.print_exc()

    def init_from_local_algo_base(self):
        """
        Warm start: Loads pre-existing algorithms from a local JSON file to seed the population.
        """
        if os.path.exists(self.local_algo_base):
            with open(self.local_algo_base, 'r', encoding='utf-8') as file:
                seeds = json.load(file)
        else:
            print(
                f"\033[91mWarning: File {self.local_algo_base} does not exist, directly starting LLM-based algorithm initialization\033[0m")
            return

        operator = 'load'
        for seed_individual in seeds:
            seed_str = seed_individual['function']
            program = TextFunctionProgramConverter.function_to_program(seed_str, self._template_program)
            program_str = str(program)
            func = TextFunctionProgramConverter.text_to_function(program_str)

            evaluation_return, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program
            ).result()

            if evaluation_return is not None:
                if isinstance(evaluation_return, (float, int)):
                    func.score = float(evaluation_return)
                else:
                    # register to profiler
                    func.all_ins_performance = evaluation_return.get('all_ins_performance', None)
                    func.list_performance = evaluation_return.get('list_performance', None)
                    func.score = evaluation_return.get('score', None)
            else:
                func.all_ins_performance = None
                func.list_performance = None
                func.score = None
            func.parents = []
            func.operator = operator
            func.algorithm = seed_individual['algorithm']
            func.evaluate_time = eval_time
            func.sample_time = None
            func.response = None
            func.prompt = None
            func.sample_num = self._tot_sample_nums + 1

            # register to the population
            self._pool.register_function(offspring=func, from_which_cluster=None)

            if self._profiler is not None:
                self._profiler.register_function(func)
                if isinstance(self._profiler, PartEvoProfiler):
                    self._profiler.register_population(self._pool)
                self._tot_sample_nums += 1

    def _sample_evaluate_register(self, prompt, image_prompt=None, messages=None, operator_name="", parent_number=None,
                                  from_which_cluster=None, reflction=None):
        """
        Execute the full evolutionary cycle for a single candidate:
        1. Sample: Query LLM for new algorithm design and code.
        2. Evaluate: Run the code in a secure parallel executor.
        3. Register: Store the individual in the population and log results.
        """
        sample_start = time.time()
        thought, func, response = self._sampler.get_thought_and_function(prompt, image_prompt, messages)
        sample_time = time.time() - sample_start

        if thought is None:
            print(
                '[Warning - Code 01] Failed to extract the "thought" concept. If this occurs frequently, please check the LLM output or the regex pattern.')

        if func is None:
            print(
                '[Warning - Code 02] Failed to extract the "func" implementation. If this occurs frequently, please check the LLM output or the code parsing logic.')

        if thought is None or func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        # Synchronously wait for parallel evaluation result
        evaluation_return, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        # Update function object with evaluation feedback and lineage
        if evaluation_return is not None:
            if isinstance(evaluation_return, (float, int)):
                func.score = float(evaluation_return)
            else:
                # register to profiler
                func.all_ins_performance = evaluation_return.get('all_ins_performance', None)
                func.list_performance = evaluation_return.get('list_performance', None)
                func.score = evaluation_return.get('score', None)
        else:
            func.all_ins_performance = None
            func.list_performance = None
            func.score = None

        if parent_number is not None:
            func.parents = parent_number
        func.operator = operator_name
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.response = response
        func.prompt = prompt
        func.sample_num = self._tot_sample_nums + 1
        if reflction:
            func.reflction = reflction

        # register to the population
        self._pool.register_function(offspring=func, from_which_cluster=from_which_cluster)

        # register to the log
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, PartEvoProfiler):
                self._profiler.register_population(self._pool)
            self._tot_sample_nums += 1

    def _continue_loop(self) -> bool:
        """Check if termination conditions (max generations or max samples) have been met."""
        if self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._pool.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._pool.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _partevo_multi_threaded_sampling(self, tid=0, *args, **kwargs):
        """
        Main evolutionary loop: iteratively applies search operators to the population.
        Selects parents, generates prompts based on niche-specific operators, and triggers sampling.
        """

        target_samples = kwargs.get('target_samples', float('inf'))

        while self._continue_loop() and self._tot_sample_nums < target_samples:
            try:
                # Step 1: Parent Selection from Niche
                parent_candidates, operator_name, working_cluster_id = self._pool.select_parent()
                parent_ids = [parent.sample_num for parent in parent_candidates]

                if operator_name == 'error':
                    print(f"\033[93m[Thread-{tid}] ❌ Warning: Parent selection failed. Please investigate.\033[0m")
                    continue

                # --- Operator logic: RE (Reflection-based Evolution) ---
                if operator_name == 're':
                    primary_parent = parent_candidates[0]
                    primary_parent_id = parent_ids[0]
                    reflection_prompt_messages = PartEvoPrompt.get_prompt_reflection(self._task_description_str,
                                                                                     primary_parent,
                                                                                     self._function_to_evolve
                                                                                     )
                    generated_reflection = self._sampler.get_reflection(prompt="",
                                                                        messages=reflection_prompt_messages)
                    operator_messages = PartEvoPrompt.get_prompt_re(self._task_description_str,
                                                                    primary_parent,
                                                                    self._function_to_evolve,
                                                                    generated_reflection)
                    if self._debug_mode:
                        print(f'RE Prompt: {self.messages_to_string(operator_messages)}')
                    self._sample_evaluate_register(prompt="",
                                                   messages=operator_messages,
                                                   operator_name='re',
                                                   reflction=generated_reflection,
                                                   parent_number=[primary_parent_id]
                                                   )
                    if not self._continue_loop():
                        break

                # --- Operator logic: SE (Summary-based Evolution) ---
                elif operator_name == 'se':
                    primary_parent = parent_candidates[0]
                    primary_parent_id = parent_ids[0]
                    # Fetch global context summary from the ExternalArchive
                    current_summary, requires_summary_update, summary_context_samples = self._pool.external_archive.fetch_summary_context()

                    if requires_summary_update and summary_context_samples:
                        summary_prompt_messages = PartEvoPrompt.get_prompt_summary(
                            self._task_description_str,
                            self._function_to_evolve,
                            summary_context_samples
                        )

                        generated_summary = self._sampler.get_summary(prompt="", messages=summary_prompt_messages)
                        self._pool.external_archive.update_global_summary(generated_summary)
                        if generated_summary and generated_summary.strip():
                            current_summary = generated_summary
                            print(f"\033[92m[Thread-{tid}] 🎉 Successfully generated a valid global summary.\033[0m")
                        else:
                            print(
                                f"\033[93m[Thread-{tid}]⚠️ Warning: Summary generation failed. Falling back to cached summary.\033[0m")

                    else:
                        pass
                        # print(
                        #     f"[Thread-{tid}] Using cached summary. Update threshold not met ({self._pool.external_archive._request_counter % self._pool.external_archive._summary_update_interval}/{self._pool.external_archive._summary_update_interval}).")

                    operator_messages = PartEvoPrompt.get_prompt_se(self._task_description_str, primary_parent,
                                                                    self._function_to_evolve, current_summary)

                    if self._debug_mode:
                        print(f'SE Prompt: {self.messages_to_string(operator_messages)}')

                    self._sample_evaluate_register(
                        prompt="",
                        messages=operator_messages,
                        operator_name='se',
                        parent_number=[primary_parent_id]
                    )

                    if not self._continue_loop():
                        break

                # --- Operator logic: CN (Crossover between Niches) ---
                elif operator_name == 'cn':
                    operator_messages = PartEvoPrompt.get_prompt_cn(self._task_description_str, parent_candidates,
                                                                    self._function_to_evolve)
                    if self._debug_mode:
                        print(f'CN Prompt: {self.messages_to_string(operator_messages)}')
                    self._sample_evaluate_register(
                        prompt="",
                        messages=operator_messages,
                        operator_name='cn',
                        parent_number=parent_ids
                    )
                    if not self._continue_loop():
                        break

                # --- Operator logic: LGE (Local and Global guided Evolution ) ---
                elif operator_name == 'lge':
                    operator_messages = PartEvoPrompt.get_prompt_lge(self._task_description_str,
                                                                     parent_candidates,
                                                                     self._function_to_evolve)
                    if self._debug_mode:
                        print(f'LGE Prompt: {self.messages_to_string(operator_messages)}')
                    self._sample_evaluate_register(
                        prompt="",
                        messages=operator_messages,
                        operator_name='lge',
                        parent_number=parent_ids
                    )
                    if not self._continue_loop():
                        break

                else:
                    raise Exception("ERROR: The input operators are not supported at the moment. Please check !!!!!")

            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    # exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """
        Execute sampling functions (initialization or evolution) in parallel.
        Uses standard threading to handle multiple concurrent LLM requests.
        """
        init_mode = kwargs.get('init_mode', False)
        if init_mode:
            sampler_threads = [
                Thread(target=fn, args=(tid, *args), kwargs=kwargs)
                for tid in range(self._num_init_samplers)
            ]
        else:
            sampler_threads = [
                Thread(target=fn, args=(tid, *args), kwargs=kwargs)
                for tid in range(self._num_samplers)
            ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def init_using_llms(self):
        batch_num = 0
        while len(
                self._pool.population) + len(
            self._pool.next_pop) < self._pop_size and self._tot_sample_nums <= self._initial_sample_nums_max:  # 当被注册的个数还没有超过pop_size初始种群要求大小时
            batch_num += 1
            self._multi_threaded_sampling(self._extend_init_population, init_mode=True)
            print(f"Initialization of batch {batch_num} completed")
        print(
            f'pool generation is {self._pool.generation} now, got {len(self._pool.population)} different individual, use {self._tot_sample_nums}/{self._initial_sample_nums_max}')

    def run(self):
        """
        Executes the full pipeline:
        1. Initialization (Seeds + LLM Cold Start)
        2. Evolutionary training loop
        3. Final result cleanup.
        """
        if not self._resume_mode:
            # Phase 1: Population Initialization
            print("🌱 Initializing population from database...")
            self.init_from_local_algo_base()

            if len(self._pool.population) < self._pop_size:
                print("🌱 Initializing population by LLM...")
                self.init_using_llms()

        # Phase 2: Evolutionary Search Loop
        print("🧬 Starting evolutionary training pipeline...")
        self._multi_threaded_sampling(self._partevo_multi_threaded_sampling)

        # Phase 3: Cleanup and Reporting
        if self._profiler is not None:
            self._profiler.finish()

    def using_flow(self, worst_case_percent=10, top_k=None):
        """
        Executes the 'Using Mode' pipeline:
        1. Loads the latest evolved population from local storage.
        2. Filters the top-K performing algorithms if specified.
        3. Evaluates all selected algorithms on a set of new test instances.
        4. Identifies the best-performing algorithm for each specific instance.
        5. Computes overall statistics and worst-case performance metrics.
        """
        print(f"🔍 Loading model from {self._profiler._log_dir}...")
        designed_results_path = os.path.join(self._profiler._log_dir, 'population')

        # --- STEP 1: Locate the latest population file (e.g., pop_10.json) ---
        pattern = re.compile(r'^pop_(\d+)\.json$')
        max_x = -1
        latest_file = None

        if not os.path.isdir(designed_results_path):
            print(f"Error: Directory not found: {designed_results_path}")
            return

        # Iterate over files in the directory
        for filename in os.listdir(designed_results_path):
            match = pattern.match(filename)

            # If the filename matches
            if match:
                # Extract the number (group 1) and convert to int
                current_x = int(match.group(1))

                # Check if it's the largest number found so far
                if current_x > max_x:
                    max_x = current_x
                    latest_file = filename

        # Check if any matching file was found
        if latest_file is None:
            print(f"Error: No 'pop_x.json' files found in {designed_results_path}")
            return  # Or raise an Exception

        # Construct the full path to the correct file
        full_path_to_file = os.path.join(designed_results_path, latest_file)
        print(f"Found latest file: {full_path_to_file}")

        with open(full_path_to_file, 'r') as f:
            trained_data = json.load(f)

        # --- STEP 2: Top-K Filtering ---
        # Reduce the search space by only testing the highest-scoring algorithms from training.
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            print(f"✂️  Filtering Population: Selecting top {top_k} algorithms...")
            original_size = len(trained_data)

            try:
                # Sort algorithms by training score in descending order
                trained_data.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
                trained_data = trained_data[:top_k]
                print(f"   -> Reduced population from {original_size} to {len(trained_data)}.")
            except Exception as e:
                print(f"   -> ⚠️ Warning: Could not sort by 'score'. Using original order. Error: {e}")

        using_time_start = time.time()
        print("💪 [Brute Force Mode] Evaluating selected algorithms on each instance...")

        print(f"   -> Found {len(trained_data)} unique algorithms to test.")
        ins_to_be_solve_set = self.evaluation_object.ins_to_be_solve_set
        ins_to_be_solve_id_set = [id for id in ins_to_be_solve_set.keys()]

        final_results = {}
        all_scores = []

        # --- STEP 3: Brute Force Evaluation ---
        # Iterate through each test instance and test every algorithm in the filtered pool.
        for instance_id in ins_to_be_solve_id_set:
            print(f"\n[Brute Force] Solving new instance: {instance_id}")
            best_algo_for_instance = None
            best_score_for_instance = float('-inf')
            best_perf_for_instance = None

            for i, algo_json in enumerate(trained_data):
                print(f"  -> Testing algorithm {i + 1}/{len(trained_data)}...", end='\r')
                try:
                    # Parse algorithm code and execute secure evaluation
                    program = TextFunctionProgramConverter.function_to_program(algo_json['function'],
                                                                               self._template_program)
                    func = TextFunctionProgramConverter.text_to_function(str(program))

                    score_images_dict = self._evaluator._evaluate(str(program), func.name,
                                                                  ins_to_be_evaluated_id=(instance_id,),
                                                                  training_mode=False)

                    score = score_images_dict.get('all_ins_performance', {}).get(instance_id, {}).get('score',
                                                                                                      float('-inf'))

                    # Update the best algorithm found for this specific instance
                    if score is not None and score > best_score_for_instance:
                        print(f'   Update! New Best: {score:.4f} (Algo index: {i})')
                        best_score_for_instance = score
                        best_algo_for_instance = algo_json
                        best_perf_for_instance = score_images_dict.get('all_ins_performance', {})[instance_id]
                except Exception as e:
                    print(f"\n      -> ❌ Error evaluating algorithm on instance {instance_id}: {e}")
            print()

            # Store the best result found for the current instance
            if best_algo_for_instance:
                print(
                    f"   -> ✅ Best score found: {best_score_for_instance:.4f}")
                final_results[instance_id] = {
                    'algorithm': best_algo_for_instance['algorithm'],
                    'function': best_algo_for_instance['function'],
                    'score': best_perf_for_instance.get('score'),
                }
                if best_perf_for_instance.get('score') is not None:
                    all_scores.append(best_perf_for_instance['score'])
            else:
                final_results[instance_id] = {'score': None, 'evaluate_time': None}
                print(f"   -> ⚠️ Warning: No algorithm produced a valid score for instance {instance_id}.")

        # --- STEP 4: Statistics & Worst-Case Analysis ---
        valid_scores = [s for s in all_scores if s is not None]

        if valid_scores:
            final_results['sum_score_of_all_instances'] = sum(valid_scores)
            final_results['average_score_of_all_instances'] = sum(valid_scores) / len(valid_scores)
        else:
            final_results['sum_score_of_all_instances'] = None
            final_results['average_score_of_all_instances'] = None

        final_results['each_result'] = all_scores

        # Identify instances with the lowest scores (Bottom K%) to analyze robustness
        id_score_pairs = []
        for k, v in final_results.items():
            if isinstance(k, int) and isinstance(v, dict) and v.get('score') is not None:
                id_score_pairs.append((k, v['score']))
        id_score_pairs.sort(key=lambda x: x[1])
        total_valid_count = len(id_score_pairs)
        cutoff_count = int(total_valid_count * (worst_case_percent / 100.0))
        if cutoff_count == 0 and total_valid_count > 0:
            cutoff_count = 1

        # Calculate worst-case statistics
        worst_cases = id_score_pairs[:cutoff_count]
        worst_instance_ids = [pair[0] for pair in worst_cases]
        worst_scores_values = [pair[1] for pair in worst_cases]
        worst_avg_score = sum(worst_scores_values) / len(worst_scores_values) if worst_scores_values else None

        if worst_avg_score is not None:
            print(f"\n📉 [Worst-Case Stats] Bottom {worst_case_percent}% (Count: {len(worst_cases)}):")
            print(f"   -> Average Score: {worst_avg_score}")

        final_results['worst_case_stats'] = {
            'percent_threshold': worst_case_percent,
            'count': len(worst_cases),
            'average_score': worst_avg_score,
            'instance_ids': worst_instance_ids,
            'scores': worst_scores_values
        }

        # --- STEP 5: Finalize & Logging ---
        using_time_end = time.time()
        final_results['running_time'] = using_time_end - using_time_start
        print(f"Running time: {final_results['running_time']} seconds")

        if self._profiler:
            self._profiler.using_final(final_results=final_results)
        print(f"\n💡 Using Mode finished.")

        print(
            f'There are {len(ins_to_be_solve_set)} instances to solve. \nSuccessfully solved {len(valid_scores)} instances, with an average score of {final_results["average_score_of_all_instances"]}.')

    def messages_to_string(self, messages, image_placeholder="<<<IMAGE>>>"):
        """
        Convert a structured messages list (OpenAI-style) into a single formatted string.
        Supports both string contents and list of dicts ('text' and 'image_url' content types).

        :param messages: list of dicts with 'role' and 'content'
        :param image_placeholder: str or callable, placeholder inserted for images
        :return: str
        """
        output_lines = []
        for message in messages:
            role = message.get("role", "user")
            contents = message.get("content", [])

            output_lines.append(f"[{role.upper()}]")

            if isinstance(contents, str):
                output_lines.append(contents)

            elif isinstance(contents, list):
                for item in contents:
                    if not isinstance(item, dict):
                        continue

                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            output_lines.append(text)
                    elif item.get("type") == "image_url":
                        # Optional: handle custom placeholders with description
                        url = item.get("image_url", {}).get("url", "")
                        desc = item.get("image_url", {}).get("detail", "an image")
                        if callable(image_placeholder):
                            placeholder = image_placeholder(url, desc)
                        else:
                            placeholder = f"{image_placeholder}  # {desc}"
                        output_lines.append(placeholder)

            output_lines.append("")  # blank line between messages

        return "\n".join(output_lines)
