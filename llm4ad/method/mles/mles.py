# Module Name: MLES
# Last Revision: 2026/2/9
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu, Xialiang Tong, Mingxuan Yuan, Fei Liu, Zhichao Lu, and Qingfu Zhang.
#       "Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies."
#       The Fourteenth International Conference on Learning Representations (ICLR). 2026.

# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .population import Population
from .profiler import MLESProfiler
from .prompt import MLESPrompt
from .sampler import MLESSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
import itertools

import json
import os
import re


class MLES:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 operators: tuple = ('e1', 'e2', 'm1_M', 'm2_M'),   # ('e1', 'e2', 'm1', 'm2', 'm1_M', 'm2_M', )
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 initial_sample_nums_max: int = 50,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 seed_path="",
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            initial_sample_nums_max     : maximum samples restriction during initialization.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        # Core components for evaluation and task context
        self.evaluation_object = evaluation
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description

        # Evolution constraints and parameters
        self.seed_path = seed_path
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self.operators = operators

        # Validate operator requirements (e.g., text descriptions for specific operators)
        self.check_before_running()

        # Concurrency and runtime settings
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = initial_sample_nums_max
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # Initialize core modules: Population storage, LLM sampler, and Secure evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = MLESSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # Internal counters
        self._tot_sample_nums = 0
        self._initial_sample_nums_max = max(
            self._initial_sample_nums_max,
            2 * self._pop_size
        )

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

    def check_before_running(self):
        """Verify that the evaluation object provides required text descriptions for specific operators."""
        if 'm1_text' in self.operators:
            if hasattr(self.evaluation_object, 'non_image_representation_explanation'):
                self._information_discription = self.evaluation_object.non_image_representation_explanation
            else:
                raise ValueError(
                    "When 'text' is in operators, non image information description of this task cannot be empty")

    def init_from_local_algo_base(self):
        """Bootstrap the population using pre-existing algorithms from a local JSON seed file."""
        if os.path.exists(self.seed_path):
            with open(self.seed_path, 'r', encoding='utf-8') as file:
                seeds = json.load(file)
        else:
            print(
                f"\033[91mWarning: File {self.seed_path} does not exist, directly starting LLM-based algorithm initialization\033[0m")
            return

        operator = 'load'
        for seed_individual in seeds:
            # Parse function code and reconstruct the program structure
            seed_str = seed_individual['function']
            seed_algorithm = seed_individual['algorithm']
            program = TextFunctionProgramConverter.function_to_program(seed_str, self._template_program)
            program_str = str(program)
            func = TextFunctionProgramConverter.text_to_function(program_str)

            # Evaluate the seed program and record performance
            score_images_dict, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program
            ).result()

            # Metadata assignment and population registration
            if score_images_dict is not None:
                func.score = score_images_dict['score']
                func.image64 = score_images_dict['image']
                func.observation = score_images_dict['observation']
            else:
                func.score = None

            func.operator = operator
            func.evaluate_time = eval_time
            func.algorithm = seed_algorithm
            func.sample_time = 0

            # register to the population
            self._population.register_function(func)

            if self._profiler is not None:
                self._profiler.register_function(func, program=str(program))
                if isinstance(self._profiler, MLESProfiler):
                    self._profiler.register_population(self._population)

    def _sample_evaluate_register(self, prompt, image_prompt=None, messages=None, operator_name="", parent_number=None):
        """
        Execute the full evolutionary cycle for a single candidate:
        1. Sample: Query LLM for new algorithm design and code.
        2. Evaluate: Run the code in a secure parallel executor.
        3. Register: Store the individual in the population and log results.
        """
        sample_start = time.time()
        thought, func, response = self._sampler.get_thought_and_function(prompt, image_prompt, messages)
        sample_time = time.time() - sample_start

        if thought is None or func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        # Synchronously wait for parallel evaluation result
        score_images_dict, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        # Update function object with evaluation feedback and lineage
        if score_images_dict is not None:
            func.score = score_images_dict['score']
            func.image64 = score_images_dict['image']
            func.observation = score_images_dict['observation']
        else:
            func.score = None
        if parent_number is not None:
            func.parents = parent_number
        func.operator = operator_name
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.response = response
        func.prompt = prompt

        # register to the population
        self._population.register_function(func)

        # register to the log
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, MLESProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

    def _continue_loop(self) -> bool:
        """Check if termination conditions (max generations or max samples) have been met."""
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_mles_operator(self, tid=0):
        """
        Main evolutionary loop: iteratively applies search operators to the population.
        Supports multi-threaded sampling by offsetting the operator cycle based on thread ID.
        """
        # Cycle through available operators to ensure diverse search start for each thread
        operator_cycle = itertools.cycle(self.operators)
        for _ in range(tid):
            _ = next(operator_cycle)

        while self._continue_loop():
            try:
                # get current operator
                operator = next(operator_cycle)

                if operator == 'e1_advanced':
                    # get a new func using e1
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    messages = MLESPrompt.get_prompt_e1_advanced(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1 Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", messages=messages, operator_name='e1_advanced',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'e1':
                    # get a new func using e1
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MLESPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e1',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2
                elif operator == 'e2':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MLESPrompt.get_prompt_e2(self._task_description_str, indivs,
                                                       self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e2',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2
                elif operator == 'e2_advanced':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    messages = MLESPrompt.get_prompt_e2_advanced(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2_advanced Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", messages=messages, operator_name='e2',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2 Multimodal
                elif operator == 'e2_M':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    messages = MLESPrompt.get_prompt_e2_M(self._task_description_str, indivs,
                                                          self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2 Multimodal Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages, operator_name='e2_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using m1
                elif operator == 'm1':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1 Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", operator_name='m1', messages=messages,
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using m2
                elif operator == 'm2':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2 Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", operator_name='m2', messages=messages,
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using m1_Multimodal
                elif operator == 'm1_M':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m1_M(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_Multimodel Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages, operator_name='m1_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_text':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m1_M_text_info(self._task_description_str, indiv,
                                                                    self._function_to_evolve,
                                                                    self._information_discription)
                    if self._debug_mode:
                        print(f'm1_text Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages,
                                                   operator_name='m1_text_info',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_M':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m2_M(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2_Multimodel Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages, operator_name='m2_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # no figure itself
                elif operator == 'm1_only_imagedescribtion':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_image_description(self._task_description_str, indiv,
                                                                       self._function_to_evolve)
                    description, response = self._sampler.get_image_description(prompt="", image64s=None,
                                                                                messages=messages)
                    messages = MLESPrompt.get_prompt_m1_M_image_description(self._task_description_str, indiv,
                                                                            self._function_to_evolve, description)
                    if self._debug_mode:
                        print('Description:', description)
                        print('Description response:', response)
                        print(f'm1_image_describtion Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages,
                                                   operator_name='m1_image_describtion',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_only_imagedescribtion':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_image_description(self._task_description_str, indiv,
                                                                       self._function_to_evolve)
                    description, response = self._sampler.get_image_description(prompt="", image64s=None,
                                                                                messages=messages)
                    messages = MLESPrompt.get_prompt_m2_M_image_description(self._task_description_str, indiv,
                                                                            self._function_to_evolve, description)
                    if self._debug_mode:
                        print('Description:', description)
                        print('Description response:', response)
                        print(f'm2_image_describtion Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages,
                                                   operator_name='m2_image_describtion',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_only_image':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m1_M_only_image(self._task_description_str, indiv,
                                                                     self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_only_image_Multimodel Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages,
                                                   operator_name='m1_only_image',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # --- ABLATION OPERATORS (nothought) ---
                # Variants that without the "thought" of the algorithm during the algorithm generation.
                elif operator == 'e1_nothought':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MLESPrompt.get_prompt_e1_nothought(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1_nothought Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e1_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'e2_nothought':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MLESPrompt.get_prompt_e2_nothought(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2_nothought Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e2_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_M_nothought':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m1_M_nothought(self._task_description_str, indiv,
                                                                    self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_Multimodel_nothought Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages, operator_name='m1_M_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_M_nothought':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    messages = MLESPrompt.get_prompt_m2_M_nothought(self._task_description_str, indiv,
                                                                    self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2_Multimodel_nothought Prompt: {self.messages_to_string(messages)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, messages=messages, operator_name='m2_M_nothought',
                                                   parent_number=parents_pop_register_number)
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

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _iteratively_init_population(self, tid=0):
        """
        Populate the initial generation by repeatedly sampling from the LLM.
        This loop continues until the population reaches the required size or
        the maximum initialization sample limit is exceeded.
        """
        while self._population.generation == 0:
            try:
                # Generate a 'zero-shot' initialization prompt for the task
                prompt = MLESPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                if self._debug_mode:
                    print('Init Prompt: ', prompt)

                # Sample, evaluate, and add to population if valid
                self._sample_evaluate_register(prompt, operator_name="Initialization")

                # Safety break to prevent infinite loops if the LLM fails to generate valid code
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """
        Execute sampling functions (initialization or evolution) in parallel.
        Uses standard threading to handle multiple concurrent LLM requests.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=(tid, *args), kwargs=kwargs)
            for tid in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        """
        Entry point for the MLES search process:
        1. Initialize: Load from local seeds and/or perform LLM-based cold start.
        2. Evolve: Run the iterative mutation/crossover pipeline until termination.
        3. Finalize: Shut down profilers and save final results.
        """
        if not self._resume_mode:
            # Phase 1: Population Initialization
            print("🌱 Initializing population from database...")
            self.init_from_local_algo_base()

            print("🌱 Initializing population by LLM...")
            self._multi_threaded_sampling(self._iteratively_init_population)

            # Validation: Ensure we have enough individuals to perform evolutionary operators
            if len(self._population) < self._selection_num:
                print(
                    f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return

        # Phase 2: Evolutionary Search Loop
        print("🧬 Starting evolutionary training pipeline...")
        self._multi_threaded_sampling(self._iteratively_use_mles_operator)

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

                    score = score_images_dict.get('all_ins_performance', {}).get(instance_id, {}).get('score', float('-inf'))

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
        Supports both 'text' and 'image_url' content types.

        :param messages: list of dicts with 'role' and 'content'
        :param image_placeholder: str or callable, placeholder inserted for images
        :return: str
        """
        output_lines = []
        for message in messages:
            role = message.get("role", "user")
            contents = message.get("content", [])

            output_lines.append(f"[{role.upper()}]")
            for item in contents:
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
