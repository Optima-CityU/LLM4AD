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
        self.evaluation_object = evaluation
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description

        self.seed_path = seed_path
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self.operators = operators

        self.check_before_running() # check whether satisfy the operator

        # samplers and evaluators
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

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = MLESSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = max(
            self._initial_sample_nums_max,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def check_before_running(self):
        # 检查是否有 non_image_representation_explanation 属性
        if 'm1_text' in self.operators:
            if hasattr(self.evaluation_object, 'non_image_representation_explanation'):
                self._information_discription = self.evaluation_object.non_image_representation_explanation
            else:
                raise ValueError(
                    "When 'text' is in operators, non image information description of this task cannot be empty")

    def init_from_local_algo_base(self):
        if os.path.exists(self.seed_path):
            # 打开并读取JSON文件
            with open(self.seed_path, 'r', encoding='utf-8') as file:
                seeds = json.load(file)
        else:
            print(
                f"\033[91mWarning: File {self.seed_path} does not exist, directly starting LLM-based algorithm initialization\033[0m")
            return

        operator = 'load'
        for seed_individual in seeds:
            seed_str = seed_individual['function']
            seed_algorithm = seed_individual['algorithm']
            program = TextFunctionProgramConverter.function_to_program(seed_str, self._template_program)
            program_str = str(program)
            func = TextFunctionProgramConverter.text_to_function(program_str)

            score_images_dict, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program
            ).result()

            if score_images_dict is not None:
                # register to profiler
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
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        thought, func, response = self._sampler.get_thought_and_function(prompt, image_prompt, messages)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return
        # evaluate
        # score_images_dict
        score_images_dict, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        if score_images_dict is not None:
            # register to profiler
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

        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, MLESProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

    def _continue_loop(self) -> bool:
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
        # 使用 itertools.cycle 使 operators 循环
        operator_cycle = itertools.cycle(self.operators)
        for _ in range(tid):
            # 获取当前循环的 operator
            _ = next(operator_cycle)

        while self._continue_loop():
            try:
                # 获取当前循环的 operator
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

                # TODO：测试这个算子
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

                elif operator == 'e1_nothought':
                    # get a new func using e1
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

                # get a new func using e2
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
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = MLESPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                if self._debug_mode:
                    print('Init Prompt: ', prompt)
                self._sample_evaluate_register(prompt, operator_name="Initialization")
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
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
        if not self._resume_mode:
            print("🌱 Initializing population from database...")
            self.init_from_local_algo_base()
            # do initialization
            print("🌱 Initializing population by LLM...")
            self._multi_threaded_sampling(self._iteratively_init_population)
            # terminate searching if
            if len(self._population) < self._selection_num:
                print(
                    f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return
        print("🧬 Starting evolutionary training pipeline...")
        # evolutionary search
        self._multi_threaded_sampling(self._iteratively_use_mles_operator)
        # finish
        if self._profiler is not None:
            self._profiler.finish()

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
