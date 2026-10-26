API reference
=============

Generated from the checked-out source. Optional packages are not imported.
All top-level public classes/functions and public methods are indexed.
Signatures are complete; missing docstrings remain visible as documentation gaps.

llm4ad.base.code
----------------

Function
~~~~~~~~

.. code-block:: text

    class Function()

.. code-block:: text

    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    score: Any | None = None
    evaluate_time: float | None = None
    sample_time: float | None = None
    operator: str | None = 'Unknown'

.. code-block:: text

    A parsed Python function.

.. code-block:: text

    Function.to_code_without_docstring(self) -> str

.. code-block:: text

    Return the function as a string without the docstring.

Program
~~~~~~~

.. code-block:: text

    class Program()

.. code-block:: text

    preface: str
    functions: list[Function]

.. code-block:: text

    A parsed Python program.

.. code-block:: text

    Program.find_function_index(self, function_name: str) -> int

.. code-block:: text

    Returns the index of input function name.

.. code-block:: text

    Program.get_function(self, function_name: str) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Program.exec(self) -> List[Callable]

.. code-block:: text

    Documentation gap: no method docstring is present.

TextFunctionProgramConverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class TextFunctionProgramConverter()

.. code-block:: text

    Convert text to Program instance and Function instance,
    Convert Program instance to Function instance, and Function instance to Program instance.

.. code-block:: text

    TextFunctionProgramConverter.text_to_program(cls, program_str: str) -> Program | None

.. code-block:: text

    Returns Program object by parsing input text using Python AST.


.. code-block:: text

    TextFunctionProgramConverter.text_to_function(cls, program_str: str) -> Function | None

.. code-block:: text

    Returns Function object by parsing input text using Python AST.


.. code-block:: text

    TextFunctionProgramConverter.function_to_program(cls, function: str | Function, template_program: str | Program) -> Program | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TextFunctionProgramConverter.program_to_function(cls, program: str | Program) -> Function | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.base.errors
------------------

FatalSearchError
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FatalSearchError(RuntimeError)

.. code-block:: text

    A failed run contract, such as incompatible replay or checkpoint data.

ReplayError
~~~~~~~~~~~

.. code-block:: text

    class ReplayError(FatalSearchError)

.. code-block:: text

    A terminal replay failure; search loops must propagate it to the caller.

ReplayMismatchError
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ReplayMismatchError(ReplayError, ValueError)

.. code-block:: text

    The next request differs from the recorded request.

ReplayExhaustedError
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ReplayExhaustedError(ReplayError, EOFError)

.. code-block:: text

    The search requested more responses than the journal contains.

ReplayBackendError
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ReplayBackendError(ReplayError)

.. code-block:: text

    The next journal entry records a backend failure.

llm4ad.base.evaluate
--------------------

Evaluation
~~~~~~~~~~

.. code-block:: text

    class Evaluation(ABC)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Evaluation.__init__(self, template_program: str | Program, task_description: str='', use_numba_accelerate: bool=False, use_protected_div: bool=False, protected_div_delta: float=1e-05, random_seed: int | None=None, timeout_seconds: int | float=None, *, exec_code: bool=True, safe_evaluate: bool=True, daemon_eval_process: bool=False, fork_proc: Literal['auto'] | bool='auto')

.. code-block:: text

    Evaluation interface for executing generated code.
    Args:
        use_numba_accelerate: Wrap the function with '@numba.jit(nopython=True)'.
        use_protected_div   : Modify 'a / b' => 'a / (b + delta)'. Maybe useful for mathematical tasks.
        protected_div_delta : Delta value in protected div.
        random_seed         : If is not None, set random seed in the first line of the function body.
        timeout_seconds     : Terminate the evaluation after timeout seconds.
        exec_code           : Using 'exec()' to compile the code and provide the callable function.
            If is set to 'False', the 'callable_func' argument in 'self.evaluate_program' is always 'None'.
            If is set to 'False', the user should provide the score of the program based on 'program_str' argument in 'self.evaluate_program'.
        safe_evaluate       : Evaluate in safe mode using a new process. If is set to False,
            the evaluation will not be terminated after timeout seconds. The user should consider how to
            terminate evaluating in time.
        daemon_eval_process : Set the evaluate process as a daemon process. If set to True,
            you can not set new processes in the evaluator. Which means in self.evaluate_program(),
            you can not create new processes.
        fork_proc           : This arg is valid when safe_evaluate=True, which determines to 'fork' process or 'spawn' a safe process.
            If set to 'auto', the process creating method will depend on OS. Set to 'True' to use 'fork', 'False' to use 'spawn'.

    -Assume that: use_numba_accelerate=True, self.use_protected_div=True, and self.random_seed=2024.
    -The original function:
    --------------------------------------------------------------------------------
    import numpy as np

    def f(a, b):
        a = np.random.random()
        return a / b
    --------------------------------------------------------------------------------
    -In the Evaluation phase, the modified function will be:
    --------------------------------------------------------------------------------
    import numpy as np
    import numba

    @numba.jit(nopython=True)
    def f():
        np.random.seed(2024)
        a = np.random.random()
        return _protected_div(a, b)

    def _protected_div(a, b, delta=1e-5):
        return a / (b + delta)
    --------------------------------------------------------------------------------
    As shown above, the 'import numba', 'numba.jit()' decorator, and '_protected_dev' will be added by this function.

.. code-block:: text

    Evaluation.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Evaluate a given function. You can use compiled function (function_callable),
    as well as the original function strings for evaluation.
    Args:
        program_str: The function in string. You can _ignore this argument when implementation. (See below).
        callable_func: The callable heuristic function. You can call it using `callable_func(args, kwargs)`.
    Return:
        Returns the fitness value.

    Assume that: self.use_numba_accelerate = True, self.use_protected_div = True,
    and self.random_seed = 2024, the argument 'function_str' will be something like below:
    --------------------------------------------------------------------------------
    import numpy as np
    import numba

    @numba.jit(nopython=True)
    def f(a, b):
        np.random.seed(2024)
        a = a + np.random.random()
        return _protected_div(a, b)

    def _protected_div(a, b, delta=1e-5):
        return a / (b + delta)
    --------------------------------------------------------------------------------
    As shown above, the 'import numba', 'numba.jit()' decorator,
    and '_protected_dev' will be added by this function.

SecureEvaluator
~~~~~~~~~~~~~~~

.. code-block:: text

    class SecureEvaluator()

.. code-block:: text

    Evaluate in a separate process with optional wall-clock termination.

    The historical name is retained for compatibility. This is not a security
    sandbox: candidate code retains the user's file, network and OS permissions.
    See SECURITY.md for resource limits and the threat model.

.. code-block:: text

    SecureEvaluator.__init__(self, evaluator: Evaluation, debug_mode=False, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SecureEvaluator.evaluate_program(self, program: str | Program, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SecureEvaluator.evaluate_program_record_time(self, program: str | Program, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.base.modify_code
-----------------------

ModifyCode
~~~~~~~~~~

.. code-block:: text

    class ModifyCode()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ModifyCode.add_decorator(cls, program: str, function_name: str, decorator_name: str | List[str], decorator_args: List[str | Tuple[str, Any]]=None) -> str

.. code-block:: text

    Add wrapper: @module_name.wrapper_name(...) to the function.
    Args:
        program       : The program in string.
        function_name : The name of the function that to be decorated.
        decorator_name: The name of the decorator. This argument can be list of str(s) or a str splited by '.'.
                        Example: ['numba', 'jit'], 'numba.jit', ['a', 'b', 'c'], 'a.b.c'.
        decorator_args: The args and kwargs of the decorator.

    Example 1:
    ----------------------------------------------------------------------------------
    >>> program = '''
    ... def f():
    ...     return 0'''
    >>> ModifyCode.add_decorator(program, 'f', 'torch.jit.script')
    '@torch.jit.script()\ndef f():\n    return 0'
    >>>
    ----------------------------------------------------------------------------------

    Example 2:
    ----------------------------------------------------------------------------------
    >>> program = '''
    ... def f():
    ...     return 0'''
    >>> ModifyCode.add_decorator(program, 'f', ['numba', 'jit'], [('nopython', True)])
    '@numba.jit(nopython=True)\ndef f():\n    return 0'
    >>>
    ----------------------------------------------------------------------------------

    Example 3:
    ----------------------------------------------------------------------------------
    >>> program = '''
    ... def f():
    ...     return 0'''
    >>> ModifyCode.add_decorator(program, 'f', 'a.b.c.d', [1, True, ('e', 'all'), ('f', True)])
    "@a.b.c.d(1, True, e='all', f=True)\ndef f():\n    return 0"
    >>>
    ----------------------------------------------------------------------------------

.. code-block:: text

    ModifyCode.add_import_package_statement(cls, program: str, package_name: str, as_name: str | None=None, *, check_imported: bool=True) -> str

.. code-block:: text

    Add 'import package_name as as_name' in the program code.
    Args:
        program       : The program in string.
        package_name  : The name of the package to be imported.
        as_name       : The alias of the imported package. Such as 'np' to 'numpy'.
        check_imported: Check if 'import {package_name} as {as_name}' statement has already existed,
                        this function returns the original program if it exists.

.. code-block:: text

    ModifyCode.add_numpy_random_seed_to_func(cls, program: str, func_name: str, seed: int=2024) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ModifyCode.replace_div_with_protected_div(cls, program: str, delta: float=1e-05, numba_accelerate: bool=False, return_div_func_name: bool=False) -> str | Tuple[str, str]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ModifyCode.add_np_random_seed_below_numpy_import(cls, program: str, seed: int=2024) -> str

.. code-block:: text

    Add 'import numpy as np' statement (if needed) to the program and insert 'np.random.seed(seed)' under it.
    Args:
        program: program you want to add.
        seed   : seed number.
    Returns:
        modified_code: program with 'np.random.seed(...)'.

.. code-block:: text

    ModifyCode.add_numba_decorator(cls, program: str, function_name: str | List[str]) -> str

.. code-block:: text

    This function aims to accelerate the evaluation of the searched code. This is achieved by decorating '@numba.jit()'
    to the function_to_evolve or other functions in the specification that can be speed up using numba.
    However, it should be noted that not all numpy functions support numba acceleration: such as np.piecewise().
    So use this function wisely. Haha!

    Example input program:
    ----------------------------------------------------------
        def func(a: np.ndarray):
            return a * 2
    ----------------------------------------------------------
    Example output program:
    ----------------------------------------------------------
        import numba

        numba.jit()
        def func(a: np.ndarray):
            return a * 2
    ----------------------------------------------------------

.. code-block:: text

    ModifyCode.rename_function(cls, code: str, source_name: str, target_name: str) -> str

.. code-block:: text

    Renames function calls from `source_name` to `target_name`.


.. code-block:: text

    ModifyCode.get_functions_name(cls, code: str) -> MutableSet[str]

.. code-block:: text

    Returns the set of all function name in `code`.


.. code-block:: text

    ModifyCode.yield_decorated(cls, code: str, module: str, name: str) -> Iterator[str]

.. code-block:: text

    Yields names of functions decorated with `@module.name` in `code`.


llm4ad.base.sample
------------------

LLM
~~~

.. code-block:: text

    class LLM()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LLM.__init__(self, *, do_auto_trim=True, debug_mode=False)

.. code-block:: text

    Language model interface.
    This interface defines how to interact with LLM api / deployed LLM.
    Args:
        do_auto_trim: if set to True, then automatically trim the code from response content.

.. code-block:: text

    LLM.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Returns a predicted continuation of `prompt`.
    -For example, the response content of the LLM is:
    ------------------------------------------------------------------------------------------------------------------
    Here is the function.
    def priority_v2(..., ...) -> Any:
        a = np.array([1, 2, 3])
        if len(a) > 2:
            return a / a.sum()
        else:
            return a / a.mean()
    This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------

.. code-block:: text

    LLM.draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]

.. code-block:: text

    Returns multiple predicted continuations of `prompt`.


.. code-block:: text

    LLM.close(self)

.. code-block:: text

    Defines how to close the connection to API,
    or release the GPU resources at the end of the program search.

SampleTrimmer
~~~~~~~~~~~~~

.. code-block:: text

    class SampleTrimmer()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    SampleTrimmer.__init__(self, llm: LLM)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SampleTrimmer.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Get a sample based on the provided 'LLM' instance.
    If the inner sampler sets 'auto_trim' to True, trim anything before the function body.

.. code-block:: text

    SampleTrimmer.draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]

.. code-block:: text

    Get samples based on the provided 'Sampler' instance.
    If the inner sampler sets 'auto_trim' to True, trim anything before the function body.

.. code-block:: text

    SampleTrimmer.trim_preface_of_function(cls, generated_code: str)

.. code-block:: text

    Trim the redundant descriptions/symbols/'def' declaration BEFORE the function body.
    Example of a generated content from an LLM:
    --------------------------------------------------------------------------
    This is the optimized function ...

    def priority_v2(...) -> ...:
        a = random.random()
        return a * a

    This function aims to ...
    --------------------------------------------------------------------------
    Example return of this function:
    --------------------------------------------------------------------------
        a = random.random()
        return a * a

    This function aims to ...
    --------------------------------------------------------------------------

.. code-block:: text

    SampleTrimmer.auto_trim(cls, generated_code: str) -> str

.. code-block:: text

    Automatically trim the preface of the generated content.


.. code-block:: text

    SampleTrimmer.sample_to_function(cls, generated_code: str, template_program: str | Program) -> Function | None

.. code-block:: text

    Convert the generated content (with redundant component)
    to a Function instance. If the convert fails, return None.
    Please note that the modified Function instance is not executable,
    as it lacks 'import ...' statements.

.. code-block:: text

    SampleTrimmer.sample_to_program(cls, generated_code: str, template_program: str | Program) -> Program | None

.. code-block:: text

    Convert the generated content (with redundant component)
    to a Function instance. If the convert fails, return None.

.. code-block:: text

    SampleTrimmer.trim_function_body(cls, generated_code: str) -> str | None

.. code-block:: text

    Extracts the body of the generated function, trimming anything after it.


.. code-block:: text

    SampleTrimmer.remove_docstrings(cls, func: Function | str)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.gui
----------

main_gui
~~~~~~~~

.. code-block:: text

    Executes the optimization process using dynamically loaded components based on provided GUI-configured settings.
    Args:
        llm (dict): A dictionary containing the configuration for the large language model (LLM), including:
            - 'name': The name of the LLM class.
            - Additional parameters required for LLM initialization, such as 'host', 'key', 'model', etc.
        method (dict): A dictionary containing the configuration for the optimization method, including:
            - 'name': The name of the optimization method class.
            - Additional method parameters like 'max_sample_nums', 'max_generations', 'pop_size', etc.
        evaluation (dict): A dictionary containing the configuration for the evaluation method, including:
            - 'name': The name of the evaluation class.
            - Additional evaluation parameters.
        profiler (dict): A dictionary containing the configuration for the profiler, including:
            - 'name': The name of the profiler class.
            - Additional profiler parameters like 'evaluation_name', 'method_name', etc.

    The function dynamically loads the LLM, method, evaluation, and profiler classes using their names specified in the dictionaries.
    It then initializes each component with its respective parameters and runs the optimization process.
    The function filters the parameter dictionaries to pass the appropriate arguments to each class initializer.

    Example:
        llm_config = {
            'name': 'HttpsApi',
            'host': 'api.example.com',
            'key': 'your-api-key',
            'model': 'gpt-4-mini'
        }
        method_config = {
            'name': 'EoH',
            'max_sample_nums': 100,
            'max_generations': 10,
            'pop_size': 20,
            'num_samplers': 4,
            'num_evaluators': 4
        }
        evaluation_config = {
            'name': 'OBPEvaluation'
        }
        profiler_config = {
            'name': 'EoHTensorboardProfiler'
        }

        main_gui(llm_config, method_config, evaluation_config, profiler_config)


.. code-block:: text

    main_gui(llm: dict, method: dict, evaluation: dict, profiler: dict)

llm4ad.method.__init__
----------------------

import_all_method_classes_from_subfolders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.

.. code-block:: text

    import_all_method_classes_from_subfolders(root_directory: str)

llm4ad.method.eoh.eoh
---------------------

EoH
~~~

.. code-block:: text

    class EoH()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoH.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_generations: Optional[int]=10, max_sample_nums: Optional[int]=100, pop_size: Optional[int]=5, selection_num=2, use_e2_operator: bool=True, use_m1_operator: bool=True, use_m2_operator: bool=True, num_samplers: int=1, num_evaluators: int=1, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Evolutionary of Heuristics.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                          pass 'None' to disable this termination condition.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
                          pass 'None' to disable this termination condition.
        pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
        selection_num   : number of selected individuals while crossover.
        use_e2_operator : if use e2 operator.
        use_m1_operator : if use m1 operator.
        use_m2_operator : if use m2 operator.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    EoH.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.eoh.population
----------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.survival(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.eoh.profiler
--------------------------

EoHProfiler
~~~~~~~~~~~

.. code-block:: text

    class EoHProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    EoHProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHTensorboardProfiler(TensorboardProfiler, EoHProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHWandbProfiler
~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHWandbProfiler(WandBProfiler, EoHProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.eoh.prompt
------------------------

EoHPrompt
~~~~~~~~~

.. code-block:: text

    class EoHPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHPrompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.eoh.resume
------------------------

resume_eoh
~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_eoh(eoh: EoH, path)

llm4ad.method.eoh.sampler
-------------------------

EoHSampler
~~~~~~~~~~

.. code-block:: text

    class EoHSampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHSampler.__init__(self, llm: LLM, template_program: str | Program)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHSampler.get_thought_and_function(self, prompt: str) -> Tuple[str, Function]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EoHSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.funsearch.config
------------------------------

ProgramsDatabaseConfig
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ProgramsDatabaseConfig()

.. code-block:: text

    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30000

.. code-block:: text

    Configuration of a ProgramsDatabase.

    Attributes:
        functions_per_prompt: Number of previous programs to include in prompts.
        num_islands: Number of islands to maintain as a diversity mechanism.
        reset_period: How often (in seconds) the weakest islands should be reset.
        cluster_sampling_temperature_init: Initial temperature for softmax sampling
            of clusters within an island.
        cluster_sampling_temperature_period: Period of linear decay of the cluster
            sampling temperature.

llm4ad.method.funsearch.funsearch
---------------------------------

FunSearch
~~~~~~~~~

.. code-block:: text

    class FunSearch()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FunSearch.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, num_samplers: int=4, num_evaluators: int=4, samples_per_prompt: int=4, max_sample_nums: Optional[int]=20, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Function Search.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.funsearch.FunSearchProfiler'. If you do not want to use it, you can pass a 'None'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        num_samplers    : number of independent Samplers in the experiment.
        num_evaluators  : number of independent program Evaluators in the experiment.
        resume_mode     : in resume_mode, funsearch will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    FunSearch.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.funsearch.profiler
--------------------------------

FunSearchProfiler
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FunSearchProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FunSearchProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, program_db_register_interval: int=100, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    FunSearch Profiler.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    FunSearchProfiler.register_program_db(self, program_db: ProgramsDatabase)

.. code-block:: text

    Save ProgramDB to a file.
    [
        [{'score': -300, 'functions': [xxx, xxx, xxx, ...]}, {'score': -200, 'functions': [xxx, xxx, xxx, ...]}, {...}],
        [{...}, {...}],
    ]

FunSearchTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FunSearchTensorboardProfiler(TensorboardProfiler, FunSearchProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FunSearchTensorboardProfiler.__init__(self, *, log_dir: str | None=None, initial_num_samples=0, program_db_register_interval: int=100, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    FunSearch Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

FunSearchWandbProfiler
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FunSearchWandbProfiler(WandBProfiler, FunSearchProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FunSearchWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, program_db_register_interval: int=100, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    FunSearch Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

llm4ad.method.funsearch.programs_database
-----------------------------------------

Prompt
~~~~~~

.. code-block:: text

    class Prompt()

.. code-block:: text

    code: str
    version_generated: int
    island_id: int

.. code-block:: text

    A prompt produced by the ProgramsDatabase, to be sent to Samplers.

    Attributes:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the implementations
         included in the prompt. Used to direct the newly generated funsearch_impl
         into the same island.

ProgramsDatabase
~~~~~~~~~~~~~~~~

.. code-block:: text

    class ProgramsDatabase()

.. code-block:: text

    A collection of programs, organized as islands.

.. code-block:: text

    ProgramsDatabase.__init__(self, config: config_lib.ProgramsDatabaseConfig, template: Program, function_to_evolve: str) -> None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProgramsDatabase.get_prompt(self) -> Prompt

.. code-block:: text

    Returns a prompt containing implementations from one chosen island.

.. code-block:: text

    ProgramsDatabase.islands(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProgramsDatabase.register_function(self, function: Function, island_id: int | None, score: Any) -> None

.. code-block:: text

    Registers `program` in the database.

.. code-block:: text

    ProgramsDatabase.reset_islands(self) -> None

.. code-block:: text

    Resets the weaker half of islands.

Island
~~~~~~

.. code-block:: text

    class Island()

.. code-block:: text

    A sub-population of the programs database.

.. code-block:: text

    Island.__init__(self, template: Program, function_to_evolve: str, functions_per_prompt: int, cluster_sampling_temperature_init: float, cluster_sampling_temperature_period: int) -> None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Island.clusters(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Island.get_num_programs(self) -> int

.. code-block:: text

    Implemented by RZ, help us to profile.


.. code-block:: text

    Island.register_function(self, function: Function, score: int | float) -> None

.. code-block:: text

    Stores a program on this island, in its appropriate cluster.

.. code-block:: text

    Island.get_prompt(self) -> tuple[str, int]

.. code-block:: text

    Constructs a prompt containing functions from this island.

Cluster
~~~~~~~

.. code-block:: text

    class Cluster()

.. code-block:: text

    A cluster of programs on the same island and with the same Signature.

.. code-block:: text

    Cluster.__init__(self, score: float, implementation: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Cluster.score(self) -> float

.. code-block:: text

    Reduced score of the signature that this cluster represents.

.. code-block:: text

    Cluster.programs(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Cluster.register_program(self, program: Function) -> None

.. code-block:: text

    Adds `program` to the cluster.

.. code-block:: text

    Cluster.sample_program(self) -> Function

.. code-block:: text

    Samples a program, giving higher probability to shorther programs.

llm4ad.method.funsearch.resume
------------------------------

resume_funsearch
~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_funsearch(fs: FunSearch)

llm4ad.method.hillclimb.hillclimb
---------------------------------

HillClimb
~~~~~~~~~

.. code-block:: text

    class HillClimb()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HillClimb.__init__(self, llm: LLM, evaluation: Evaluation, profiler: HillClimbProfiler=None, max_sample_nums: Optional[int]=20, num_samplers: int=4, num_evaluators: int=4, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Hill Climbing Search.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.hillclimb.HillClimbProfiler'. If you do not want to use it, you can pass a 'None'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        num_samplers    : number of independent Samplers in the experiment.
        num_evaluators  : number of independent program Evaluators in the experiment.
        resume_mode     : in resume_mode, hillclimb will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    HillClimb.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.hillclimb.resume
------------------------------

resume_hillclimb
~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_hillclimb(hc: HillClimb)

llm4ad.method.hsevo.hsevo
-------------------------

HSEvo
~~~~~

.. code-block:: text

    class HSEvo()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HSEvo.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_sample_nums: Optional[int]=450, pop_size: Optional[int]=10, init_pop_size: Optional[int]=30, mutation_rate: float=0.5, hm_size: int=5, hmcr: float=0.7, par: float=0.5, bandwidth: float=0.2, max_iter: int=5, num_samplers: int=4, num_evaluators: int=4, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    HSEvo: Diversity-Driven Harmony Search + Genetic Algorithm using LLMs.

    Args:
        llm             : an instance of 'llm4ad.base.LLM'.
        evaluation      : an instance of 'llm4ad.base.Evaluation'.
        profiler        : an instance of 'llm4ad.method.hsevo.HSEvoProfiler'. Pass 'None' to disable.
        max_sample_nums : terminate after this many evaluated functions (maps to HSEvo's 'max_fe').
        pop_size        : population size used for selection/crossover.
        init_pop_size   : number of individuals generated for the initial population (with rotating personas).
        mutation_rate   : fraction of pop_size mutated from the elitist each generation.
        hm_size         : harmony-memory size for the harmony search operator.
        hmcr            : harmony memory considering rate.
        par             : pitch adjustment rate.
        bandwidth       : pitch adjustment bandwidth (fraction of each parameter range).
        max_iter        : number of harmony-search improvisation iterations.
        num_samplers    : number of threads used for batched LLM sampling.
        num_evaluators  : number of workers used for parallel evaluation.
        resume_mode     : if True, skip the initial population creation (see note in `run`).
        debug_mode      : if True, print detailed information.
        multi_thread_or_process_eval: 'thread' or 'process' pool for evaluation.
        **kwargs        : extra args passed to 'llm4ad.base.SecureEvaluator' (e.g. 'fork_proc').

.. code-block:: text

    HSEvo.evaluate_population(self, population: list[dict], hs_try_idx: int=None) -> list[dict]

.. code-block:: text

    Evaluate a population using LLM4AD's SecureEvaluator (parallel).

    Mirrors HSEvo's `evaluate_population` accounting (one `function_evals`
    increment per individual, best-obj logging), but the inner evaluation is
    delegated to `SecureEvaluator` and `obj = -score`.

.. code-block:: text

    HSEvo.response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict

.. code-block:: text

    Convert an LLM response (or a substituted code string) to an individual.

.. code-block:: text

    HSEvo.mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict

.. code-block:: text

    Mark an individual as invalid (verbatim).

.. code-block:: text

    HSEvo.init_population(self) -> None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.update_iter(self) -> None

.. code-block:: text

    Update after each iteration (verbatim).

.. code-block:: text

    HSEvo.random_select(self, population: list[dict]) -> list[dict]

.. code-block:: text

    Random selection with equal probability (verbatim).

    Note: HSEvo's black-box branch (which also filters `obj < seed_obj`) is
    omitted because LLM4AD has no notion of `problem_type`; we always keep
    the valid-individuals branch.

.. code-block:: text

    HSEvo.flash_reflection(self, population: list[dict]) -> None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.comprehensive_reflection(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.crossover(self, population: list[dict]) -> list[dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.mutate(self) -> list[dict]

.. code-block:: text

    Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals.

.. code-block:: text

    HSEvo.sel_individual_hs(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.initialize_harmony_memory(self, bounds)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.responses_to_population(self, responses, try_hs_idx=None) -> list[dict]

.. code-block:: text

    Convert responses (here, substituted code strings) to a population.

.. code-block:: text

    HSEvo.create_population_hs(self, str_code, parameter_ranges, harmony_memory, try_hs_idx=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.find_best_obj(self, population_hs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.create_new_harmony(self, harmony_memory, bounds)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.update_harmony_memory(self, population_hs, harmony_memory, new_harmony, func_block, parameter_ranges, try_hs_idx)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.harmony_search(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.evolve(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HSEvo.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.hsevo.profiler
----------------------------

HSEvoProfiler
~~~~~~~~~~~~~

.. code-block:: text

    class HSEvoProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HSEvoProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    HSEvo Profiler.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    HSEvoProfiler.register_population(self, population: List[Dict], generation: int)

.. code-block:: text

    Checkpoint the current HSEvo population (list of individual dicts).

HSEvoTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class HSEvoTensorboardProfiler(TensorboardProfiler, HSEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HSEvoTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    HSEvoTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

HSEvoWandbProfiler
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class HSEvoWandbProfiler(WandBProfiler, HSEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HSEvoWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    HSEvoWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.hsevo.prompt
--------------------------

make_func_signature
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Return a signature template with a ``{version}`` placeholder, e.g.
    ``def priority_v{version}(item: float, bins: np.ndarray) -> np.ndarray:``.

.. code-block:: text

    make_func_signature(function: Function) -> str

make_seed_func
~~~~~~~~~~~~~~

.. code-block:: text

    Render the template function as ``{name}_v1`` inside a python code block.

    HSEvo's seed_func.txt holds a complete trivial design. We render the LLM4AD
    template function (with its docstring + body) as ``_v1`` and wrap it in a
    ```python``` fence so that HSEvo's `extract_code_from_generator` extracts the
    full body via its regex path (rather than the def->first-return fallback).

.. code-block:: text

    make_seed_func(function: Function) -> str

make_func_desc
~~~~~~~~~~~~~~

.. code-block:: text

    Describe the target function's I/O. Prefer the template docstring; fall
    back to the task description.

.. code-block:: text

    make_func_desc(function: Function, task_description: str) -> str

llm4ad.method.hsevo.util
------------------------

format_messages
~~~~~~~~~~~~~~~

.. code-block:: text

    Build a chat message list with a system and a user turn.

    This mirrors HSEvo's `format_messages`, but takes the system/user strings
    directly (the original took a Hydra `cfg` it never used). LLM4AD's
    `HttpsApi`/`OpenAIAPI` backends accept this list directly as the `prompt`
    argument of `draw_sample`, which preserves HSEvo's system/user separation.

.. code-block:: text

    format_messages(system: str, user: str) -> List[Dict[str, str]]

extract_code_from_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Extract code from the response of the code generator (verbatim HSEvo).

    The only deviation from upstream is that the prepended ``scipy``/``torch``
    imports are guarded with ``try/except`` so that environments without those
    optional packages (e.g. the default LLM4AD env, which ships without torch)
    do not fail to execute every generated heuristic.

.. code-block:: text

    extract_code_from_generator(content)

filter_code
~~~~~~~~~~~

.. code-block:: text

    Remove lines containing signature and import statements (verbatim HSEvo).

.. code-block:: text

    filter_code(code_string)

extract_to_hs
~~~~~~~~~~~~~

.. code-block:: text

    Parse the harmony-search LLM response (verbatim HSEvo).

    Expects two ```python``` blocks: (1) the parameterised function with
    ``{param}`` placeholders in its default values and (2) a ``parameter_ranges``
    dict. Returns ``(parameter_ranges, function_block)`` or ``(None, None)``.

.. code-block:: text

    extract_to_hs(input_string: str)

llm4ad.method.lhns.elite_set
----------------------------

EliteSet
~~~~~~~~

.. code-block:: text

    class EliteSet()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EliteSet.__init__(self, elite_set_size, elite_set: List[LHNSProgram] | EliteSet | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EliteSet.elite_set(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EliteSet.update(self, function: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EliteSet.register_function(self, func: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EliteSet.has_duplicate_function(self, func: str | LHNSFunction) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EliteSet.selection(self) -> LHNSFunction

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.lhns.func_ruin
----------------------------

LHNSFunction
~~~~~~~~~~~~

.. code-block:: text

    class LHNSFunction(Function)

.. code-block:: text

    features: str = ''

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSFunction.convert_function_to_lhnsfunction(cls, function: Function) -> LHNSFunction

.. code-block:: text

    Convert a Function object to a LHNSFunction object.


LHNSProgram
~~~~~~~~~~~

.. code-block:: text

    class LHNSProgram(Program)

.. code-block:: text

    functions: list[LHNSFunction]

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSProgram.convert_program_to_lhnsprogram(cls, program: Program) -> LHNSProgram

.. code-block:: text

    Convert a Program object to a LHNSProgram object.


LHNSTextFunctionProgramConverter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class LHNSTextFunctionProgramConverter(TextFunctionProgramConverter)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSTextFunctionProgramConverter.text_to_program(cls, program_str: str) -> LHNSProgram | None

.. code-block:: text

    Returns Program object by parsing input text using Python AST.


.. code-block:: text

    LHNSTextFunctionProgramConverter.text_to_function(cls, program_str: str) -> LHNSFunction | None

.. code-block:: text

    Returns Function object by parsing input text using Python AST.


LHNSFunctionRuin
~~~~~~~~~~~~~~~~

.. code-block:: text

    class LHNSFunctionRuin()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSFunctionRuin.delete_function_snips(cls, function: LHNSFunction, cooling_rate: float) -> (LHNSFunction, int)

.. code-block:: text

    Delete lines from the function body.
    Args:
        function: The function to delete lines from.
        cooling_rate: The number of lines to delete.
    Returns:
        The modified function without lines deleted.

.. code-block:: text

    LHNSFunctionRuin.find_code_features(cls, prev_function: LHNSFunction, new_function: LHNSFunction) -> List[str]

.. code-block:: text

    Find the code features of the new function that are not in the previous function.
    Args:
        prev_function: The previous function.
        new_function: The new function.
    Returns:
        The code features of the new function that are not in the previous function.

.. code-block:: text

    LHNSFunctionRuin.merge_features(cls, function: LHNSFunction, features: List[str]) -> LHNSFunction

.. code-block:: text

    Merge the features into the function.
    Args:
        function: The function to merge features into.
        features: The features to merge.
    Returns:
        The modified function with features merged.

llm4ad.method.lhns.lhns
-----------------------

LHNS
~~~~

.. code-block:: text

    class LHNS()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNS.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_sample_nums: Optional[int]=100, cooling_rate: float=0.1, elite_set_size: int=5, method: str='vns', num_samplers: int=1, num_evaluators: int=1, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Evolutionary of Heuristics.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.eoh.LHNSProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                          pass 'None' to disable this termination condition.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
                          pass 'None' to disable this termination condition.
        pop_size        : population size, if set to 'None', LHNS will automatically adjust this parameter.
        selection_num   : number of selected individuals while crossover.
        use_e2_operator : if use e2 operator.
        use_m1_operator : if use m1 operator.
        use_m2_operator : if use m2 operator.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    LHNS.simulated_annealing(self, next_func: LHNSFunction, cooling_rate: float, trans_count: int) -> (bool, int)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNS.method_vns(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNS.method_ils(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNS.method_ts(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNS.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.lhns.profiler
---------------------------

LHNSProfiler
~~~~~~~~~~~~

.. code-block:: text

    class LHNSProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSProfiler.__init__(self, log_dir: Optional[str]=None, *, evaluation_name='Problem', method_name='LHNS', initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    LHNS Profiler
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        method_name        : the name of the search method.
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

LHNSTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class LHNSTensorboardProfiler(TensorboardProfiler, LHNSProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSTensorboardProfiler.__init__(self, log_dir: str | None=None, *, evaluation_name='Problem', method_name='LHNS', initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    LHNS Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        method_name        : the name of the search method.
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    LHNSTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

LHNSWandbProfiler
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class LHNSWandbProfiler(WandBProfiler, LHNSProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, evaluation_name='Problem', method_name='LHNS', initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    LHNS Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        method_name        : the name of the search method.
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    LHNSWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.lhns.prompt
-------------------------

LHNSPrompt
~~~~~~~~~~

.. code-block:: text

    class LHNSPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSPrompt.get_prompt_i1(cls, task_prompt: str, template_function: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSPrompt.get_prompt_merge(cls, task_prompt: str, indi: LHNSFunction, prev_indi: LHNSFunction, template_function: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSPrompt.get_prompt_rr(cls, task_prompt: str, indi: LHNSFunction, cooling_rate: float, template_function: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSPrompt.get_prompt_m(cls, task_prompt: str, indi: LHNSFunction, template_function: LHNSFunction)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.lhns.resume
-------------------------

resume_lhns
~~~~~~~~~~~

.. code-block:: text

    Reject incomplete LHNS checkpoints instead of constructing an invalid run.

.. code-block:: text

    resume_lhns(lhns, path)

resume_eoh
~~~~~~~~~~

.. code-block:: text

    Legacy misnamed entry point; LHNS checkpoint restoration is unsupported.

.. code-block:: text

    resume_eoh(eoh, path)

llm4ad.method.lhns.sampler
--------------------------

LHNSSampler
~~~~~~~~~~~

.. code-block:: text

    class LHNSSampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LHNSSampler.__init__(self, sampler: LLM, template_program: str | LHNSProgram)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSSampler.llm(self)

.. code-block:: text

    Underlying backend, shared with recording and search cleanup.

.. code-block:: text

    LHNSSampler.get_thought_and_function(self, prompt: str) -> Tuple[str, LHNSFunction]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LHNSSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.llamea.evaluation
-------------------------------

generate_evaluator
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    A LLaMEA instance works on llamea.Solution object, this generator
    takes the instance of evaluation, that have evaluate member mapping
    Callable -> float, and returns a function that takes that `float` value
    to update the `Solution` with appropriate fitness.

.. code-block:: text

    generate_evaluator(for_instance: Evaluation)

llm4ad.method.llamea.llamea
---------------------------

LLaMEA
~~~~~~

.. code-block:: text

    class LLaMEA(LLaMEA_Algorithm)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LLaMEA.__init__(self, llm: LLM, evaluator: Evaluation, iterations: int=50, n_parents: int=5, n_offsprings: int=5, role_prompt: str='', task_prompt: str='', example_prompt: str | None=None, minimization: bool=False, elitism: bool=True, **kwargs)

.. code-block:: text

    Args:
        evaluation_function: A function for scoring the fitness of the llm generated heuristic.
        llm: An instance of llamea.LLM one of the llms in Ollama, OpenAI, Gemini, DeepSeek,
        evaluation: An instance of llm4ad.base.Evaluation, which defines the way to calculate the score of a generated function.
        iterations: Iteration Count for evolution process,
        n_parents: Number of individuals in parent population (λ),
        n_offsprings: Number of individuals in offspring population (µ),
        role_prompt: LLM role prompt like: "You are an excellent scientific programmer tasked to solve the problem of GVRP.",
        task_prompt: Task prompt is llm4ad.tasks.*.template.task_description for solving a problem,
        example_prompt: Example propmt is llm4ad.tasks.*.template.template_program for solving a problem,
        minimisation: Flag to define direction of optimality.
        elitism: A bool flag to run algorithm in (λ + µ) if set True, else (λ , µ).

llm4ad.method.llamea.llamea_llm
-------------------------------

LlameaLLM
~~~~~~~~~

.. code-block:: text

    class LlameaLLM(HttpsApi)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LlameaLLM.__init__(self, host, key, model, timeout=60, code_pattern=None, name_pattern=None, desc_pattern=None, cs_pattern=None, logger=None, **kwargs)

.. code-block:: text

    Https API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        timeout: API timeout.

.. code-block:: text

    LlameaLLM.set_logger(self, logger)

.. code-block:: text

    Sets the logger object to log the conversation.

    Args:
        logger (Logger): A logger object to log the conversation.

.. code-block:: text

    LlameaLLM.sample_solution(self, session_messages: list, parent_ids: list | None=None, HPO: bool=False, base_code: str | None=None, diff_mode: bool=False)

.. code-block:: text

    Generate or mutate a solution using the language model.

    Args:
        session_messages: Conversation history for the LLM.
        parent_ids: Identifier(s) of parent solutions.
        HPO: If ``True``, attempt to extract a configuration space.
        base_code: Existing code to patch when ``diff_mode`` is ``True``.
        diff_mode: When ``True``, interpret the LLM response as a unified
            diff patch to apply to ``base_code`` rather than full source
            code.

    Returns:
        tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

    Raises:
        NoCodeException: If the language model fails to return any code.
        Exception: Captures and logs any other exceptions that occur during the interaction.

.. code-block:: text

    LlameaLLM.extract_configspace(self, message)

.. code-block:: text

    Extracts the configuration space definition in json from a given message string using regular expressions.

    Args:
        message (str): The message string containing the algorithm code.

    Returns:
        ConfigSpace: Extracted configuration space object.

.. code-block:: text

    LlameaLLM.extract_algorithm_code(self, message)

.. code-block:: text

    Extracts algorithm code from a given message string using regular expressions.

    Args:
        message (str): The message string containing the algorithm code.

    Returns:
        str: Extracted algorithm code.

    Raises:
        NoCodeException: If no code block is found within the message.

.. code-block:: text

    LlameaLLM.extract_algorithm_description(self, message)

.. code-block:: text

    Extracts algorithm description from a given message string using regular expressions.

    Args:
        message (str): The message string containing the algorithm name and code.

    Returns:
        str: Extracted algorithm name or empty string.

llm4ad.method.llamea.sampler
----------------------------

LLaMEASampler
~~~~~~~~~~~~~

.. code-block:: text

    class LLaMEASampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LLaMEASampler.__init__(self, llm)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.mcts
---------------------------

MCTSNode
~~~~~~~~

.. code-block:: text

    class MCTSNode()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MCTSNode.__init__(self, algorithm, code, obj, depth=0, individual=None, is_root=False, parent=None, visit=0, raw_info=None, Q=0)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTSNode.add_child(self, child_node: MCTSNode)

.. code-block:: text

    Documentation gap: no method docstring is present.

MCTS
~~~~

.. code-block:: text

    class MCTS()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MCTS.__init__(self, root_answer, alpha, lambad0)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS.backpropagate(self, node: MCTSNode)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS.uct(self, node: MCTSNode, eval_remain)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS.is_fully_expanded(self, node: MCTSNode)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.mcts_ahd
-------------------------------

MCTS_AHD
~~~~~~~~

.. code-block:: text

    class MCTS_AHD()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MCTS_AHD.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_sample_nums: Optional[int]=100, init_size: Optional[float]=4, pop_size: Optional[int]=10, selection_num: int=2, num_samplers: int=1, num_evaluators: int=1, alpha: float=0.5, lambda_0: float=0.1, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Evolutionary of Heuristics.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
                          pass 'None' to disable this termination condition.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
                          pass 'None' to disable this termination condition.
        init_size       : population size, if set to 'None', EoH will automatically adjust this parameter.
        pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
        selection_num   : number of selected individuals while crossover.
        alpha           : a parameter for the UCT formula, which is used to balance exploration and exploitation.
        lambda_0        : a parameter for the UCT formula, which is used to balance exploration and exploitation.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    MCTS_AHD.check_duplicate(self, population, code)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS_AHD.check_duplicate_obj(self, population, code)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS_AHD.population_management_s1(self, pop_input, size)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS_AHD.expand(self, mcts: MCTS, node_set, cur_node: MCTSNode, option: str)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MCTS_AHD.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.population
---------------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, init_pop_size, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.next_gen_pop(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.survival(self, pop_size: int=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.survival_s1(self, pop_size: int=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function, pop: List[Function]=None) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self, pop: List[Function]=None) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection_e1(self, pop: List[Function]=None) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.profiler
-------------------------------

MAProfiler
~~~~~~~~~~

.. code-block:: text

    class MAProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MAProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    MCTS_AHD Profiler
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    MAProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

MATensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MATensorboardProfiler(TensorboardProfiler, MAProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MATensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    MCTS_AHD Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    MATensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

MAWandbProfiler
~~~~~~~~~~~~~~~

.. code-block:: text

    class MAWandbProfiler(WandBProfiler, MAProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MAWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    MCTS_AHD Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    MAWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.prompt
-----------------------------

MAPrompt
~~~~~~~~

.. code-block:: text

    class MAPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MAPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MAPrompt.get_prompt_s1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mcts_ahd.resume
-----------------------------

resume_ma
~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_ma(ma: MCTS_AHD, path)

llm4ad.method.mcts_ahd.sampler
------------------------------

MASampler
~~~~~~~~~

.. code-block:: text

    class MASampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MASampler.__init__(self, llm: LLM, template_program: str | Program)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MASampler.get_thought_and_function(self, task_description: str, prompt: str) -> Tuple[str, Function]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MASampler.get_prompt_refine(self, task_prompt: str, idea: str, code: str)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MASampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.meoh.meoh
-----------------------

MEoH
~~~~

.. code-block:: text

    class MEoH()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoH.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_generations: int | None=10, max_sample_nums: int | None=100, pop_size: int=20, selection_num=2, use_e2_operator: bool=True, use_m1_operator: bool=True, use_m2_operator: bool=True, num_samplers: int=1, num_evaluators: int=1, num_objs: int=2, *, resume_mode: bool=False, initial_sample_num: int | None=None, debug_mode: bool=False, multi_thread_or_process_eval: str='thread', **kwargs)

.. code-block:: text

    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.meoh.MEoHProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        pop_size        : population size.
        selection_num   : number of selected individuals while crossover.
        use_e2_operator : if use e2 operator.
        use_m1_operator : if use m1 operator.
        use_m2_operator : if use m2 operator.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    MEoH.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.meoh.population
-----------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.elitist(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.meoh.profiler
---------------------------

MEoHProfiler
~~~~~~~~~~~~

.. code-block:: text

    class MEoHProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoHProfiler.__init__(self, log_dir: str | None=None, num_objs=2, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

MEoHTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MEoHTensorboardProfiler(TensorboardProfiler, MEoHProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoHTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

MEoHWandbProfiler
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MEoHWandbProfiler(WandBProfiler, MEoHProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoHWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.meoh.prompt
-------------------------

MEoHPrompt
~~~~~~~~~~

.. code-block:: text

    class MEoHPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoHPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHPrompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.meoh.resume
-------------------------

resume_meoh
~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_meoh(meoh: MEoH)

llm4ad.method.meoh.sampler
--------------------------

MEoHSampler
~~~~~~~~~~~

.. code-block:: text

    class MEoHSampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MEoHSampler.__init__(self, llm: LLM, template_program: str | Program)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHSampler.get_thought_and_function(self, prompt: str) -> Tuple[str, Function]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MEoHSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mles.mles
-----------------------

MLES
~~~~

.. code-block:: text

    class MLES()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MLES.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_generations: Optional[int]=10, max_sample_nums: Optional[int]=100, pop_size: Optional[int]=5, selection_num=2, operators: tuple=('e1', 'e2', 'm1_M', 'm2_M'), num_samplers: int=1, num_evaluators: int=1, *, resume_mode: bool=False, initial_sample_nums_max: int=50, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', seed_path='', **kwargs)

.. code-block:: text

    Evolutionary of Heuristics.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                          pass 'None' to disable this termination condition.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
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

.. code-block:: text

    MLES.check_before_running(self)

.. code-block:: text

    Verify that the evaluation object provides required text descriptions for specific operators.

.. code-block:: text

    MLES.init_from_local_algo_base(self)

.. code-block:: text

    Bootstrap the population using pre-existing algorithms from a local JSON seed file.

.. code-block:: text

    MLES.run(self)

.. code-block:: text

    Entry point for the MLES search process:
    1. Initialize: Load from local seeds and/or perform LLM-based cold start.
    2. Evolve: Run the iterative mutation/crossover pipeline until termination.
    3. Finalize: Shut down profilers and save final results.

.. code-block:: text

    MLES.using_flow(self, worst_case_percent=10, top_k=None)

.. code-block:: text

    Executes the 'Using Mode' pipeline:
    1. Loads the latest evolved population from local storage.
    2. Filters the top-K performing algorithms if specified.
    3. Evaluates all selected algorithms on a set of new test instances.
    4. Identifies the best-performing algorithm for each specific instance.
    5. Computes overall statistics and worst-case performance metrics.

.. code-block:: text

    MLES.messages_to_string(self, messages, image_placeholder='<<<IMAGE>>>')

.. code-block:: text

    Convert a structured messages list (OpenAI-style) into a single formatted string.
    Supports both 'text' and 'image_url' content types.

    :param messages: list of dicts with 'role' and 'content'
    :param image_placeholder: str or callable, placeholder inserted for images
    :return: str

llm4ad.method.mles.population
-----------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Manages a collection of designed algorithms.
    Handles individual registration, duplicate detection, environmental selection,
    and rank-based parent selection for evolution.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Initialize the population container.
    :param pop_size: Maximum number of individuals allowed in the active population.
    :param generation: Initial generation counter.
    :param pop: Optional initial set of individuals.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Returns the current active individuals (the 'survivors').

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Returns the current evolutionary generation count.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Registers a new individual into the population.
    Triggers environmental selection once the buffer reaches pop_size.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Checks for redundancy based on code string equality or identical fitness scores.

.. code-block:: text

    Population.selection(self, number=1, best_must=False, mode='exp', pressure=0.5) -> List[Function]

.. code-block:: text

    Selects parents for the next evolutionary step using rank-based probability distributions.

    :param number: Number of individuals to select.
    :param best_must: If True, ensures the global best is included in the output.
    :param mode: Probability distribution mode ('exp' for exponential, 'linear', or uniform).
    :param pressure: Selection pressure for 'exp' mode (higher = more likely to pick top individuals).

llm4ad.method.mles.profiler
---------------------------

MLESProfiler
~~~~~~~~~~~~

.. code-block:: text

    class MLESProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MLESProfiler.__init__(self, log_dir: Optional[str]=None, *, evaluation_name='Problem', method_name='EoH', initial_num_samples=0, log_style='complex', create_random_path=True, run_mode='Training', **kwargs)

.. code-block:: text

    MLES Profiler
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        method_name        : the name of the search method.
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    MLESProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESProfiler.using_final(self, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHTensorboardProfiler(TensorboardProfiler, MLESProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHWandbProfiler
~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHWandbProfiler(WandBProfiler, MLESProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.mles.prompt
-------------------------

MLESPrompt
~~~~~~~~~~

.. code-block:: text

    class MLESPrompt()

.. code-block:: text

    Template factory for Multimodal LLM prompts.
    Converts evolutionary search operations (Initialization, Crossover, Mutation, Modification)
    into structured instructions for the LLM.

.. code-block:: text

    MLESPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESPrompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    [INIT] Zero-shot Initialization Prompt.
    Requests the LLM to design a baseline algorithm from scratch based
    solely on the task description.

.. code-block:: text

    MLESPrompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    [CROSSOVER] Diversity-driven Crossover.
    Provides multiple parent algorithms and explicitly asks the LLM
    to create something 'totally different' to avoid local optima.

.. code-block:: text

    MLESPrompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESPrompt.get_prompt_e1_advanced(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_e2_advanced(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Return Messages

.. code-block:: text

    MLESPrompt.get_prompt_e2_M(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m1_M(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m1_M_only_image(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m1_M_text_info(cls, task_prompt: str, indi: Function, template_function: Function, information_explanation)

.. code-block:: text

    Use text information [no-image] to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_image_description(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m1_M_image_description(cls, task_prompt: str, indi: Function, template_function: Function, description: str)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m2_M(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m2_M_image_description(cls, task_prompt: str, indi: Function, template_function: Function, description: str)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_e1_nothought(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESPrompt.get_prompt_e2_nothought(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MLESPrompt.get_prompt_m1_M_nothought(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

.. code-block:: text

    MLESPrompt.get_prompt_m2_M_nothought(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Use Figures to instruct the design progress

llm4ad.method.mles.sampler
--------------------------

MLESSampler
~~~~~~~~~~~

.. code-block:: text

    class MLESSampler()

.. code-block:: text

    The MLESSampler serves as the interface for interacting with the LLM.
    It handles the execution of sampling requests and uses regular expressions
    to extract structured thoughts, image descriptions, and executable code
    from raw LLM text outputs.

.. code-block:: text

    MLESSampler.__init__(self, sampler: LLM, template_program: str | Program)

.. code-block:: text

    Initialize the sampler with an LLM engine and a code template.

    :param sampler: The LLM instance used to generate responses.
    :param template_program: A template structure used to reconstruct
                             Function objects from raw code strings.

.. code-block:: text

    MLESSampler.llm(self)

.. code-block:: text

    Underlying backend, shared with recording and search cleanup.

.. code-block:: text

    MLESSampler.get_thought_and_function(self, prompt: str, image64s: List=None, messages: List=None) -> tuple[str | None, Function | None, str]

.. code-block:: text

    Main sampling method: Sends a prompt to the LLM and extracts both
    the algorithmic reasoning (thought) and the resulting code.

    :return: A tuple containing (extracted_thought, Function_object, raw_response_string).

.. code-block:: text

    MLESSampler.get_image_description(self, prompt: str, image64s: List=None, messages: List=None)

.. code-block:: text

    Used for multimodal feedback loops. Extracts a textual analysis of
    visual data (e.g., control trajectories) generated by the LLM.

    :return: A tuple containing (extracted_description, raw_response_string).

.. code-block:: text

    MLESSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts content within single curly braces {like this}.
    This corresponds to the 'algorithmic concept' requested in MLESPrompt.

.. code-block:: text

    MLESSampler.trim_description_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts content within double curly braces {{like this}}.
    Used for multi-line textual descriptions of visual results.

llm4ad.method.moead.moead
-------------------------

MOEAD
~~~~~

.. code-block:: text

    class MOEAD()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEAD.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_generations: int | None=10, max_sample_nums: int | None=100, pop_size: int=20, selection_num=5, use_e2_operator: bool=True, use_m1_operator: bool=True, use_m2_operator: bool=True, num_samplers: int=1, num_evaluators: int=1, num_objs: int=2, *, resume_mode: bool=False, initial_sample_num: int | None=None, debug_mode: bool=False, multi_thread_or_process_eval: str='thread', **kwargs)

.. code-block:: text

    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.moead.MOEADProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        pop_size        : population size.
        selection_num   : number of selected individuals while crossover.
        use_e2_operator : if use e2 operator.
        use_m1_operator : if use m1 operator.
        use_m2_operator : if use m2 operator.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    MOEAD.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.moead.population
------------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.elitist(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self, pref: np.array) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.moead.profiler
----------------------------

MOEADProfiler
~~~~~~~~~~~~~

.. code-block:: text

    class MOEADProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEADProfiler.__init__(self, log_dir: str | None=None, num_objs=2, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

MOEADTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MOEADTensorboardProfiler(TensorboardProfiler, MOEADProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEADTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

MOEADWandbProfiler
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MOEADWandbProfiler(WandBProfiler, MOEADProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEADWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.moead.prompt
--------------------------

MOEADPrompt
~~~~~~~~~~~

.. code-block:: text

    class MOEADPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEADPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADPrompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.moead.resume
--------------------------

resume_moead
~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_moead(moead: MOEAD)

llm4ad.method.moead.sampler
---------------------------

MOEADSampler
~~~~~~~~~~~~

.. code-block:: text

    class MOEADSampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEADSampler.__init__(self, llm: LLM, template_program: str | Program)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADSampler.get_thought_and_function(self, prompt: str) -> Tuple[str, Function]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEADSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.nsga2.nsga2
-------------------------

NSGA2
~~~~~

.. code-block:: text

    class NSGA2()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_generations: int | None=10, max_sample_nums: int | None=100, pop_size: int=20, selection_num=5, use_e2_operator: bool=True, use_m1_operator: bool=True, use_m2_operator: bool=True, num_samplers: int=1, num_evaluators: int=1, num_objs: int=2, *, resume_mode: bool=False, initial_sample_num: int | None=None, debug_mode: bool=False, multi_thread_or_process_eval: str='thread', **kwargs)

.. code-block:: text

    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.nsga2.NSGA2Profiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        pop_size        : population size.
        selection_num   : number of selected individuals while crossover.
        use_e2_operator : if use e2 operator.
        use_m1_operator : if use m1 operator.
        use_m2_operator : if use m2 operator.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    NSGA2.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.nsga2.population
------------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.elitist(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.nsga2.profiler
----------------------------

NSGA2Profiler
~~~~~~~~~~~~~

.. code-block:: text

    class NSGA2Profiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2Profiler.__init__(self, log_dir: str | None=None, num_objs=2, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Profiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

NSGA2TensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class NSGA2TensorboardProfiler(TensorboardProfiler, NSGA2Profiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2TensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2TensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

NSGA2WandbProfiler
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class NSGA2WandbProfiler(WandBProfiler, NSGA2Profiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2WandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2WandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.nsga2.prompt
--------------------------

NSGA2Prompt
~~~~~~~~~~~

.. code-block:: text

    class NSGA2Prompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2Prompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_prompt_i1(cls, task_prompt: str, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Prompt.get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.nsga2.resume
--------------------------

resume_nsga2
~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_nsga2(nsga2: NSGA2)

llm4ad.method.nsga2.sampler
---------------------------

NSGA2Sampler
~~~~~~~~~~~~

.. code-block:: text

    class NSGA2Sampler()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    NSGA2Sampler.__init__(self, llm: LLM, template_program: str | Program)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Sampler.get_thought_and_function(self, prompt: str) -> Tuple[str, Function]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    NSGA2Sampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.partevo.base
--------------------------

Evoind
~~~~~~

.. code-block:: text

    class Evoind()

.. code-block:: text

    function: Function
    cluster_id: Optional[int] = None
    reflection: str = ''
    feature: List[Any] = field(default_factory=list)

.. code-block:: text

    Evolutionary Individual used within the PartEvo framework.
    Wraps a Function object and stores its metadata within a specific algorithm cluster.

    An Evoind is the basic unit for selection, elimination, and other operations
    within the algorithm pool.

.. code-block:: text

    Evoind.set_feature(self, feature_given)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Evoind.set_reflection(self, reflection_given)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.partevo.clustermanager
------------------------------------

get_bert_embeddings
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    An optional algorithm feature mapping for algorithm clustering.
    Generates text embeddings using a BERT model for semantic analysis of algorithm descriptions (Thoughts).

    Args:
        texts: A list of strings to be embedded.
        model_path: Local path to BERT model. Defaults to 'bert-base-uncased'.

    Returns:
        A numpy array of shape (len(texts), 768) containing the CLS token embeddings.

.. code-block:: text

    get_bert_embeddings(texts: List[str], model_path: str=None) -> np.ndarray

individual_feature
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Calculates multi-modal features for the population to facilitate niching/clustering.
    Supported types:
    - 'ast': Structural similarity via CodeBLEU.
    - 'language': Semantic similarity via BERT.
    - 'random': Gaussian noise baseline.
    - 'objective': Performance-based features.

.. code-block:: text

    individual_feature(population: List[Evoind], feature_type: Tuple[str, ...]=('ast',), save_path: str='', bert_model_path: str=None)

ClusterManager
~~~~~~~~~~~~~~

.. code-block:: text

    class ClusterManager()

.. code-block:: text

    Global Orchestrator for the Partition-to-Evolve (PartEvo) framework.

    Workflow:
    1. Maintains a global population and an offspring buffer.
    2. Executes 'Cold Start' until sufficient individuals are collected.
    3. Performs Clustering based on extracted code features.
    4. Manages ClusterUnits, which handle sub-population evolution and operator selection.
    5. Implements resource allocation (resource tilt) based on cluster performance.

.. code-block:: text

    ClusterManager.__init__(self, pop_size: int=16, n_clusters: int=4, intra_operators: Tuple[str, ...]=('re', 'se', 'cc', 'lge'), intra_operators_parent_num: Dict[str, int]=None, intra_operators_frequency: Optional[Dict]=None, use_resource_tilt: bool=False, resource_tilt_alpha: float=2.0, feature_type: Tuple[str, ...]=('ast',), bert_model_path: str=None, debug_flag: bool=False)

.. code-block:: text

    Initialize the Cluster Manager.

    Args:
        pop_size: Target size for the global population.
        n_clusters: Number of niches/clusters to maintain.
        intra_operators: List of available evolutionary operators.
        use_resource_tilt: If True, high-performing clusters get more sampling opportunities.

.. code-block:: text

    ClusterManager.initial_population_clustering(self)

.. code-block:: text

    Creating initial niching.
    Converts Functions to Evoinds, extracts multi-modal features, and initializes ClusterUnits via Clustering (such as, K-Means)

.. code-block:: text

    ClusterManager.select_parent(self) -> Tuple[List[Function], str, int]

.. code-block:: text

    Selects parent individuals for the next evolutionary step.
    Handles both 'Cold Start' (random selection) and 'Clustered' (unit-specific selection) phases.

.. code-block:: text

    ClusterManager.register_function(self, offspring: Function, from_which_cluster: int=None)

.. code-block:: text

    Registers a new function into the global population, external archive and according niches.
    Handles deduplication and elitist replacement.

.. code-block:: text

    ClusterManager.debug_status(self)

.. code-block:: text

    Prints diagnostic information about cluster health and scores.

llm4ad.method.partevo.clusterunit
---------------------------------

ClusterUnit
~~~~~~~~~~~

.. code-block:: text

    class ClusterUnit()

.. code-block:: text

    An Algorithm Pool Unit that manages a sub-population of a specific algorithmic branch (niche).

    Responsibilities:
    - Performs parent selection for the specific cluster using various strategies.
    - Manages the local population including registration, deduplication, and survival of the fittest.
    - Tracks stagnation and improvement metrics for the niche.

.. code-block:: text

    ClusterUnit.__init__(self, cluster_id: int, max_pop_size: int, intra_operators: Tuple[str, ...], intra_operators_parent_num: Dict[str, int], pop: List[Evoind] | ClusterUnit | None=None)

.. code-block:: text

    Initializes the ClusterUnit (Niche).

    Args:
        cluster_id: Unique identifier for this cluster/niche.
        max_pop_size: Maximum number of individuals allowed in the sub-population.
        intra_operators: Tuple of operator names (e.g., 're', 'se', 'cc', 'lge').
        intra_operators_parent_num: Dictionary mapping operators to required parent counts.
        pop: Initial population or existing ClusterUnit to clone from.

.. code-block:: text

    ClusterUnit.population(self)

.. code-block:: text

    Returns a thread-safe copy of the current sub-population list.

.. code-block:: text

    ClusterUnit.selection(self, existing_functions: Optional[List[Function]]=None, best_must: bool=False, mode: str=None, help_inter: bool=False, help_number: int=1, tournament_k: int=3, target_instances: Optional[List[Any]]=None) -> Tuple[List[Function], str | None, bool]

.. code-block:: text

    Unified selection interface for the clusterUnit.

    Args:
        existing_functions: List of functions to exclude from selection.
        best_must: If True, ensures the current cluster-best is included in selection.
        mode: Selection strategy ('top', 'tournament', 'random', 'exp', etc.).
        help_inter: Set to True if this cluster is providing individuals for another cluster.
        help_number: Number of individuals to provide in help mode.
        target_instances: Specific problem instances to prioritize performance on.

    Returns:
        Tuple: (selected_parent_functions, current_operator_name, need_external_collaboration)

.. code-block:: text

    ClusterUnit.has_duplicate_function(self, func: str | Evoind) -> bool

.. code-block:: text

    Checks if an individual with identical code already exists in the cluster.

.. code-block:: text

    ClusterUnit.register_individual(self, new_individual: Evoind)

.. code-block:: text

    Registers a new individual into the population.
    Implements 'Identical Score Replacement' to maintain niche diversity.

.. code-block:: text

    ClusterUnit.do_pop_management(self)

.. code-block:: text

    Cleans the population by removing duplicates and truncating to max size.
    Survival is based strictly on score.

.. code-block:: text

    ClusterUnit.get_best_individual(self) -> Evoind | None

.. code-block:: text

    Returns the highest-scoring individual in the Niche.

llm4ad.method.partevo.externalArchive
-------------------------------------

ExternalArchive
~~~~~~~~~~~~~~~

.. code-block:: text

    class ExternalArchive()

.. code-block:: text

    ExternalArchive (Global External Memory Pool)

    Acts at the ClusterManager level to record the global evolutionary trajectory.
    Implements a 'Waterfall' and 'Hard Negatives' mechanism:
    1. Elites: Maintains strict purity, keeping only the Global Top-K candidates.
    2. Hard Negatives: High-scoring candidates ousted from the Elite pool are demoted here.
    3. SE Context: Provides cached, high-contrast context (Elites vs. Hard Negatives)
       for the Semantic Exploration (SE) operator, fully thread-safe with fault tolerance.

.. code-block:: text

    ExternalArchive.__init__(self, max_elites: int=5, max_hard_negatives: int=30, summary_update_interval: int=12)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ExternalArchive.register(self, candidate: Function)

.. code-block:: text

    Main entry point: evaluate and register a new candidate function.

.. code-block:: text

    ExternalArchive.fetch_summary_context(self) -> Tuple[str, bool, Dict[str, List[Function]]]

.. code-block:: text

    Called by the outer 'se' operator.
    Returns: (current_cached_summary, requires_update_flag, context_samples_dict)

.. code-block:: text

    ExternalArchive.update_global_summary(self, generated_summary: Optional[str])

.. code-block:: text

    Called by the LLM thread to write back the generated summary and release lock.
    Handles failed LLM generations gracefully.

.. code-block:: text

    ExternalArchive.global_summary(self) -> str

.. code-block:: text

    Read-only property to get the current summary without triggering updates.

llm4ad.method.partevo.partevo
-----------------------------

PartEvo
~~~~~~~

.. code-block:: text

    class PartEvo()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PartEvo.__init__(self, llm: LLM, evaluation: Evaluation, profiler: Union[ProfilerBase, PartEvoProfiler]=None, max_generations: Optional[int]=10, max_sample_nums: Optional[int]=100, pop_size: int=16, operators: tuple=('re', 'se', 'cc', 'lge'), operators_parent_num: Optional[Dict]=None, operators_frequency: Optional[Dict]=None, num_samplers: int=1, num_evaluators: int=1, resume_mode: bool=False, initial_sample_nums_max: int=50, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', partition_method: str='kmeans', partition_number: int=4, local_algo_base='', feature_used: Tuple[Literal['ast', 'language'], ...]=('ast',), use_resource_tilt: bool=False, bert_model_path: str='', **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvo.init_from_local_algo_base(self)

.. code-block:: text

    Warm start: Loads pre-existing algorithms from a local JSON file to seed the population.

.. code-block:: text

    PartEvo.init_using_llms(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvo.run(self)

.. code-block:: text

    Executes the full pipeline:
    1. Initialization (Seeds + LLM Cold Start)
    2. Evolutionary training loop
    3. Final result cleanup.

.. code-block:: text

    PartEvo.using_flow(self, worst_case_percent=10, top_k=None)

.. code-block:: text

    Executes the 'Using Mode' pipeline:
    1. Loads the latest evolved population from local storage.
    2. Filters the top-K performing algorithms if specified.
    3. Evaluates all selected algorithms on a set of new test instances.
    4. Identifies the best-performing algorithm for each specific instance.
    5. Computes overall statistics and worst-case performance metrics.

.. code-block:: text

    PartEvo.messages_to_string(self, messages, image_placeholder='<<<IMAGE>>>')

.. code-block:: text

    Convert a structured messages list (OpenAI-style) into a single formatted string.
    Supports both string contents and list of dicts ('text' and 'image_url' content types).

    :param messages: list of dicts with 'role' and 'content'
    :param image_placeholder: str or callable, placeholder inserted for images
    :return: str

llm4ad.method.partevo.profiler
------------------------------

PartEvoProfiler
~~~~~~~~~~~~~~~

.. code-block:: text

    class PartEvoProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PartEvoProfiler.__init__(self, log_dir: Optional[str]=None, *, evaluation_name='Problem', method_name='EoH', initial_num_samples=0, log_style='complex', create_random_path=True, run_mode='Training', **kwargs)

.. code-block:: text

    MLES Profiler
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        method_name        : the name of the search method.
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    PartEvoProfiler.register_population(self, pop: ClusterManager)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoProfiler.using_final(self, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHTensorboardProfiler(TensorboardProfiler, PartEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

EoHWandbProfiler
~~~~~~~~~~~~~~~~

.. code-block:: text

    class EoHWandbProfiler(WandBProfiler, PartEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EoHWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    EoH Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    EoHWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.partevo.prompt
----------------------------

PartEvoPrompt
~~~~~~~~~~~~~

.. code-block:: text

    class PartEvoPrompt()

.. code-block:: text

    Template factory for Multimodal LLM prompts.
    Converts evolutionary search operations (Initialization, Crossover, Mutation, Modification)
    into structured instructions for the LLM.

.. code-block:: text

    PartEvoPrompt.create_instruct_prompt(cls, prompt: str) -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoPrompt.get_system_prompt(cls) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoPrompt.get_prompt_batch_init(cls, task_prompt: str, template_function: 'Function', current_population: List['Function'], branch_novelty=30)

.. code-block:: text

    Generate a structured prompt to initialize a batch of novel algorithms.

.. code-block:: text

    PartEvoPrompt.get_prompt_reflection(cls, task_prompt: str, func: 'Function', template_function: 'Function')

.. code-block:: text

    Generate a structured prompt to ask the LLM for targeted suggestions (reflection) on a given algorithm.

.. code-block:: text

    PartEvoPrompt.get_prompt_summary(cls, task_prompt: str, template_function: 'Function', summary_context_samples: Dict[str, List['Function']], current_summary: str='') -> List[Dict]

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoPrompt.get_prompt_re(cls, task_prompt: str, parent_func: Function, template_function: Function, reflection: str)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoPrompt.get_prompt_se(cls, task_prompt: str, parent_func: 'Function', template_function: 'Function', global_summary: str) -> List[Dict]

.. code-block:: text

    Generate a structured prompt for the Semantic Exploration (SE) operator.
    Uses a 'God View' global summary to guide the modification of the current algorithm.

.. code-block:: text

    PartEvoPrompt.get_prompt_cn(cls, task_prompt: str, parents: List[Function], template_function: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoPrompt.get_prompt_lge(cls, task_prompt: str, parents: List['Function'], template_function: 'Function') -> List[Dict]

.. code-block:: text

    Generates the prompt for the Local-Global Exploration (LGE) operator.
    Inspired by Particle Swarm Optimization (PSO), it evolves parents[0] (the current algorithm)
    by drawing insights from superior references (Local Best and/or Global Best) in parents[1:].

llm4ad.method.partevo.sampler
-----------------------------

PartEvoSampler
~~~~~~~~~~~~~~

.. code-block:: text

    class PartEvoSampler()

.. code-block:: text

    The MLESSampler serves as the interface for interacting with the LLM.
    It handles the execution of sampling requests and uses regular expressions
    to extract structured thoughts, image descriptions, and executable code
    from raw LLM text outputs.

.. code-block:: text

    PartEvoSampler.__init__(self, sampler: LLM, template_program: str | Program)

.. code-block:: text

    Initialize the sampler with an LLM engine and a code template.

    :param sampler: The LLM instance used to generate responses.
    :param template_program: A template structure used to reconstruct
                             Function objects from raw code strings.

.. code-block:: text

    PartEvoSampler.llm(self)

.. code-block:: text

    Underlying backend, shared with recording and search cleanup.

.. code-block:: text

    PartEvoSampler.get_thought_and_function(self, prompt: str, image64s: List=None, messages: List=None) -> tuple[str | None, Function | None, str]

.. code-block:: text

    Main sampling method: Sends a prompt to the LLM and extracts both
    the algorithmic reasoning (thought) and the resulting code.

    :return: A tuple containing (extracted_thought, Function_object, raw_response_string).

.. code-block:: text

    PartEvoSampler.get_image_description(self, prompt: str, image64s: List=None, messages: List=None)

.. code-block:: text

    Used for multimodal feedback loops. Extracts a textual analysis of
    visual data (e.g., control trajectories) generated by the LLM.

    :return: A tuple containing (extracted_description, raw_response_string).

.. code-block:: text

    PartEvoSampler.get_reflection(self, prompt: str, messages: List=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoSampler.get_summary(self, prompt: str, messages: List=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PartEvoSampler.trim_thought_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts content within <concept>...</concept> tags.
    This corresponds to the 'algorithmic concept' requested in the Prompt.

.. code-block:: text

    PartEvoSampler.trim_reflection_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts the content inside <reflection>...</reflection> tags.

.. code-block:: text

    PartEvoSampler.trim_summary_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts the content inside <reflection>...</reflection> tags.

.. code-block:: text

    PartEvoSampler.trim_description_from_response(cls, response: str) -> str | None

.. code-block:: text

    Extracts content within <description>...</description> tags.
    Used for multi-line textual descriptions of visual results.

llm4ad.method.randsample.randsample
-----------------------------------

RandSample
~~~~~~~~~~

.. code-block:: text

    class RandSample()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    RandSample.__init__(self, llm: LLM, evaluation: Evaluation, profiler: RandSampleProfiler=None, num_samplers: int=4, num_evaluators: int=4, max_sample_nums: Optional[int]=20, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Random Sampling
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.randsample.RandSampleProfiler'. If you do not want to use it, you can pass a 'None'.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
        num_samplers    : number of independent Samplers in the experiment.
        num_evaluators  : number of independent program Evaluators in the experiment.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    RandSample.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.randsample.resume
-------------------------------

resume_randsample
~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_randsample(rs: RandSample)

llm4ad.method.reevo.population
------------------------------

Population
~~~~~~~~~~

.. code-block:: text

    class Population()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    Population.__init__(self, pop_size, generation=0, pop: List[Function] | Population | None=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.population(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.elite_function(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.generation(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.survival(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.register_function(self, func: Function)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.has_duplicate_function(self, func: str | Function) -> bool

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    Population.selection(self) -> Function

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.reevo.profiler
----------------------------

ReEvoProfiler
~~~~~~~~~~~~~

.. code-block:: text

    class ReEvoProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ReEvoProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    ReEvo Profiler
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    ReEvoProfiler.register_population(self, pop: Population)

.. code-block:: text

    Documentation gap: no method docstring is present.

ReEvoTensorboardProfiler
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ReEvoTensorboardProfiler(TensorboardProfiler, ReEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ReEvoTensorboardProfiler.__init__(self, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    Profiler for Tensorboard.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    ReEvoTensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

ReEvoWandbProfiler
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class ReEvoWandbProfiler(WandBProfiler, ReEvoProfiler)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ReEvoWandbProfiler.__init__(self, wandb_project_name: str, log_dir: str | None=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    Profiler for Wandb.
    Args:
        wandb_project_name : the name of the wandb project
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        **kwargs           : kwargs for wandb

.. code-block:: text

    ReEvoWandbProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.reevo.prompt
--------------------------

ReEvoPrompt
~~~~~~~~~~~

.. code-block:: text

    class ReEvoPrompt()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ReEvoPrompt.get_pop_init_prompt(cls, task_prompt: str, template_function: Function) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ReEvoPrompt.get_short_term_reflection_prompt(cls, task_prompt: str, indivs: List[Function]) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ReEvoPrompt.get_crossover_prompt(cls, task_prompt: str, short_term_reflection_prompt: str, indivs: List[Function]) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ReEvoPrompt.get_long_term_reflection_prompt(cls, task_prompt: str, prior_long_term_reflection: str, new_short_term_reflections: List[str]) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ReEvoPrompt.get_elist_mutation_prompt(cls, task_prompt: str, long_term_reflection_prompt: str, elite_function: Function) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.reevo.reevo
-------------------------

ReEvo
~~~~~

.. code-block:: text

    class ReEvo()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ReEvo.__init__(self, llm: LLM, evaluation: Evaluation, profiler: ProfilerBase=None, max_sample_nums: Optional[int]=100, pop_size: Optional[int]=20, mutation_rate: float=0.5, num_samplers: int=1, num_evaluators: int=1, *, resume_mode: bool=False, debug_mode: bool=False, multi_thread_or_process_eval: Literal['thread', 'process']='thread', **kwargs)

.. code-block:: text

    Reflective Evolution.
    Args:
        llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
        evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
        profiler        : an instance of 'llm4ad.method.reevo.ReEvoProfiler'. If you do not want to use it, you can pass a 'None'.
        max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                          pass 'None' to disable this termination condition.
        max_sample_nums : maximum reserved candidate attempts, including malformed output and failed evaluations; reflection calls are separate.
                          pass 'None' to disable this termination condition.
        pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
        resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
        debug_mode      : if set to True, we will print detailed information.
        multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
            multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
            setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
            Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
            and you set this argument to 'thread'.
        **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.

.. code-block:: text

    ReEvo.run(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.method.reevo.resume
--------------------------

resume_reevo
~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    resume_reevo(reevo: ReEvo, path)

llm4ad.task.__init__
--------------------

import_all_evaluation_classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.

.. code-block:: text

    import_all_evaluation_classes(root_directory)

llm4ad.task.machine_learning.acrobot.evaluation
-----------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on car mountain problem.

.. code-block:: text

    evaluate(env: gym.Env, action_select: callable) -> float

AcrobotEvaluation
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class AcrobotEvaluation(Evaluation)

.. code-block:: text

    Evaluator for car mountain problem.

.. code-block:: text

    AcrobotEvaluation.__init__(self, max_steps=500, timeout_seconds=20, **kwargs)

.. code-block:: text

    Args:
        - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
        - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

    Attributes:
        - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.

.. code-block:: text

    AcrobotEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.machine_learning.acrobot.test
-----------------------------------------

choose_action
~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    choose_action(ct1: float, st1: float, ct2: float, st2: float, avt1: float, avt2: float, last_action: int) -> int

llm4ad.task.machine_learning.car_mountain.evaluation
----------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on car mountain problem.

.. code-block:: text

    evaluate(env: gym.Env, action_select: callable) -> float

CarMountainEvaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class CarMountainEvaluation(Evaluation)

.. code-block:: text

    Evaluator for car mountain problem.

.. code-block:: text

    CarMountainEvaluation.__init__(self, max_steps=500, timeout_seconds=20, **kwargs)

.. code-block:: text

    Args:
        - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
        - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

    Attributes:
        - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.

.. code-block:: text

    CarMountainEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.machine_learning.car_mountain.q-learning
----------------------------------------------------

QLearning
~~~~~~~~~

.. code-block:: text

    class QLearning()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    QLearning.__init__(self, actions_space, learning_rate=0.01, reward_decay=0.99, e_greedy=0.6)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.choose_action(self, state)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.toBins(self, clip_min, clip_max, num)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.digit(self, x, bin)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.digitize_state(self, observation)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.learn(self, state, action, r, next_state)

.. code-block:: text

    Documentation gap: no method docstring is present.

train
~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    train()

test
~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    test()

run_test
~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    run_test()

llm4ad.task.machine_learning.car_mountain.test
----------------------------------------------

QLearning
~~~~~~~~~

.. code-block:: text

    class QLearning()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    QLearning.__init__(self, actions_space, learning_rate=0.01, reward_decay=0.99, e_greedy=0.6)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.choose_action(self, state)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.toBins(self, clip_min, clip_max, num)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.digit(self, x, bin)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.digitize_state(self, observation)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    QLearning.learn(self, state, action, r, next_state)

.. code-block:: text

    Documentation gap: no method docstring is present.

train
~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    train()

taste
~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    taste()

choose_action
~~~~~~~~~~~~~

.. code-block:: text

    Return the action for the car to proceed the next move.
    Args:
        pos: Car's position, a float ranges between [-1.2, 0.6].
        v: Car's velocity, a float ranges between [-0.07, 0.07].
        last_action: Car's next move, a int ranges between [0, 1, 2].
    Return:
         An integer representing the selected action for the car.
         0: accelerate to left
         1: don't accelerate
         2: accelerate to right

.. code-block:: text

    choose_action(pos: float, v: float, last_action: int) -> int

run_test
~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    run_test()

llm4ad.task.machine_learning.car_mountain_continue.evaluation
-------------------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on car mountain problem.

.. code-block:: text

    evaluate(env: gym.Env, action_select: callable) -> float

CarMountainCEvaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class CarMountainCEvaluation(Evaluation)

.. code-block:: text

    Evaluator for car mountain problem.

.. code-block:: text

    CarMountainCEvaluation.__init__(self, max_steps=500, timeout_seconds=20, **kwargs)

.. code-block:: text

    Args:
        - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
        - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

    Attributes:
        - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.

.. code-block:: text

    CarMountainCEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.machine_learning.car_mountain_continue.test
-------------------------------------------------------

choose_action
~~~~~~~~~~~~~

.. code-block:: text

    Return the action for the car to proceed the next move.
    Args:
        pos: Car's position, a float ranges between [-1.2, 0.6].
        v: Car's velocity, a float ranges between [-0.07, 0.07].
        last_action: Car's next move, a int ranges between [0, 1, 2].
    Return:
         An integer representing the selected action for the car.
         0: accelerate to left
         1: don't accelerate
         2: accelerate to right

.. code-block:: text

    choose_action(pos: float, v: float, last_action: float) -> [float]

run_test
~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    run_test()

llm4ad.task.machine_learning.car_racing.evaluation
--------------------------------------------------

RacingCarEvaluation
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class RacingCarEvaluation(Evaluation)

.. code-block:: text

    Evaluator for CarRacing/custom control problems.

.. code-block:: text

    RacingCarEvaluation.__init__(self, whocall='Eoh', max_steps=1200, timeout_seconds=180, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RacingCarEvaluation.evaluate(self, action_select: callable, ins_to_be_evaluated_id: Set | List | None=None, training_mode=True) -> Optional[dict]

.. code-block:: text

    🔒 MOSTLY BOILERPLATE: Aggregates results across instances.
    🛠️ Users only need to modify the final return dictionary if they want to track extra custom data.

.. code-block:: text

    RacingCarEvaluation.evaluate_single(self, action_select: callable, env_seed=42, skip_frame=1)

.. code-block:: text

    # =========================================================================
    # 🛠️ USER DEFINED (ii): Single Episode Evaluation & Image Generation
    # This is the core logic. You must run your environment, collect rewards,
    # generate a plot/image demonstrating the behavior, and return it as base64.
    # =========================================================================

.. code-block:: text

    RacingCarEvaluation.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.machine_learning.car_racing.original_text
-----------------------------------------------------

heuristic_action
~~~~~~~~~~~~~~~~

.. code-block:: text

    Select a heuristic action as [steering, gas, brake].

.. code-block:: text

    heuristic_action(observation)

heuristic_action_4o
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    heuristic_action_4o(observation)

display_observation
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Display the current observation.

.. code-block:: text

    display_observation(observation)

llm4ad.task.machine_learning.moon_lander.evaluation
---------------------------------------------------

MoonLanderEvaluation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MoonLanderEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the Lunar Lander control problem.

.. code-block:: text

    MoonLanderEvaluation.__init__(self, whocall='Eoh', max_steps=200, timeout_seconds=300, **kwargs)

.. code-block:: text

    Args:
        - 'max_steps' (int): Maximum number of steps allowed per episode in the MountainCar-v0 environment (default is 500).
        - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

    Attributes:
        - 'env' (gym.Env): The MountainCar-v0 environment with a modified maximum episode length.

.. code-block:: text

    MoonLanderEvaluation.evaluate(self, action_select: callable, ins_to_be_evaluated_id: Set | List | None=None, training_mode=True) -> Optional[dict]

.. code-block:: text

    🔒 MOSTLY BOILERPLATE: Aggregates results across instances.

.. code-block:: text

    MoonLanderEvaluation.evaluate_single(self, action_select: callable, env_seed=42)

.. code-block:: text

    # =========================================================================
    # 🛠️ USER DEFINED (ii): Single Episode Evaluation & Image Generation
    # Run the environment, track fuel/rewards, and generate the image canvas.
    # =========================================================================

.. code-block:: text

    MoonLanderEvaluation.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MoonLanderEvaluation.create_base64(self, original_input, fitness, recoder, which_image)

.. code-block:: text

    Helper to convert the numpy canvas into a base64 encoded string for MLES.

.. code-block:: text

    MoonLanderEvaluation.feature_pipeline(self, seed, env_max_episode_steps=100)

.. code-block:: text

    (Advanced) No action → feature extraction for clustering.

.. code-block:: text

    MoonLanderEvaluation.visualize_instance_features_base64(self, mode: str='combined') -> str

.. code-block:: text

    (Advanced) Visualization tool for trajectory features.

.. code-block:: text

    MoonLanderEvaluation.show_instance_features(self, mode: str='combined', duichen=False)

.. code-block:: text

    (Advanced) Interactive UI for instance features.

.. code-block:: text

    MoonLanderEvaluation.show_clustered_features(self, json_file_path: str, data_source: str, cluster_key: str)

.. code-block:: text

    Visualize instance features with colors assigned from JSON cluster data.

    The data source and JSON cluster-assignment key are selected independently.

    Args:
        json_file_path: Path to final_output.json or using_final_results.json.
        data_source: 'training' selects self.instance_feature; 'testing' selects
            self.to_be_solve_ins_feature.
        cluster_key: JSON assignment key: 'cluster_id_instances' for training,
            or 'apply_cluster_of_each_instance' / 'match_cluster_of_each_instance'
            for evaluation results.

.. code-block:: text

    MoonLanderEvaluation.filter_custom_instances_by_x(self, instance_input: list | dict, threshold: float=4.5) -> tuple[list, list]

.. code-block:: text

    Generate features and split instances by their horizontal position.

    Args:
        instance_input: List of seeds or a dictionary mapping instance IDs to seeds.
        threshold: Absolute horizontal-position threshold; defaults to 4.5.

    Returns:
        (satisfied_ids, not_satisfied_ids). The first group has X outside
        [-threshold, threshold]; the second group lies within that interval.

llm4ad.task.machine_learning.moon_lander.feature_pipeline
---------------------------------------------------------

moon_lander_feature
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on moon lander problem.

.. code-block:: text

    moon_lander_feature(seed, env_max_episode_steps=200)

llm4ad.task.machine_learning.moon_lander.test
---------------------------------------------

choose_action
~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    choose_action(state, reward, last_action)

llm4ad.task.machine_learning.pendulum.evaluation
------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    evaluate(env: gym.Env, action_select: callable) -> float | None

evaluate_single
~~~~~~~~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on the pendulum swing-up problem.

.. code-block:: text

    evaluate_single(env: gym.Env, action_select: callable) -> float

PendulumEvaluation
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class PendulumEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the pendulum swing-up problem.

.. code-block:: text

    PendulumEvaluation.__init__(self, max_steps=500, timeout_seconds=20, **kwargs)

.. code-block:: text

    Args:
        - 'max_steps' (int): Maximum number of steps allowed per episode in the Pendulum-v1 environment (default is 200).
        - '**kwargs' (dict): Additional keyword arguments passed to the parent class initializer.

    Attributes:
        - 'env' (gym.Env): The Pendulum-v1 environment with a modified maximum episode length.

.. code-block:: text

    PendulumEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.machine_learning.pendulum.test
------------------------------------------

choose_action
~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    choose_action(x: float, y: float, angular_velocity: float, last_action: float) -> float

llm4ad.task.optimization.admissible_set.evaluation
--------------------------------------------------

ASPEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class ASPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for online bin packing problem.

.. code-block:: text

    ASPEvaluation.__init__(self, timeout_seconds=60, dimension=15, weight=10, **kwargs)

.. code-block:: text

    Args:
        - 'dimension' (int): The dimension of tested case (default is 15).
        - 'weight' (int): The wight of tested case (default is 10).

.. code-block:: text

    ASPEvaluation.expand_admissible_set(self, pre_admissible_set: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]

.. code-block:: text

    Expands a pre-admissible set into an admissible set.

.. code-block:: text

    ASPEvaluation.get_surviving_children(self, extant_elements, new_element, valid_children)

.. code-block:: text

    Returns the indices of `valid_children` that remain valid after adding `new_element` to `extant_elements`.

.. code-block:: text

    ASPEvaluation.evaluate(self, priority: callable) -> int

.. code-block:: text

    Generates a symmetric constant-weight admissible set I(n, w).

.. code-block:: text

    ASPEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.bp_1d_construct.evaluation
---------------------------------------------------

BP1DEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class BP1DEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the 1D Bin Packing Problem.

.. code-block:: text

    BP1DEvaluation.__init__(self, timeout_seconds: int=60, n_bins: int=500, n_instance: int=8, n_items: int=500, bin_capacity: int=100, **kwargs)

.. code-block:: text

    Args:
        n_bins: The number of available bins at the beginning.

.. code-block:: text

    BP1DEvaluation.plot_bins(self, bins: List[List[int]], bin_capacity: int)

.. code-block:: text

    Plot the bins and their contents.

    Args:
        bins: A list of bins, where each bin is a list of item weights.
        bin_capacity: The capacity of each bin.

.. code-block:: text

    BP1DEvaluation.pack_items(self, item_weights: List[int], bin_capacity: int, eva: Callable, n_bins: int) -> Tuple[int, List[List[int]]]

.. code-block:: text

    Pack items into bins using a constructive heuristic.

    Args:
        item_weights: A list of item weights.
        bin_capacity: The capacity of each bin.
        eva: The constructive heuristic function to select the next item and bin.
        n_bins: The number of available bins at the beginning.

    Returns:
        A tuple containing:
        - The total number of bins used.
        - A list of bins, where each bin is a list of item weights.

.. code-block:: text

    BP1DEvaluation.evaluate(self, eva: Callable) -> float

.. code-block:: text

    Evaluate the constructive heuristic for the 1D Bin Packing Problem.

    Args:
        instance_data: List of tuples containing the item weights and bin capacity.
        n_ins: Number of instances to evaluate.
        eva: The constructive heuristic function to evaluate.
        n_bins: The number of available bins at the beginning.

    Returns:
        The average number of bins used across all instances.

.. code-block:: text

    BP1DEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.bp_1d_construct.get_instance
-----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_items: int, bin_capacity: int)

.. code-block:: text

    Initialize the GetData class for the 1D Bin Packing Problem.

    Args:
        n_instance: Number of instances to generate.
        n_items: Number of items.
        bin_capacity: Capacity of each bin.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the 1D Bin Packing Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - item_weights: A list of item weights.
        - bin_capacity: The capacity of each bin.

llm4ad.task.optimization.bp_2d_construct.evaluation
---------------------------------------------------

BP2DEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class BP2DEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the 2D Bin Packing Problem.

.. code-block:: text

    BP2DEvaluation.__init__(self, timeout_seconds: int=120, n_bins: int=100, n_instance: int=8, n_items: int=100, bin_width: int=100, bin_height: int=100, **kwargs)

.. code-block:: text

    Args:
        n_bins: The number of available bins at the beginning.

.. code-block:: text

    BP2DEvaluation.plot_solution(self, bins: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]], bin_dimensions: Tuple[int, int])

.. code-block:: text

    Plot the final packing solution for 2D bin packing.

    Args:
        bins: A list of bins, where each bin is a list of tuples containing the corner and dimensions of packed items.
        bin_dimensions: A tuple representing the (width, height) of the bin.

.. code-block:: text

    BP2DEvaluation.pack_items_2d(self, item_dimensions: List[Tuple[int, int]], bin_dimensions: Tuple[int, int], eva: Callable, n_bins: int) -> Tuple[int, List[List[Tuple[int, int]]]] | None

.. code-block:: text

    Pack items into bins using a constructive heuristic for the 2D Bin Packing Problem.
    The bins are represented as a discrete point matrix to track feasible areas.

    Args:
        item_dimensions: A list of tuples, where each tuple represents the (width, height) of an item.
        bin_dimensions: A tuple representing the (width, height) of the bin.
        eva: The constructive heuristic function to select the next item and bin.
        n_bins: The number of available bins at the beginning.

    Returns:
        A tuple containing:
        - The total number of bins used.
        - A list of bins, where each bin is a list of item dimensions.
        Returns None if any item remains unpacked.

.. code-block:: text

    BP2DEvaluation.evaluate_2d(self, eva: Callable) -> float | None

.. code-block:: text

    Evaluate the constructive heuristic for the 2D Bin Packing Problem.

    Args:
        eva: callable function of constructive heuristic.

    Returns:
        The average number of bins used across all instances.

.. code-block:: text

    BP2DEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.bp_2d_construct.get_instance
-----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_items: int, bin_width: int, bin_height: int)

.. code-block:: text

    Initialize the GetData class for the 2D Bin Packing Problem.

    Args:
        n_instance: Number of instances to generate.
        n_items: Number of items.
        bin_width: Width of each bin.
        bin_height: Height of each bin.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the 2D Bin Packing Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - item_dimensions: A list of tuples, where each tuple represents the (width, height) of an item.
        - bin_dimensions: A tuple representing the (width, height) of the bin.

llm4ad.task.optimization.cflp_construct.evaluation
--------------------------------------------------

CFLPEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class CFLPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the Capacitated Facility Location Problem.

.. code-block:: text

    CFLPEvaluation.__init__(self, timeout_seconds: int=60, n_instance: int=16, n_facilities: int=50, n_customers: int=50, max_capacity: int=100, max_demand: int=20, max_cost: int=50, **kwargs)

.. code-block:: text

    Initialize the evaluator.

.. code-block:: text

    CFLPEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CFLPEvaluation.plot_solution(self, facility_capacities: List[int], customer_demands: List[int], assignments: List[List[int]], assignment_costs: List[List[int]])

.. code-block:: text

    Plot the final solution of assignments for the Capacitated Facility Location Problem.

    Args:
        facility_capacities: A list of facility capacities.
        customer_demands: A list of customer demands.
        assignments: A list of assignments, where each assignment is a list of customer indices assigned to a facility.
        assignment_costs: A 2D list (matrix) of costs, where the cost of assigning customer j to facility i is assignment_costs[i][j].

.. code-block:: text

    CFLPEvaluation.assign_customers(self, facility_capacities: List[int], customer_demands: List[int], assignment_costs: List[List[int]], eva: Callable) -> Tuple[int, List[List[int]]] | None

.. code-block:: text

    Assign customers to facilities using a constructive heuristic.

    Args:
        facility_capacities: A list of facility capacities.
        customer_demands: A list of customer demands.
        assignment_costs: A 2D list (matrix) of costs, where the cost of assigning customer j to facility i is assignment_costs[i][j].
        eva: The constructive heuristic function to select the next customer-facility assignment.

    Returns:
        A tuple containing:
        - The total cost of the assignments.
        - A list of assignments, where each assignment is a list of customer indices assigned to a facility.
        Returns None if the heuristic stops before assigning every customer.

.. code-block:: text

    CFLPEvaluation.evaluate_cflp(self, eva: Callable) -> float | None

.. code-block:: text

    Evaluate the constructive heuristic for the Capacitated Facility Location Problem.

    Args:
        instance_data: List of dictionaries containing facility capacities, customer demands, and assignment costs.
        n_ins: Number of instances to evaluate.
        eva: The constructive heuristic function to evaluate.

    Returns:
        The average total cost across all instances.

llm4ad.task.optimization.cflp_construct.get_instance
----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_facilities: int, n_customers: int, max_capacity: int, max_demand: int, max_cost: int)

.. code-block:: text

    Initialize the GetDataCFLP class for the Capacitated Facility Location Problem.

    Args:
        n_instance: Number of instances to generate.
        n_facilities: Number of facilities.
        n_customers: Number of customers.
        max_capacity: Maximum capacity of any facility.
        max_demand: Maximum demand of any customer.
        max_cost: Maximum cost for assigning a customer to a facility.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Capacitated Facility Location Problem.

    Returns:
        A list of dictionaries, where each dictionary contains:
        - facility_capacities: A list of capacities for each facility.
        - customer_demands: A list of demands for each customer.
        - assignment_costs: A 2D list (matrix) of costs, where the cost of assigning
          customer j to facility i is assignment_costs[i][j].

llm4ad.task.optimization.co_bench.aircraft_landing_co_bench.evaluation
----------------------------------------------------------------------

ALEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class ALEvaluationCB(Evaluation)

.. code-block:: text

    Evaluator for aircraft landing.

.. code-block:: text

    ALEvaluationCB.__init__(self, timeout_seconds=300, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    ALEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ALEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ALEvaluationCB.load_data(self, input_str)

.. code-block:: text

    Reads the aircraft landing scheduling problem instance from a string.
    The string contains a single case with the following format:
        Line 1: <num_planes> <freeze_time>
        For each plane (i = 1, …, num_planes):
            - A line with 6 numbers:
                  appearance_time earliest_landing_time target_landing_time
                  latest_landing_time penalty_cost_early penalty_cost_late
            - One or more subsequent lines containing exactly num_planes separation times.
              (Separation times for plane i with respect to planes 1..num_planes. They may span multiple lines.)
    Returns:
        A dictionary containing the keys:
            - "num_planes"  : int
            - "freeze_time" : float
            - "planes"      : list of dicts (one per plane)
            - "separation"  : list of lists of floats

.. code-block:: text

    ALEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a proposed aircraft landing schedule.
    Expects the following keys in kwargs:
        - num_planes  : int, number of planes.
        - num_runways : int, number of runways.
        - freeze_time : float.
        - planes      : list of dicts, each containing:
                          "earliest", "target", "latest", "penalty_early", "penalty_late".
        - separation  : list of lists (floats), where separation[i][j] is the required gap after plane i lands
                        before plane j can land when they are assigned to the same runway.
        - schedule    : dict mapping plane_id (1-indexed) to a dict with keys:
                          "landing_time" (float) and "runway" (int).
    The evaluation performs these checks:
        1. Each plane's landing time is within its allowed time window.
        2. Each plane is assigned to a runway in the range [1, num_runways].
        3. For every two distinct planes i and j assigned to the same runway,
           if plane i lands at or before plane j then the gap must be at least
           the required separation time.
    The total penalty is computed as follows for each plane:
        - If landing_time < target: penalty = (target - landing_time) * penalty_early.
        - If landing_time > target: penalty = (landing_time - target) * penalty_late.
        - If landing_time == target: no penalty.
    Returns:
        The total penalty (a float) if the schedule is feasible.
    Raises:
        ValueError with an informative message if any constraint is violated.

.. code-block:: text

    ALEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ALEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.assignment_problem_co_bench.evaluation
------------------------------------------------------------------------

APEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class APEvaluationCB(Evaluation)

.. code-block:: text

    Evaluator for assignment problem.

.. code-block:: text

    APEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    APEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    APEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    APEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads input string content and separates it into multiple cases for the assignment problem.
    The input is expected to contain one or more cases. Each case has the following format:
      - The first non-empty line of the case is a single integer n (the number of items/agents).
      - The remaining tokens in the case provide the cost information. This can be in one of two formats:
          1. Dense Format: Exactly n*n numeric tokens (row-major order).
          2. Sparse Format: A sequence of tokens in groups of three: (i, j, cost). Any (i,j)
             not specified is assigned a cost equal to 1000 times the maximum provided cost in that row.
    Cases in the input are separated by one or more blank lines.
    Parameters:
      input_string (str): The input content as string.
    Returns:
      A list of dictionaries, each containing:
          - "n": int, the number of items.
          - "cost_matrix": numpy.ndarray of shape (n, n) with the assignment costs.

.. code-block:: text

    APEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates the solution of the Assignment Problem for a single case.
    Parameters:
      - case (dict): A dictionary containing the case data with keys:
            * "n": int, the number of items/agents.
            * "cost_matrix": numpy.ndarray, the cost matrix.
      - solution (dict): A dictionary representing the solution returned by solve(), with keys:
            * "total_cost": numeric, the total cost reported by the solution.
            * "assignment": list of tuples (i, j) where i is the item (1-indexed) and j is the assigned agent (1-indexed).
    Returns:
      A numeric score representing the total cost computed from the cost_matrix based on the provided assignment.
    The function performs the following checks:
      - Each item (1 to n) must be assigned exactly once.
      - Each agent (1 to n) must be assigned exactly once.
      - The computed total cost (from the cost_matrix and assignment) must match the reported total_cost
        (within a small tolerance). If not, the computed total is used.

.. code-block:: text

    APEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    APEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.assortment_problem_co_bench.evaluation
------------------------------------------------------------------------

AssortPEvaluationCB
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class AssortPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    AssortPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    AssortPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    AssortPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    AssortPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads the input data for one or more cases from a TXT file.
    The file format is as follows for each case:
      1. A line with three tokens: m n waste_cost
         - m: number of piece types (int)
         - n: number of stock rectangles (int)
         - waste_cost: cost per unit area of waste (float)
      2. Next n lines: each with "length width fixed_cost" for a stock rectangle.
      3. Next m lines: each with "length width min max value" for a piece.
    If the file contains multiple cases, they should be separated by at least one blank line.
    Returns:
      A list of dictionaries, one per case. Each dictionary contains:
        - "m": int
        - "n": int
        - "waste_cost": float
        - "stocks": list of dicts (each with keys 'length', 'width', 'fixed_cost')
        - "pieces": list of dicts (each with keys 'length', 'width', 'min', 'max', 'value')

.. code-block:: text

    AssortPEvaluationCB.eval_func(self, m, n, waste_cost, stocks, pieces, objective, placements)

.. code-block:: text

    Evaluates the solution for the arrangement optimization problem using waste area percentage as the metric.
    The overall waste area percentage is computed as:
         overall_waste_percentage = (total unused area) / (total stock area)
    where a lower percentage indicates better utilization. This metric disregards piece values and fixed costs.
    Infeasible solutions (due to piece count constraint violations or using more than 2 distinct stock types)
    will raise an exception.
    Inputs:
      - m (int): Number of piece types.
      - n (int): (Not used directly) originally denoted the number of stock rectangles, but now placements are stock instances.
      - waste_cost (float): Not used in this metric.
      - stocks (list of dict): Each dict represents a stock type with keys:
            'length', 'width', 'fixed_cost'.
            There is an infinite supply of each stock type.
      - pieces (list of dict): Each dict represents a piece type with keys:
            'length', 'width', 'min', 'max', 'value'.
      - placements (dict): Mapping from stock instance id (1-indexed) to a dictionary with keys:
            'stock_type' : (1-indexed id of the stock type used for this instance),
            'placements' : a list of placements for pieces within that stock instance.
                           Each placement is a dict with keys:
                              'piece'       (piece type, 1-indexed),
                              'x'           (x-coordinate of the bottom-left corner),
                              'y'           (y-coordinate of the bottom-left corner),
                              'orientation' (0 for normal, 1 for rotated 90°).
      - objective (float): Not used in this metric (provided for reference).
    Returns:
      The overall waste area percentage (float) if all constraints are satisfied.
    Raises:
      Exception: If a placement violates boundary or overlap conditions, if piece count constraints are not met,
                 or if more than 2 distinct stock types are used.

.. code-block:: text

    AssortPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    AssortPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.bp_1d_co_bench.evaluation
-----------------------------------------------------------

BP1DEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class BP1DEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    BP1DEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    BP1DEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    BP1DEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    BP1DEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load test cases from string content for the bin packing problem.
    The input format:
      1. The first nonempty line is an integer P, the number of test cases.
      2. For each test case:
         a. A line with the problem identifier (e.g., "u120_00").
         b. A line with three space-separated numbers: bin_capacity, num_items, best_known.
            (Note: bin_capacity and item sizes may be given as floats.)
         c. Then num_items lines, each with a number representing an item size.
    Returns:
      A list of dictionaries. Each dictionary contains the input data for one test case with keys:
        - 'id':           Problem identifier (string)
        - 'bin_capacity': Bin capacity (float)
        - 'num_items':    Number of items (int)
        - 'best_known':   Best known number of bins (int)
        - 'items':        List of item sizes (list of floats)

.. code-block:: text

    BP1DEvaluationCB.eval_func(self, id, bin_capacity, num_items, best_known, items, num_bins, bins)

.. code-block:: text

    Evaluate the bin packing solution for a single test case.
    Parameters (from the input case and the solution):
      - id:           Problem identifier (string)
      - bin_capacity: Bin capacity (int)
      - num_items:    Number of items (int)
      - best_known:   Best known number of bins (int)
      - items:        List of item sizes (list of ints)
      - num_bins:     Number of bins used in the solution (int)
      - bins:         List of lists; each inner list contains 1-based item indices assigned to that bin.
    Returns:
      A scalar score (int). The score is the total number of bins used.
      If the solution is invalid (e.g., item indices are wrong, items not used exactly once, or bin capacity exceeded),
      a penalty of 1,000,000 is added to the score.

.. code-block:: text

    BP1DEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.capacitated_warehouse_location_co_bench.evaluation
------------------------------------------------------------------------------------

CWLEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class CWLEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CWLEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CWLEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CWLEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CWLEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads one or more problem cases from the input string.
    Expected Input String Format for each case:
      Line 1: Two integers: m n
      Next m lines: Each line contains two numbers: capacity fixed_cost for a warehouse.
      Next n lines: Each line contains: demand (a number) followed by m numbers representing the cost of
                  allocating the customer's demand to each warehouse.
    If the input string contains multiple cases, the cases appear sequentially.
    Returns:
      A list of dictionaries, each corresponding to one case. Each dictionary has the keys:
         - 'm': Number of potential warehouses (int)
         - 'n': Number of customers (int)
         - 'warehouses': List of dictionaries; each with keys 'capacity' and 'fixed_cost'
         - 'customers': List of dictionaries; each with keys 'demand' and 'costs' (list of floats)

.. code-block:: text

    CWLEvaluationCB.eval_func(self, m, n, warehouses, customers, warehouse_open, assignments, **kwargs)

.. code-block:: text

    Evaluates the solution for the Capacitated Warehouse Location Problem with Splittable Customer Demand,
    using a weighted average cost for each customer.
    For each customer:
      - The sum of allocations across warehouses must equal the customer's demand.
      - The assignment cost is computed as the weighted average of the per-unit costs,
        i.e., for each warehouse i, the fraction of demand allocated from i multiplied by its cost.
      - No positive allocation is allowed for a warehouse that is closed.
    Additionally, for each warehouse:
      - The total allocated demand must not exceed its capacity.
    The total cost is computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum over customers of the weighted average assignment cost)
    Input Parameters:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: List of dictionaries (each with 'capacity' and 'fixed_cost')
      - customers: List of dictionaries (each with 'demand' and 'costs' (list of floats representing per-unit cost))
      - warehouse_open: List of m integers (0 or 1) indicating whether each warehouse is closed or open.
      - assignments: List of n lists (each of length m) where assignments[j][i] represents the amount of
                     customer j's demand allocated to warehouse i.
      - kwargs: Other parameters (not used here).
    Returns:
      A floating-point number representing the total cost if the solution is feasible.
    Raises:
      Exception: If any of the following conditions are violated:
          - The sum of allocations for any customer does not equal its demand.
          - Any positive allocation is made to a closed warehouse.
          - Any warehouse's total allocated demand exceeds its capacity.

.. code-block:: text

    CWLEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CWLEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.common_due_date_scheduling_co_bench.evaluation
--------------------------------------------------------------------------------

CDDSEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class CDDSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CDDSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CDDSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CDDSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CDDSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads the input file and returns a list of cases.
    Each case is represented as a dictionary containing:
         - 'jobs': a list of tuples (p, a, b) for each job.
         - 'h': a float parameter for due date computation (default set to 0.6).
    The input format:
         • The first token is an integer T indicating the number of cases.
         • For each case:
               – The first integer is n, the number of jobs.
               – The following n lines each contain three space-separated integers: p, a, and b.
    Returns:
         List[dict]: A list where each element is a dictionary with the keys 'jobs' and 'h'.

.. code-block:: text

    CDDSEvaluationCB.eval_func(self, jobs, schedule, h=0.6)

.. code-block:: text

    Evaluates the quality of a schedule for the restricted single‐machine common due date problem.
    Parameters:
         - jobs (List[Tuple[int, int, int]]): List of jobs, each represented as (p, a, b).
         - schedule (List[int]): A permutation (1-based indices) representing the processing order.
         - h (float): Factor for computing the common due date d = floor(sum(p) * h).
    Returns:
         int: The total penalty computed for the schedule.
    The evaluation:
         1. Compute d = floor(total_processing_time * h).
         2. Process jobs in the given order, accumulating processing times.
         3. For each job, if the cumulative time C is less than d, add a penalty a × (d − C);
            if C is greater than d, add a penalty b × (C − d); no penalty is incurred if C equals d.
         4. Sum the penalties to yield the total score.

.. code-block:: text

    CDDSEvaluationCB.norm_score(self, results)

.. code-block:: text

    Given a dictionary `results` where each key is a test case filename (e.g., "sch10.txt")
    and the value is a tuple (scores, error_message), this function returns a new dictionary
    with the normed results. For each test case, the normed score for each k is computed as:
        norm = (optimal score for h=0.6) / (model's score)
    The optimal scores for h=0.6 are pre-defined for each job instance size n.
    If a score in the list is not numeric (e.g., "Timeout (10s)"), that entry is skipped.
    Parameters:
      results (dict): A dictionary where keys are filenames (e.g., "sch10.txt") and values
                      are tuples (scores, error_message).
    Returns:
      dict: A dictionary with the same keys, where each value is a list of normed scores
            computed for the numeric entries only.

.. code-block:: text

    CDDSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.constrained_guillotine_cutting_co_bench.evaluation
------------------------------------------------------------------------------------

CGCEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class CGCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CGCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CGCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CGCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CGCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads one or more cases from an input file for the Constrained Guillotine Cutting problem.
    The input file contains one or more cases concatenated together. Each case is structured as follows:
      - The first token: an integer m (the number of piece types).
      - The next two tokens: stock_length and stock_width.
      - Then 4*m tokens follow, where each group of 4 tokens represents a piece type:
            piece_length, piece_width, maximum permitted count, piece_value.
    Parameters:
      input_path (str): Path to the input TXT file.
    Returns:
      List[dict]: A list where each element is a dictionary with the following keys:
          - m: int, number of piece types.
          - stock_length: int, length of the stock sheet.
          - stock_width: int, width of the stock sheet.
          - piece_types: list of dicts, each with keys 'length', 'width', 'max', 'value'.

.. code-block:: text

    CGCEvaluationCB.eval_func(self, m, stock_length, stock_width, piece_types, total_value, placements)

.. code-block:: text

    Evaluates a solution for the Fixed Orientation Guillotine Cutting problem by verifying all constraints.
    Raises an error immediately upon any constraint violation.
    Parameters:
      m (int): Number of piece types.
      stock_length (int): Length of the stock rectangle.
      stock_width (int): Width of the stock rectangle.
      piece_types (list of dict): Each dict has keys:
          'length': int, piece length.
          'width' : int, piece width.
          'max'   : int, maximum permitted count for the piece type.
          'value' : int, value of the piece.
      total_value (int): The reported total value from the solution.
      placements (list): List of placements. Each placement is a tuple or list of 6 integers:
            (piece_type_index, x, y, placed_length, placed_width, orientation_flag)
            where orientation_flag must be 0 (rotation not allowed).
    Returns:
      int: The computed total value if the solution is valid.
    Constraints verified:
      - Each placement is a well-formed 6-tuple of integers.
      - The piece type index is within the valid range.
      - The orientation flag is 0 (rotation is not allowed).
      - The placed dimensions match the expected dimensions.
      - Each piece is completely within the stock boundaries.
      - No two pieces overlap.
      - The count of pieces of each type does not exceed its allowed maximum.
      - The reported total_value exactly equals the computed sum of placed piece values.
      - The set of placements satisfies the guillotine cutting condition.

.. code-block:: text

    CGCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CGCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.constrained_non_guillotine_cutting_co_bench.evaluation
----------------------------------------------------------------------------------------

CNCEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class CNCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CNCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CNCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CNCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CNCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads input data from a text file and returns a list of test case dictionaries.
    The input format:
      - First line: integer T (number of test cases)
      - For each test case:
          * A line with integer m (number of pieces)
          * A line with two integers: stock_length and stock_width
          * m subsequent lines, each with 5 integers:
                length, width, min_required, max_allowed, value
    Returns:
      List[Dict]: A list where each element is a dictionary with keys:
          'stock_length': int,
          'stock_width': int,
          'pieces': list of dicts, each dict has:
                'length': int,
                'width': int,
                'min': int,
                'max': int,
                'value': int

.. code-block:: text

    CNCEvaluationCB.eval_func(self, stock_length, stock_width, pieces, placements)

.. code-block:: text

    Evaluates the solution for a single test case.
    Parameters:
      - stock_length (int): Length of the stock rectangle.
      - stock_width (int): Width of the stock rectangle.
      - pieces (list of dict): List of piece definitions.
      - placements (list): List of placements; each placement is a 4-tuple:
                           (piece_type, x, y, r)
    Returns:
      float: The overall score, computed as the sum of values of all placed pieces,
             if the solution is feasible.
    Raises:
      ValueError: If any constraint is violated.

.. code-block:: text

    CNCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CNCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.container_loading_co_bench.evaluation
-----------------------------------------------------------------------

CLEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class CLEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CLEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CLEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CLEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CLEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads container loading problem data from string content.
    The input format:
      1. The first line is an integer P, the number of test problems.
      2. For each test problem:
         a. A header line with two integers: problem_index and a seed (the seed may be ignored).
            (Note: Some files may only provide one number; the seed is optional.)
         b. A line with three integers: container_length, container_width, container_height.
         c. A line with a single integer n: the number of box types.
         d. Then n lines follow, each with 7 or 8 integers in this order:
                box_type, d1, f1, d2, f2, d3, f3 [, count]
            If only 7 numbers are provided, a default count of 1 is assumed.
    Returns:
      A list of dictionaries, one per test case. Each dictionary has the following keys:
         - 'problem_index': int,
         - 'container': tuple (container_length, container_width, container_height),
         - 'box_types': dict mapping each box_type to a dict with keys:
               'dims': [d1, d2, d3],
               'flags': [f1, f2, f3],
               'count': count

.. code-block:: text

    CLEvaluationCB.eval_func(self, problem_index, container, box_types, placements, **kwargs)

.. code-block:: text

    Evaluates a container loading solution for a single test case.
    Parameters:
      - problem_index: the integer identifier of the test case.
      - container: a tuple (container_length, container_width, container_height).
      - box_types: a dictionary mapping box types to their specifications.
      - placements: a list of placement dictionaries; each must include:
            'box_type', 'container_id', 'x', 'y', 'z', 'v', 'hswap'
    Returns:
      A scalar float value representing the volume utilization ratio if the solution is valid.
      If any placement is invalid (e.g., incorrect orientation, out-of-bound placement,
      overlapping boxes, or exceeding available count), the function returns 0.0.
    Evaluation Details:
      - For each placement, verifies that the chosen vertical dimension (v) is allowed.
      - Computes the oriented dimensions:
            horizontal dimensions are the two not chosen as vertical (swapped if hswap == 1),
            vertical dimension is dims[v].
      - Checks that each box is entirely within the container.
      - Checks that boxes do not overlap (touching is allowed).
      - Verifies that the number of placed boxes for each type does not exceed the available count.
      - The score is computed as (total placed volume) / (container volume).

.. code-block:: text

    CLEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.container_loading_with_weight_restrictions_co_bench.evaluation
------------------------------------------------------------------------------------------------

CLWREvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class CLWREvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CLWREvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CLWREvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CLWREvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CLWREvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads the input data file for the Container Loading problem.
    The input file may contain one or more test cases. For each test case:
      - The first non-empty line contains three floats: container length, width, height (in cm).
      - The next non-empty line contains an integer n (number of box types) and a float (total cargo volume in m³).
      - The following n non-empty lines each contain 11 whitespace-separated values:
            Box length, length_flag, Box width, width_flag, Box height, height_flag,
            count, weight, lb1, lb2, lb3.
    Returns:
       A list where each element is a dictionary containing the input data for one test case with keys:
         'container', 'n', 'cargo_vol', and 'box_types'.

.. code-block:: text

    CLWREvaluationCB.get_box_dimensions(self, box, orientation)

.. code-block:: text

    Given a box type (dictionary) and an orientation (1, 2, or 3),
    returns a tuple (dx, dy, dz, lb, volume) where:
      - (dx, dy) are the horizontal dimensions,
      - dz is the vertical dimension,
      - lb is the load-bearing ability for that orientation,
      - volume is the original box volume.
    Orientation conventions:
      1: Box length is vertical (dz = length; horizontal: width, height).
      2: Box width is vertical (dz = width; horizontal: length, height).
      3: Box height is vertical (dz = height; horizontal: length, width).

.. code-block:: text

    CLWREvaluationCB.boxes_overlap(self, b1, b2)

.. code-block:: text

    Determines if two boxes overlap in space.
    Each box is represented as a dict with keys:
      x, y, z, dx, dy, dz.
    Returns True if the boxes overlap (i.e. intersect in all three dimensions, not just touch).

.. code-block:: text

    CLWREvaluationCB.eval_func(self, container, n, cargo_vol, box_types, instance, util, m, placements)

.. code-block:: text

    Hard evaluation for a container–loading solution.
    This function checks all constraints and raises an error immediately when any
    constraint is violated. The constraints include:
      - Validity of the box type index.
      - Box orientation (via get_box_dimensions).
      - Box placement completely within container boundaries.
      - Not exceeding the available counts for each box type.
      - Proper support: every box not on the floor must be fully and uniquely supported.
      - Overlap: boxes may only overlap if one is exactly supporting the other.
      - Load-bearing capacity: the weight on each box must not exceed its capacity.
    If all constraints are met, the function returns the container volume utilization,
    i.e., (total placed box volume) / (container volume).
    Inputs:
      - container: tuple (L, W, H) in cm.
      - n: number of box types.
      - cargo_vol: total cargo volume (m³) (not used in evaluation).
      - box_types: list of box type dictionaries.
      - instance: instance number (int) (not used in evaluation).
      - util: reported utilization (float) (ignored here).
      - m: number of placements.
      - placements: list of placements; each placement is a dict with keys:
            'box_type' (int, 1-indexed),
            'orientation' (int),
            'x', 'y', 'z' (floats).
    Returns:
      A float representing the container utilization if all constraints are satisfied.

.. code-block:: text

    CLWREvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.corporate_structuring_co_bench.evaluation
---------------------------------------------------------------------------

CSEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class CSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads input string content that may contain one or more cases.
    File Format for each case:
      - Line 1: Two space-separated numbers: N target
      - Next N lines: For each country i (1-indexed), four space-separated values:
                         tax_code, foreign_income_tax_rate, domestic_income_tax_rate, profit
      - Remaining tokens: N*N floating-point numbers representing the withholding tax matrix.
        (These numbers can be spread across one or more lines.)
    Returns:
      A list of dictionaries. Each dictionary corresponds to one test case and has the keys:
         - "N": (int) number of countries.
         - "target": (int) target country (1-indexed).
         - "countries": (dict) mapping each country id to its tuple of (tax_code, foreign_rate, domestic_rate, profit).
         - "withholding": (dict of dict) where withholding[i][j] is the withholding tax rate from country i to j.

.. code-block:: text

    CSEvaluationCB.eval_func(self, N, target, countries, withholding, structure)

.. code-block:: text

    Evaluates the score of a given tree structure.
    Inputs:
      - N: Number of countries.
      - target: The designated target country (1-indexed) that is the root (its parent is 0).
      - countries: A dict mapping country id (1-indexed) to a tuple:
              (tax_code, foreign_income_tax_rate, domestic_income_tax_rate, profit)
      - withholding: A dict of dicts where withholding[i][j] is the withholding tax rate
                     applied when country i sends dividends to j.
      - structure: A dict representing the corporate tree. Each key is a country (child) and its
                   value is its immediate parent (for the target, parent is 0).
    Returns:
      The score, defined as:
          total_profit = (sum of profits for all countries) - (total_tax)
      where total_tax is the sum of domestic tax and extra foreign tax paid in the tree.

.. code-block:: text

    CSEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.crew_scheduling_co_bench.evaluation
---------------------------------------------------------------------

CSchedulingEvaluationCB
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class CSchedulingEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CSchedulingEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    CSchedulingEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSchedulingEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSchedulingEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads input data from a provided text string. This function supports multiple cases.
    The input file format for each case is as follows:
      - The first line contains two numbers: the number of tasks (N) and the maximum allowed duty time (time_limit).
      - The next N lines contain two numbers each: start time and finish time for each task (tasks are indexed from 1 to N).
      - The remaining lines describe transition arcs between tasks in the format: "i j cost".
    Cases are assumed to be separated by one or more blank lines.
    Returns:
      list: A list of dictionaries, each dictionary corresponds to one case with keys:
            "N", "time_limit", "tasks", "arcs".

.. code-block:: text

    CSchedulingEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates the quality (i.e. total cost and feasibility) of a crew scheduling solution.
    Raises an error immediately if any feasibility constraint is violated.
    Input kwargs must include:
      - N (int): Number of tasks.
      - K (int): The exact number of crews required.
      - time_limit (float): Maximum allowed duty time.
      - tasks (dict): Mapping from task ID to (start_time, finish_time).
      - arcs (dict): Mapping from (from_task, to_task) to transition cost.
      - crews (list): List of lists, where each inner list is the sequence of task IDs for one crew.
    Returns:
      float: The total transition cost if the solution is feasible.

.. code-block:: text

    CSchedulingEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CSchedulingEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.equitable_partitioning_problem_co_bench.evaluation
------------------------------------------------------------------------------------

EPPEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class EPPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    EPPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    EPPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EPPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EPPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads input string content where each non-empty line represents an individual with space-separated binary attributes.
    In case the input string contains multiple cases (separated by one or more blank lines), this function will
    separate them into distinct cases.
    Parameters:
        input_string (str): The string content with the input data.
    Returns:
        list: A list of dictionaries. Each dictionary represents one case with the key 'data' mapping to a 2D list
              (matrix) of binary attributes (0 or 1). For example:
              [
                  {"data": [[0, 1, 0], [1, 0, 1], ...]},
                  {"data": [[1, 1], [0, 1], ...]},
                  ...
              ]
    Raises:
        Exception: If the string cannot be read, or if any line is invalid, contains non-integer tokens,
                   tokens not in {0, 1}, or if any row has an inconsistent number of attributes.

.. code-block:: text

    EPPEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a partitioning solution for the equitable distribution problem using the new imbalance metric.
    Expected Parameters (provided via kwargs):
      - data (list of list of int): A matrix of binary attributes for individuals.
      - assignment (list of int): A list of positive integers representing group assignments for each individual.
    Evaluation Metric:
      For each attribute (column), compute the number of 1's per group. Then, compute the mean of these counts.
      The imbalance for the attribute is defined as the average of the absolute differences between each group's count and the mean count.
      The final score is the sum of these imbalances over all attributes.
      (A lower score indicates a more balanced partitioning.)
    Returns:
      total_imbalance: The computed total imbalance (float).
    Raises:
      Exception: If any expected parameter is missing, if the assignment format is invalid, or if the number of groups is not 8.

.. code-block:: text

    EPPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    EPPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.euclidean_steiner_problem_co_bench.evaluation
-------------------------------------------------------------------------------

ESPEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class ESPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ESPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    ESPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ESPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ESPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads the input string and returns a list of individual test problems.
    The input file may contain one or more cases. Each case is expected to follow the format:
       Line 1: An integer m representing the number of test problems in the case.
       For each test problem:
           Line 1: An integer n representing the number of points.
           Next n lines: Two space-separated floating-point numbers for the x- and y-coordinates.
    Returns:
       A list of dictionaries. Each dictionary corresponds to one test problem and contains:
         - "points": a list of (x, y) tuples representing the terminals.
    The function ignores empty lines and supports multiple cases concatenated in one file.

.. code-block:: text

    ESPEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates candidate solutions for the Euclidean Steiner Problem.
    Expected kwargs:
      - problems: a list of test problems; each test problem is a dict with key "points"
                  which holds a list of (x, y) tuples representing the original terminals.
      - solutions: a list of candidate solutions, one for each test problem.
                   Each candidate solution is a dict with:
                       - "steiner_points": a list of (x, y) tuples representing the additional Steiner points.
    Evaluation:
      For each test problem:
         1. Compute MST_original, the total length of the Minimum Spanning Tree (MST) computed
            on the original terminals.
         2. Compute candidate_value, the total length of the MST computed on the union of
            the original terminals and the candidate Steiner points.
            (Both MST computations use Euclidean distance where the distance between (x1,y1) and (x2,y2)
            is sqrt((x1-x2)^2 + (y1-y2)^2).)
         3. A valid candidate must have candidate_value ≤ MST_original (within a small tolerance).
            If not, a ValueError is raised.
            Otherwise, the quality ratio is computed as candidate_value / MST_original.
            (A lower ratio indicates a better solution.)
      The overall score is the average of the ratios over all test problems.
    Returns:
      overall_score (float): The average ratio over all test problems.

.. code-block:: text

    ESPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ESPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.flow_shop_scheduling_co_bench.evaluation
--------------------------------------------------------------------------

FSSEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class FSSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FSSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    FSSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FSSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FSSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads a file containing multiple test cases for the flow shop scheduling problem.
    The file format:
      - A header line: "number of jobs, number of machines, initial seed, upper bound and lower bound :"
      - Next line: five numbers (n, m, seed, upper_bound, lower_bound)
      - A line that starts with "processing times :"
      - Then m lines of processing times. Each line contains n integers (processing times for one machine across all jobs).
    The function returns a list of test cases, where each test case is a dictionary with:
      - "n" (int): number of jobs
      - "m" (int): number of machines
      - "matrix" (list of list of int): processing times in a n x m matrix (each row corresponds to a job)
      - "upper_bound" (int)
      - "lower_bound" (int)

.. code-block:: text

    FSSEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a flow shop scheduling solution for a single instance.
    Input kwargs must include:
      - n (int): Number of jobs.
      - m (int): Number of machines.
      - matrix (list of list of int): Processing times matrix.
      - job_sequence (list of int): A 1-indexed permutation of job indices, as returned by solve.
    The evaluation metric (makespan) is computed using the classical flow shop recurrence:
      - C[0][0] = processing_time(job_1, machine_0)
      - For the first job on machines j > 0: C[0][j] = C[0][j-1] + processing_time(job_1, machine_j)
      - For subsequent jobs on the first machine: C[i][0] = C[i-1][0] + processing_time(job_(i+1), machine_0)
      - For all other entries: C[i][j] = max(C[i-1][j], C[i][j-1]) + processing_time(job_(i+1), machine_j)
    Returns:
      float: The computed makespan for the provided solution.

.. code-block:: text

    FSSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.generalised_assignment_problem_co_bench.evaluation
------------------------------------------------------------------------------------

GAPEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class GAPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GAPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    GAPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GAPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GAPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load and parse the input file for the Generalised Assignment Problem (GAP).
    The input is expected to be a whitespace‐delimited text file with the following format:
      - The first token is an integer P, indicating the number of cases.
      - For each case, the following tokens are provided sequentially:
          • Two integers: m (number of agents) and n (number of jobs).
          • m×n numbers representing the cost matrix (row by row).
          • m×n numbers representing the resource consumption matrix (row by row).
          • m numbers representing the capacities for each agent.
    Parameters:
      input_file_path: (str) Path to the input text file.
    Returns:
      A list of dictionaries. Each dictionary corresponds to one case and contains the keys:
          'm', 'n', 'cost_matrix', 'consumption_matrix', and 'capacities'.

.. code-block:: text

    GAPEvaluationCB.eval_func(self, m, n, cost_matrix, consumption_matrix, capacities, assignments, **kwargs)

.. code-block:: text

    Evaluate a solution for a single case of the Generalised Assignment Problem (GAP).
    Parameters:
      - m: (int) Number of agents.
      - n: (int) Number of jobs.
      - cost_matrix: (list of list of float) The cost matrix of size m×n.
      - consumption_matrix: (list of list of float) The resource consumption matrix of size m×n.
      - capacities: (list of float) The resource capacities for each of the m agents.
      - assignments: (list of int) A list of n integers (using 1-indexing) representing the agent
                     assigned to each job.
    Evaluation:
      - TotalCost is computed as the sum of cost_matrix[agent-1][j] for each job j.
      - For each agent i, ResourceConsumption[i] is the sum of consumption_matrix[i][j] for jobs assigned to agent i.
      - If an agent’s ResourceConsumption exceeds its capacity, a ValueError is raised.
      - For a maximization problem, the score is simply the TotalCost.
        (For minimization problems, you might use the negative of TotalCost.)
    Returns:
      A numeric score (float) evaluating the quality of the solution.

.. code-block:: text

    GAPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GAPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.graph_colouring_co_bench.evaluation
---------------------------------------------------------------------

GCEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class GCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    GCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads the input DIMACS string, which may contain one or more cases.
    Each case is separated by a header line (starting with "p"). For each case, the function:
      - Ignores blank lines and comment lines (starting with "c").
      - Parses the header line ("p edge <n> <m>") if present; if absent, determines n from edge listings.
      - Parses each edge line (starting with "e") to extract the edge (u,v).
      - Builds an adjacency list mapping each vertex (from 1 to n) to its adjacent vertices.
    Returns:
        A list where each element is a dictionary containing the data for one case.
        Each dictionary has at least the following keys:
            - 'n': int, number of vertices.
            - 'edges': list of (u, v) tuples.
            - 'adjacency': dict mapping vertex (1..n) to a set of adjacent vertices.

.. code-block:: text

    GCEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a solution for a single case.
    Expected kwargs:
        - 'n': int, number of vertices.
        - 'adjacency': dict mapping each vertex (1..n) to a set of adjacent vertices.
        - Plus all key-value pairs from the solution dictionary produced by solve,
          mapping vertex ids to assigned positive integer colors.
    Evaluation:
        - Verifies that every vertex from 1 to n is assigned a positive integer color.
        - For each edge (u,v), if the assigned colors are the same, counts as a conflict.
        - Let C be the total number of conflicts and k be the number of distinct colors used.
        - If C > 0, the solution is invalid and an error is raised.
        - If C == 0, the score is simply k (lower is better).
    Returns:
        A scalar score (integer or float) representing the evaluation of the solution.

.. code-block:: text

    GCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.hybrid_reentrant_shop_scheduling_co_bench.evaluation
--------------------------------------------------------------------------------------

HRSSEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class HRSSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HRSSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    HRSSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HRSSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HRSSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads the input string and parses one or more problem instances.
    The input is expected to have one or more instances separated by lines that contain only dashes (e.g., "-----").
    Each instance must include exactly 4 nonempty lines:
      1. Header line: "Number of jobs: X  Number of machines: Y"
      2. Initialization time: "Initialization time: Z"
      3. Setup times: "Setup times: t1 t2 ... tX"
      4. Processing times: "Processing times: p1 p2 ... pX"
    Returns:
      A list of dictionaries. Each dictionary corresponds to a problem instance and contains the keys:
         - 'n_jobs': integer
         - 'n_machines': integer
         - 'init_time': integer
         - 'setup_times': list of integers
         - 'processing_times': list of integers

.. code-block:: text

    HRSSEvaluationCB.eval_func(self, n_jobs, n_machines, init_time, setup_times, processing_times, permutation, batch_assignment)

.. code-block:: text

    1. Initialization on one of m identical primary machines:
         - Jobs are processed in natural order (1, 2, …, n_jobs) using list scheduling.
         - In this phase, each job takes 'init_time'. The machine assignment is determined
           by the list scheduling, and that assignment is used for the final main processing.
      2. Setup on the remote server:
         - Jobs are processed in the order specified by 'permutation' (a 1-indexed list).
         - A job's setup can start only after its initialization is complete and when the
           remote server is free. The setup time for job j is given as setup_times[j-1].
      3. Main processing on primary machines:
         - Each job is processed on the same primary machine that performed its initialization.
         - Within each machine, jobs are processed in the natural order (i.e., in order of their job indices).
         - The processing time for job j is given by processing_times[j-1].
    The makespan is defined as the time when the last job completes its main processing.
    Parameters:
      - n_jobs: Integer; number of jobs.
      - n_machines: Integer; number of primary machines.
      - init_time: Integer; initialization time for each job.
      - setup_times: List of integers; setup times for each job on the remote server.
      - processing_times: List of integers; processing times for each job in main processing.
      - permutation: List of integers of length n_jobs; a permutation (1-indexed) representing the order
                     in which jobs are processed on the remote server.
    Returns:
      A scalar (float or integer) representing the makespan (total completion time).
    Raises:
      ValueError: if any input constraint is not met.

.. code-block:: text

    HRSSEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    HRSSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.job_shop_scheduling_co_bench.evaluation
-------------------------------------------------------------------------

JSSEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class JSSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    JSSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    JSSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    JSSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    JSSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    JSSEvaluationCB.eval_func(self, n_jobs, n_machines, times, machines, start_times, **kwargs)

.. code-block:: text

    Evaluates the solution for a job shop scheduling problem.
    Input:
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        times (list of list of int): Processing times for each operation.
            Dimensions: n_jobs x n_machines.
        machines (list of list of int): Machine assignments for each operation.
            Dimensions: n_jobs x n_machines.
        start_times (list of list of int): Proposed start times for each operation.
            Dimensions: n_jobs x n_machines.
        kwargs: Other parameters that may be provided, which are ignored here.
    Output:
        score (int): The makespan, defined as the maximum completion time across all jobs.
    Raises:
        ValueError: If any scheduling constraints are violated.

.. code-block:: text

    JSSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.maximal_independent_set_co_bench.evaluation
-----------------------------------------------------------------------------

MISEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class MISEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MISEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    MISEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MISEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MISEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load data method for compatibility with comprehensive testing.
    Since MIS task loads pickle files directly in __init__, this method
    returns cases from the dictionary format.

    Args:
        input_string: Dataset content (not used, but required for interface)

    Returns:
        list: List of dataset entries for compatibility

.. code-block:: text

    MISEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate a Maximum Independent Set solution for correctness.
    Args:
        name (str): Name of the test case
        graph (networkx.Graph): The graph that was solved
        mis_nodes (list): List of nodes claimed to be in the maximum independent set
        mis_size (int): Claimed size of the maximum independent set
    Returns:
        actual_size (int): The actual size of the provided solution
        # dict: Evaluation results containing:
        #     - is_valid (bool): Whether the solution is a valid independent set
        #     - actual_size (int): The actual size of the provided solution
        #     - score (int): The score of the solution (0 if invalid, actual_size if valid)
        #     - error (str, optional): Error message if any constraint is violated

.. code-block:: text

    MISEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MISEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.multi_demand_multidimensional_knapsack_problem_co_bench.evaluation
----------------------------------------------------------------------------------------------------

MDMKPEvaluationCB
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MDMKPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MDMKPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    MDMKPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MDMKPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MDMKPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load MDMKP test instances from a given file and split each base instance into
    6 separate optimization problems. Each split instance is a dictionary with the keys:
        - 'n': number of decision variables.
        - 'm': number of <= constraints.
        - 'q': number of active >= constraints for this variant (1, m//2, or m).
        - 'A_leq': list of lists representing the <= constraints coefficients.
        - 'b_leq': list representing the right-hand side for the <= constraints.
        - 'A_geq': list of lists representing the active >= constraints coefficients (first q rows).
        - 'b_geq': list representing the active right-hand side values (first q values).
        - 'cost_vector': objective function coefficients for this variant.
        - 'cost_type': either "positive" or "mixed".
    The file format is assumed to be:
      - The first line contains an integer K: the number of base test instances.
      - For each instance:
          * A line with two integers: n (number of variables) and m (number of <= constraints).
          * m lines with n integers each: coefficients for the <= constraints.
          * One line with m integers: right-hand side for the <= constraints.
          * m lines with n integers each: coefficients for the >= constraints.
          * One line with m integers: right-hand side for the >= constraints.
          * 6 lines with n integers each: cost vectors.
            - The first 3 lines correspond to the positive cost case (for q = 1, m//2, m).
            - The next 3 lines correspond to the mixed cost case (for q = 1, m//2, m).
    Returns:
         A list of dictionaries, each representing one optimization problem variant.

.. code-block:: text

    MDMKPEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate a solution for a given MDMKP instance.
    This function expects keyword arguments that include both the instance data and the solution.
    It is designed to be used as:
        eval_func(**case, **output)
    where the instance 'case' contains the following keys:
        - n: int
             Number of decision variables.
        - m: int
             Number of <= constraints.
        - q: int
             Number of active >= constraints (subset of the full set).
        - A_leq: list of lists of int (dimensions: m x n)
             Coefficient matrix for <= constraints.
        - b_leq: list of int (length m)
             Right-hand side for <= constraints.
        - A_geq: list of lists of int (dimensions: q x n)
             Coefficient matrix for >= constraints.
        - b_geq: list of int (length q)
             Right-hand side for >= constraints.
        - cost_vector: list of int (length n)
             Objective function coefficients.
        - cost_type: str
             A string indicating the cost type ("positive" or "mixed").
    And the solver output (the solution) is expected to include at least:
        - x: list of int
             Binary decision vector (0 or 1) of length n.
    The evaluation process is as follows:
      1. Check that the decision vector is of the proper length and binary.
      2. Verify that each <= constraint is satisfied:
             For each i in 0,...,m-1, sum_{j=0}^{n-1} A_leq[i][j]*x[j] <= b_leq[i]
      3. Verify that each >= constraint is satisfied:
             For each i in 0,...,q-1, sum_{j=0}^{n-1} A_geq[i][j]*x[j] >= b_geq[i]
      4. If all constraints are satisfied, compute the objective value as:
             score = sum_{j=0}^{n-1} cost_vector[j] * x[j]
      5. Return the computed score.
    If any constraint is violated or the input format is incorrect, the function raises a ValueError.
    Returns:
        score: int or float, the objective value computed from the solution.

.. code-block:: text

    MDMKPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MDMKPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.multidimensional_knapsack_problem_co_bench.evaluation
---------------------------------------------------------------------------------------

MKPEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class MKPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MKPEvaluationCB.__init__(self, timeout_seconds=300, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    MKPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MKPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MKPEvaluationCB.load_data2(self, input_path)

.. code-block:: text

    Loads instance(s) from the OR-Library mknap2.txt file.
    This file contains many lines of commentary and then one or more instances.
    Each instance is defined (after removing comments) as:
      <#knapsacks> <#objects>
      <weights of objects>       (there will be exactly #objects numbers)
      <knapsack capacities>        (exactly #knapsacks numbers)
      <matrix of resource consumptions>   (#objects rows, each with #knapsacks numbers)
      [<known optimum>]           (an extra token, optional)
    In our formulation:
      - the number of decision variables (n) is set to the number of objects,
      - the number of constraints (m) is set to the number of knapsacks,
      - the profit coefficients p are taken equal to the object weights,
      - the constraint coefficients r are taken from the matrix (transposed so that each
        constraint i gets a list of consumptions for all objects),
      - the right-hand sides b are the knapsack capacities.
    Returns:
      A list of dictionaries. Each dictionary corresponds to one problem instance and
      has the keys:
         'n' : int, number of objects (decision variables)
         'm' : int, number of knapsacks (constraints)
         'p' : list of floats, profit coefficients (length n)
         'r' : list of m lists of floats, where each inner list is of length n (constraint coefficients)
         'b' : list of floats, knapsack capacities (length m)
      If the instance file also provides an optimum value, it is stored under key 'opt'.

.. code-block:: text

    MKPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads the input string and returns a list of test cases.
    Each case is represented as a dictionary containing:
        - 'n': number of decision variables.
        - 'm': number of constraints.
        - 'p': list of floats, profit coefficients.
        - 'r': list of m lists of floats, constraint coefficients.
        - 'b': list of floats, right-hand side values.

.. code-block:: text

    MKPEvaluationCB.eval_func(self, n, m, p, r, b, x, opt=None)

.. code-block:: text

    Evaluates the solution for a multidimensional knapsack problem instance.
    Inputs:
      - n: int, number of decision variables.
      - m: int, number of constraints.
      - p: list of floats, profit coefficients (length n).
      - r: list of m lists of floats, each representing the constraint coefficients.
      - b: list of floats, right-hand side values for each constraint (length m).
      - x: list of ints (0 or 1), the solution decisions (length n).
      - opt (float, optional): The known optimal (or best-known) objective value.
        This parameter is provided by instances loaded via load_data2, if available.
    Evaluation:
      - The objective value is computed as:
            sum(p[j] * x[j] for j in range(n))
      - For each constraint i, the total resource consumption is computed as:
            sum(r[i][j] * x[j] for j in range(n))
      - If any constraint i is violated (i.e., the consumption exceeds b[i]), an error is raised.
      - If all constraints are satisfied, the score is equal to the objective value.
    Returns:
      - If opt is not provided (None), returns a float representing the overall quality score.
      - If opt is provided, returns a tuple:
            (score, gap)
        where gap is defined as (score - opt), which indicates how far (or above)
        the computed score is relative to the known optimum.

.. code-block:: text

    MKPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MKPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.open_shop_scheduling_co_bench.evaluation
--------------------------------------------------------------------------

OSSEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class OSSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    OSSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    OSSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OSSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OSSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OSSEvaluationCB.eval_func(self, n_jobs, n_machines, times, machines, start_times, **kwargs)

.. code-block:: text

    Evaluates the solution for a open shop scheduling problem.
    Input:
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        times (list of list of int): Processing times for each operation.
            Dimensions: n_jobs x n_machines.
        machines (list of list of int): Machine assignments for each operation.
            Dimensions: n_jobs x n_machines.
        start_times (list of list of int): Proposed start times for each operation.
            Dimensions: n_jobs x n_machines.
        kwargs: Other parameters that may be provided, which are ignored here.
    Output:
        score (int): The makespan, defined as the maximum completion time across all jobs.
    Raises:
        ValueError: If any scheduling constraints are violated.

.. code-block:: text

    OSSEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.p_median_capacitated_co_bench.evaluation
--------------------------------------------------------------------------

PMCEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class PMCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PMCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PMCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PMCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PMCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load one or more instances of the Capacitated P-Median Problem from a text file.
    The input file structure is:
      Line 1: An integer M, the number of problem instances in the file.
      Then, for each instance:
          - A header line with two values: <problem_number> <best_known_solution_value>
          - A line with three values: <n> <p> <Q>
          - n subsequent lines each with: <customer_number> <x_coordinate> <y_coordinate> <demand>
    Returns:
      A list of dictionaries. Each dictionary contains the keys:
         - 'best_known': float
         - 'n': int
         - 'p': int
         - 'Q': float
         - 'customers': list of tuples (customer_id, x, y, demand)

.. code-block:: text

    PMCEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate the solution for a single instance of the Capacitated P-Median Problem.
    This function expects the following keyword arguments (combined from the instance data and the solution):
      - best_known (float): Best known solution value (for reference).
      - n (int): Number of customers.
      - p (int): Number of medians.
      - Q (float): Capacity of each median.
      - customers (list of tuples): Each tuple is (customer_id, x, y, demand).
      - objective (numeric): The objective value (total cost) reported by the solution.
      - medians (list of int): List of chosen medians (customer IDs), exactly p elements.
      - assignments (list of int): List of assignments for each customer (length n), where each entry is one of the chosen medians.
    The evaluation performs the following:
      1. Verifies that each assignment is to one of the selected medians.
      2. Checks that the total demand assigned to each median does not exceed Q.
      3. Recomputes the total cost as the sum, over all customers, of the Euclidean distance (rounded down)
         from the customer to its assigned median.
      4. Computes the score as: score = best_known / computed_total_cost.
    Returns:
      A scalar float representing the score for the solution.

.. code-block:: text

    PMCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.p_median_uncapacitated_co_bench.evaluation
----------------------------------------------------------------------------

PMUEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class PMUEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PMUEvaluationCB.__init__(self, timeout_seconds=300, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PMUEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PMUEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PMUEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads one or more cases from the input file for the p-median problem, optimized for speed.
    This version uses NumPy to perform the Floyd–Warshall algorithm in a vectorized manner,
    which is significantly faster than the pure-Python triple nested loops for moderate-to-large graphs.
    The input is expected to have one or more cases. Each case starts with a header line
    containing three numbers: n m p, where:
        - n: number of vertices,
        - m: number of edges,
        - p: number of medians to choose.
    This is followed by at least m non-empty lines, each specifying an edge in the format:
        i j cost
    (If there are more than m edge lines, only the first m valid ones are used.)
    For each case, the function builds the complete cost matrix by:
      - Initializing an n x n NumPy array with infinity (and 0 on the diagonal).
      - Processing m valid edges (using the last occurrence for duplicate edges).
      - Running a vectorized Floyd–Warshall algorithm to compute all-pairs shortest paths.
    Returns:
        A list of dictionaries, one per case. Each dictionary contains:
            - 'n': int, number of vertices.
            - 'm': int, number of edges.
            - 'p': int, number of medians to choose.
            - 'dist': list of lists, the complete cost matrix (n x n), converted from a NumPy array.

.. code-block:: text

    PMUEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a candidate solution for the uncapacitated p-median problem.
    Parameters:
        candidate_data (dict): Contains the input data for a single case with keys:
            - 'n': int, number of vertices.
            - 'm': int, number of edges.
            - 'p': int, number of medians to choose.
            - 'dist': list of lists, the complete cost matrix (n x n).
        solution (dict): The candidate solution with key:
            - 'medians': list of exactly p distinct integers (each between 1 and n).
    Returns:
        float: The total assignment cost, i.e., the sum over all vertices of the shortest distance
               to the nearest chosen median.
    Raises:
        ValueError: If the solution is invalid due to incorrect format, duplicates, out-of-range values,
                    or if any vertex is unreachable from all medians.

.. code-block:: text

    PMUEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PMUEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.packing_unequal_circles_area_co_bench.evaluation
----------------------------------------------------------------------------------

PUCAEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class PUCAEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PUCAEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PUCAEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PUCAEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PUCAEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load and parse the input file containing one or multiple cases.
    File Format:
      - The file is a plain-text file with non-empty lines.
      - Each case starts with a header line containing exactly four numbers:
            n cx cy R
        where:
          • n  is the number of circles (an integer),
          • cx and cy are the container's center coordinates (floats),
          • R  is the container's radius (float).
      - The next n non-empty lines each contain one real number representing
        the radius of a circle.
    Returns:
      A list of cases, where each case is a dictionary with keys:
          "n"     : int, number of circles.
          "cx"    : float, container center x-coordinate.
          "cy"    : float, container center y-coordinate.
          "R"     : float, container radius.
          "radii" : list of float, the radii of the circles.

.. code-block:: text

    PUCAEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate the solution for the Maximise Area problem of Unequal Circle Packing.
    Input (merged from the case data and the solution):
      - n     : int, the total number of circles.
      - cx    : float, x-coordinate of the container's center.
      - cy    : float, y-coordinate of the container's center.
      - R     : float, the container's radius.
      - radii : list of float, radii for each circle.
      - coords: list of (x, y) tuples, the centers of the circles as produced by solve.
                A circle is considered unpacked if its center equals (-1, -1) (within tolerance).
    Evaluation Details:
      1. Identify packed circles: a circle is considered packed if its center is not (-1, -1)
         (within a small tolerance tol).
      2. For every packed circle:
         - Verify container feasibility:
             Ensure that sqrt((x - cx)^2 + (y - cy)^2) + r_i <= R (within tolerance).
         - Record its container clearance: clearance = R - (distance from (cx, cy) + r_i).
      3. For every pair of packed circles, verify non-overlap:
             Ensure that the distance between centers >= r_i + r_j (within tolerance).
         And record the pair clearance: (distance - (r_i + r_j)).
      4. If any feasibility constraint is violated (beyond tol), raise an Exception.
      5. Compute the primary score as the total area of packed circles:
             total_area = sum(π * (r_i)^2 for each packed circle).
         Then, use the minimum clearance (across all container and pair clearances) as a tie-breaker.
         (For example, final score = total_area + ε * (minimum clearance), with ε small.)
      6. Return the final score (a higher score indicates a better solution).
    Returns:
      float: the evaluation score.

.. code-block:: text

    PUCAEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PUCAEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.packing_unequal_circles_co_bench.evaluation
-----------------------------------------------------------------------------

PUCEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class PUCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PUCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PUCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PUCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PUCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load and parse the input file containing one or multiple cases.
    File Format:
      - The file is a plain-text file with non-empty lines.
      - Each case starts with a header line containing exactly four numbers:
            n cx cy R
        where:
          • n  is the number of circles (an integer),
          • cx and cy are the container's center coordinates (floats),
          • R  is the container's radius (float).
      - The next n non-empty lines each contain one real number representing
        the radius of a circle.
    Returns:
      A list of cases, where each case is a dictionary with keys:
          "n"     : int, number of circles.
          "cx"    : float, container center x-coordinate.
          "cy"    : float, container center y-coordinate.
          "R"     : float, container radius.
          "radii" : list of float, the radii of the circles.

.. code-block:: text

    PUCEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate the solution for the Maximise Number problem of Unequal Circle Packing.
    Input (merged from the case data and the solution):
      - n     : int, the total number of circles.
      - cx    : float, x-coordinate of the container's center.
      - cy    : float, y-coordinate of the container's center.
      - R     : float, the container's radius.
      - radii : list of float, radii for each circle (assumed sorted in increasing order).
      - coords: list of (x, y) tuples, the centers of the circles as produced by solve.
    Evaluation Details:
      1. Identify “packed” circles. By convention, a circle is considered packed if its coordinate
         is not equal to the default (cx, cy) (within tolerance). For the maximize number problem,
         the optimal solution packs a prefix of the sorted circles.
      2. Verify the prefix property: if any circle i is packed, then all circles with index < i must also be packed.
      3. For every packed circle:
         - Check container feasibility:
             Ensure that sqrt((x-cx)^2 + (y-cy)^2) + r_i <= R.
         - Record the clearance: R - (distance to (cx,cy) + r_i).
      4. For every pair of packed circles, verify non-overlap:
             Ensure that distance((x_i,y_i), (x_j,y_j)) >= r_i + r_j.
         And record the pair clearance.
      5. If any feasibility constraint is violated (beyond a small tolerance), raise an error.
      6. Let the primary score be the number of circles packed (i.e. the prefix length).
         Use the minimum clearance among packed circles as a tie-breaker.
         (For example, final score = (number packed) + ε*(minimum clearance), with ε small.)
    Returns:
      float: the evaluation score (a higher score indicates a better solution).
             The main component is the number of circles feasibly packed.

.. code-block:: text

    PUCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.packing_unequal_rectangles_and_squares_area_co_bench.evaluation
-------------------------------------------------------------------------------------------------

PURSAEvaluationCB
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class PURSAEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PURSAEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PURSAEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PURSAEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PURSAEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads input string content that may contain multiple cases for the packing problem.
    Each case is formatted as follows:
      - A header line with four values: n, cx, cy, R
          n   : number of items (rectangles or squares)
          cx, cy : container center coordinates
          R   : container radius
      - Next n non-empty lines: each line represents an item:
          * For a square: one number (side length) — interpreted as (side, side)
          * For a rectangle: two numbers (length and width)
    Returns:
      A list of cases. Each case is a dictionary with the following keys:
         - 'n'    : int, number of items
         - 'cx'   : float, x-coordinate of container center
         - 'cy'   : float, y-coordinate of container center
         - 'R'    : float, container radius
         - 'items': list of tuples, where each tuple is (L, W) for the respective item.

.. code-block:: text

    PURSAEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a candidate solution for the "maximize total area" rectangle/square packing problem.
    The function expects:
      data: a dict with keys:
            - n        : int, number of items (rectangles or squares)
            - cx, cy   : floats, coordinates of the container center
            - R        : float, radius of the container
            - items    : list of tuples, each (L, W) giving dimensions of an item
            - shape    : string, either "rectangle" or "square"
            - rotation : bool, whether 90° rotation is allowed
      sol: a dict with key 'placements' containing a list of exactly n tuples.
           Each tuple is (x, y, theta) where:
             - (x, y) is the center position for the item (if packed),
             - theta is the rotation angle in degrees (counter-clockwise from the horizontal).
             - For an unpacked item, x and y must be exactly -1 and theta is ignored (or should be 0).
    The evaluation process checks all feasibility constraints:
      1. The number of placements equals n.
      2. Each placement tuple must have three numerical values.
      3. For each item:
         - If it is "unpacked" (x == -1 and y == -1), it contributes no area.
         - If it is "packed" (x,y != -1), then:
            a. If rotation is not allowed, theta must be 0 (within a tiny tolerance).
            b. If rotation is allowed, theta must be either 0 or 90 (within tolerance).
            c. The entire item (with given dimensions and rotation) must lie completely inside
               the circular container (centered at (cx, cy) with radius R).
      4. No two packed items may overlap (their interiors should be disjoint).
    If any constraint is violated, the function raises a ValueError with an appropriate message.
    If all constraints are met, the function returns the total area of the packed items.
    (This is the score that we wish to maximize.)
    Note: The evaluation is designed to be robust against malicious modifications
          by the solve function. Only valid solutions (with zero penalties) receive a score.

.. code-block:: text

    PURSAEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.packing_unequal_rectangles_and_squares_co_bench.evaluation
--------------------------------------------------------------------------------------------

PURSEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class PURSEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    PURSEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    PURSEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PURSEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    PURSEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads input string content that may contain multiple cases for the packing problem.
    Each case is formatted as follows:
      - A header line with four values: n, cx, cy, R
          n   : number of items (rectangles or squares)
          cx, cy : container center coordinates
          R   : container radius
      - Next n non-empty lines: each line represents an item:
          * For a square: one number (side length) — interpreted as (side, side)
          * For a rectangle: two numbers (length and width)
    Returns:
      A list of cases. Each case is a dictionary with the following keys:
         - 'n'    : int, number of items
         - 'cx'   : float, x-coordinate of container center
         - 'cy'   : float, y-coordinate of container center
         - 'R'    : float, container radius
         - 'items': list of tuples, where each tuple is (L, W) for the respective item.

.. code-block:: text

    PURSEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a solution for the "maximise number of items packed" rectangle (or square)
    packing problem in a circular container.
    Parameters:
      input_data: dict with keys:
         - n         : int, total number of available items.
         - cx, cy    : floats, coordinates of the container center.
         - R         : float, container radius.
         - items     : list of tuples, where each tuple (L, W) gives the dimensions of an item.
                       (For squares, L == W.)
         - shape     : str, either "rectangle" or "square".
         - rotation  : bool, whether 90° rotation is allowed.
      solution_output: dict with key 'placements' containing a list of exactly n tuples.
         Each tuple is (x, y, theta), where:
           - (x, y) are the center coordinates.
           - theta is the rotation angle in degrees (counter-clockwise from horizontal).
           - For any item that is not packed, x and y should be set to -1 (theta can be 0).
    Returns:
      score: int, the number of valid (packed) items.
    Raises:
      ValueError: if any constraint is violated.

.. code-block:: text

    PURSEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.resource_constrained_shortest_path_co_bench.evaluation
----------------------------------------------------------------------------------------

RCSPEvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class RCSPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    RCSPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    RCSPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RCSPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RCSPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load one or more cases from a TXT input file for the Resource Constrained Shortest Path problem.
    The input file format (per case) is as follows:
      1. Three numbers: n (number of vertices), m (number of arcs), K (number of resources)
      2. For each resource (k = 1,...,K): the lower limit on the resource consumed on the chosen path.
      3. For each resource (k = 1,...,K): the upper limit on the resource consumed on the chosen path.
      4. For each vertex (i = 1,...,n): K numbers indicating the resource consumption incurred at that vertex.
      5. For each arc (j = 1,...,m): (3 + K) numbers:
             - starting vertex,
             - ending vertex,
             - cost of the arc,
             - K numbers indicating the resource consumption incurred on the arc.
    Note:
      In many of the RCSP test files, the file is a stream of numbers separated by whitespace rather than fixed lines.
      This implementation reads the entire file and splits it into tokens.
    Returns:
      A list of cases. Each case is a dictionary with keys:
         "n", "m", "K", "lower_bounds", "upper_bounds", "vertex_resources", "graph"

.. code-block:: text

    RCSPEvaluationCB.eval_func(self, n, m, K, lower_bounds, upper_bounds, vertex_resources, graph, total_cost, path)

.. code-block:: text

    Evaluate the solution for one case of the Resource Constrained Shortest Path problem.
    Parameters:
      n, m, K                : Input parameters defining the problem instance.
      lower_bounds           : List of lower resource bounds (length K).
      upper_bounds           : List of upper resource bounds (length K).
      vertex_resources       : List (length n) of lists (each of length K) with resource consumption per vertex.
      graph                  : Dictionary mapping each vertex (1-indexed) to its outgoing arcs.
                               Each arc is a tuple (end_vertex, cost, [arc resource consumptions]).
      total_cost             : The total cost value reported by the solution (not used in validation).
      path                   : List of vertex indices (integers) defining the computed path.
    Returns:
      The total arc cost along the path if the solution is valid.
    Raises:
      ValueError: If the solution is invalid (i.e. the path does not start at vertex 1, does not end at vertex n,
                  contains an undefined arc, or the cumulative resource consumption (from both vertices and arcs)
                  is not within the specified bounds for each resource).

.. code-block:: text

    RCSPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RCSPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.set_covering_co_bench.evaluation
------------------------------------------------------------------

SCEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class SCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    SCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    SCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads one or more set covering test cases from string content.
    The input can contain one or more cases. Each case must follow one of three formats:
      Format A (SCP/Beasley):
        - Header (first nonempty line): two integers, m and n.
        - Next: a cost vector of n integers (which may span multiple lines).
        - Then: for each row, a line that starts with an integer k (number of columns covering the row)
                followed by k space‑separated 1-indexed column indices.
      Format B (Real-world rail problems):
        - Header: two integers, m and n.
        - Next n nonempty lines: each line describes a column by giving:
              cost, the number of rows the column covers, and then that many 1-indexed row indices.
        - Row coverage is then built by aggregating the information from each column.
      Format C (Dense row format):
        - Header: two integers, m and n.
        - Next m nonempty lines: each line lists the 1-indexed column indices that cover that row.
        - In this format, every column has an implicit unit cost.
    If the input contains multiple cases, it is assumed that the cases are separated
    by at least one blank line.
    Returns:
      A list of cases, where each case is a dictionary with keys:
         - "m": number of rows (int)
         - "n": number of columns (int)
         - "costs": list of column costs (list of int)
         - "row_cover": list of lists; each inner list contains the 1-indexed column numbers covering that row.

.. code-block:: text

    SCEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates the solution for a single test case.
    Parameters:
      - m: (int) number of rows.
      - n: (int) number of columns.
      - costs: (list of int) where costs[j] is the cost for column j+1.
      - row_cover: (list of list of int) where row_cover[i] contains the 1-indexed columns covering row i+1.
      - selected_columns: a list of chosen 1-indexed column numbers.
    Evaluation:
      1. Compute the total cost as the sum of the costs for each selected column.
      2. Verify that every row is covered by at least one of the selected columns.
         If any row is uncovered, the function raises an error indicating the constraint violation.
    Returns:
      A scalar value representing the computed score (total cost) if all constraints are met.
    Raises:
      KeyError: if "selected_columns" is not provided in kwargs.
      ValueError: if any selected column is out of valid bounds or if any row is left uncovered.

.. code-block:: text

    SCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.set_partitioning_co_bench.evaluation
----------------------------------------------------------------------

SPEvaluationCB
~~~~~~~~~~~~~~

.. code-block:: text

    class SPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    SPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    SPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load and validate one or multiple set partitioning cases from a TXT file.
    The file may contain multiple cases. Each case is structured as follows:
      - The first non-empty line of a case contains two integers: num_rows and num_columns.
      - Then, for each of the num_columns columns, there is one line containing:
            cost (int), count (int), followed by exactly 'count' integers (the row indices covered).
    Each case is parsed and validated independently. If any inconsistency or formatting error is found,
    a ValueError is raised.
    Returns:
      cases (list): A list of dictionaries, each representing one case with keys:
                     - 'num_rows': int
                     - 'num_columns': int
                     - 'columns_info': dict mapping column index (1-indexed) -> (cost, set(row_indices))

.. code-block:: text

    SPEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluate a solution for a set partitioning problem case.
    Expected kwargs:
      - num_rows (int): Total number of rows.
      - num_columns (int): Total number of columns.
      - columns_info (dict): Dictionary mapping column index (1-indexed) to a tuple (cost, set(row indices)).
      - selected_columns (list): List of selected column indices (should be in strictly increasing order).
    Raises:
      ValueError: If any constraints are violated, such as an invalid output format,
                  a column index error, or if any row is not covered exactly once.
    Returns:
      score (int): The computed score, which is the total cost of the selected columns.
                   Lower scores are better.

.. code-block:: text

    SPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SPEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.travelling_salesman_problem_co_bench.evaluation
---------------------------------------------------------------------------------

TSPEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class TSPEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    TSPEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    TSPEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TSPEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TSPEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Load TSP instances from a file.
    Args:
        file_path (str): Path to the file containing TSP instances
    Returns:
        list: List of dictionaries, each containing a TSP instance with:
            - 'nodes': List of (x, y) coordinates
            - 'tour': List of node indices representing the optimal tour (if available)

.. code-block:: text

    TSPEvaluationCB.eval_func(self, nodes, label_tour, tour)

.. code-block:: text

    Evaluate a predicted TSP tour against a reference tour.
    Args:
        nodes (list): List of (x, y) coordinates representing cities in the TSP problem
                     Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
        label_tour (list): Reference/optimal tour as list of node indices
                          Format: [0, 3, 1, ...] (may be None if no reference available)
        tour (list): Predicted tour from the solver as list of node indices
                         Format: [0, 3, 1, ...]
    Returns:
        float: Optimality gap percentage ((predicted_cost/optimal_cost - 1) * 100)
               or just the predicted cost if no label_tour is provided

.. code-block:: text

    TSPEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.uncapacitated_warehouse_location_co_bench.evaluation
--------------------------------------------------------------------------------------

UWLEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class UWLEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    UWLEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    UWLEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UWLEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UWLEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads one or more problem cases from the input file.
    Expected Input File Format for each case:
      Line 1: Two integers: m n
      Next m lines: Each line contains two numbers: capacity fixed_cost for a warehouse.
      Next n lines: Each line contains: demand (a number) followed by m numbers representing the cost of
                  allocating the customer's demand to each warehouse.
    If the input file contains multiple cases, the cases appear sequentially in the file.
    Returns:
      A list of dictionaries, each corresponding to one case. Each dictionary has the keys:
         - 'm': Number of potential warehouses (int)
         - 'n': Number of customers (int)
         - 'warehouses': List of dictionaries; each with keys 'capacity' and 'fixed_cost'
         - 'customers': List of dictionaries; each with keys 'demand' and 'costs' (list of floats)

.. code-block:: text

    UWLEvaluationCB.eval_func(self, m, n, warehouses, customers, warehouse_open, assignments, **kwargs)

.. code-block:: text

    Evaluates the solution for the Uncapacitated Warehouse Location Problem.
    For each customer:
      - The customer must be assigned to exactly one open warehouse.
      - The assignment cost is the cost associated with the warehouse to which the customer is assigned.
      - No assignment is allowed for a warehouse that is closed.
    The total cost is computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of assignment costs for all customers)
    Input Parameters:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: List of dictionaries, each with keys:
            'fixed_cost': The fixed cost for opening the warehouse.
            'capacity': Provided but ignored in this problem.
      - customers: List of dictionaries, each with keys:
            'costs': A list of floats representing the cost of assigning the customer entirely to each warehouse.
            'demand': Provided but ignored in this problem.
      - warehouse_open: List of m integers (0 or 1) indicating whether each warehouse is closed or open.
      - assignments: List of n lists (each of length m) where assignments[j][i] is 1 if customer j is assigned to warehouse i, and 0 otherwise.
      - kwargs: Other parameters (not used here).
    Returns:
      A floating-point number representing the total cost if the solution is feasible.
    Raises:
      Exception: If any of the following conditions are violated:
          - The sum of assignments for any customer is not exactly 1.
          - Any positive assignment is made to a closed warehouse.
          - Any assignment value is not binary (0 or 1).

.. code-block:: text

    UWLEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UWLEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.unconstrained_guillotine_cutting_co_bench.evaluation
--------------------------------------------------------------------------------------

UGCEvaluationCB
~~~~~~~~~~~~~~~

.. code-block:: text

    class UGCEvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    UGCEvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    UGCEvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UGCEvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UGCEvaluationCB.load_data(self, input_string)

.. code-block:: text

    Loads one or more problem cases from the input file.
    The input is expected to contain one or more cases.
    Each case has the following format:
      - Line 1: An integer m (number of pieces).
      - Line 2: Two integers: stock_width and stock_height.
      - Next m lines: Each line contains three space-separated integers: l, w, value.
    Cases are concatenated one after the other (ignoring blank lines).
    Parameters:
        input_path (str): Path to the input file.
    Returns:
        list: A list of dictionaries. Each dictionary corresponds to one case and contains:
            - "m" (int): number of pieces.
            - "stock_width" (int): width of the stock rectangle.
            - "stock_height" (int): height of the stock rectangle.
            - "pieces" (dict): mapping from piece_id (1-indexed) to a dict with keys 'l', 'w', 'value'.

.. code-block:: text

    UGCEvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates a candidate solution for the guillotine cutting problem.
    This function computes the total value of the placed pieces while enforcing
    the following constraints by raising errors when violated:
      1. Each placement must be entirely within the stock rectangle.
      2. Placements must not overlap.
      3. Each piece may be used at most once.
      4. Each placement must have a valid orientation (0 or 1).
    Parameters (passed as keyword arguments):
        - m (int): Number of pieces.
        - stock_width (int): Width of the stock rectangle.
        - stock_height (int): Height of the stock rectangle.
        - pieces (dict): Dictionary mapping piece_id to {'l', 'w', 'value'}.
        - placements (list): List of placements, where each placement is a dict with keys:
              'piece_id', 'x', 'y', 'orientation'.
    Returns:
        float: Total value of the placed pieces if all constraints are met.
    Raises:
        ValueError: If any of the constraints (format, boundary, overlap, duplicate usage, or orientation)
                    are violated.

.. code-block:: text

    UGCEvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    UGCEvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.co_bench.utils
---------------------------------------

robust_request
~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    robust_request(func, *args, **kwargs)

load_subdir_as_text
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Load files from a subdirectory in a Hugging Face dataset as text format.

    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        skip_ext: File extensions to skip (default: (".py",))
        streaming: Whether to use streaming mode

    Returns:
        A dict where keys are original filenames and values are loaded datasets

    Example:
        ds = load_subdir_as_text("CO-Bench/CO-Bench", "Aircraft landing")
        # Returns: {"airland1.txt": Dataset(...), "airland2.txt": Dataset(...), ...}

.. code-block:: text

    load_subdir_as_text(repo_id: str, subdir: str, *, skip_ext: tuple[str, ...]=('.py',), streaming: bool=False)

load_subdir_as_pickle
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Load pickle files from a subdirectory in a Hugging Face dataset.

    Args:
        repo_id: The repository ID on Hugging Face (e.g., "CO-Bench/CO-Bench")
        subdir: The subdirectory path within the dataset
        include_subdirs: Tuple of subdirectory names to include (if empty, includes all)
        streaming: Whether to use streaming mode

    Returns:
        A dict where keys are subdirectory names and values are dicts of
        {filename: loaded_pickle_content}

    Example:
        result = load_subdir_as_pickle("CO-Bench/CO-Bench", "Maximal independent set",
                                     include_subdirs=("er_test", "er_large_test"))
        # Returns: {"er_test": {"file1.gpickle": graph1, ...}, "er_large_test": {...}}

.. code-block:: text

    load_subdir_as_pickle(repo_id: str, subdir: str, *, include_subdirs: tuple[str, ...]=(), streaming: bool=False)

llm4ad.task.optimization.co_bench.vehicle_routing_period_routing_co_bench.evaluation
------------------------------------------------------------------------------------

VRPREvaluationCB
~~~~~~~~~~~~~~~~

.. code-block:: text

    class VRPREvaluationCB(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    VRPREvaluationCB.__init__(self, timeout_seconds=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    VRPREvaluationCB.evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPREvaluationCB.evaluate(self, eva: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPREvaluationCB.load_data(self, input_string)

.. code-block:: text

    Reads a period vehicle routing problem file and returns a dictionary with the problem data.
    The file is expected to have the following format:
      Line 1: Two integers: <num_customers> <period_length>
              (Note: the depot is specified as customer_id = 0.)
      Line 2: A list of period_length integers representing the number of vehicles on each day.
      Line 3: A single number representing the constant capacity of every vehicle.
      Lines 4 onward: Each line represents a vertex (depot or customer) in the format:
                        customer_id x_coordinate y_coordinate demand possible_schedule_list
                        For the depot (customer_id = 0) the demand and schedule are omitted or ignored.
                        e.g., depot line: 0 30 40 0
                              customer line: 1 37 52 7 [[1, 0], [0, 1]]
    Parameters:
      input_string (str): The input content as string.
    Returns:
      A dictionary with keys:
          - "period_length" (int)
          - "vehicles_per_day" (list of ints)
          - "vehicle_capacity" (number)
          - "depot": dict with keys: "id", "x", "y"
          - "customers": list of customer dictionaries (for customer id ≠ 0)
                      Each customer dictionary contains keys:
                          "id": int, the customer id.
                          "x": float, the x coordinate.
                          "y": float, the y coordinate.
                          "demand": float, the customer demand.
                          "schedules": list of lists, each sub-list is a binary schedule for the period.

.. code-block:: text

    VRPREvaluationCB.eval_func(self, **kwargs)

.. code-block:: text

    Evaluates the solution of the Period Vehicle Routing Problem for a single case.
    Input kwargs should include:
      - from data:
            "depot": dict with keys "id", "x", "y".
            "customers": list of customer dictionaries (each with keys "id", "x", "y", "demand", "schedules").
            "vehicles_per_day": list of ints (indicating the number of available vehicles per day).
            "vehicle_capacity": numeric, the capacity of each vehicle.
            "period_length": int, the number of days.
      - from solve:
            "selected_schedules": a mapping from customer id to the chosen schedule (a list of binary integers).
            "tours": a mapping from day (1-indexed) to a list of tours;
                     each tour is a list of vertex ids (integers), starting and ending at depot (id 0),
                     with no intermediate depot visits.
    The evaluator checks the following:
      1. For each customer (other than the depot), verifies that there is a chosen schedule,
         and that the chosen schedule is one of that customer's candidate schedules.
      2. For each day:
           - Verifies that the number of tours does not exceed the available vehicles for that day.
           - Checks that every customer whose chosen schedule requires service is visited exactly once.
      3. Each tour must:
           - Start at the depot (id 0) and end at the depot (id 0).
           - Not include any depot visit in the middle (the depot may appear only as the first and the last vertex).
           - Not visit the same customer more than once.
      4. Each tour must satisfy the capacity constraint: the total customer demand on the tour does not exceed vehicle_capacity.
      5. Finally, the evaluator computes the total tour length (using Euclidean distance) over all days.
    Returns:
      A numeric value representing the total tour length computed from the solution.
    Raises an error if any constraint is violated.

.. code-block:: text

    VRPREvaluationCB.norm_score(self, results)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPREvaluationCB.get_dev(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.cvrp_construct.evaluation
--------------------------------------------------

CVRPEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class CVRPEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    CVRPEvaluation.__init__(self, timeout_seconds=20, n_instance=16, problem_size=50, capacity=40, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CVRPEvaluation.plot_solution(self, instance: np.ndarray, route: list, demands: list, vehicle_capacity: int)

.. code-block:: text

    Plot the solution of the Capacitated Vehicle Routing Problem (CVRP).

    Args:
        instance: A 2D array of node coordinates (including the depot).
        route: A list representing the sequence of nodes visited in the route.
        demands: A list of demands for each node.
        vehicle_capacity: The capacity of the vehicle.

.. code-block:: text

    CVRPEvaluation.tour_cost(self, instance, solution)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CVRPEvaluation.route_construct(self, distance_matrix, demands, vehicle_capacity, heuristic)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CVRPEvaluation.evaluate(self, heuristic)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    CVRPEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.cvrp_construct.get_instance
----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_cities, capacity)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    each instance -> (coordinates, distances, demands, capacity)

llm4ad.task.optimization.jssp_construct.evaluation
--------------------------------------------------

JSSPEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class JSSPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for Job Shop Scheduling Problem.

.. code-block:: text

    JSSPEvaluation.__init__(self, timeout_seconds=20, n_instance=16, n_jobs=50, n_machines=10, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    JSSPEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    JSSPEvaluation.plot_solution(self, schedule: List[List[Tuple[int, int, int]]], n_jobs: int, n_machines: int)

.. code-block:: text

    Plots the schedule as a Gantt chart.

    Args:
        schedule: The schedule generated by select_next_operation.
        n_jobs: Number of jobs.
        n_machines: Number of machines.

.. code-block:: text

    JSSPEvaluation.schedule_jobs(self, processing_times, n_jobs, n_machines, eva)

.. code-block:: text

    Schedule jobs on machines using a greedy constructive heuristic.

    Args:
        processing_times: A list of lists representing the processing times of each job on each machine.
        n_jobs: Number of jobs.
        n_machines: Number of machines.

    Returns:
        The makespan, which is the total time required to complete all jobs.

.. code-block:: text

    JSSPEvaluation.evaluate(self, eva: Callable) -> float

.. code-block:: text

    Evaluate the constructive heuristic for JSSP.

    Args:
        instance_data: List of tuples containing the processing times, number of jobs, and number of machines.
        n_ins: Number of instances to evaluate.
        n_jobs: Number of jobs.
        n_machines: Number of machines.
        eva: The constructive heuristic function to evaluate.

    Returns:
        The average makespan across all instances.

llm4ad.task.optimization.jssp_construct.get_instance
----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_jobs: int, n_machines: int)

.. code-block:: text

    Initialize the GetData class for JSSP.

    Args:
        n_instance: Number of instances to generate.
        n_jobs: Number of jobs.
        n_machines: Number of machines.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Job Shop Scheduling Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - processing_times: A list of lists representing the processing times of each job on each machine.
        - n_jobs: Number of jobs.
        - n_machines: Number of machines.

llm4ad.task.optimization.knapsack_construct.evaluation
------------------------------------------------------

KnapsackEvaluation
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class KnapsackEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the Knapsack Problem.

.. code-block:: text

    KnapsackEvaluation.__init__(self, timeout_seconds=20, n_instance=32, n_items=50, knapsack_capacity=100, **kwargs)

.. code-block:: text

    Initialize the evaluator for the Knapsack Problem.

.. code-block:: text

    KnapsackEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnapsackEvaluation.plot_solution(self, item_weights: list, item_values: list, selected_indices: list, knapsack_capacity: int)

.. code-block:: text

    Plot the solution of the Knapsack problem.

    Args:
        item_weights: A list of item weights.
        item_values: A list of item values.
        selected_indices: A list of indices of selected items.
        knapsack_capacity: The capacity of the knapsack.

.. code-block:: text

    KnapsackEvaluation.pack_items(self, item_weights: List[int], item_values: List[int], knapsack_capacity: int, eva: Callable) -> Tuple[int, List[int]]

.. code-block:: text

    Select items for the knapsack using a constructive heuristic.

    Args:
        item_weights: A list of item weights.
        item_values: A list of item values.
        knapsack_capacity: The capacity of the knapsack.
        eva: The constructive heuristic function to select the next item.

    Returns:
        A tuple containing:
        - The total value of the selected items.
        - A list of selected item indices.

.. code-block:: text

    KnapsackEvaluation.evaluate(self, eva: Callable) -> float

.. code-block:: text

    Evaluate the constructive heuristic for the Knapsack Problem.

    Args:
        instance_data: List of tuples containing the item weights, values, and knapsack capacity.
        n_ins: Number of instances to evaluate.
        eva: The constructive heuristic function to evaluate.

    Returns:
        The average total value of selected items across all instances.

llm4ad.task.optimization.knapsack_construct.get_instance
--------------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_items: int, knapsack_capacity: int)

.. code-block:: text

    Initialize the GetData class for the Knapsack Problem.

    Args:
        n_instance: Number of instances to generate.
        n_items: Number of items.
        knapsack_capacity: Capacity of the knapsack.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Knapsack Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - item_weights: A list of item weights.
        - item_values: A list of item values.
        - knapsack_capacity: The capacity of the knapsack.

llm4ad.task.optimization.max_cut.evaluation
-------------------------------------------

MaxCutEvaluation
~~~~~~~~~~~~~~~~

.. code-block:: text

    class MaxCutEvaluation(Evaluation)

.. code-block:: text

    Evaluator for maximum cut problem using local search.

.. code-block:: text

    MaxCutEvaluation.__init__(self, timeout_seconds=30, n_instance=16, n_nodes=100, edge_probability=0.5, max_flips=None, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MaxCutEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MaxCutEvaluation.compute_cut_value(self, adjacency: dict, cut_S: list, cut_T: list) -> float

.. code-block:: text

    Compute total weight of edges crossing between S and T.

.. code-block:: text

    MaxCutEvaluation.compute_flip_gain(self, node: int, adjacency: dict, cut_S_set: set, cut_T_set: set) -> float

.. code-block:: text

    Compute the gain of flipping a node (positive means improvement).

.. code-block:: text

    MaxCutEvaluation.get_improving_candidates(self, adjacency: dict, cut_S: list, cut_T: list) -> list

.. code-block:: text

    Get all nodes whose flip would improve (or not worsen) the cut.

.. code-block:: text

    MaxCutEvaluation.evaluate(self, eva: callable) -> float

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.max_cut.get_instance
---------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_nodes, edge_probability=0.5)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.online_bin_packing.evaluation
------------------------------------------------------

OBPEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class OBPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for online bin packing problem.

.. code-block:: text

    OBPEvaluation.__init__(self, timeout_seconds=30, n_instances=5, n_items=5000, capacity=100, **kwargs)

.. code-block:: text

    Args:
        - 'data_file' (str): The data file to load (default is 'weibull_5k_train.pkl').
        - 'data_key' (str): The key of the data to load (default is 'data_key').

    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    OBPEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OBPEvaluation.plot_solution(self, bins_packed: np.ndarray, items: list, capacity: int, max_unused_bins: int=5)

.. code-block:: text

    Plot the solution of the 1D Online Bin Packing Problem, omitting unused bins.

    Args:
        bins_packed: A numpy array of remaining capacities in the bins after packing.
        items: A list of item sizes.
        capacity: The capacity of each bin.
        max_unused_bins: Maximum number of unused bins to include in the plot (for sampling).

.. code-block:: text

    OBPEvaluation.get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray

.. code-block:: text

    Returns indices of bins in which item can fit.

.. code-block:: text

    OBPEvaluation.online_binpack(self, items: tuple[float, ...], bins: np.ndarray, priority: callable) -> tuple[list[list[float, ...], ...], np.ndarray]

.. code-block:: text

    Performs online binpacking of `items` into `bins`.

.. code-block:: text

    OBPEvaluation.evaluate(self, priority: callable) -> float

.. code-block:: text

    Evaluate heuristic function on a set of online binpacking instances.

llm4ad.task.optimization.online_bin_packing.generate_weibull_instances
----------------------------------------------------------------------

generate_weibull_dataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    generate_weibull_dataset(num_instances, num_items, capacity_limit)

llm4ad.task.optimization.online_bin_packing_2O.evaluation
---------------------------------------------------------

get_valid_bin_indices
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Returns indices of bins in which item can fit.

.. code-block:: text

    get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray

online_binpack
~~~~~~~~~~~~~~

.. code-block:: text

    Performs online binpacking of `items` into `bins`.

.. code-block:: text

    online_binpack(items: tuple[float, ...], bins: np.ndarray, priority: callable) -> tuple[list[list[float, ...], ...], np.ndarray]

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate heuristic function on a set of online binpacking instances.

.. code-block:: text

    evaluate(instances: dict, priority: callable) -> np.ndarray

OBP_2O_Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class OBP_2O_Evaluation(Evaluation)

.. code-block:: text

    Evaluator for online bin packing problem.

.. code-block:: text

    OBP_2O_Evaluation.__init__(self, timeout_seconds=60, data_file='weibull_train.pkl', data_key='weibull_5k_train', **kwargs)

.. code-block:: text

    Args:
        - 'data_file' (str): The data file to load (default is 'weibull_5k_train.pkl').
        - 'data_key' (str): The key of the data to load (default is 'data_key').

    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    OBP_2O_Evaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.online_bin_packing_2O.generate_weibull_instances
-------------------------------------------------------------------------

generate_weibull_dataset
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    generate_weibull_dataset(num_instances, num_items, capacity_limit)

llm4ad.task.optimization.orienteering_construct.evaluation
----------------------------------------------------------

OrienteeringEvaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class OrienteeringEvaluation(Evaluation)

.. code-block:: text

    Evaluator for constructive heuristics for the Orienteering Problem.

.. code-block:: text

    OrienteeringEvaluation.__init__(self, timeout_seconds=30, n_instance=16, problem_size=50, max_length_ratio=0.35, seed=2024, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OrienteeringEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OrienteeringEvaluation.get_feasible_unvisited_nodes(self, current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, remaining_budget: float) -> np.ndarray

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OrienteeringEvaluation.construct_solution(self, instance: dict, select_next_node: callable) -> tuple[list[int], float, float] | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OrienteeringEvaluation.evaluate(self, select_next_node: callable) -> float | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.orienteering_construct.get_instance
------------------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, problem_size: int, max_length_ratio: float=0.35, seed: int=2024)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.ovrp_construct.evaluation
--------------------------------------------------

OVRPEvaluation
~~~~~~~~~~~~~~

.. code-block:: text

    class OVRPEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    OVRPEvaluation.__init__(self, timeout_seconds=20, problem_size=50, n_instance=16, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OVRPEvaluation.plot_solution(self, instance: np.ndarray, route: list, demands: list, vehicle_capacity: int)

.. code-block:: text

    Plot the solution of the Open Vehicle Routing Problem (Open VRP).

    Args:
        instance: A 2D array of node coordinates (including the depot).
        route: A list representing the sequence of nodes visited in the route.
        demands: A list of demands for each node.
        vehicle_capacity: The capacity of the vehicle.

.. code-block:: text

    OVRPEvaluation.tour_cost(self, instance, solution)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OVRPEvaluation.route_construct(self, distance_matrix, demands, vehicle_capacity, heuristic)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OVRPEvaluation.evaluate(self, heuristic)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OVRPEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.ovrp_construct.get_instance
----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_cities)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    each instance -> (coordinates, distances, demands, capacity)

llm4ad.task.optimization.pymoo_moead.evaluation
-----------------------------------------------

MOEAD_PYMOO_Evaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class MOEAD_PYMOO_Evaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    MOEAD_PYMOO_Evaluation.__init__(self, timeout_seconds=100, n_var=10, n_obj=3, n_partitions=12, pop_size=100, n_gen=100, seed=None, **kwargs)

.. code-block:: text

    Parameter Description:
    This evaluator now receives a decomposition function via the evaluate_program interface.

.. code-block:: text

    MOEAD_PYMOO_Evaluation.evaluate(self, decomposition_func: Callable=None) -> float

.. code-block:: text

    Return hypervolume (larger is better) and store detailed results in self.last_result.

.. code-block:: text

    MOEAD_PYMOO_Evaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    MOEAD_PYMOO_Evaluation.plot_solutions(self, solutions)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.pymoo_moead.get_instance
-------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_var, n_obj)

.. code-block:: text

    Initialize parameters for the DTLZ problem.
    Args:
        n_var (int): The number of decision variables.
        n_obj (int): The number of objectives.

.. code-block:: text

    GetData.get_problem_instance(self)

.. code-block:: text

    Generate and return a DTLZ4 problem instance using the pymoo library.

llm4ad.task.optimization.qap_construct.evaluation
-------------------------------------------------

QAPEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class QAPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the Quadratic Assignment Problem.

.. code-block:: text

    QAPEvaluation.__init__(self, timeout_seconds=60, n_facilities=20, n_instance=8, **kwargs)

.. code-block:: text

    Initializes the QAP evaluator.

.. code-block:: text

    QAPEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Evaluates the program (constructive heuristic) for the QAP.

    Args:
        program_str: Program string (not used here, but kept for compatibility).
        callable_func: The constructive heuristic function to evaluate.

    Returns:
        The average total cost across all instances.

.. code-block:: text

    QAPEvaluation.plot_solution(self, flow_matrix: np.ndarray, distance_matrix: np.ndarray, assignment: List[int])

.. code-block:: text

    Plot the solution of the Quadratic Assignment Problem.

    Args:
        flow_matrix: Flow matrix between facilities.
        distance_matrix: Distance matrix between locations.
        assignment: Assignment of facilities to locations.

.. code-block:: text

    QAPEvaluation.qap_evaluate(self, current_assignment: List[int], flow_matrix: np.ndarray, distance_matrix: np.ndarray, eva: Callable) -> List[int]

.. code-block:: text

    Evaluate the assignment for the Quadratic Assignment Problem using a constructive heuristic.

    The heuristic `eva` is expected to return a complete assignment (a permutation of locations)
    in one call.

.. code-block:: text

    QAPEvaluation.evaluate_qap(self, eva: Callable) -> float

.. code-block:: text

    Evaluate the constructive heuristic for the Quadratic Assignment Problem.

    Args:
        instance_data: List of tuples containing the flow and distance matrices.
        n_ins: Number of instances to evaluate.
        eva: The constructive heuristic function to evaluate.

    Returns:
        The average total cost across all instances.

llm4ad.task.optimization.qap_construct.get_instance
---------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_facilities: int)

.. code-block:: text

    Initialize the QAPDataGenerator class for the Quadratic Assignment Problem.

    Args:
        n_instance: Number of instances to generate.
        n_facilities: Number of facilities (and locations).

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Quadratic Assignment Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - flow_matrix: A 2D numpy array representing the flow between facilities.
        - distance_matrix: A 2D numpy array representing the distance between locations.

llm4ad.task.optimization.set_cover_construct.evaluation
-------------------------------------------------------

SCPEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class SCPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for the Set Covering Problem.

.. code-block:: text

    SCPEvaluation.__init__(self, timeout_seconds=30, n_instance: int=16, n_elements: int=50, n_subsets: int=50, max_subset_size: int=8, **kwargs)

.. code-block:: text

    Args:
        n_instance: Number of instances to generate.
        n_elements: Number of elements in the universal set.
        n_subsets: Number of subsets in the collection.
        max_subset_size: Maximum size of each subset.

.. code-block:: text

    SCPEvaluation.evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None

.. code-block:: text

    Evaluate a constructive heuristic for the Set Covering Problem.

    Args:
        program_str: A string representation of the heuristic (unused here).
        callable_func: The constructive heuristic function to evaluate.

    Returns:
        The average number of subsets used.

.. code-block:: text

    SCPEvaluation.plot_solution(self, universal_set: List[int], selected_subsets: List[List[int]], all_subsets: List[List[int]])

.. code-block:: text

    Plot the final solution of the Set Covering Problem, including selected and unselected subsets.

    Args:
        universal_set: The universal set of elements.
        selected_subsets: The list of selected subsets that cover the universal set.
        all_subsets: The list of all subsets (including unselected ones).

.. code-block:: text

    SCPEvaluation.cover_subsets(self, universal_set: List[int], subsets: List[List[int]], eva: Callable) -> Tuple[int, List[List[int]]] | None

.. code-block:: text

    Select subsets to cover the universal set using a constructive heuristic.

    Args:
        universal_set: The universal set of elements to cover.
        subsets: A list of subsets, where each subset is a list of elements.
        eva: The constructive heuristic function to select the next subset.

    Returns:
        A tuple containing:
        - The total number of subsets used.
        - A list of selected subsets.
        Returns None if the heuristic stops before covering every coverable element.

.. code-block:: text

    SCPEvaluation.evaluate(self, eva: Callable) -> float | None

.. code-block:: text

    Evaluate the constructive heuristic for the Set Covering Problem.

    Args:
        instance_data: List of tuples containing the universal set and subsets.
        n_ins: Number of instances to evaluate.
        eva: The constructive heuristic function to evaluate.

    Returns:
        The average number of subsets used across all instances.

llm4ad.task.optimization.set_cover_construct.get_instance
---------------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_elements: int, n_subsets: int, max_subset_size: int)

.. code-block:: text

    Initialize the GetData class for the Set Covering Problem.

    Args:
        n_instance: Number of instances to generate.
        n_elements: Number of elements in the universal set.
        n_subsets: Number of subsets in the collection.
        max_subset_size: Maximum size of each subset.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Set Covering Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - universal_set: A list of elements in the universal set.
        - subsets: A list of subsets, where each subset is a list of elements.

llm4ad.task.optimization.set_cover_construct.test
-------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance: int, n_jobs: int, n_machines: int)

.. code-block:: text

    Initialize the GetData class for JSSP.

    Args:
        n_instance: Number of instances to generate.
        n_jobs: Number of jobs.
        n_machines: Number of machines.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Generate instances for the Job Shop Scheduling Problem.

    Returns:
        A list of tuples, where each tuple contains:
        - processing_times: A list of lists representing the processing times of each job on each machine.
        - n_jobs: Number of jobs.
        - n_machines: Number of machines.

determine_next_operation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Determine the next operation to schedule based on a greedy heuristic.

    Args:
        current_status: A dictionary representing the current status of each machine and job.
        feasible_operations: A list of feasible operations that can be scheduled next.

    Returns:
        The next operation to schedule, represented as a tuple (job_id, machine_id, processing_time).

.. code-block:: text

    determine_next_operation(current_status, feasible_operations)

schedule_jobs
~~~~~~~~~~~~~

.. code-block:: text

    Schedule jobs on machines using a greedy constructive heuristic.

    Args:
        processing_times: A list of lists representing the processing times of each job on each machine.
        n_jobs: Number of jobs.
        n_machines: Number of machines.

    Returns:
        The makespan, which is the total time required to complete all jobs.

.. code-block:: text

    schedule_jobs(processing_times, n_jobs, n_machines)

llm4ad.task.optimization.tsp_construct.evaluation
-------------------------------------------------

TSPEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class TSPEvaluation(Evaluation)

.. code-block:: text

    Evaluator for traveling salesman problem.

.. code-block:: text

    TSPEvaluation.__init__(self, timeout_seconds=30, n_instance=16, problem_size=50, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    TSPEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TSPEvaluation.tour_cost(self, instance, solution, problem_size)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TSPEvaluation.generate_neighborhood_matrix(self, instance)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TSPEvaluation.evaluate(self, eva: callable) -> float

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.tsp_construct.get_instance
---------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_cities)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.tsp_gls_2O.evaluation
----------------------------------------------

calculate_cost
~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    calculate_cost(inst: TSPInstance, path: np.ndarray) -> float

solve_with_time
~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    solve_with_time(inst: TSPInstance, eva) -> Tuple[float, float]

evaluate
~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    evaluate(instance_data, n_ins, prob_size, eva: callable) -> np.ndarray

TSP_GLS_2O_Evaluation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class TSP_GLS_2O_Evaluation(Evaluation)

.. code-block:: text

    Evaluator for traveling salesman problem.

.. code-block:: text

    TSP_GLS_2O_Evaluation.__init__(self, **kwargs)

.. code-block:: text

    Args:
        None
    Raises:
        AttributeError: If the data key does not exist.
        FileNotFoundError: If the specified data file is not found.

.. code-block:: text

    TSP_GLS_2O_Evaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.tsp_gls_2O.get_instance
------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_cities)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

TSPInstance
~~~~~~~~~~~

.. code-block:: text

    class TSPInstance()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    TSPInstance.__init__(self, positions: npt.NDArray[np.float64]) -> None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.tsp_gls_2O.gls
---------------------------------------

guided_local_search
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    guided_local_search(distmat: FloatArray, guide: FloatArray, update_edge_distance, perturbation_moves: int=30, iter_limit: int=1000) -> npt.NDArray[np.uint16]

guided_local_search_with_time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    guided_local_search_with_time(distmat: FloatArray, guide: FloatArray, update_edge_distance, perturbation_moves: int=30, iter_limit: int=1000) -> Tuple[npt.NDArray[np.uint16], float]

multi_start_guided_local_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    multi_start_guided_local_search(dist: FloatArray, guide: FloatArray, n_starts: int=10, perturbation_moves=30, iter_limit=1000)

llm4ad.task.optimization.vrptw_construct.evaluation
---------------------------------------------------

VRPTWEvaluation
~~~~~~~~~~~~~~~

.. code-block:: text

    class VRPTWEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    VRPTWEvaluation.__init__(self, timeout_seconds=30, problem_size=50, n_instance=16, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPTWEvaluation.tour_cost(self, distance_matrix, solution, time_service, time_windows)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPTWEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VRPTWEvaluation.evaluate(self, heuristic)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.optimization.vrptw_construct.get_instance
-----------------------------------------------------

GetData
~~~~~~~

.. code-block:: text

    class GetData()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    GetData.__init__(self, n_instance, n_cities)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    GetData.generate_instances(self)

.. code-block:: text

    each instance -> (coordinates, distances, demands, capacity)

llm4ad.task.science_discovery.bactgrow.evaluation
-------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(data: dict, equation: callable) -> float | None

BGEvaluation
~~~~~~~~~~~~

.. code-block:: text

    class BGEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    BGEvaluation.__init__(self, timeout_seconds=20, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    BGEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.feynman_srsd.base
-----------------------------------------------

KnownEquation
~~~~~~~~~~~~~

.. code-block:: text

    class KnownEquation(object)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    KnownEquation.__init__(self, num_vars, sampling_objs, kwargs_list=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.get_eq_name(self, prefix=None, suffix=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.get_var_count(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.get_op_count(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.check_num_vars_consistency(self, debug=False)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.get_domain_range(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.check_if_valid(self, values)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.create_dataset(self, sample_size, patience=10)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.traverse_tree(self, node, dot, from_idx=None, node_list=None, num_digits=4)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.find_stationary_points(self, excludes_saddle_points=False)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    KnownEquation.from_sympy_eq(cls, sympy_eq, sampling_objs, reindexes=True)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.feynman_srsd.evaluation
-----------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(data: dict, equation: callable) -> float | None

FeynmanEvaluation
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    FeynmanEvaluation.__init__(self, timeout_seconds=20, test_id=1, sample_size=5000, **kwargs)

.. code-block:: text

    Args:
        timeout_seconds: evaluate time limit.
        test_id: test equation id ranges from [1, 120].

.. code-block:: text

    FeynmanEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.feynman_srsd.feynman_equations
------------------------------------------------------------

register_feynman_eq_class
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    register_feynman_eq_class(cls)

FeynmanICh6Eq20
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh6Eq20(KnownEquation)

.. code-block:: text

    - Equation: I.6.20
    - Raw: exp(-(theta / sigma) ** 2 / 2) / (sqrt(2 * pi) * sigma)
    - Num. Vars: 2
    - Vars:
        - x[0]: theta (float)
        - x[1]: sigma (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanICh6Eq20.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh6Eq20.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh6Eq20a
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh6Eq20a(KnownEquation)

.. code-block:: text

    - Equation: I.6.20a
    - Raw: exp(-theta ** 2 / 2) / sqrt(2 * pi)
    - Num. Vars: 1
    - Vars:
        - x[0]: theta (float)
    - Constraints:

.. code-block:: text

    FeynmanICh6Eq20a.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh6Eq20a.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh6Eq20b
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh6Eq20b(KnownEquation)

.. code-block:: text

    - Equation: I.6.20b
    - Raw: exp(-((theta - theta1) / sigma) ** 2 / 2) / (sqrt(2 * pi) * sigma)
    - Num. Vars: 3
    - Vars:
        - x[0]: theta (float)
        - x[1]: theta1 (float)
        - x[2]: sigma (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanICh6Eq20b.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh6Eq20b.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh8Eq14
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh8Eq14(KnownEquation)

.. code-block:: text

    - Equation: I.8.14
    - Raw: sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: x2 (float)
        - x[1]: x1 (float)
        - x[2]: y2 (float)
        - x[3]: y1 (float)
    - Constraints:

.. code-block:: text

    FeynmanICh8Eq14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh8Eq14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh9Eq18
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh9Eq18(KnownEquation)

.. code-block:: text

    - Equation: I.9.18
    - Raw: 6.6743e-11 * m1 * m2 / ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    - Num. Vars: 9
    - Vars:
        - x[0]: m1 (float, positive)
        - x[1]: m2 (float, positive)
        - x[2]: x2 (float)
        - x[3]: x1 (float)
        - x[4]: y2 (float)
        - x[5]: y1 (float)
        - x[6]: z2 (float)
        - x[7]: z1 (float)
    - Constraints:
        - (x[2] - x[3]) ** 2 + (x[4] - x[5]) ** 2 + (x[6] - x[7]) ** 2 != 0

.. code-block:: text

    FeynmanICh9Eq18.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh9Eq18.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh10Eq7
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh10Eq7(KnownEquation)

.. code-block:: text

    - Equation: I.10.7
    - Raw: m_0 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: m_0 (float, positive)
        - x[1]: v (float, positive)
    - Constraints:
        - 1 - x[1] ** 2 / 2.99792458e8 ** 2 > 0

.. code-block:: text

    FeynmanICh10Eq7.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh10Eq7.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh11Eq19
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh11Eq19(KnownEquation)

.. code-block:: text

    - Equation: I.11.19
    - Raw: x1 * y1 + x2 * y2 + x3 * y3
    - Num. Vars: 6
    - Vars:
        - x[0]: x1 (float)
        - x[1]: y1 (float)
        - x[2]: x2 (float)
        - x[3]: y2 (float)
        - x[4]: x3 (float)
        - x[5]: y3 (float)
    - Constraints:

.. code-block:: text

    FeynmanICh11Eq19.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh11Eq19.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh12Eq1
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh12Eq1(KnownEquation)

.. code-block:: text

    - Equation: I.12.1
    - Raw: mu * Nn
    - Num. Vars: 2
    - Vars:
        - x[0]: mu (float, positive)
        - x[1]: Nn (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh12Eq1.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh12Eq1.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh12Eq2
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh12Eq2(KnownEquation)

.. code-block:: text

    - Equation: I.12.2
    - Raw: q1 * q2 * r / (4 * pi * 8.854e-12 * r ** 3)
    - Num. Vars: 3
    - Vars:
        - x[0]: q1 (float)
        - x[1]: q2 (float)
        - x[2]: r (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanICh12Eq2.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh12Eq2.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh12Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh12Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.12.4
    - Raw: q1 * r / (4 * pi * 8.854e-12 * r ** 3)
    - Num. Vars: 2
    - Vars:
        - x[0]: q1 (float)
        - x[1]: r (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanICh12Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh12Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh12Eq5
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh12Eq5(KnownEquation)

.. code-block:: text

    - Equation: I.12.5
    - Raw: q2 * Ef
    - Num. Vars: 2
    - Vars:
        - x[0]: q2 (float)
        - x[1]: Ef (float)
    - Constraints:

.. code-block:: text

    FeynmanICh12Eq5.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh12Eq5.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh12Eq11
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh12Eq11(KnownEquation)

.. code-block:: text

    - Equation: I.12.11
    - Raw: q * (Ef + B * v * sin(theta))
    - Num. Vars: 5
    - Vars:
        - x[0]: q (float)
        - x[1]: Ef (float)
        - x[2]: B (float, positive)
        - x[3]: v (float, positive)
        - x[4]: theta (float)
    - Constraints:

.. code-block:: text

    FeynmanICh12Eq11.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh12Eq11.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh13Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh13Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.13.4
    - Raw: 1 / 2 * m * (v ** 2 + u ** 2 + w ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: v (float,)
        - x[2]: u (float)
        - x[3]: w (float)
    - Constraints:

.. code-block:: text

    FeynmanICh13Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh13Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh13Eq12
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh13Eq12(KnownEquation)

.. code-block:: text

    - Equation: I.13.12
    - Raw: 6.67430e-11 * m1 * m2 * (1 / r2 - 1 / r1)
    - Num. Vars: 4
    - Vars:
        - x[0]: m1 (float, positive)
        - x[1]: m2 (float, positive)
        - x[2]: r2 (float, positive)
        - x[3]: r1 (float, positive)
    - Constraints:
        - x[2] != 0
        - x[3] != 0

.. code-block:: text

    FeynmanICh13Eq12.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh13Eq12.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh14Eq3
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh14Eq3(KnownEquation)

.. code-block:: text

    - Equation: I.14.3
    - Raw: 9.8066 * m * z
    - Num. Vars: 2
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: z (float)
    - Constraints:

.. code-block:: text

    FeynmanICh14Eq3.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh14Eq3.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh14Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh14Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.14.4
    - Raw: 1 / 2 * k_spring * x ** 2
    - Num. Vars: 2
    - Vars:
        - x[0]: k_spring (float, positive)
        - x[1]: x (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh14Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh14Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh15Eq10
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh15Eq10(KnownEquation)

.. code-block:: text

    - Equation: I.15.10
    - Raw: m_0 * v / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: m_0 (float, positive)
        - x[1]: v (float)
    - Constraints:
        - 1 - x[1] ** 2 / 2.99792458e8 ** 2 > 0

.. code-block:: text

    FeynmanICh15Eq10.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh15Eq10.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh15Eq3t
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh15Eq3t(KnownEquation)

.. code-block:: text

    - Equation: I.15.3t
    - Raw: (t - u * x / c ** 2) / sqrt(1 - u ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: t (float, positive)
        - x[1]: u (float)
        - x[2]: x (float)
    - Constraints:
        - 1 - x[1] ** 2 / 2.99792458e8 ** 2 >= 0

.. code-block:: text

    FeynmanICh15Eq3t.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh15Eq3t.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh15Eq3x
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh15Eq3x(KnownEquation)

.. code-block:: text

    - Equation: I.15.3x
    - Raw: (x - u * t) / sqrt(1 - u ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: x (float)
        - x[1]: u (float)
        - x[2]: t (float)
    - Constraints:
        - 1 - x[1] ** 2 / 2.99792458e8 ** 2 > 0

.. code-block:: text

    FeynmanICh15Eq3x.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh15Eq3x.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh16Eq6
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh16Eq6(KnownEquation)

.. code-block:: text

    - Equation: I.16.6
    - Raw: (u + v) / (1 + u * v / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: u (float)
        - x[1]: v (float)
    - Constraints:
        - 1 + x[0] * x[1] != 0

.. code-block:: text

    FeynmanICh16Eq6.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh16Eq6.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh18Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh18Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.18.4
    - Raw: (m1 * r1 + m2 * r2) / (m1 + m2)
    - Num. Vars: 4
    - Vars:
        - x[0]: m1 (float, positive)
        - x[1]: r1 (float)
        - x[2]: m2 (float, positive)
        - x[3]: r2 (float)
    - Constraints:
        - x[0] + x[2] != 0

.. code-block:: text

    FeynmanICh18Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh18Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh18Eq12
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh18Eq12(KnownEquation)

.. code-block:: text

    - Equation: I.18.12
    - Raw: r * F * sin(theta)
    - Num. Vars: 3
    - Vars:
        - x[0]: r (float, positive)
        - x[1]: F (float)
        - x[2]: theta (float)
    - Constraints:

.. code-block:: text

    FeynmanICh18Eq12.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh18Eq12.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh18Eq16
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh18Eq16(KnownEquation)

.. code-block:: text

    - Equation: I.18.16
    - Raw: m * r * v * sin(theta)
    - Num. Vars: 4
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: r (float, positive)
        - x[2]: v (float)
        - x[3]: theta (float)
    - Constraints:

.. code-block:: text

    FeynmanICh18Eq16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh18Eq16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh24Eq6
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh24Eq6(KnownEquation)

.. code-block:: text

    - Equation: I.24.6
    - Raw: 1 / 2 * m * (omega ** 2 + omega_0 ** 2) * 1 / 2 * x ** 2
    - Num. Vars: 4
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: omega (float)
        - x[2]: omega_0 (float)
        - x[3]: x (float)
    - Constraints:

.. code-block:: text

    FeynmanICh24Eq6.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh24Eq6.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh25Eq13
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh25Eq13(KnownEquation)

.. code-block:: text

    - Equation: I.25.13
    - Raw: q / C
    - Num. Vars: 2
    - Vars:
        - x[0]: q (float)
        - x[1]: C (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanICh25Eq13.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh25Eq13.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh26Eq2
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh26Eq2(KnownEquation)

.. code-block:: text

    - Equation: I.26.2
    - Raw: sin(theta1) / sin(theta2)
    - Num. Vars: 2
    - Vars:
        - x[0]: theta1 (float, positive)
        - x[1]: theta2 (float, positive)
    - Constraints:
        - x[0] * np.sin(x[1]) >= -np.pi /2
        - x[0] * np.sin(x[1]) <= np.pi/2

.. code-block:: text

    FeynmanICh26Eq2.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh26Eq2.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh27Eq6
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh27Eq6(KnownEquation)

.. code-block:: text

    - Equation: I.27.6
    - Raw: 1 / (1 / d1 + n / d2)
    - Num. Vars: 3
    - Vars:
        - x[0]: d1 (float, positive)
        - x[1]: n (float, positive)
        - x[2]: d2 (float, positive)
    - Constraints:
        - x[0] != 0
        - x[2] != 0

.. code-block:: text

    FeynmanICh27Eq6.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh27Eq6.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh29Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh29Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.29.4
    - Raw: omega / 2.99792458e8
    - Num. Vars: 1
    - Vars:
        - x[0]: omega (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh29Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh29Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh29Eq16
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh29Eq16(KnownEquation)

.. code-block:: text

    - Equation: I.29.16
    - Raw: sqrt(x1 ** 2 + x2 ** 2 + 2 * x1 * x2 * cos(theta1 - theta2))
    - Num. Vars: 4
    - Vars:
        - x[0]: x1 (float, positive)
        - x[1]: x2 (float, positive)
        - x[2]: theta1 (float)
        - x[3]: theta2 (float)
    - Constraints:

.. code-block:: text

    FeynmanICh29Eq16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh29Eq16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh30Eq3
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh30Eq3(KnownEquation)

.. code-block:: text

    - Equation: I.30.3
    - Raw: Int_0 * sin(n * theta / 2) ** 2 / sin(theta / 2) ** 2
    - Num. Vars: 3
    - Vars:
        - x[0]: Int_0 (float, positive)
        - x[1]: n (integer, positive)
        - x[2]: theta (float)
    - Constraints:
        - np.sin(x[2] / 2) != 0

.. code-block:: text

    FeynmanICh30Eq3.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh30Eq3.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh30Eq5
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh30Eq5(KnownEquation)

.. code-block:: text

    - Equation: I.30.5
    - Raw: lambda / (n * sin(theta))
    - Num. Vars: 3
    - Vars:
        - x[0]: lambda (float, positive)
        - x[1]: n (integer, positive)
        - x[2]: theta (float, positive)
    - Constraints:
        - x[1] != 0
        - x[2] != 0
        - x[2] != pi

.. code-block:: text

    FeynmanICh30Eq5.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh30Eq5.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh32Eq5
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh32Eq5(KnownEquation)

.. code-block:: text

    - Equation: I.32.5
    - Raw: q ** 2 * a ** 2 / (6 * pi * 8.854e-12 * 2.99792458e8 ** 3)
    - Num. Vars: 4
    - Vars:
        - x[0]: q (float)
        - x[1]: a (float, positive)
    - Constraints:
        - x[2] != 0
        - x[3] != 0

.. code-block:: text

    FeynmanICh32Eq5.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh32Eq5.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh32Eq17
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh32Eq17(KnownEquation)

.. code-block:: text

    - Equation: I.32.17
    - Raw: (1 / 2 * 8.854e-12 * 2.99792458e8 * Ef ** 2) * (8 * pi * r ** 2 / 3) * (omega ** 4 / (omega ** 2 - omega_0 ** 2) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: Ef (float)
        - x[1]: r (float, positive)
        - x[2]: omega (float)
        - x[3]: omega_0 (float)
    - Constraints:
        - x[2] ** 2 - x[3] ** 2 != 0

.. code-block:: text

    FeynmanICh32Eq17.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh32Eq17.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh34Eq10
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh34Eq10(KnownEquation)

.. code-block:: text

    - Equation: I.34.10
    - Raw: omega_0 / (1 - v / 2.99792458e8)
    - Num. Vars: 2
    - Vars:
        - x[0]: omega_0 (float, positive)
        - x[1]: v (float)
    - Constraints:
        - 2.99792458e8 - x[1] != 0

.. code-block:: text

    FeynmanICh34Eq10.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh34Eq10.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh34Eq8
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh34Eq8(KnownEquation)

.. code-block:: text

    - Equation: I.34.8
    - Raw: q * v * B / p
    - Num. Vars: 4
    - Vars:
        - x[0]: q (float)
        - x[1]: v (float)
        - x[2]: B (float)
        - x[3]: p (float)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanICh34Eq8.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh34Eq8.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh34Eq14
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh34Eq14(KnownEquation)

.. code-block:: text

    - Equation: I.34.14
    - Raw: (1 + v / 2.99792458e8) / sqrt(1 - v ** 2 / 2.99792458e8 ** 2) * omega_0
    - Num. Vars: 2
    - Vars:
        - x[0]: v (float)
        - x[1]: omega_0 (float, positive)
    - Constraints:
        - 2.99792458e8 ** 2 - x[0] ** 2 > 0
        - x[1] != 0

.. code-block:: text

    FeynmanICh34Eq14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh34Eq14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh34Eq27
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh34Eq27(KnownEquation)

.. code-block:: text

    - Equation: I.34.27
    - Raw: (6.626e-34 / (2 * pi)) * omega
    - Num. Vars: 1
    - Vars:
        - x[0]: omega (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh34Eq27.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh34Eq27.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh37Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh37Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.37.4
    - Raw: I1 + I2 + 2 * sqrt(I1 * I2) * cos(delta)
    - Num. Vars: 3
    - Vars:
        - x[0]: I1 (float, positive)
        - x[1]: I2 (float, positive)
        - x[2]: delta (float)
    - Constraints:
        - x[0]*x[1] >= 0

.. code-block:: text

    FeynmanICh37Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh37Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh38Eq12
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh38Eq12(KnownEquation)

.. code-block:: text

    - Equation: I.38.12
    - Raw: 4 * pi * 8.854e-12 * (6.626e-34 / (2 * pi)) ** 2 / (m * q ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: q (float)
    - Constraints:
        - x[0] != 0
        - x[1] != 0

.. code-block:: text

    FeynmanICh38Eq12.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh38Eq12.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh39Eq10
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh39Eq10(KnownEquation)

.. code-block:: text

    - Equation: I.39.10
    - Raw: 3 / 2 * pr * V
    - Num. Vars: 2
    - Vars:
        - x[0]: pr (float, positive)
        - x[1]: V (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh39Eq10.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh39Eq10.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh39Eq11
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh39Eq11(KnownEquation)

.. code-block:: text

    - Equation: I.39.11
    - Raw: 1 / (gamma - 1) * pr * V
    - Num. Vars: 3
    - Vars:
        - x[0]: gamma (float, positive)
        - x[1]: pr (float, positive)
        - x[2]: V (float, positive)
    - Constraints:
        - x[0] - 1 != 0

.. code-block:: text

    FeynmanICh39Eq11.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh39Eq11.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh39Eq22
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh39Eq22(KnownEquation)

.. code-block:: text

    - Equation: I.39.22
    - Raw: n * 1.380649e-23 * T / V
    - Num. Vars: 3
    - Vars:
        - x[0]: n (integer, positive)
        - x[1]: T (float, positive)
        - x[2]: V (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanICh39Eq22.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh39Eq22.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh40Eq1
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh40Eq1(KnownEquation)

.. code-block:: text

    - Equation: I.40.1
    - Raw: n_0 * exp(-m * 9.80665 * x / (1.380649e-23 * T))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_0 (float, positive)
        - x[1]: m (float, positive)
        - x[2]: x (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanICh40Eq1.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh40Eq1.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh41Eq16
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh41Eq16(KnownEquation)

.. code-block:: text

    - Equation: I.41.16
    - Raw: 6.626e-34 / (2 * pi) * omega ** 3 / (pi ** 2 * 2.99792458e8 ** 2 * (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1))
    - Num. Vars: 2
    - Vars:
        - x[0]: omega (float, positive)
        - x[1]: T (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanICh41Eq16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh41Eq16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh43Eq16
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh43Eq16(KnownEquation)

.. code-block:: text

    - Equation: I.43.16
    - Raw: mu_drift * q * Volt / d
    - Num. Vars: 4
    - Vars:
        - x[0]: mu_drift (float)
        - x[1]: q (float)
        - x[2]: Volt (float)
        - x[3]: d (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanICh43Eq16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh43Eq16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh43Eq31
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh43Eq31(KnownEquation)

.. code-block:: text

    - Equation: I.43.31
    - Raw: mob * 1.380649e-23 * T
    - Num. Vars: 2
    - Vars:
        - x[0]: mob (float, positive)
        - x[1]: T (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanICh43Eq31.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh43Eq31.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh43Eq43
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh43Eq43(KnownEquation)

.. code-block:: text

    - Equation: I.43.43
    - Raw: 1 / (gamma - 1) * 1.380649e-23 * v / A
    - Num. Vars: 3
    - Vars:
        - x[0]: gamma (float, positive)
        - x[1]: v (float, positive)
        - x[2]: A (float, positive)
    - Constraints:
        - x[0] - 1 != 0
        - x[2] != 0

.. code-block:: text

    FeynmanICh43Eq43.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh43Eq43.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh44Eq4
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh44Eq4(KnownEquation)

.. code-block:: text

    - Equation: I.44.4
    - Raw: n * 1.380649e-23 * T * ln(V2 / V1)
    - Num. Vars: 4
    - Vars:
        - x[0]: n (integer -> real due to its order, positive)
        - x[1]: T (float, positive)
        - x[2]: V2 (float, positive)
        - x[3]: V1 (float, positive)
    - Constraints:
        - x[3] != 0
        - x[2] / x[3] > 0

.. code-block:: text

    FeynmanICh44Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh44Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh47Eq23
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh47Eq23(KnownEquation)

.. code-block:: text

    - Equation: I.47.23
    - Raw: sqrt(gamma * pr / rho)
    - Num. Vars: 3
    - Vars:
        - x[0]: gamma (float, positive)
        - x[1]: pr (float, positive)
        - x[2]: rho (float, positive)
    - Constraints:
        - x[0] * x[1] / x[2] >= 0
        - x[2] != 0

.. code-block:: text

    FeynmanICh47Eq23.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh47Eq23.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh48Eq2
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh48Eq2(KnownEquation)

.. code-block:: text

    - Equation: I.48.2
    - Raw: m * 2.99792458e8 ** 2 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: v (float, positive)
    - Constraints:
        - 1 - x[1] ** 2 / 2.99792458e8 ** 2 > 0

.. code-block:: text

    FeynmanICh48Eq2.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh48Eq2.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanICh50Eq26
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanICh50Eq26(KnownEquation)

.. code-block:: text

    - Equation: I.50.26
    - Raw: x1 * (cos(omega * t) + alpha * cos(omega * t) ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: x1 (float, positive)
        - x[1]: omega (float)
        - x[2]: t (float, positive)
        - x[3]: alpha (float)
    - Constraints:

.. code-block:: text

    FeynmanICh50Eq26.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanICh50Eq26.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh2Eq42
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh2Eq42(KnownEquation)

.. code-block:: text

    - Equation: II.2.42
    - Raw: kappa * (T2 - T1) * A / d
    - Num. Vars: 5
    - Vars:
        - x[0]: kappa (float, positive)
        - x[1]: T2 (float, positive)
        - x[2]: T1 (float, positive)
        - x[3]: A (float, positive)
        - x[4]: d (float, positive)
    - Constraints:
        - x[4] != 0

.. code-block:: text

    FeynmanIICh2Eq42.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh2Eq42.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh3Eq24
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh3Eq24(KnownEquation)

.. code-block:: text

    - Equation: II.3.24
    - Raw: Pwr / (4 * pi * r ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: Pwr (float)
        - x[1]: r (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanIICh3Eq24.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh3Eq24.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh4Eq23
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh4Eq23(KnownEquation)

.. code-block:: text

    - Equation: II.4.23
    - Raw: q / (4 * pi * 8.854e-12 * r)
    - Num. Vars: 2
    - Vars:
        - x[0]: q (float)
        - x[1]: r (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanIICh4Eq23.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh4Eq23.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh6Eq11
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh6Eq11(KnownEquation)

.. code-block:: text

    - Equation: II.6.11
    - Raw: 1 / (4 * pi * 8.854e-12) * p_d * cos(theta) / r ** 2
    - Num. Vars: 3
    - Vars:
        - x[0]: p_d (float)
        - x[1]: theta (float)
        - x[2]: r (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIICh6Eq11.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh6Eq11.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh6Eq15a
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh6Eq15a(KnownEquation)

.. code-block:: text

    - Equation: II.6.15a
    - Raw: p_d / (4 * pi * 8.854e-12) * 3 * z / r ** 5 * sqrt(x ** 2 + y ** 2)
    - Num. Vars: 5
    - Vars:
        - x[0]: p_d (float)
        - x[1]: z (float, positive)
        - x[2]: r (float, positive)
        - x[3]: x (float, positive)
        - x[4]: y (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIICh6Eq15a.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh6Eq15a.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh6Eq15b
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh6Eq15b(KnownEquation)

.. code-block:: text

    - Equation: II.6.15b
    - Raw: p_d / (4 * pi * 8.854e-12) * 3 * cos(theta) * sin(theta) / r ** 3
    - Num. Vars: 3
    - Vars:
        - x[0]: p_d (float)
        - x[1]: theta (float)
        - x[2]: r (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIICh6Eq15b.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh6Eq15b.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh8Eq7
~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh8Eq7(KnownEquation)

.. code-block:: text

    - Equation: II.8.7
    - Raw: 3 / 5 * q ** 2 / (4 * pi * 8.854e-12 * d)
    - Num. Vars: 2
    - Vars:
        - x[0]: q (float)
        - x[1]: d (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanIICh8Eq7.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh8Eq7.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh8Eq31
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh8Eq31(KnownEquation)

.. code-block:: text

    - Equation: II.8.31
    - Raw: 8.854e-12 * Ef ** 2 / 2
    - Num. Vars: 1
    - Vars:
        - x[0]: Ef (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh8Eq31.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh8Eq31.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh10Eq9
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh10Eq9(KnownEquation)

.. code-block:: text

    - Equation: II.10.9
    - Raw: sigma_den / 8.854e-12 * 1 / (1 + chi)
    - Num. Vars: 2
    - Vars:
        - x[0]: sigma_den (float)
        - x[1]: chi (float, positive)
    - Constraints:
        - 1 + x[1] != 0

.. code-block:: text

    FeynmanIICh10Eq9.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh10Eq9.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh11Eq3
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh11Eq3(KnownEquation)

.. code-block:: text

    - Equation: II.11.3
    - Raw: q * Ef / (m * (omega_0 ** 2 - omega ** 2))
    - Num. Vars: 5
    - Vars:
        - x[0]: q (float)
        - x[1]: Ef (float, positive)
        - x[2]: m (float, positive)
        - x[3]: omega_0 (float)
        - x[4]: omega (float)
    - Constraints:
        - x[2] != 0
        - x[3] ** 2 - x[4] ** 2 != 0

.. code-block:: text

    FeynmanIICh11Eq3.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh11Eq3.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh11Eq17
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh11Eq17(KnownEquation)

.. code-block:: text

    - Equation: II.11.17
    - Raw: n_0 * (1 + p_d * Ef * cos(theta) / (1.380649e-23 * T))
    - Num. Vars: 5
    - Vars:
        - x[0]: n_0 (float, positive)
        - x[1]: p_d (float)
        - x[2]: Ef (float)
        - x[3]: theta (float)
        - x[4]: T (float, positive)
    - Constraints:
        - x[4] != 0

.. code-block:: text

    FeynmanIICh11Eq17.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh11Eq17.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh11Eq20
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh11Eq20(KnownEquation)

.. code-block:: text

    - Equation: II.11.20
    - Raw: n_rho * p_d ** 2 * Ef / (3 * 1.380649e-23 * T)
    - Num. Vars: 4
    - Vars:
        - x[0]: n_rho (integer -> real due to its order, positive)
        - x[1]: p_d (float)
        - x[2]: Ef (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIICh11Eq20.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh11Eq20.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh11Eq27
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh11Eq27(KnownEquation)

.. code-block:: text

    - Equation: II.11.27
    - Raw: n * alpha / (1 - (n * alpha / 3)) * 8.854e-12 * Ef
    - Num. Vars: 3
    - Vars:
        - x[0]: n (integer -> real due to its order, positive)
        - x[1]: alpha (float, positive)
        - x[2]: Ef (float, positive)
    - Constraints:
        - 1 - (x[0] * x[1] / 3) != 0

.. code-block:: text

    FeynmanIICh11Eq27.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh11Eq27.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh11Eq28
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh11Eq28(KnownEquation)

.. code-block:: text

    - Equation: II.11.28
    - Raw: 1 + n * alpha / (1 - (n * alpha / 3))
    - Num. Vars: 2
    - Vars:
        - x[0]: n (integer -> real due to its order, positive)
        - x[1]: alpha (float, positive)
    - Constraints:
        - 1-(x[0]*x[1]/3) != 0

.. code-block:: text

    FeynmanIICh11Eq28.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh11Eq28.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh13Eq17
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh13Eq17(KnownEquation)

.. code-block:: text

    - Equation: II.13.17
    - Raw: 1 / (4 * pi * 8.854e-12 * 2.99792458e8 ** 2) * 2 * I / r
    - Num. Vars: 2
    - Vars:
        - x[0]: I (float)
        - x[1]: r (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanIICh13Eq17.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh13Eq17.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh13Eq23
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh13Eq23(KnownEquation)

.. code-block:: text

    - Equation: II.13.23
    - Raw: rho_c_0 / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: rho_c_0 (float, positive)
        - x[1]: v (float, positive)
    - Constraints:
        - 2.99792458e8 ** 2 - x[1] ** 2 > 0

.. code-block:: text

    FeynmanIICh13Eq23.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh13Eq23.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh13Eq34
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh13Eq34(KnownEquation)

.. code-block:: text

    - Equation: II.13.34
    - Raw: rho_c_0 * v / sqrt(1 - v ** 2 / 2.99792458e8 ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: rho_c_0 (float, positive)
        - x[1]: v (float, positive)
    - Constraints:
        - 2.99792458e8 ** 2 - x[1] ** 2 > 0

.. code-block:: text

    FeynmanIICh13Eq34.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh13Eq34.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh15Eq4
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh15Eq4(KnownEquation)

.. code-block:: text

    - Equation: II.15.4
    - Raw: -mom * B * cos(theta)
    - Num. Vars: 3
    - Vars:
        - x[0]: mom (float)
        - x[1]: B (float)
        - x[2]: theta (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh15Eq4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh15Eq4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh15Eq5
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh15Eq5(KnownEquation)

.. code-block:: text

    - Equation: II.15.5
    - Raw: -p_d * Ef * cos(theta)
    - Num. Vars: 3
    - Vars:
        - x[0]: p_d (float)
        - x[1]: Ef (float)
        - x[2]: theta (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh15Eq5.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh15Eq5.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh21Eq32
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh21Eq32(KnownEquation)

.. code-block:: text

    - Equation: II.21.32
    - Raw: q / (4 * pi * 8.854e-12 * r * (1 - v / 2.99792458e8))
    - Num. Vars: 3
    - Vars:
        - x[0]: q (float)
        - x[1]: r (float, positive)
        - x[2]: v (float, positive)
    - Constraints:
        - x[1] != 0
        - 2.99792458e8 - x[2] > 0

.. code-block:: text

    FeynmanIICh21Eq32.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh21Eq32.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh24Eq17
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh24Eq17(KnownEquation)

.. code-block:: text

    - Equation: II.24.17
    - Raw: sqrt(omega ** 2 / 2.99792458e8 ** 2 - pi ** 2 / d ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: omega (float)
        - x[1]: d (float, positive)
    - Constraints:
        - x[0] ** 2 / 2.99792458e8 ** 2 - np.pi ** 2 / x[1] ** 2 >= 0
        - x[1] != 0

.. code-block:: text

    FeynmanIICh24Eq17.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh24Eq17.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh27Eq16
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh27Eq16(KnownEquation)

.. code-block:: text

    - Equation: II.27.16
    - Raw: 8.854e-12 * 2.99792458e8 * Ef ** 2
    - Num. Vars: 1
    - Vars:
        - x[0]: Ef (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh27Eq16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh27Eq16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh27Eq18
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh27Eq18(KnownEquation)

.. code-block:: text

    - Equation: II.27.18
    - Raw: 8.854e-12 * Ef ** 2
    - Num. Vars: 1
    - Vars:
        - x[0]: Ef (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh27Eq18.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh27Eq18.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh34Eq2a
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh34Eq2a(KnownEquation)

.. code-block:: text

    - Equation: II.34.2a
    - Raw: q * v / (2 * pi * r)
    - Num. Vars: 3
    - Vars:
        - x[0]: q (float)
        - x[1]: v (float)
        - x[2]: r (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIICh34Eq2a.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh34Eq2a.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh34Eq2
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh34Eq2(KnownEquation)

.. code-block:: text

    - Equation: II.34.2
    - Raw: q * v * r / 2
    - Num. Vars: 3
    - Vars:
        - x[0]: q (float)
        - x[1]: v (float)
        - x[2]: r (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIICh34Eq2.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh34Eq2.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh34Eq11
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh34Eq11(KnownEquation)

.. code-block:: text

    - Equation: II.34.11
    - Raw: g_ * q * B / (2 * m)
    - Num. Vars: 4
    - Vars:
        - x[0]: g_ (float)
        - x[1]: q (float)
        - x[2]: B (float)
        - x[3]: m (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIICh34Eq11.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh34Eq11.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh34Eq29a
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh34Eq29a(KnownEquation)

.. code-block:: text

    - Equation: II.34.29a
    - Raw: q * 6.626e-34 / (4 * pi * m)
    - Num. Vars: 2
    - Vars:
        - x[0]: q (float)
        - x[1]: m (float, positive)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanIICh34Eq29a.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh34Eq29a.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh34Eq29b
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh34Eq29b(KnownEquation)

.. code-block:: text

    - Equation: II.34.29b
    - Raw: g_ * 9.2740100783e-24 * B * Jz / (6.626e-34 / (2 * pi))
    - Num. Vars: 3
    - Vars:
        - x[0]: g_ (float)
        - x[1]: B (float)
        - x[2]: Jz (float)
    - Constraints:

.. code-block:: text

    FeynmanIICh34Eq29b.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh34Eq29b.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh35Eq18
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh35Eq18(KnownEquation)

.. code-block:: text

    - Equation: II.35.18
    - Raw: n_0 / (exp(mom * B / (1.380649e-23 * T)) + exp(-mom * B / (1.380649e-23 * T)))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_0 (integer -> real due to its order, positive)
        - x[1]: mom (float, positive)
        - x[2]: B (float, positive)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIICh35Eq18.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh35Eq18.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh35Eq21
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh35Eq21(KnownEquation)

.. code-block:: text

    - Equation: II.35.21
    - Raw: n_rho * mom * tanh(mom * B / (1.380649e-23 * T))
    - Num. Vars: 4
    - Vars:
        - x[0]: n_rho (integer -> real due to its order, positive)
        - x[1]: mom (float, positive)
        - x[2]: B (float, positive)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIICh35Eq21.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh35Eq21.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh36Eq38
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh36Eq38(KnownEquation)

.. code-block:: text

    - Equation: II.36.38
    - Raw: mom * H / (1.380649e-23 * T) + (mom * alpha) / (8.854e-12 * 2.99792458e8 ** 2 * 1.380649e-23 * T) * M
    - Num. Vars: 5
    - Vars:
        - x[0]: mom (float)
        - x[1]: H (float)
        - x[2]: T (float, positive)
        - x[3]: alpha (float, positive)
        - x[4]: M (integer -> real due to its order, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIICh36Eq38.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh36Eq38.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh37Eq1
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh37Eq1(KnownEquation)

.. code-block:: text

    - Equation: II.37.1
    - Raw: mom * (1 + chi) * B
    - Num. Vars: 3
    - Vars:
        - x[0]: mom (float)
        - x[1]: chi (float)
        - x[2]: B (float)
    - Constraints:

.. code-block:: text

    FeynmanIICh37Eq1.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh37Eq1.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh38Eq3
~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh38Eq3(KnownEquation)

.. code-block:: text

    - Equation: II.38.3
    - Raw: Y * A * x / d
    - Num. Vars: 4
    - Vars:
        - x[0]: Y (float, positive)
        - x[1]: A (float, positive)
        - x[2]: x (float)
        - x[3]: d (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIICh38Eq3.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh38Eq3.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIICh38Eq14
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIICh38Eq14(KnownEquation)

.. code-block:: text

    - Equation: II.38.14
    - Raw: Y / (2 * (1 + sigma))
    - Num. Vars: 2
    - Vars:
        - x[0]: Y (float, positive)
        - x[1]: sigma (float)
    - Constraints:
        - 1 + x[1] != 0

.. code-block:: text

    FeynmanIICh38Eq14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIICh38Eq14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh4Eq32
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh4Eq32(KnownEquation)

.. code-block:: text

    - Equation: III.4.32
    - Raw: 1 / (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1)
    - Num. Vars: 2
    - Vars:
        - x[0]: omega (float, positive)
        - x[1]: T (float, positive)
    - Constraints:
        - x[0] != 0
        - x[1] != 0

.. code-block:: text

    FeynmanIIICh4Eq32.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh4Eq32.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh4Eq33
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh4Eq33(KnownEquation)

.. code-block:: text

    - Equation: III.4.33
    - Raw: (6.626e-34 / (2 * pi)) * omega / (exp((6.626e-34 / (2 * pi)) * omega / (1.380649e-23 * T)) - 1)
    - Num. Vars: 2
    - Vars:
        - x[0]: omega (float, positive)
        - x[1]: T (float, positive)
    - Constraints:
        - x[0] != 0
        - x[1] != 0

.. code-block:: text

    FeynmanIIICh4Eq33.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh4Eq33.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh7Eq38
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh7Eq38(KnownEquation)

.. code-block:: text

    - Equation: III.7.38
    - Raw: 2 * mom * B / (6.626e-34 / (2 * pi))
    - Num. Vars: 2
    - Vars:
        - x[0]: mom (float)
        - x[1]: B (float)
    - Constraints:

.. code-block:: text

    FeynmanIIICh7Eq38.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh7Eq38.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh8Eq54
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh8Eq54(KnownEquation)

.. code-block:: text

    - Equation: III.8.54
    - Raw: sin(E_n * t / (6.626e-34 / (2 * pi))) ** 2
    - Num. Vars: 2
    - Vars:
        - x[0]: E_n (float)
        - x[1]: t (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIIICh8Eq54.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh8Eq54.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh9Eq52
~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh9Eq52(KnownEquation)

.. code-block:: text

    - Equation: III.9.52
    - Raw: (p_d * Ef * t / (6.626e-34 / (2 * pi))) ** 2 * sin((omega - omega_0) * t / 2) ** 2 / ((omega - omega_0) * t / 2) ** 2
    - Num. Vars: 5
    - Vars:
        - x[0]: p_d (float)
        - x[1]: Ef (float)
        - x[2]: t (float, positive)
        - x[3]: omega (float, positive)
        - x[4]: omega_0 (float, positive)
    - Constraints:
        - x[2] != 0
        - x[3] - x[4] != 0

.. code-block:: text

    FeynmanIIICh9Eq52.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh9Eq52.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh10Eq19
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh10Eq19(KnownEquation)

.. code-block:: text

    - Equation: III.10.19
    - Raw: mom * sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    - Num. Vars: 4
    - Vars:
        - x[0]: mom (float)
        - x[1]: Bx (float)
        - x[2]: By (float)
        - x[3]: Bz (float)
    - Constraints:

.. code-block:: text

    FeynmanIIICh10Eq19.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh10Eq19.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh12Eq43
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh12Eq43(KnownEquation)

.. code-block:: text

    - Equation: III.12.43
    - Raw: n * (6.626e-34 / (2 * pi))
    - Num. Vars: 1
    - Vars:
        - x[0]: n (integer)
    - Constraints:

.. code-block:: text

    FeynmanIIICh12Eq43.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh12Eq43.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh13Eq18
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh13Eq18(KnownEquation)

.. code-block:: text

    - Equation: III.13.18
    - Raw: 2 * E_n * d ** 2 * k / (6.626e-34 / (2 * pi))
    - Num. Vars: 3
    - Vars:
        - x[0]: E_n (float)
        - x[1]: d (float, positive)
        - x[2]: k (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIIICh13Eq18.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh13Eq18.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh14Eq14
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh14Eq14(KnownEquation)

.. code-block:: text

    - Equation: III.14.14
    - Raw: I_0 * (exp(q * Volt / (1.380649e-23 * T)) - 1)
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float)
        - x[1]: q (float, positive)
        - x[2]: Volt (float)
        - x[3]: T (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIIICh14Eq14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh14Eq14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh15Eq12
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh15Eq12(KnownEquation)

.. code-block:: text

    - Equation: III.15.12
    - Raw: 2 * U * (1 - cos(k * d))
    - Num. Vars: 3
    - Vars:
        - x[0]: U (float, positive)
        - x[1]: k (float, positive)
        - x[2]: d (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIIICh15Eq12.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh15Eq12.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh15Eq14
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh15Eq14(KnownEquation)

.. code-block:: text

    - Equation: III.15.14
    - Raw: (6.626e-34 / (2 * pi)) ** 2 / (2 * E_n * d ** 2)
    - Num. Vars: 2
    - Vars:
        - x[0]: E_n (float, positive)
        - x[1]: d (float, positive)
    - Constraints:
        - x[0] != 0
        - x[1] != 0

.. code-block:: text

    FeynmanIIICh15Eq14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh15Eq14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh15Eq27
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh15Eq27(KnownEquation)

.. code-block:: text

    - Equation: III.15.27
    - Raw: 2 * pi * alpha / (n * d)
    - Num. Vars: 3
    - Vars:
        - x[0]: alpha (integer)
        - x[1]: n (integer, positive)
        - x[2]: d (float, positive)
    - Constraints:
        - x[1] != 0
        - x[2] != 0

.. code-block:: text

    FeynmanIIICh15Eq27.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh15Eq27.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh17Eq37
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh17Eq37(KnownEquation)

.. code-block:: text

    - Equation: III.17.37
    - Raw: beta * (1 + alpha * cos(theta))
    - Num. Vars: 3
    - Vars:
        - x[0]: beta (float, positive)
        - x[1]: alpha (float)
        - x[2]: theta (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanIIICh17Eq37.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh17Eq37.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh19Eq51
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh19Eq51(KnownEquation)

.. code-block:: text

    - Equation: III.19.51
    - Raw: -m * q ** 4 / (2 * (4 * pi * 8.854e-12) ** 2 * (6.626e-34 / (2 * pi)) ** 2) * (1 / n ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: q (float)
        - x[2]: n (integer, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanIIICh19Eq51.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh19Eq51.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanIIICh21Eq20
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanIIICh21Eq20(KnownEquation)

.. code-block:: text

    - Equation: III.21.20
    - Raw: -rho_c_0 * q * A_vec / m
    - Num. Vars: 4
    - Vars:
        - x[0]: rho_c_0 (float)
        - x[1]: q (float, positive)
        - x[2]: A_vec (float)
        - x[3]: m (float, positive)
    - Constraints:
        - x[3] != 0

.. code-block:: text

    FeynmanIIICh21Eq20.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanIIICh21Eq20.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus1
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus1(KnownEquation)

.. code-block:: text

    - Equation: Rutherford scattering
    - Raw: (Z_1 * Z_2 * alpha * 1.054571817e-34 * 2.99792458e8 / (4 * E_n * sin(theta / 2) ** 2)) ** 2
    - Num. Vars: 4
    - Vars:
        - x[0]: Z_1 (integer, positive)
        - x[1]: Z_2 (integer, positive)
        - x[2]: E_n (float, positive)
        - x[3]: theta (float, positive)
    - Constraints:
        - x[2] != 0
        - np.sin(x[3] / 2) != 0

.. code-block:: text

    FeynmanBonus1.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus1.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus2
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus2(KnownEquation)

.. code-block:: text

    - Equation: 3.55 Goldstein
    - Raw: m * k_G / L ** 2 * (1 + sqrt(1 + 2 * E_n * L ** 2 / (m * k_G ** 2)) * cos(theta1 - theta2))
    - Num. Vars: 6
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: k_G (float, positive)
        - x[2]: L (float, positive)
        - x[3]: E_n (float, positive)
        - x[4]: theta1 (float, positive)
        - x[5]: theta2 (float, positive)
    - Constraints:
        - x[0] != 0
        - x[1] != 0
        - x[3] * x[2] ** 2 / (x[0] * x[1] ** 2) >= -1 / 2

.. code-block:: text

    FeynmanBonus2.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus2.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus3
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus3(KnownEquation)

.. code-block:: text

    - Equation: 3.64 Goldstein
    - Raw: d * (1 - alpha ** 2) / (1 + alpha * cos(theta1 - theta2))
    - Num. Vars: 4
    - Vars:
        - x[0]: d (float, positive)
        - x[1]: alpha (float, positive)
        - x[2]: theta1 (float, positive)
        - x[3]: theta2 (float, positive)
    - Constraints:
        - x[1] * np.cos(x[2] - x[3]) != -1

.. code-block:: text

    FeynmanBonus3.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus3.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus4
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus4(KnownEquation)

.. code-block:: text

    - Equation: 3.16 Goldstein
    - Raw: sqrt(2 / m * (E_n - U - L ** 2 / (2 * m * r ** 2)))
    - Num. Vars: 5
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: E_n (float, positive)
        - x[2]: U (float, positive)
        - x[3]: L (float, positive)
        - x[4]: r (float, positive)
    - Constraints:
        - 2 / x[0] * (x[1] - x[2] - x[3] ** 2 / (2 * x[0] * x[4] ** 2)) >= 0
        - x[0] != 0
        - x[4] != 0

.. code-block:: text

    FeynmanBonus4.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus4.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus5
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus5(KnownEquation)

.. code-block:: text

    - Equation: 3.74 Goldstein
    - Raw: 2 * pi * d ** (3 / 2) / sqrt(6.67430e-11 * (m1 + m2))
    - Num. Vars: 3
    - Vars:
        - x[0]: d (float, positive)
        - x[2]: m1 (float, positive)
        - x[3]: m2 (float, positive)
    - Constraints:
        - x[1] != 0
        - x[2] + x[3] != 0
        - x[1] * (x[2] + x[3]) > 0

.. code-block:: text

    FeynmanBonus5.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus5.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus6
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus6(KnownEquation)

.. code-block:: text

    - Equation: 3.99 Goldstein
    - Raw: sqrt(1 + 2 * epsilon ** 2 * E_n * L ** 2 / (m * (Z_1 * Z_2 * q ** 2) ** 2))
    - Python:
    - Num. Vars: 7
    - Vars:
        - x[0]: epsilon (float)
        - x[1]: E_n (float, positive)
        - x[2]: L (float, positive)
        - x[3]: m (float, positive)
        - x[4]: Z_1 (integer, positive)
        - x[5]: Z_2 (integer, positive)
        - x[6]: q (float)
    - Constraints:
        - x[3] != 0
        - x[4] != 0
        - x[5] != 0
        - x[6] != 0
        - 1 + 2 * x[0] * x[1] * x[2] ** 2 / (x[3] * (x[4] * x[5] * x[6] ** 2) ** 2) >= 0

.. code-block:: text

    FeynmanBonus6.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus6.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus7
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus7(KnownEquation)

.. code-block:: text

    - Equation: Friedman Equation
    - Raw: sqrt(8 * pi * 6.67430e-11 * rho / 3 - alpha * 2.99792458e8 ** 2 / d ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: rho (float, positive)
        - x[1]: alpha (Integer)
        - x[2]: d (float, positive)
    - Constraints:
        - x[2] != 0
        - 6.67430e-11 * x[0] * x[2] ** 2 / 3 >= x[1] * 2.99792458e8 ** 2

.. code-block:: text

    FeynmanBonus7.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus7.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus8
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus8(KnownEquation)

.. code-block:: text

    - Equation: Compton Scattering
    - Raw: E_n / (1 + E_n / (9.10938356e-31 * 2.99792458e8 ** 2) * (1 - cos(theta)))
    - Num. Vars: 2
    - Vars:
        - x[0]: E_n (float, positive)
        - x[1]: theta (float)
    - Constraints:
        - x[0] * (1 - np.cos(x[1])) / (9.10938356e-31 * 2.99792458e8 ** 2) != -1

.. code-block:: text

    FeynmanBonus8.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus8.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus9
~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus9(KnownEquation)

.. code-block:: text

    - Equation: Gravitational wave ratiated power
    - Raw: -32/5 * 6.67430e-11 ** 4 / 2.99792458e8 ** 5 * (m1 * m2) ** 2 * (m1 + m2) / r ** 5
    - Num. Vars: 3
    - Vars:
        - x[0]: m1 (float, positive)
        - x[1]: m2 (float, positive)
        - x[2]: r (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanBonus9.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus9.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus10
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus10(KnownEquation)

.. code-block:: text

    - Equation: Relativistic aberration
    - Raw: (cos(theta2) - v / 2.99792458e8) / (1 - v / 2.99792458e8 * cos(theta2))
    - Num. Vars: 2
    - Vars:
        - x[0]: theta2 (float, positive)
        - x[1]: v (float, positive)
    - Constraints:
        - x[1] / 2.99792458e8 * np.cos(x[0]) != 1
        - (np.cos(x[0]) - x[1] / 2.99792458e8) / (1 - x[1] / 2.99792458e8 * np.cos(x[0])) >= 1
        - (np.cos(x[0]) - x[1] / 2.99792458e8) / (1 - x[1] / 2.99792458e8 * np.cos(x[0])) <= 1

.. code-block:: text

    FeynmanBonus10.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus10.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus11
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus11(KnownEquation)

.. code-block:: text

    - Equation: N-slit diffraction
    - Raw: I_0 * (sin(alpha / 2) * sin(n * delta / 2) / (alpha / 2 * sin(delta / 2))) ** 2
    - Num. Vars: 4
    - Vars:
        - x[0]: I_0 (float, positive)
        - x[1]: alpha (float, positive)
        - x[2]: n (integer, positive)
        - x[3]: delta (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanBonus11.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus11.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus12
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus12(KnownEquation)

.. code-block:: text

    - Equation: 2.11 Jackson
    - Raw: q / (4 * pi * epsilon * y ** 2) * (4 * pi * epsilon * Volt * d - q * d * y ** 3 / (y ** 2 - d ** 2) ** 2)
    - Num. Vars: 5
    - Vars:
        - x[0]: q (float)
        - x[1]: epsilon (float, positive)
        - x[2]: y (float, positive)
        - x[3]: Volt (float)
        - x[4]: d (float, positive)
    - Constraints:
        - x[2] != 0

.. code-block:: text

    FeynmanBonus12.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus12.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus13
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus13(KnownEquation)

.. code-block:: text

    - Equation: 3.45 Jackson
    - Raw: 1 / (4 * pi * epsilon) * q / sqrt(r ** 2 + d ** 2 - 2 * r * d * cos(alpha))
    - Num. Vars: 5
    - Vars:
        - x[0]: epsilon (float, positive)
        - x[1]: q (float)
        - x[2]: r (float, positive)
        - x[3]: d (float, positive)
        - x[4]: alpha (float, positive)
    - Constraints:
        - x[2] ** 2 + x[3] ** 2 - 2 * x[2] * x[3] * np.cos(x[4]) > 0

.. code-block:: text

    FeynmanBonus13.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus13.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus14
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus14(KnownEquation)

.. code-block:: text

    - Equation: 4.60' Jackson
    - Raw: Ef * cos(theta) * (-r + d ** 3 / r ** 2 * (alpha - 1) / (alpha + 2))
    - Num. Vars: 5
    - Vars:
        - x[0]: Ef (float)
        - x[1]: theta (float, positive)
        - x[2]: r (float, positive)
        - x[3]: d (float, positive)
        - x[4]: alpha (float, positive)
    - Constraints:
        - x[2] != 0
        - x[4] != -2

.. code-block:: text

    FeynmanBonus14.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus14.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus15
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus15(KnownEquation)

.. code-block:: text

    - Equation: 11.38 Jackson
    - Raw: sqrt(1 - v ** 2 / 2.99792458e8 ** 2) * omega / (1 + v / 2.99792458e8 * cos(theta))
    - Num. Vars: 3
    - Vars:
        - x[0]: v (float, positive)
        - x[1]: omega (float, positive)
        - x[2]: theta (float, positive)
    - Constraints:
        - x[0] / 2.99792458e8 * np.cos(x[2]) != -1
        - 2.99792458e8 ** 2 - x[0] ** 2 >= 0

.. code-block:: text

    FeynmanBonus15.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus15.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus16
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus16(KnownEquation)

.. code-block:: text

    - Equation: 8.56 Goldstein
    - Raw: sqrt((p - q * A_vec) ** 2 * 2.99792458e8 ** 2 + m ** 2 * 2.99792458e8 ** 4) + q * Volt
    - Num. Vars: 5
    - Vars:
        - x[0]: p (float)
        - x[1]: q (float)
        - x[2]: A_vec (float)
        - x[3]: m (float, positive)
        - x[4]: Volt (float)
    - Constraints:

.. code-block:: text

    FeynmanBonus16.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus16.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus17
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus17(KnownEquation)

.. code-block:: text

    - Equation: 12.80' Goldstein
    - Raw: 1 / (2 * m) * (p ** 2 + m ** 2 * omega ** 2 * x ** 2 * (1 + alpha * x / y))
    - Num. Vars: 6
    - Vars:
        - x[0]: m (float, positive)
        - x[1]: p (float)
        - x[2]: omega (float, positive)
        - x[3]: x (float)
        - x[4]: alpha (float)
        - x[5]: y (float, positive)
    - Constraints:
        - x[0] != 0
        - x[5] != 0

.. code-block:: text

    FeynmanBonus17.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus17.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus18
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus18(KnownEquation)

.. code-block:: text

    - Equation: 15.2.1 Weinberg
    - Raw: 3 / (8 * pi * 6.67430e-11) * (2.99792458e8 ** 2 * k_f / r ** 2 + H_G ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: k_f (float)
        - x[1]: r (float, positive)
        - x[2]: H_G (float)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanBonus18.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus18.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus19
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus19(KnownEquation)

.. code-block:: text

    - Equation: 15.2.2 Weinberg
    - Raw: -1 / (8 * pi * 6.67430e-11) * (2.99792458e8 ** 4 * k_f / r ** 2 + H_G ** 2 * 2.99792458e8 ** 2 * (1 - 2 * alpha))
    - Num. Vars: 4
    - Vars:
        - x[0]: k_f (float)
        - x[1]: r (float, positive)
        - x[2]: H_G (float)
        - x[3]: alpha (float)
    - Constraints:
        - x[1] != 0

.. code-block:: text

    FeynmanBonus19.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus19.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

FeynmanBonus20
~~~~~~~~~~~~~~

.. code-block:: text

    class FeynmanBonus20(KnownEquation)

.. code-block:: text

    - Equation: Klein-Nishina (13.132 Schwarz)
    - Raw: 1 / (4 * pi) * 7.2973525693e-3 ** 2 * 6.626e-34 ** 2 / (9.10938356e-31 ** 2 * 2.99792458e8 ** 2) * (omega_0 / omega) ** 2 * (omega_0 / omega + omega / omega_0 - sin(beta) ** 2)
    - Num. Vars: 3
    - Vars:
        - x[0]: omega_0 (float, positive)
        - x[1]: omega (float, positive)
        - x[2]: beta (float, positive)
    - Constraints:

.. code-block:: text

    FeynmanBonus20.__init__(self, sampling_objs=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    FeynmanBonus20.eq_func(self, x)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.feynman_srsd.registry
---------------------------------------------------

register_sampling_func
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    register_sampling_func(func)

register_sampling_class
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    register_sampling_class(cls)

register_eq_class
~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    register_eq_class(cls)

get_sampling_obj
~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    get_sampling_obj(key, **kwargs)

get_eq_obj
~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    get_eq_obj(key, **kwargs)

llm4ad.task.science_discovery.feynman_srsd.sampling
---------------------------------------------------

default_sampling
~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    default_sampling(sample_size, min_value=0.1, max_value=10.0)

default_positive_sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    default_positive_sampling(sample_size, min_value=0.1, max_value=10.0)

default_negative_sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    default_negative_sampling(sample_size, min_value=0.1, max_value=10.0)

simple_sampling
~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    simple_sampling(sample_size, min_value=0.0, max_value=1.0)

simple_positive_sampling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    simple_positive_sampling(sample_size, min_value=0.0, max_value=1.0)

simple_negative_sampling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    simple_negative_sampling(sample_size, min_value=0.0, max_value=1.0)

integer_sampling
~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    integer_sampling(sample_size, min_value=1, max_value=100)

integer_positive_sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    integer_positive_sampling(sample_size, min_value=1, max_value=100)

integer_negative_sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    integer_negative_sampling(sample_size, min_value=1, max_value=100)

DefaultSampling
~~~~~~~~~~~~~~~

.. code-block:: text

    class DefaultSampling(object)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    DefaultSampling.__init__(self, min_value, max_value, uses_positive=True, uses_negative=True)

.. code-block:: text

    Documentation gap: no method docstring is present.

SimpleSampling
~~~~~~~~~~~~~~

.. code-block:: text

    class SimpleSampling(object)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    SimpleSampling.__init__(self, min_value, max_value, uses_positive=True, uses_negative=True)

.. code-block:: text

    Documentation gap: no method docstring is present.

IntegerSampling
~~~~~~~~~~~~~~~

.. code-block:: text

    class IntegerSampling(object)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    IntegerSampling.__init__(self, min_value, max_value, uses_positive=True, uses_negative=True)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.ode_1d.evaluation
-----------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(program_str: str, data: dict, equation: callable) -> float | None

ODEEvaluation
~~~~~~~~~~~~~

.. code-block:: text

    class ODEEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ODEEvaluation.__init__(self, timeout_seconds=60, test_id=1, **kwargs)

.. code-block:: text

    Args:
        timeout_seconds: evaluate time limit.
        test_id: test equation id ranges from [1, 16].

.. code-block:: text

    ODEEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.oscillator1.evaluation
----------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(data: dict, equation: callable) -> float | None

OscillatorEvaluation1
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class OscillatorEvaluation1(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    OscillatorEvaluation1.__init__(self, timeout_seconds=20, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OscillatorEvaluation1.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.oscillator2.evaluation
----------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(data: dict, equation: callable) -> float | None

OscillatorEvaluation2
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class OscillatorEvaluation2(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    OscillatorEvaluation2.__init__(self, timeout_seconds=20, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OscillatorEvaluation2.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.task.science_discovery.stresstrain.evaluation
----------------------------------------------------

evaluate
~~~~~~~~

.. code-block:: text

    Evaluate the equation on data observations.

.. code-block:: text

    evaluate(data: dict, equation: callable) -> float | None

SSEvaluation
~~~~~~~~~~~~

.. code-block:: text

    class SSEvaluation(Evaluation)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    SSEvaluation.__init__(self, timeout_seconds=20, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    SSEvaluation.evaluate_program(self, program_str: str, callable_func: callable) -> Any | None

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.checkpoint
-----------------------

latest_snapshot
~~~~~~~~~~~~~~~

.. code-block:: text

    Select only numbered snapshots; unrelated files cannot change ordering.

.. code-block:: text

    latest_snapshot(log_dir, directory='population', prefix='pop')

restore_population
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Restore saved individuals directly, without re-running survival/selection.

.. code-block:: text

    restore_population(population_class, log_dir, pop_size, *, init_pop_size=None, template_program=None)

restore_attempt_count
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Use persisted attempt counts; old checkpoints fall back to history order.

.. code-block:: text

    restore_attempt_count(log_dir, legacy_count, method_name)

load_samples
~~~~~~~~~~~~

.. code-block:: text

    Return sample dictionaries ordered by recorded sample order.

    Best-only logs are excluded. Malformed files raise instead of silently
    constructing a partial experiment history. Zero scores are preserved.

.. code-block:: text

    load_samples(log_dir)

llm4ad.tools.llm.__init__
-------------------------

import_all_llm_classes_from_subfolders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.

.. code-block:: text

    import_all_llm_classes_from_subfolders(root_directory)

llm4ad.tools.llm.llm_api_baiduqianfan
-------------------------------------

BaiduQianfanAPI
~~~~~~~~~~~~~~~

.. code-block:: text

    class BaiduQianfanAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    BaiduQianfanAPI.__init__(self, host='qianfan.baidubce.com', key=None, model=None, path='/v2/chat/completions', timeout=60, **kwargs)

.. code-block:: text

    Baidu Qianfan API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        path   : API path for chat completions.
        timeout: API timeout.

.. code-block:: text

    BaiduQianfanAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

QianfanAPI
~~~~~~~~~~

.. code-block:: text

    class QianfanAPI(BaiduQianfanAPI)

.. code-block:: text

    Alias for BaiduQianfanAPI.

llm4ad.tools.llm.llm_api_https
------------------------------

HttpsApi
~~~~~~~~

.. code-block:: text

    class HttpsApi(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    HttpsApi.__init__(self, host, key, model, timeout=60, **kwargs)

.. code-block:: text

    Https API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        timeout: API timeout.

.. code-block:: text

    HttpsApi.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

llm4ad.tools.llm.llm_api_openai
-------------------------------

OpenAIAPI
~~~~~~~~~

.. code-block:: text

    class OpenAIAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    OpenAIAPI.__init__(self, base_url: str, api_key: str, model: str, timeout=60, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OpenAIAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.llm.llm_api_qwen
-----------------------------

QwenAPI
~~~~~~~

.. code-block:: text

    class QwenAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    QwenAPI.__init__(self, host='dashscope.aliyuncs.com', key=None, model=None, path='/compatible-mode/v1/chat/completions', timeout=60, **kwargs)

.. code-block:: text

    Qwen API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        path   : API path for chat completions.
        timeout: API timeout.

.. code-block:: text

    QwenAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

DashScopeAPI
~~~~~~~~~~~~

.. code-block:: text

    class DashScopeAPI(QwenAPI)

.. code-block:: text

    Alias for QwenAPI.

BailianAPI
~~~~~~~~~~

.. code-block:: text

    class BailianAPI(QwenAPI)

.. code-block:: text

    Alias for QwenAPI.

llm4ad.tools.llm.llm_api_tencentcloud
-------------------------------------

TencentCloudAPI
~~~~~~~~~~~~~~~

.. code-block:: text

    class TencentCloudAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    TencentCloudAPI.__init__(self, host='tokenhub.tencentmaas.com', key=None, model=None, path='/v1/chat/completions', timeout=60, **kwargs)

.. code-block:: text

    Tencent Cloud API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        path   : API path for chat completions.
        timeout: API timeout.

.. code-block:: text

    TencentCloudAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

llm4ad.tools.llm.llm_api_volcengine
-----------------------------------

VolcengineAPI
~~~~~~~~~~~~~

.. code-block:: text

    class VolcengineAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    VolcengineAPI.__init__(self, host='ark.cn-beijing.volces.com', key=None, model=None, path='/api/v3/chat/completions', timeout=60, **kwargs)

.. code-block:: text

    Volcengine API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        path   : API path for chat completions.
        timeout: API timeout.

.. code-block:: text

    VolcengineAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

DoubaoAPI
~~~~~~~~~

.. code-block:: text

    class DoubaoAPI(VolcengineAPI)

.. code-block:: text

    Alias for VolcengineAPI.

llm4ad.tools.llm.llm_api_zhipu
------------------------------

ZhipuAPI
~~~~~~~~

.. code-block:: text

    class ZhipuAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ZhipuAPI.__init__(self, host='open.bigmodel.cn', key=None, model=None, path='/api/paas/v4/chat/completions', timeout=60, **kwargs)

.. code-block:: text

    Zhipu AI API
    Args:
        host   : host name. please note that the host name does not include 'https://'
        key    : API key.
        model  : LLM model name.
        path   : API path for chat completions.
        timeout: API timeout.

.. code-block:: text

    ZhipuAPI.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Sends a request to the LLM and retrieves the generated response.

    This method supports multiple input formats for backward compatibility:
    1. Explicit 'messages' list via kwargs.
    2. A message list passed directly as the 'prompt'.
    3. Multimodal inputs (text + base64 images).
    4. Simple string prompts.

    Args:
        prompt: The text prompt or a list of message dictionaries.
        **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

    Returns:
        The string content of the LLM's response.

llm4ad.tools.llm.local_ollama
-----------------------------

LocalOllamaLLM
~~~~~~~~~~~~~~

.. code-block:: text

    class LocalOllamaLLM(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LocalOllamaLLM.__init__(self, model_name: str, **ollama_llm_init_params)

.. code-block:: text

    Deploy Ollama model on local devices.
    Args:
        model_name            : name of local Ollama model checkpoint.
        ollama_llm_init_params: initialization params for `langchain_ollama.OllamaLLM`.

.. code-block:: text

    LocalOllamaLLM.draw_sample(self, prompt: str | Any, *args, **kwargs) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.llm.local_vllm.vllm_llama
--------------------------------------

completions
~~~~~~~~~~~

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    completions()

llm4ad.tools.llm.local_vllm.vllm_local_http_api
-----------------------------------------------

VLLMManager
~~~~~~~~~~~

.. code-block:: text

    class VLLMManager()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    VLLMManager.__init__(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VLLMManager.deploy_models(self, model_path: str, tknz_path: str, gpus: List[int], ports: List[int], gpu_mem_utils: float | List[float]=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VLLMManager.release_resources_(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    VLLMManager.release_resources(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

LocalVLLMAPI
~~~~~~~~~~~~

.. code-block:: text

    class LocalVLLMAPI(LLM)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    LocalVLLMAPI.__init__(self, model_path, tknz_path, gpus: List[int], ports: List[int], **kwargs)

.. code-block:: text

    Deploy multiple LLMs on multiple GPUs using VLLM backends.
    Currently only support deploying each LLM on single GPU.

    Args:
        model_path: pretrained model path
        tknz_path : tokenizer path
        gpus      : gpu ids where you want to deploy an LLM.
        ports:    :

.. code-block:: text

    LocalVLLMAPI.draw_sample(self, prompt, *args, **kwargs) -> str

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    LocalVLLMAPI.close(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.llm.ollama_http
----------------------------

OllamaHTTP
~~~~~~~~~~

.. code-block:: text

    class OllamaHTTP(LLM)

.. code-block:: text

    Use a separately installed Ollama server without downloading models.

    Args:
        model: An already installed model name.
        endpoint: Local server URL.
        seed: Server sampling seed; not a guarantee across inference versions.
        temperature: Generation temperature.
        timeout: Request deadline in seconds. Errors propagate to the caller.

.. code-block:: text

    OllamaHTTP.__init__(self, model, endpoint='http://127.0.0.1:11434', *, seed=17, temperature=0.0, timeout=120)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    OllamaHTTP.model_info(self)

.. code-block:: text

    Return the installed model digest and server version for provenance.

.. code-block:: text

    OllamaHTTP.draw_sample(self, prompt, *args, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.llm.replay
-----------------------

RecordingLLM
~~~~~~~~~~~~

.. code-block:: text

    class RecordingLLM(LLM)

.. code-block:: text

    Wrap an LLM and durably record requests/responses in a new JSONL file.

    Args:
        llm: Backend implementing ``draw_sample`` and ``close``.
        path: A new file; existing recordings are never overwritten or appended.
        model_metadata: Explicit model/revision/generation metadata, without keys.
        run_metadata: Optional task, seed, dataset and budget configuration.

.. code-block:: text

    RecordingLLM.__init__(self, llm: LLM, path, *, model_metadata=None, run_metadata=None)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RecordingLLM.draw_sample(self, prompt, *args, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    RecordingLLM.close(self)

.. code-block:: text

    Close the journal and underlying backend after outstanding calls finish.

ReplayLLM
~~~~~~~~~

.. code-block:: text

    class ReplayLLM(LLM)

.. code-block:: text

    Replay a complete recording in request-start order, without any API client.

    Changed prompts/arguments, corrupt journals and exhausted recordings raise
    explicit errors. Replay never silently falls back to online generation.

.. code-block:: text

    ReplayLLM.__init__(self, path)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ReplayLLM.remaining(self)

.. code-block:: text

    Number of unconsumed requests.

.. code-block:: text

    ReplayLLM.draw_sample(self, prompt, *args, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.profiler.__init__
------------------------------

import_all_profiler_classes_from_subfolders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    This function is kept for compatibility but the dynamic import
    is now handled by the package's __init__.py.

.. code-block:: text

    import_all_profiler_classes_from_subfolders(root_directory)

llm4ad.tools.profiler.profile
-----------------------------

ProfilerBase
~~~~~~~~~~~~

.. code-block:: text

    class ProfilerBase()

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    ProfilerBase.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style: Literal['simple', 'complex']='complex', create_random_path=True, num_objs=1, **kwargs)

.. code-block:: text

    Base profiler for recording experimental results.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    ProfilerBase.record_parameters(self, llm, prob, method)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProfilerBase.register_function(self, function: Function, program: str='', *, resume_mode=False)

.. code-block:: text

    Record an obtained function.


.. code-block:: text

    ProfilerBase.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProfilerBase.get_logger(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProfilerBase.resume(self, *args, **kwargs)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    ProfilerBase.load_logfile(cls, logdir, valid_only=False) -> Tuple[List[str], List[float]]

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.profiler.tensorboard_profiler
------------------------------------------

TensorboardProfiler
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    class TensorboardProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    TensorboardProfiler.__init__(self, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, **kwargs)

.. code-block:: text

    Base profiler for recording experimental results.
    Args:
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...

.. code-block:: text

    TensorboardProfiler.get_logger(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    TensorboardProfiler.register_function(self, function: Function, program='', *, resume_mode=False)

.. code-block:: text

    Record an obtained function. This is a synchronized function.


.. code-block:: text

    TensorboardProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.profiler.wandb_profiler
------------------------------------

WandBProfiler
~~~~~~~~~~~~~

.. code-block:: text

    class WandBProfiler(ProfilerBase)

.. code-block:: text

    Documentation gap: no docstring is present in this source definition.

.. code-block:: text

    WandBProfiler.__init__(self, wandb_project_name: str, log_dir: Optional[str]=None, *, initial_num_samples=0, log_style='complex', create_random_path=True, fork_proc: Literal['auto'] | bool='auto', **wandb_init_kwargs)

.. code-block:: text

    Weights and Biases profiler.
    Args:
        wandb_project_name : the project name in which you sync your results.
        log_dir            : the directory of current run
        initial_num_samples: the sample order start with `initial_num_samples`.
        create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        fork_proc          : whether to fork the wandb process.

.. code-block:: text

    WandBProfiler.get_logger(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

.. code-block:: text

    WandBProfiler.register_function(self, function: Function, program='', *, resume_mode=False)

.. code-block:: text

    Record an obtained function. This is a synchronized function.


.. code-block:: text

    WandBProfiler.finish(self)

.. code-block:: text

    Documentation gap: no method docstring is present.

llm4ad.tools.reproducibility
----------------------------

redact
~~~~~~

.. code-block:: text

    Redact credential fields recursively; prompt/response text is retained verbatim.

    Callers must still inspect artifacts before sharing: free text can itself contain
    private information. This function is not a secret detector for arbitrary text.

.. code-block:: text

    redact(value)

seed_all
~~~~~~~~

.. code-block:: text

    Seed Python and NumPy. GPU kernels and remote model sampling are separate.

.. code-block:: text

    seed_all(seed: int)

sha256_file
~~~~~~~~~~~

.. code-block:: text

    Hash a dataset, weight file or artifact without loading it all into memory.

.. code-block:: text

    sha256_file(path)

source_fingerprint
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Fingerprint installed source/configs, including uncommitted local changes.

.. code-block:: text

    source_fingerprint()

write_run_manifest
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Write a versioned manifest atomically; return the exact recorded dictionary.

    ``model`` should include the model ID, immutable revision and sampling parameters.
    ``repository`` is optional: no unrelated current-directory Git state is assumed.

.. code-block:: text

    write_run_manifest(path, *, config, model, seed, datasets=(), repository=None)

