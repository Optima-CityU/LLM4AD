from __future__ import annotations

from typing import Any, List, Tuple
import numpy as np

from llm4ad.base import Evaluation
# Assuming these imports exist in your project structure
from template import template_program, task_description, aim_java_relative_path, java_dir
import os
import subprocess
import sys
import glob
import shutil
import re
import concurrent.futures

__all__ = ['Ails2EvaluationDebug']


class Ails2EvaluationDebug(Evaluation):
    """
    Evaluator for AILSII Java - DEBUG MODE.
    Runs only ONE instance to check for compilation errors and logic bugs.
    """

    def __init__(self,
                 debug_instance_name: str = None,  # e.g., "X-n101-k25.vrp"
                 default_instance_timeout=30,
                 dump_java_output_on_finish: bool = True,  # Force True for debug
                 jdk_bin_path: str = r";C:\Program Files\Common Files\Oracle\Java\javapath",
                 instances_subdir: str = "XLDemo",
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=default_instance_timeout
        )

        self.debug_instance_name = debug_instance_name
        self.default_instance_timeout = default_instance_timeout
        self.dump_java_output_on_finish = dump_java_output_on_finish
        self.java_commands = []
        self.jdk_bin_path = jdk_bin_path
        self.max_parallel_instances = 1  # Force serial for debugging
        self.instances_subdir = instances_subdir

        # Simplified timeout for debug mode
        self.timeout_seconds = default_instance_timeout + 60
        print(f"--- DEBUG EVALUATOR INITIALIZED ---")
        print(f"Target Instance: {self.debug_instance_name if self.debug_instance_name else 'First available'}")

    def copy_dir_multiple_times(self, n: int):
        """Standard copy logic"""
        dst_base_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(dst_base_dir, java_dir)

        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

        os.makedirs(dst_base_dir, exist_ok=True)

        # For debug, we usually only need index 0
        for i in range(n):
            dst_dir = os.path.join(dst_base_dir, os.path.basename(src_dir) + f"_{i}")
            if os.path.exists(dst_dir):
                print(f"{dst_dir} already exists, removing it first.")
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"Copied {src_dir} -> {dst_dir}")

    def run_command(self, commands, python_timeout: int) -> Tuple[str, int]:
        """Standard run command logic"""
        last_line_score = ""
        full_stdout = ""
        full_stderr = ""
        process = None

        try:
            print(f"Running Command: {' '.join(commands)}")  # Print command for debug
            process = subprocess.Popen(
                commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="gbk"  # Adjust encoding if necessary (e.g., utf-8)
            )

            full_stdout, full_stderr = process.communicate(timeout=python_timeout)
            return_code = process.returncode

            if return_code != 0:
                print(f"Java process failed. Return code: {return_code}", file=sys.stderr)
                return "CRASH", return_code

        except subprocess.TimeoutExpired:
            print(f"Java process timed out.", file=sys.stderr)
            process.kill()
            try:
                full_stdout, full_stderr = process.communicate(timeout=5)
            except:
                pass
            return_code = -1

        except Exception as e:
            print(f"Python error: {e}", file=sys.stderr)
            if process: process.kill()
            return "PY_ERROR", -2

        # Dump output for debugging
        if self.dump_java_output_on_finish:
            print(f"\n[Java STDOUT]:\n{full_stdout}")
            if full_stderr:
                print(f"[Java STDERR]:\n{full_stderr}", file=sys.stderr)

        if full_stdout:
            for line in reversed(full_stdout.strip().split('\n')):
                line_full = line.strip()
                if not line_full: continue
                try:
                    # Adjust this parsing logic based on your Java output format
                    # Assuming format: "Result; 12345.0"
                    if ";" in line_full:
                        score_part = line_full.split(';', 1)[1].strip()
                        if score_part:
                            last_line_score = score_part
                            break
                except IndexError:
                    pass

        if last_line_score:
            return last_line_score, return_code
        else:
            return "NO_SCORE_OUTPUT", return_code

    def evaluate(self, java_script: str, subprocess_index=0) -> float | None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(current_path, java_dir + f"_{subprocess_index}")
        target_change_java = os.path.join(target_dir, aim_java_relative_path)

        classpath_separator = ';' if sys.platform == "win32" else ':'
        compile_output_dir = os.path.join(target_dir, "bin")
        os.makedirs(compile_output_dir, exist_ok=True)

        libs_dir = os.path.join(target_dir, "libs")
        libs_path_glob = os.path.join(libs_dir, "*")

        if os.path.isdir(libs_dir) and glob.glob(libs_path_glob):
            classpath_compile = libs_path_glob
            classpath_run = f"{compile_output_dir}{classpath_separator}{libs_path_glob}"
        else:
            classpath_compile = ""
            classpath_run = f"{compile_output_dir}"

        # --- 1. Build Java Commands (Modified for Debug) ---
        if not self.java_commands:
            instances_dir = os.path.join(target_dir, self.instances_subdir)
            instance_files = glob.glob(os.path.join(instances_dir, "*.vrp"))

            if not instance_files:
                print(f"Error: No .vrp files in {instances_dir}")
                return None

            # ==========================================
            # === DEBUG: FILTER FOR SINGLE INSTANCE ===
            # ==========================================
            if self.debug_instance_name:
                # Filter list to find specific file
                filtered = [f for f in instance_files if self.debug_instance_name in os.path.basename(f)]
                if filtered:
                    instance_files = [filtered[0]]  # Take the match
                    print(f"Debug Mode: Selected instance {os.path.basename(instance_files[0])}")
                else:
                    print(f"Warning: Instance {self.debug_instance_name} not found. Using first available.")
                    instance_files = [instance_files[0]]
            else:
                # No specific name given, just take the first one
                print(
                    f"Debug Mode: No specific instance named. Using first available: {os.path.basename(instance_files[0])}")
                instance_files = [instance_files[0]]
            # ==========================================

            main_class = "SearchMethod.AILSII"
            instance_commands = []

            for instance_file_path in instance_files:
                # Use default timeout for debug to avoid waiting too long
                java_limit_time = self.default_instance_timeout
                python_timeout = java_limit_time + 10

                command = [
                    "java",
                    "-cp", classpath_run,
                    main_class,
                    "-file", instance_file_path,
                    "-rounded", "true",
                    "-limit", str(java_limit_time),
                    "-stoppingCriterion", "Time"
                ]
                instance_commands.append((command, python_timeout))

            self.java_commands = instance_commands

        # --- 2. Setup JDK ---
        if "JDK_PATH" not in os.environ:
            os.environ['PATH'] += self.jdk_bin_path
            os.environ["JDK_PATH"] = "set"

        # --- 3. Inject and Compile ---
        src_path = os.path.join(target_dir, "src", "AILS-II_origin", "src")
        sources_file = os.path.join(target_dir, "sources.txt")

        try:
            with open(target_change_java, "w", encoding="gbk") as f:
                f.write(java_script)
        except Exception as e:
            print(f"Write failed: {e}")
            return None

        try:
            java_files = glob.glob(os.path.join(src_path, "**/*.java"), recursive=True)
            with open(sources_file, 'w', encoding="gbk") as fs:
                fs.write("\n".join(java_files))
        except Exception as e:
            print(f"Source collection failed: {e}")
            return None

        compile_cmd = ["javac", "-d", compile_output_dir, "-sourcepath", src_path]
        if classpath_compile:
            compile_cmd.extend(["-cp", classpath_compile])
        compile_cmd.append(f"@{sources_file}")

        print("Compiling Java...")
        compile_process = subprocess.run(
            compile_cmd, capture_output=True, text=True, encoding="gbk", errors='replace'
        )

        if compile_process.returncode != 0:
            print("COMPILATION FAILED!", file=sys.stderr)
            print(compile_process.stderr, file=sys.stderr)
            return None
        else:
            print("Compilation Successful.")

        # --- 4. Run Single Instance (No ThreadPool needed for 1 item) ---
        print(f"Starting execution of {len(self.java_commands)} instance(s)...")

        results = []
        for cmd, cmd_timeout in self.java_commands:
            # Run directly in main thread for easier debugging
            res = self.run_command(cmd, cmd_timeout)
            results.append(res)

        # --- 5. Result ---
        fitness = []
        for (last_line, return_code) in results:
            try:
                fitness.append(float(last_line))
            except:
                print(f"Invalid result: {last_line}")

        if not fitness:
            return None

        return -float(np.mean(fitness))

    def evaluate_program(self, program_str: str, **kwargs) -> Any | None:
        return self.evaluate(program_str, subprocess_index=0)


if __name__ == '__main__':
    # LLM Generated Code
    java_script = """
package Perturbation;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import Data.Instance;
import DiversityControl.OmegaAdjustment;
import Improvement.IntraLocalSearch;
import SearchMethod.Config;
import Solution.Node;
import Solution.Solution;


public class Ruinnew extends Perturbation {
    public Ruinnew(Instance instance, Config config, HashMap<String, OmegaAdjustment> omegaSetup, IntraLocalSearch intraLocalSearch) {
        super(instance, config, omegaSetup, intraLocalSearch);
        this.perturbationType = PerturbationType.Ruinnew;
    }
    public void applyPerturbation(Solution s) {
        setSolution(s);
        if (size <= 0) return;
        // Simple random removal for debug test
        int toRemove = 10;
        for(int i=0; i<toRemove; i++) {
            Node n = solution[rand.nextInt(solution.length)];
            if(n.route != null) {
               f += n.route.remove(n);
               candidates[countCandidates++] = n;
            }
        }
        setOrder();
        addCandidates();
        assignSolution(s);
    }
}
    """

    # --- CONFIGURATION FOR DEBUGGING ---
    # 1. Set the specific instance name you want to test (or None for the first one)
    #    Example: "X-n214-k11.vrp"
    target_instance = None

    # 2. Initialize Debug Evaluator
    debug_eval = Ails2EvaluationDebug(
        debug_instance_name=target_instance,
        default_instance_timeout=15,  # Short timeout for quick debug
        dump_java_output_on_finish=True
    )

    print("Initializing Sandbox...")
    try:
        debug_eval.copy_dir_multiple_times(1)  # Only need 1 copy
    except Exception as e:
        print(f"Init failed: {e}")
        sys.exit(1)

    print(">>> STARTING SINGLE INSTANCE DEBUG <<<")
    score = debug_eval.evaluate_program(java_script)
    print(f">>> DEBUG FINISHED. Score: {score} <<<")
