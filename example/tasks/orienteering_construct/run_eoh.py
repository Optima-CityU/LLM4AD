# SPDX-License-Identifier: BSD-2-Clause
import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.orienteering_construct import OrienteeringEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.eoh import EoH


def main():
    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='sk-xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='xxx',  # your llm, e.g., 'gpt-4o-mini'
                   timeout=60)

    task = OrienteeringEvaluation(
        timeout_seconds=20,
        n_instance=8,
        problem_size=30,
        max_length_ratio=0.35,
        seed=2024,
    )

    method = EoH(llm=llm,
                 profiler=ProfilerBase(log_dir='logs/orienteering_eoh', log_style='complex'),
                 evaluation=task,
                 max_sample_nums=20,
                 max_generations=5,
                 pop_size=2,
                 num_samplers=1,
                 num_evaluators=1)

    method.run()


if __name__ == '__main__':
    main()
