import subprocess
import argparse
import sys
import json
import os
import json
import copy
import time
from functools import wraps

from typing import Dict, Any, List

from generator.et_generator import et_generator

from generator.megatron_core.arguments import (
    parse_json_args as parse_megatron_args,
    validate_args,
    validate_args_extern
)

from utils import (
    get_perf,
    modify_comm_group_json,
    generate_trace_output_rule,
    generate_trace_template_mapping,
    reformat_visualize_file,
    parse_args,
)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'> Function: {func.__name__} took {total_time*1000:.2f} micro seconds')
        return result
    return timeit_wrapper

@timeit
def subprocess_run_with_log(bash, log_file):
    with open(os.path.join(os.getcwd(), log_file), 'w') as f:
        subprocess.run(bash, stdout=f)

@timeit
def trace_generator(json_args: Dict[str, str], trace_generate: bool =True) -> str:

    output_file_path = os.path.join(os.getcwd(), json_args["output_filepath"])
    if not os.path.exists(output_file_path):
        os.mkdir(output_file_path)

    # if not trace_generate:
    #     print(">> Skip trace generation and conversion.")
    #     return os.path.join(output_file_path, json_args["output_filename"])

    # print(">> Pytorch+ trace generating...")
    et_generator(
        output_file_path = output_file_path,
        trace_generate = trace_generate, 
        json_args=json_args
    )
    # print(">> Pytorch+ trace generation completed.")

    # print(">> trace converting...")
    # logger = get_logger(json_args["log_filename"])
    # logger.debug(" ".join(sys.argv))
    # for i in range(len(json_output_files)):
    #     json_output_file = json_output_files[i]
    #     try:
    #         et_output_file = json_args["output_filename"] + f".{i}.et"
    #         converter = PyTorchConverter(json_output_file, os.path.join(output_file_path, et_output_file), logger)
    #         converter.convert()
    #     except Exception:
    #         traceback.print_exc()
    #         logger.debug(traceback.format_exc())
    #         sys.exit(1)
    # print(f">> trace conversion completed with files storing in {output_file_path}")
    return os.path.join(output_file_path, json_args["output_filename"])

def main() -> None:
    parser = argparse.ArgumentParser(description="Execution LLM Simulator")
    parser.add_argument("--config-file", type=str, default=None, required=True, help="configuration file")
    parsed_args = parser.parse_args()
    json_args = json.load(open(parsed_args.config_file, 'r'))

    # preprocess megatron args
    parse_args(json_args)
    megatron_args = parse_megatron_args(json_args["trace_generator_config"])
    validate_args(megatron_args)
    validate_args_extern(megatron_args)
    json_args["trace_generator_config"]["megatron_args"] = megatron_args

    #generate trace
    print("> trace generator running...")
    workload = trace_generator(json_args["trace_generator_config"], json_args["trace_generator_config"]["trace_generate"])
    print(f"> trace generator completed with log-file storing in {json_args['trace_generator_config']['log_filename']}.")
    
    #simulation
    modify_comm_group_json(json_args)
    trace_template_mapping = generate_trace_template_mapping(json_args)
    trace_output_rule, num_npus = generate_trace_output_rule(json_args)
    llm_sim_bash = \
    [
        json_args['simulator_config']["binary"], 
        f"--workload-configuration={workload}", 
        f"--system-configuration={json_args['simulator_config']['system']}", 
        f"--network-configuration={json_args['simulator_config']['network']}", 
        f"--remote-memory-configuration={json_args['simulator_config']['remote_memory']}", 
        f"--comm-group-configuration={json_args['simulator_config']['comm_group']}", 
        f"--trace-output-rule={trace_output_rule}", 
        f"--trace-template-mapping={trace_template_mapping}",
        f"--trace-output-file={json_args['simulator_config']['trace_output']}"
    ]
    with open(os.path.join(json_args['trace_generator_config']['output_filepath'], "llm_sim_cmd.log"), 'w') as f:
        f.write(" ".join(llm_sim_bash))
    print("> LLM simulating...")
    subprocess_run_with_log(llm_sim_bash, "llm_sim.log")
    print(f"> LLM simulation completed with log-file storing in {os.path.join(os.getcwd(), 'llm_sim.log')}")

    # visualization
    # assert visual_config['num_npus']==trace_gen_config["pipeline-model-parallel-size"], \
    #     f"visualization --num_npus {visual_config['num_npus']} arg fail. (in generation {trace_gen_config['pp_world_size']})"
    visualize_bash = \
    [
        "chakra_timeline_visualizer",
        f"--input_filename={json_args['simulator_config']['trace_output']}",
        f"--output_filename={json_args['time_line_visualizer_config']['output_filename']}",
        f"--num_npus={num_npus}",
        f"--npu_frequency={1000}",
    ]
    print("> timeline visualizer running...")
    subprocess_run_with_log(visualize_bash, "visualizer.log")
    print(  "> timeline visualizer completed with " + \
            f"log-file storing in {os.path.join(os.getcwd(), 'visualizer.log')} and " + \
            f"visualizer file storing in {json_args['time_line_visualizer_config']['output_filename']}")
    
    # get performance
    get_perf(json_args)

    # reformat file
    reformat_visualize_file(json_args)
    
if __name__ == "__main__":
    main()
