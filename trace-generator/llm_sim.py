import subprocess
import argparse
import sys
import json
import os
import json
import copy

from typing import Dict, Any, List

from generator.et_generator import (
    et_generator,
    calculate_transformer_flops,
)
from generator.megatron_core.arguments import (
    parse_json_args as parse_megatron_args,
    validate_args,
    validate_args_extern
)

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

def get_throughput(
    json_args: Dict[str, str]
):
    with open(json_args['time_line_visualizer_config']['output_filename'], 'r') as f:
        data = json.load(f)
        elapsed_time = data["traceEvents"][-1]["ts"] / 1E3 # unit: second
        megatron_args = json_args["trace_generator_config"]["megatron_args"]
        (total_flops, embedding) = calculate_transformer_flops(args=megatron_args, enable_extra_param=False)
        total_flops = (total_flops + embedding)*megatron_args.train_iters
        throughput = (
            total_flops /
            (elapsed_time * 10**12 * megatron_args.world_size)
        )
        elapsed_time_per_iteration = elapsed_time/megatron_args.train_iters
        print(f"throughput per GPU (TFLOP/s/GPU): {throughput:.2f}")
        print(f"elapsed time per iteration (ms): {elapsed_time_per_iteration*1E3:.2f}")

def get_perf(
    json_args: Dict[str, str]
):
    print("------------------------ results ------------------------", flush=True)

    get_throughput(json_args)

    # get trace output
    with open(json_args['time_line_visualizer_config']['output_filename'], 'r', encoding='utf-8') as file:
        trace = json.load(file)
    
    # setup perf dict
    perf_dict_sample: Dict[str, int] = {
        "forward-compute": 0,
        "backward-compute": 0,
        "send": 0,
        "recv": 0,
        "all-grads-sync": 0,
        "params-all-gather": 0,
    }
    perf_dict: Dict[int, Dict[str, int]] = {}
    world_size = 0
    if json_args['time_line_visualizer_config']["single_pp_group"]:
        world_size = json_args["trace_generator_config"]["pipeline-model-parallel-size"]
    else:
        world_size = json_args["trace_generator_config"]["world-size"]
    for i in range(world_size):
        perf_dict[i] = copy.copy(perf_dict_sample)
    # get all perf according to rank
    for event in trace["traceEvents"]:
        if event["name"].find("COMP_") != -1:
            if event["name"].find("forward") != -1:
                perf_dict[event["pid"]]["forward-compute"] += event["dur"]
            elif event["name"].find("backward") != -1:
                perf_dict[event["pid"]]["backward-compute"] += event["dur"]
        elif event["name"].find("COMM_") != -1:
            if event["name"].find("send") != -1:
                perf_dict[event["pid"]]["send"] += event["dur"]
            elif event["name"].find("recv") != -1:
                perf_dict[event["pid"]]["recv"] += event["dur"]
            elif event["name"].find("grad_reducescatter") != -1:
                perf_dict[event["pid"]]["all-grads-sync"] += event["dur"]
            elif event["name"].find("param_allgather") != -1:
                perf_dict[event["pid"]]["params-all-gather"] += event["dur"]
    # print(perf_dict)
    perf_min_max_dict: Dict[str, Dict[str, int]] = {
        "forward-compute":      {"min": sys.float_info.max, "max": 0},
        "backward-compute":     {"min": sys.float_info.max, "max": 0},
        "send":                 {"min": sys.float_info.max, "max": 0},
        "recv":                 {"min": sys.float_info.max, "max": 0},
        "all-grads-sync":       {"min": sys.float_info.max, "max": 0},
        "params-all-gather":    {"min": sys.float_info.max, "max": 0},
    }
    for perf in perf_dict.values():
        for key in perf.keys():
            perf_min_max_dict[key]["max"] = max(perf[key], perf_min_max_dict[key]["max"])
            perf_min_max_dict[key]["min"] = min(perf[key], perf_min_max_dict[key]["min"])
    for key in perf_min_max_dict.keys():
        for min_max_key in perf_min_max_dict[key].keys():
            perf_min_max_dict[key][min_max_key] = (
                perf_min_max_dict[key][min_max_key] 
                / json_args["trace_generator_config"]["train-iters"]
            )
    for key in perf_min_max_dict.keys():
        msg = (
            key.ljust(30, '-')
            + f'({perf_min_max_dict[key]["min"]:.2f}, {perf_min_max_dict[key]["max"]:.2f})'
        )
        print(msg)

    print("-------------------- end of results ---------------------", flush=True)

def modify_comm_group_json(json_args: Dict[str, Any] = {}):
    comm_group : Dict[str, Any] = {}
    if os.path.exists(json_args['simulator_config']['comm_group']):
        with open(json_args['simulator_config']['comm_group'], 'r') as file:
            comm_group = json.load(file)
    comm_group['dp_group'] = json_args["trace_generator_config"]["comm_group"]["all_dp_group"]
    comm_group['embedding_group'] = json_args["trace_generator_config"]["comm_group"]["all_embed_group"]
    comm_group['position_embedding_group'] = json_args["trace_generator_config"]["comm_group"]["all_pos_embed_group"]
    
    with open(json_args['simulator_config']['comm_group'], 'w') as file:
        file.write(json.dumps(comm_group, indent=4))

def generate_trace_output_rule(
    json_args: Dict[str, Any] = {}
) -> tuple[str, int]:
    single_pp_group=json_args['time_line_visualizer_config']["single_pp_group"]
    output_rank_num : int = 0
    output_rank_list: List[int] = []

    if single_pp_group:
        # get single pp group
        from generator.et_generator import get_primary_pp_group
        output_rank_list = get_primary_pp_group(json_args["trace_generator_config"])
        output_rank_num = len(output_rank_list)
    else:
        output_rank_num = json_args["trace_generator_config"]["world-size"]
        output_rank_list = list(range(output_rank_num))

    # write single pp group in to rule file
    with open(json_args["trace_generator_config"]['output_filepath'] + "/trace-output-rule.txt", "w") as file:
        file.write(str(len(output_rank_list))+"\n")
        for rank in output_rank_list:
            file.write(str(rank)+" ")
        file.write("\n")
    
    return [json_args["trace_generator_config"]['output_filepath'] + "/trace-output-rule.txt", output_rank_num]

def generate_trace_template_mapping(
    json_args: Dict[str, Any] = {}
) -> str:
    from generator.megatron_core.p2p_communication import is_creating_template
    if not is_creating_template():
        return "None"
    with open(json_args["trace_generator_config"]['output_filepath'] + "/trace-template-mapping.txt", "w") as file:
        template_mapping = json_args["trace_generator_config"]["template_mapping"]
        file.write(str(len(template_mapping)) + "\n") 
        for key in template_mapping.keys():
            file.write(str(key)+" "+" ".join(str(x) for x in template_mapping[key])+"\n")
    return json_args["trace_generator_config"]['output_filepath'] + "/trace-template-mapping.txt"

def reformat_visualize_file(
    json_args: Dict[str, Any] = {}
):
    # single_pp_group=json_args['time_line_visualizer_config']["single_pp_group"]
    # get trace output
    with open(json_args['time_line_visualizer_config']['output_filename'], 'r', encoding='utf-8') as file:
        trace = json.load(file)
    # # remove all event which pid not in pp group
    # if single_pp_group:
    #     comm_group = json.load(open(json_args['simulator_config']['comm_group'], 'r'))
    #     # get single pp group
    #     from generator.et_generator import get_primary_pp_group
    #     pp_group: List[int] = get_primary_pp_group(json_args["trace_generator_config"])

    #     single_pp_group_event = []
    #     for event in trace["traceEvents"]:
    #         if event["pid"] in pp_group and event["pid"] % json_args["trace_generator_config"]["tensor-model-parallel-size"] == 0:
    #             single_pp_group_event.append(event)
    #     trace["traceEvents"] = single_pp_group_event
    # # change tid for dummy and comp node
    # for event in trace["traceEvents"]:
    #     if event["name"].find("dummy") != -1:
    #         event["tid"] = 100
    #     elif event["name"].find("COMP_NODE_") != -1:
    #         event["tid"] = 0
    # reformat
    with open(json_args['time_line_visualizer_config']['output_filename'], 'w') as file:
        file.write(json.dumps(trace, indent=4))

def parse_args(
    json_args: Dict[str, Any] = {}
):
    parse_keys = [
        "trace_generate", "use_chakra", 
        "use_simplified_trace", 
        "use_simplified_template_trace", 
        "use_step_snapshot",
        "step_snapshot_data"
    ]
    for key in parse_keys:
        assert key in json_args["trace_generator_config"].keys(), f"{key} not found in trace_generator_config."
    
    # chakra trace
    tg_args = json_args["trace_generator_config"]
    if tg_args["use_chakra"] != 0:
        assert tg_args["use_simplified_trace"] == 0, \
            "use_simplified_trace should set to 0 when use_chakra is 1."
        assert tg_args["use_simplified_template_trace"] == 0, \
            "use_simplified_template_trace should set to 0 when use_chakra is 1."
        json_args["trace_generator_config"]["is-simplified"] = False
    else:
        json_args["trace_generator_config"]["is-simplified"] = True

    sim_system_config = None
    with open(json_args['simulator_config']['system'], 'r') as f:
        sim_system_config = json.load(f)

    # simplified trace
    if tg_args["use_simplified_trace"] != 0:
        sim_system_config["simplified-trace"] = 1
    else:
        sim_system_config["simplified-trace"] = 0

    # simplified template trace
    if tg_args["use_simplified_template_trace"] != 0:
        from generator.megatron_core.p2p_communication import set_creating_template
        set_creating_template()
        sim_system_config["template-trace"] = 1
    else:
        from generator.megatron_core.p2p_communication import unset_creating_template
        unset_creating_template()
        sim_system_config["template-trace"] = 0
    
    with open(json_args['simulator_config']['system'], 'w') as f:
        f.write(json.dumps(sim_system_config, indent=4))

    # snapshot
    from generator.utils import (
        set_snapshot_data,
        set_use_snapshot
    )
    if json_args["trace_generator_config"]["use_step_snapshot"]:
        assert len(json_args["trace_generator_config"]["step_snapshot_data"]) == 3, \
            "step_snapshot_data should set to 3 integers (pp0, ppn, ppend)."
        set_use_snapshot(True)
        set_snapshot_data(json_args["trace_generator_config"]["step_snapshot_data"])
    else:
        set_use_snapshot(False)
        set_snapshot_data([])

    if "total_file_size_gb_limit" not in json_args["trace_generator_config"].keys():
        json_args["trace_generator_config"]["total_file_size_gb_limit"] = 10
    json_args["remove_json_files"] = None
    # exit()

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
    print(" ".join(llm_sim_bash))
    print("> LLM simulating...")
    with open(os.path.join(os.getcwd(), "llm_sim.log"), 'w') as f:
        subprocess.run(llm_sim_bash, stdout=f)
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
    with open(os.path.join(os.getcwd(), "visualizer.log"), 'w') as f:
        subprocess.run(visualize_bash, stdout=f)
    print(  "> timeline visualizer completed with " + \
            f"log-file storing in {os.path.join(os.getcwd(), 'visualizer.log')} and " + \
            f"visualizer file storing in {json_args['time_line_visualizer_config']['output_filename']}")
    
    # get performance
    get_perf(json_args)

    # reformat file
    reformat_visualize_file(json_args)
    
if __name__ == "__main__":
    main()
