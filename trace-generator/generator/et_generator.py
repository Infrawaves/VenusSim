import os
import sys
import json
import traceback
import logging
from argparse import Namespace
from typing import Any, Dict, List, Set, Tuple
from generator.generator import TraceNodeAttr, TraceNode, PytorchPlusTrace
from chakra_convert_standalone.converter.pytorch_converter import PyTorchConverter
from chakra_convert_standalone.converter.converter import get_logger

from generator.megatron_core.schedules import (
    get_forward_backward_func
)
from generator.megatron_core.parallel_state import (
    initialize_model_parallel, 
    destroy_model_parallel,
    get_pipeline_model_parallel_group,
    get_data_parallel_group,
    is_embedding_group_initialized,
    get_embedding_group,
    is_position_embedding_group_initialized,
    get_position_embedding_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank
)
from generator.megatron_core.p2p_communication import (
    check_send_recv_map_in_trace,
    check_send_recv_map,
    is_creating_template,
    clear_send_recv_tag
)

from generator.megatron_core.training import train
from generator.utils import merge_branch_nodes
from generator.megatron_core.distrib_optimizer import (
    set_param_all_gather_num,
    get_param_all_gather_num
)
from generator.overlap_grad_reduce_helper import (
    set_grad_reduce_scatter_num,
    get_grad_reduce_scatter_num
)

# GPU num is world size
TRANSFORMER_FLOPS_PER_GPU = None
TRANSFORMER_PARAM_NUM_PER_MODEL_CHUNK = None
# GPU num is the num of gpus which are first or last.
EMBEDDING_FLOPS_PER_GPU = None
# Used for calculate tflops
# used for softmax_flops
CALCULATE_TRANSFORMER_FLOPS_PARAM = {
    "total": 1.28,
    "attn_softmax": 6,
    "embedding": 3, 
    "stable_embedding": 70
}

def set_transformer_flops_per_gpu(transformer_flops: int):
    global TRANSFORMER_FLOPS_PER_GPU
    TRANSFORMER_FLOPS_PER_GPU = transformer_flops

def get_transformer_flops_per_gpu():
    global EMBEDDING_FLOPS_PER_GPU
    return TRANSFORMER_FLOPS_PER_GPU

def set_embedding_flops_per_gpu(embedding_flops: int):
    global EMBEDDING_FLOPS_PER_GPU
    EMBEDDING_FLOPS_PER_GPU = embedding_flops

def get_embedding_flops_per_gpu():
    global EMBEDDING_FLOPS_PER_GPU
    return EMBEDDING_FLOPS_PER_GPU

def set_transformer_param_num_per_model_chunk(transformer_param_num: int):
    global TRANSFORMER_PARAM_NUM_PER_MODEL_CHUNK
    TRANSFORMER_PARAM_NUM_PER_MODEL_CHUNK = transformer_param_num

def get_transformer_param_num_per_model_chunk():
    global TRANSFORMER_PARAM_NUM_PER_MODEL_CHUNK
    return TRANSFORMER_PARAM_NUM_PER_MODEL_CHUNK

def et_convertor(
    rank: int, 
    json_output_file: str,
    output_file_path: str,
    logger: logging.Logger, 
    json_args: Dict[str, Any] = {}
):
    try:
        et_output_file = json_args["output_filename"] + f".{rank}.et"
        converter = PyTorchConverter(json_output_file, os.path.join(output_file_path, et_output_file), logger)
        converter.convert()
    except Exception:
        traceback.print_exc()
        logger.debug(traceback.format_exc())
        sys.exit(1)
    
    if rank == 0:
        # Manually save json files
        if "keep_json_files" in json_args.keys():
            json_args["remove_json_files"] = not json_args["keep_json_files"]
        # Automatically save json files
        elif os.path.getsize(json_output_file)/(1024**3)* \
            json_args["megatron_args"].world_size <= json_args["total_file_size_gb_limit"]:
            json_args["remove_json_files"] = False
        else:
            print(f">> json files exceed the size limit {json_args['total_file_size_gb_limit']} GB.")
            json_args["remove_json_files"] = True
        
        if json_args["remove_json_files"]:
            print(">> remove json files. (To save json files, set \"keep_json_files\" to 1)")
    
    if json_args["remove_json_files"]:
        os.remove(json_output_file)

def get_primary_pp_group(
    json_args: Dict[str, Any] = {}
) -> List[int]:
    args = json_args["megatron_args"]
    destroy_model_parallel()
    initialize_model_parallel(
        world_size                              =   args.world_size,
        npu_rank                                =   0,
        tensor_model_parallel_size              =   args.tensor_model_parallel_size,
        pipeline_model_parallel_size            =   args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size    =   args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank      =   None, # only for "encoder and decoder" models
        context_parallel_size                   =   args.context_parallel_size,
        expert_model_parallel_size              =   1, # not apply yet
    )
    primary_pp_group: List[int] = get_pipeline_model_parallel_group()
    destroy_model_parallel()
    return primary_pp_group

def modify_comm_group_json(
    dp_group: List[List[int]] = [],
    embed_group: List[List[int]] = [],
    pos_embed_group: List[List[int]] = []
):
    from llm_sim import get_global_json_args
    json_args = get_global_json_args()
    with open(json_args['simulator_config']['comm_group'], 'r') as file:
        comm_group = json.load(file)
    comm_group['dp_group'] = dp_group
    comm_group['embedding_group'] = embed_group
    comm_group['position_embedding_group'] = pos_embed_group
    
    with open(json_args['simulator_config']['comm_group'], 'w') as file:
        file.write(json.dumps(comm_group, indent=4))

def et_generator(
    output_file_path: str = "",
    trace_generate: bool = True, 
    json_args: Dict[str, Any] = {}
) -> List[str]:
    
    if trace_generate:
        print(">> Pytorch+ trace generating...")
    else:
        print(">> Skip trace generation and conversion.")

    logger = get_logger(json_args["log_filename"])
    logger.debug(" ".join(sys.argv))

    all_dp_group: List[List[int]] = []
    all_embed_group: List[List[int]] = []
    all_pos_embed_group: List[List[int]] = []
    all_pp_group: List[List[int]] = []

    primary_pp_group: List[int] = get_primary_pp_group(json_args)
    all_pp_group.append(primary_pp_group)
    # rank, [template_id, prev_rank, next_rank]
    template_mapping: Dict[int, List[int]] = {}

    args = json_args["megatron_args"]
    for i in range(args.world_size):
        # set up comm group
        destroy_model_parallel()
        initialize_model_parallel(
            world_size                              =   args.world_size,
            npu_rank                                =   i,
            tensor_model_parallel_size              =   args.tensor_model_parallel_size,
            pipeline_model_parallel_size            =   args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size    =   args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank      =   None, # only for "encoder and decoder" models
            context_parallel_size                   =   args.context_parallel_size,
            expert_model_parallel_size              =   1, # not apply yet
        )

        # trace generate
        if trace_generate and (
            (is_creating_template() and i in primary_pp_group) \
            or not is_creating_template()
        ):
            # init trace
            trace = PytorchPlusTrace()
            root_node = TraceNode("[pytorch|profiler|execution_graph|thread]").form_cpu_node().form_comp_node(1024, 1024)
            trace.add_node_with_ctrl_deps(root_node)

            # init p2p tag counter while creating template
            if is_creating_template():
                clear_send_recv_tag()
            
            train(trace=trace, rank=i, args=args)

            if not is_creating_template():
                check_send_recv_map_in_trace(trace, i)

            # # add dummy node to keep trace running
            # ctrl_node_list = ["[pytorch|profiler|execution_graph|thread]"]
            # dummy_node_list = []
            # for j in range(100):
            #     node = TraceNode("comp_dummy_node_"+str(j)).form_cpu_node().form_comp_node(1024, 1024)
            #     dummy_node_list.append(node)
            # merge_branch_nodes(ctrl_node_list, dummy_node_list, trace)

            # output in file
            output_file_is_template: str = ("~template~" if is_creating_template() else "")
            if json_args["is-simplified"]:
                output_file = os.path.join(
                    output_file_path, \
                    json_args["output_filename"] + output_file_is_template + f".{i}.et"
                )
                with open(output_file, "w") as f:
                    f.write(trace.get_simplified_trace())
            else:
                output_file = os.path.join(
                    output_file_path, \
                    json_args["output_filename"] + output_file_is_template + f"_{i}.json"
                )
                with open(output_file, "w") as f:
                    f.write(json.dumps(trace.get_json(), indent=4))
                
                et_convertor(
                    rank=i, 
                    json_output_file=output_file, 
                    output_file_path=output_file_path, 
                    logger=logger, 
                    json_args=json_args
                )

        #record pp dp embed pos-embed group
        this_pp_group = get_pipeline_model_parallel_group()
        this_dp_group = get_data_parallel_group()
        if this_pp_group not in all_pp_group:
            all_pp_group.append(this_pp_group)
        if this_dp_group not in all_dp_group:
            all_dp_group.append(this_dp_group)
        if is_embedding_group_initialized():
            this_embed_group = get_embedding_group()
            if this_embed_group not in all_embed_group:
                all_embed_group.append(this_embed_group)
        if is_position_embedding_group_initialized():
            this_pos_embed_group = get_position_embedding_group()
            if this_pos_embed_group not in all_pos_embed_group:
                all_pos_embed_group.append(this_pos_embed_group)
        
        #get template mapping
        if is_creating_template():
            index = this_pp_group.index(i)
            info = [primary_pp_group[index], get_pipeline_model_parallel_prev_rank(), get_pipeline_model_parallel_next_rank()]
            template_mapping[i] = info

    if trace_generate and not is_creating_template():
        check_send_recv_map(args.pipeline_model_parallel_size)

    # modify comm group for astra-sim
    json_args["comm_group"] = {
        "all_pp_group": [],
        "all_dp_group" : [],
        "all_embed_group" : [],
        "all_pos_embed_group" : []
    }
    json_args["comm_group"]["all_pp_group"] = all_pp_group
    json_args["comm_group"]["all_dp_group"] = all_dp_group
    json_args["comm_group"]["all_embed_group"] = all_embed_group
    json_args["comm_group"]["all_pos_embed_group"] = all_pos_embed_group

    json_args["template_mapping"] = (template_mapping if is_creating_template() else None)

    if trace_generate:
        print(">> Pytorch+ trace generation completed.")

# Function to calculate Floating-Point Operations for Transformer with MoE
def calculate_transformer_flops(
    args: Namespace, 
    enable_extra_param: bool = True
) -> Tuple[int, int]:
    
    # Attention projection size.
    kv_channels = args.hidden_size//args.num_attention_heads
    query_projection_size = kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size

    # Group Query Attention.
    if args.num_query_groups is None:
        args.num_query_groups = args.num_attention_heads

    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu == 1 else 1
    shared_expert_ffn_hidden_size = (
        0 if args.moe_shared_expert_intermediate_size is None else args.moe_shared_expert_intermediate_size
    )

    # Expansion factor, explained in the code comment.
    expansion_factor = 3 * 2 * 2

    # Calculating FLOPs.

    # attn & MLP & expert
    total_flops_without_embedding = (
        expansion_factor
        * args.global_batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (
                    1 + (args.num_query_groups / args.num_attention_heads) + (args.seq_length / args.hidden_size)
                ) * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            # Shared Experts.
            + ((shared_expert_ffn_hidden_size / args.hidden_size) * gated_linear_multiplier)
        )
    )
    # attn softmax
    attention_softmax_flops = (
        expansion_factor
        * args.global_batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            args.seq_length / ( args.hidden_size )
        )
    )
    # embedding
    embedding_flops = (
        expansion_factor
        * args.global_batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Logit.
            (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )
    
    stable_embedding_softmax_flops = (
        expansion_factor
        * args.global_batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Logit.
            (args.hidden_size / (2 * args.num_layers * args.hidden_size))
        )
        * args.stable_embedding
    )

    if enable_extra_param is False:
        attention_softmax_flops = 0
        stable_embedding_softmax_flops = 0
    else:
        global CALCULATE_TRANSFORMER_FLOPS_PARAM
        total_flops_without_embedding *= CALCULATE_TRANSFORMER_FLOPS_PARAM["total"]
        attention_softmax_flops *= CALCULATE_TRANSFORMER_FLOPS_PARAM["attn_softmax"]
        embedding_flops *= CALCULATE_TRANSFORMER_FLOPS_PARAM["embedding"]
        stable_embedding_softmax_flops *= CALCULATE_TRANSFORMER_FLOPS_PARAM["stable_embedding"]
    
    return (total_flops_without_embedding + 
            attention_softmax_flops, 
            embedding_flops + 
            stable_embedding_softmax_flops)

def calculate_transformer_param_num(
    args: Namespace, 
    verbose: bool = False
) -> int:
    # Attention projection size.
    kv_channels = args.hidden_size//args.num_attention_heads
    query_projection_size = kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size

    # Group Query Attention.
    if args.num_query_groups is None:
        args.num_query_groups = args.num_attention_heads

    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu == 1 else 1

    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            # Attention.
            (
                (1 + (args.num_query_groups / args.num_attention_heads))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            # Transformer layernorms.
            + (2 / args.hidden_size)
            # Final layernorm.
            + (1 / (args.num_layers * args.hidden_size))
        )
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"> Number of parameters in transformer layers in billions: "
            f"{num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"> Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"> Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    return num_parameters_in_transformer_layers, num_parameters_in_embedding_layers
