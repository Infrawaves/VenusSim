from typing import Any, Dict, List
from generator.generator import PytorchPlusTrace, TraceNode
from argparse import Namespace
import copy

FLOPS_PER_FWD_NODE = None
EMBEDDING_PER_FWD_NODE = None
USE_SNAPSHOT = None
SNAPSHOT_DATA = None

comp_unique_id: int = 0

def set_flops_per_fwd_node(flops_per_fwd_node: int):
    global FLOPS_PER_FWD_NODE
    FLOPS_PER_FWD_NODE = flops_per_fwd_node

def get_flops_per_fwd_node():
    return FLOPS_PER_FWD_NODE

def set_embedding_per_fwd_node(embedding_per_fwd_node: int):
    global EMBEDDING_PER_FWD_NODE
    EMBEDDING_PER_FWD_NODE = embedding_per_fwd_node

def get_embedding_per_fwd_node():
    return EMBEDDING_PER_FWD_NODE

def set_snapshot_data(snapshot_data: List[int]):
    global SNAPSHOT_DATA
    SNAPSHOT_DATA = copy.deepcopy(snapshot_data)

def get_snapshot_data():
    return SNAPSHOT_DATA

def set_use_snapshot(use_snapshot: bool):
    global USE_SNAPSHOT
    USE_SNAPSHOT = use_snapshot

def is_use_snapshot():
    return USE_SNAPSHOT


def get_computation_backward(forward_computation):
    """
    根据前向计算量，计算后向计算量。

    参数:
    - forward_time: 前向计算量

    返回:
    - backward_time: 后向计算量 (为前向计算量的 2 倍)
    """
    return 2 * forward_computation

def form_comp_node(
    comp_name: str = "", 
    args: Namespace = Namespace(), 
    is_backward: bool = False, 
    is_first_pp_stage: bool = False,
    is_last_pp_stage: bool = False
) -> TraceNode:
    global comp_unique_id
    comp_unique_id += 1
    '''
    # old version
    assert args.num_layers % (args.pipeline_model_parallel_size*args.virtual_pipeline_model_parallel_size) == 0, "form_comp_node: num_layers_per_chunk is not int."
    assert args.num_attention_heads % args.num_query_groups == 0, "form_comp_node: r is not int."
    if args.sequence_parallel:
        assert args.seq_length % args['tp_world_size'] == 0, \
            "form_comp_node: seq_length is not int while enabling sequence parallel."
    tensor_size = get_computation_forward(
        batch_size              =   args.micro_batch_size,
        num_layers_per_chunk    =   args.num_layers // (args.pipeline_model_parallel_size*args.virtual_pipeline_model_parallel_size),
        seq_length              =   args.seq_length // (args['tp_world_size'] if args.sequence_parallel else 1),
        hidden_dim              =   args.hidden_size,
        ffn_hidden              =   args.ffn_hidden_size,
        V                       =   args.vocab_size,
        r                       =   args.num_attention_heads // args.num_query_groups,
        tp                      =   args.tensor_model_parallel_size
    )
    if has_embedding:
        tensor_size = tensor_size[0]
    else:
        tensor_size = tensor_size[1]

    if is_backward:
        tensor_size = get_computation_backward(tensor_size)
    '''
    if is_use_snapshot():
        dur: int = -1
        if is_first_pp_stage:
            dur = get_snapshot_data()[0]
        elif is_last_pp_stage:
            dur = get_snapshot_data()[2]
        else:
            dur = get_snapshot_data()[1]
        if is_backward:
            dur = dur*1E6*(2/3)
        else:
            dur = dur*1E6*(1/3)
        comp_node = TraceNode(f"comp_replay_{comp_unique_id}_{comp_name}").form_replay_node(int(dur))
    else:
        num_ops = FLOPS_PER_FWD_NODE + (EMBEDDING_PER_FWD_NODE if (is_first_pp_stage or is_last_pp_stage) else 0)
        if is_backward:
            num_ops = get_computation_backward(num_ops)
        
        comp_node = TraceNode(f"comp_{comp_unique_id}_{comp_name}").form_comp_node(int(num_ops), int(num_ops))
    # trace.add_node_with_ctrl_deps(comp_node, ctrl_deps)
    return comp_node

def form_comp_node_custom(comp_name: str = "", num_ops: int = 1, tensor_size: int = 1):
    global comp_unique_id
    comp_unique_id += 1
    comp_node = TraceNode(f"comp_{comp_unique_id}_{comp_name}").form_comp_node(num_ops, tensor_size)
    # trace.add_node_with_ctrl_deps(comp_node, ctrl_deps)
    return comp_node

def merge_branch_nodes(
    ctrl_deps: List[str], 
    branch_nodes: List[TraceNode], 
    trace: PytorchPlusTrace
):
    '''
    branch_nodes: [b-0, b-1, b-2, ..., b-z]
    trace: [n-0, n-1, n-2, ..., n-m]
    ctrl_deps: [n-0, n-1]
    
    after merge:
    trace:  [n-0, n-1, n-2, ..., n-m]--------[merge node]
              |    |                      |
              --------[b-0, ..., b-z]------
    '''
    assert len(branch_nodes) >= 1 and len(ctrl_deps) >= 1, \
        "branch_nodes or ctrl_deps is illegal."
    last_node_name_before_merge = trace.get_last_node_name()
    merge_node_ctrl_name_list: List[str] = [last_node_name_before_merge]

    for i in range(len(branch_nodes)):
        branch_node = branch_nodes[i]
        if i == 0:
            trace.add_node_with_ctrl_deps(branch_node, ctrl_deps)
        else:
            trace.add_node_with_ctrl_deps(branch_node, [branch_nodes[i-1].name])
            
    merge_node_ctrl_name_list.append(branch_nodes[-1].name)

    merge_node = form_comp_node_custom(comp_name="merge_branch", num_ops=1, tensor_size=1)
    trace.add_node_with_ctrl_deps(merge_node, merge_node_ctrl_name_list)