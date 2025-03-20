# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# import operator
# from functools import reduce
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from generator.generator import TraceNode, PytorchPlusTrace

# import torch

# from megatron import core
# from megatron.core import ModelParallelConfig
from generator.megatron_core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_global_rank
)
import generator.megatron_core.parallel_state as parallel_state

# Types
# Shape = Union[List[int], torch.Size]
Shape = Union[List[int], None]

# format: {rank_id: {(src, dst): tag_num}}
send_map: Dict[int, Dict[Tuple, int]] = {}
recv_map: Dict[int, Dict[Tuple, int]] = {}
comp_unique_id: int = 0

CREATING_TEMPLATE: bool = False

def is_creating_template():
    global CREATING_TEMPLATE
    return CREATING_TEMPLATE

def set_creating_template():
    global CREATING_TEMPLATE
    CREATING_TEMPLATE = True

def unset_creating_template():
    global CREATING_TEMPLATE
    CREATING_TEMPLATE = False

'''
def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape
'''

'''
def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs
'''

def get_shape_elem_num_helper(tensor_shape: List[int]) -> int:
    if len(tensor_shape) == 0:
        return 0
    elem_num = 1
    for dim_size in tensor_shape:
        elem_num *= dim_size
    return elem_num

def get_send_recv_tag(src: int, dst: int, tag_map: Dict[Tuple, int]) -> int:
    if (src, dst) not in tag_map.keys():
        tag_map[(src, dst)] = 0
    tag = tag_map[(src, dst)]
    tag_map[(src, dst)] += 1
    return tag

def clear_send_recv_tag() -> None:
    global send_map, recv_map
    send_map.clear()
    recv_map.clear()

def get_send_recv_node_name(
    src:int, dst: int, tag: int, 
    is_send: bool = False, is_recv:bool = False
) -> str:
    if(is_send):
        return f"nccl:send_from_{src}_to_{dst}_tag_{tag}"
    if(is_recv):
        return f"nccl:recv_from_{src}_to_{dst}_tag_{tag}"
    assert False, "Should not reach here."

def form_send_node(src: int, dst: int, num: int, byte: int) -> TraceNode:
    global send_map
    # get unique tag
    if src not in send_map.keys():
        send_map[src] = {}
    tag = get_send_recv_tag(src, dst, send_map[src])
    # get node
    send_node = TraceNode(get_send_recv_node_name(src, dst, tag, is_send=True)).form_comm_send_node(src, dst, tag, num, byte)
    # if overlap, insert to trace later
    # if not is_overlap:
    #     trace.add_node_with_ctrl_deps(send_node)
    # else:
    #     ctrl_node_name: str = trace.get_last_node_name()
    #     wait_handles.append((ctrl_node_name, send_node))
    return send_node

def form_recv_node(src: int, dst: int, num: int, byte: int) -> TraceNode:
    global recv_map
    # get unique tag
    if dst not in recv_map.keys():
        recv_map[dst] = {}
    tag = get_send_recv_tag(src, dst, recv_map[dst])
    # get node
    recv_node = TraceNode(get_send_recv_node_name(src, dst, tag, is_recv=True)).form_comm_recv_node(src, dst, tag, num, byte)
    # if overlap, insert to trace later
    # if not is_overlap:
    #     trace.add_node_with_ctrl_deps(recv_node)
    # else:
    #     ctrl_node_name: str = trace.get_last_node_name()
    #     wait_handles.append((ctrl_node_name, recv_node))
    return recv_node

# old version
# def compute_time_forward(batch_size: int, num_layers: int, seq_length: int, hidden_dim: int, ffn_hidden: int, V: int, r: int, pp: int, tp: int, peak_compute: float, mfu: float) -> float:
#     """
#     计算Transformer模型的迭代计算量,分别返回:
#     1. 带附加项2sbhV/tp的前向计算量。
#     2. 不带附加项的前向计算量。
    
#     参数:
#     - batch_size: 批次大小 (b)
#     - num_layers: Transformer模型的层数 (L)
#     - seq_length: 序列长度 (s) (rank实际处理的序列长度)
#     - hidden_dim: 隐藏层维度 (h)
#     - ffn_hidden: FFN层的隐藏维度 (f)
#     - V: 词表大小
#     - r: num_head/num_query 新的系数 (r)
#     - pp: 流水线并行大小
#     - tp: 张量并行大小
#     - peak_compute: 峰值算力 (Peak)
#     - mfu: 百分数

#     返回:
#     - 带附加项的前向计算量
#     - 不带附加项的前向计算量
#     """
#     # 检查输入参数是否有效
#     if pp <= 0 or tp <= 0 or r <= 0:
#         raise ValueError("参数pp、tp和r必须大于0")
    
#     if peak_compute <= 0 or mfu <= 0:
#         raise ValueError("峰值算力和MFU必须大于0")
    
#     # 计算附加项 2sbhV / tp
#     additional_term = 2 * seq_length * batch_size * hidden_dim * V / tp
    
#     # 计算公式中的四项
#     inner_expression = (4 * seq_length * batch_size * hidden_dim**2 / r +  # 第一项
#                         4 * seq_length * batch_size * hidden_dim**2 +     # 第二项
#                         4 * seq_length**2 * batch_size * hidden_dim +     # 第三项
#                         4 * seq_length * batch_size * hidden_dim * ffn_hidden)  # 第四项
    
#     # 计算不带附加项的计算量
#     total_computation_without_additional = (num_layers / pp) * (inner_expression / tp)
    
#     # 计算带附加项的计算量
#     total_computation_with_additional = (num_layers / pp) * ((inner_expression / tp) + additional_term)
    
#     # 分别将两者除以峰值算力和MFU，并再除以10^12
#     result_with_additional = total_computation_with_additional / (peak_compute * mfu) / 1e12
#     result_without_additional = total_computation_without_additional / (peak_compute * mfu) / 1e12
    
#     return result_with_additional, result_without_additional

# old version
# def compute_backward_time(forward_time):
#     """
#     根据前向计算时间，计算后向计算的时间。

#     参数:
#     - forward_time: 前向计算的时间

#     返回:
#     - backward_time: 后向计算的时间 (为前向计算时间的 2 倍)
#     """
#     return 2 * forward_time

# old version
# def get_computation_forward(batch_size: int, num_layers_per_chunk: int, seq_length: int, hidden_dim: int, ffn_hidden: int, V: int, r: int, tp: int) -> int:
#     """
#     计算Transformer模型的迭代计算量,分别返回:
#     1. 带附加项2sbhV/tp的前向计算量。
#     2. 不带附加项的前向计算量。
    
#     参数:
#     - batch_size: 批次大小 (b)
#     - num_layers: Transformer模型的层数 (L)
#     - seq_length: 序列长度 (s) (rank实际处理的序列长度)
#     - hidden_dim: 隐藏层维度 (h)
#     - ffn_hidden: FFN层的隐藏维度 (f)
#     - V: 词表大小
#     - r: num_head/num_query (r)
#     - pp: 流水线并行大小
#     - tp: 张量并行大小

#     返回:
#     - 带附加项的前向计算量
#     - 不带附加项的前向计算量
#     """
#     # 检查输入参数是否有效
#     if tp <= 0 or r <= 0:
#         raise ValueError("参数tp和r必须大于0")
    
#     # if peak_compute <= 0 or mfu <= 0:
#     #     raise ValueError("峰值算力和MFU必须大于0")
    
#     # 计算附加项 2sbhV / tp
#     assert V % tp == 0, "get_computation_forward: V % tp != 0."
#     additional_term = 2 * seq_length * batch_size * hidden_dim * V // tp
    
#     # 计算公式中的四项
#     assert 4 * seq_length * batch_size * hidden_dim**2 % r == 0, "get_computation_forward: 4 * seq_length * batch_size * hidden_dim**2 % r != 0."
#     inner_expression = (4 * seq_length * batch_size * hidden_dim**2 // r +  # 第一项
#                         4 * seq_length * batch_size * hidden_dim**2 +     # 第二项
#                         4 * seq_length**2 * batch_size * hidden_dim +     # 第三项
#                         4 * seq_length * batch_size * hidden_dim * ffn_hidden)  # 第四项
    
#     # 计算不带附加项的计算量
#     assert inner_expression % tp == 0, "get_computation_forward: inner_expression % tp != 0."
#     total_computation_without_additional = num_layers_per_chunk * (inner_expression // tp)
    
#     # 计算带附加项的计算量
#     total_computation_with_additional = num_layers_per_chunk * ((inner_expression // tp) + additional_term)
    
#     return total_computation_with_additional, total_computation_without_additional


def _p2p_ops(
    *,
    tensor_send_prev: Optional[int],
    tensor_recv_prev: Optional[int],
    tensor_send_next: Optional[int],
    tensor_recv_next: Optional[int],
    group: List[int]
) -> List[TraceNode]:
    
    reqs: List[TraceNode] = []
    if is_creating_template():
        self_rank = 0
        next_rank = 1
        prev_rank = -1
    else:
        self_rank = get_global_rank()
        next_rank = get_pipeline_model_parallel_next_rank()
        prev_rank = get_pipeline_model_parallel_prev_rank()

    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            # send_next_req = torch.distributed.isend(
            #     tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            # )
            send_next_req = form_send_node(self_rank, next_rank, tensor_send_next, 2)
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            # recv_prev_req = torch.distributed.irecv(
            #     tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            # )
            recv_prev_req = form_recv_node(prev_rank, self_rank, tensor_recv_prev, 2)
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            # send_prev_req = torch.distributed.isend(
            #     tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            # )
            send_prev_req = form_send_node(self_rank, prev_rank, tensor_send_prev, 2)
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            # recv_next_req = torch.distributed.irecv(
            #     tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            # )
            recv_next_req = form_recv_node(next_rank, self_rank, tensor_recv_next, 2)
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            # recv_prev_req = torch.distributed.irecv(
            #     tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            # )
            recv_prev_req = form_recv_node(prev_rank, self_rank, tensor_recv_prev, 2)
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            # send_next_req = torch.distributed.isend(
            #     tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            # )
            send_next_req = form_send_node(self_rank, next_rank, tensor_send_next, 2)
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            # recv_next_req = torch.distributed.irecv(
            #     tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            # )
            recv_next_req = form_recv_node(next_rank, self_rank, tensor_recv_next, 2)
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            # send_prev_req = torch.distributed.isend(
            #     tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            # )
            send_prev_req = form_send_node(self_rank, prev_rank, tensor_send_prev, 2)
            reqs.append(send_prev_req)
    return reqs

def _communicate(
    *,
    tensor_send_next: Optional[int],
    tensor_send_prev: Optional[int],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    # config: ModelParallelConfig,
    wait_on_reqs: bool = True,
    trace: PytorchPlusTrace
) -> Tuple[Optional[int], Optional[int], Tuple[str,Optional[List[TraceNode]]]]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # if not config.variable_seq_lengths:
    #     recv_prev_shape = tensor_shape
    #     recv_next_shape = tensor_shape
    # else:
    #     recv_prev_shape, recv_next_shape = _communicate_shapes(
    #         tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
    #     )

    # assert config.variable_seq_lengths
    recv_prev_shape = tensor_shape
    recv_next_shape = tensor_shape

    if recv_prev:
        # if config.pipeline_dtype is None:
        #     raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        # tensor_recv_prev = torch.empty(
        #     recv_prev_shape,
        #     requires_grad=True,
        #     device=torch.cuda.current_device(),
        #     dtype=config.pipeline_dtype,
        # )
        # tensor_recv_prev = np.zeros(tuple(recv_prev_shape))
        tensor_recv_prev = get_shape_elem_num_helper(recv_prev_shape)
    if recv_next:
        # if config.pipeline_dtype is None:
        #     raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        # tensor_recv_next = torch.empty(
        #     recv_next_shape,
        #     requires_grad=True,
        #     device=torch.cuda.current_device(),
        #     dtype=config.pipeline_dtype,
        # )
        # tensor_recv_next = np.zeros(tuple(recv_next_shape))
        tensor_recv_next = get_shape_elem_num_helper(recv_next_shape)

    # Send tensors in both the forward and backward directions as appropriate.
    # if config.use_ring_exchange_p2p:

    #     def _ring_exchange_wrapper(**kwargs):
    #         torch.distributed.ring_exchange(**kwargs)
    #         return []

    #     p2p_func = _ring_exchange_wrapper
    # elif config.batch_p2p_comm:
    #     assert wait_on_reqs
    #     p2p_func = _batched_p2p_ops
    # else:
    #     p2p_func = _p2p_ops
    p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=get_pipeline_model_parallel_group(),
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            trace.add_node_with_ctrl_deps(req)
        reqs = None

    # if config.batch_p2p_comm and config.batch_p2p_sync:
    #     # To protect against race condition when using batch_isend_irecv().
    #     # User should assert that we have a modern enough PyTorch to not need this
    #     torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, (trace.get_last_node_name(), reqs)

def recv_forward(
    tensor_shape: Shape, 
    # config: ModelParallelConfig,
    trace: PytorchPlusTrace
) -> int:
    """ Receive tensor from previous rank in pipeline (forward receive).


    See _communicate for argument details.
    """

    if parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        # if config.timers is not None:
        #     config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            # config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('forward-recv').stop()
    return input_tensor

def recv_backward(
    tensor_shape: Shape, 
    # config: ModelParallelConfig,
    trace: PytorchPlusTrace
) -> int:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        # if config.timers is not None:
        #     config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            # config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('backward-recv').stop()
    return output_tensor_grad

def send_forward(
    output_tensor: int, 
    # config: ModelParallelConfig
    trace: PytorchPlusTrace
) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not parallel_state.is_pipeline_last_stage():
        # if config.timers is not None:
        #     config.timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            # config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('forward-send').stop()

def send_backward(
    input_tensor_grad: int,
    # config: ModelParallelConfig
    trace: PytorchPlusTrace
) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not parallel_state.is_pipeline_first_stage():
        # if config.timers is not None:
        #     config.timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            # config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('backward-send').stop()

def send_forward_recv_backward(
    output_tensor: int,
    tensor_shape: Shape, 
    # config: ModelParallelConfig
    trace: PytorchPlusTrace
) -> int:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        # if config.timers is not None:
        #     config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            #config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad

def send_backward_recv_forward(
    input_tensor_grad: int,
    tensor_shape: Shape, 
    # config: ModelParallelConfig
    trace: PytorchPlusTrace
) -> int:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        # if config.timers is not None:
        #     config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            # config=config,
            trace=trace
        )
        # if config.timers is not None:
        #     config.timers('backward-send-forward-recv').stop()
    return input_tensor

def send_forward_recv_forward(
    output_tensor: int,
    recv_prev: bool,
    tensor_shape: Shape,
    # config: ModelParallelConfig,
    trace: PytorchPlusTrace, 
    overlap_p2p_comm: bool = False,
) -> Tuple[int, Tuple[str, List[TraceNode]]]:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    # if config.timers is not None:
    #     config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        # config=config,
        trace=trace
    )
    # if config.timers is not None:
    #     config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor, ()

def send_backward_recv_backward(
    input_tensor_grad: int,
    recv_next: bool,
    tensor_shape: Shape,
    # config: ModelParallelConfig,
    trace: PytorchPlusTrace, 
    overlap_p2p_comm: bool = False,
) -> Tuple[int, Tuple[str, List[TraceNode]]]:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    # if config.timers is not None:
    #     config.timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        # config=config,
        trace=trace
    )
    # if config.timers is not None:
    #     config.timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad, ()

def send_forward_backward_recv_forward_backward(
    output_tensor: int,
    input_tensor_grad: int,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    # config: ModelParallelConfig,
    trace: PytorchPlusTrace
) -> Tuple[int, int]:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    # if config.timers is not None:
    #     config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        # config=config,
        trace=trace
    )
    # if config.timers is not None:
    #     config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad

def check_send_recv_map_in_trace(
    trace: PytorchPlusTrace,
    rank: int
):
    global send_map, recv_map
    assert rank in send_map.keys(), f"rank {rank} not found in send_map."
    assert rank in recv_map.keys(), f"rank {rank} not found in recv_map."
    for (src, dst) in send_map[rank]:
        for tag in range(send_map[rank][(src, dst)]):
            assert trace.contain_node(get_send_recv_node_name(src, dst, tag, is_send=True))
    for (src, dst) in recv_map[rank]:
        for tag in range(recv_map[rank][(src, dst)]):
            assert trace.contain_node(get_send_recv_node_name(src, dst, tag, is_recv=True))

def check_send_recv_map(check_world_size: int):
    global send_map, recv_map
    assert len(send_map) >= check_world_size, "send_map is too small."
    assert len(recv_map) >= check_world_size, "recv_map is too small."
    for rank in range(check_world_size):
        check_send_map: Dict[Tuple, int] = send_map[rank]
        for (src, dst) in check_send_map.keys():
            if src != rank:
                print(f"Wrong src in rank {rank}, (src: {src}, dst: {dst}) pair.")
            if (src, dst) not in recv_map[dst]:
                print(f"(src: {src}, dst: {dst}) pair not found in recv map {dst}.")
            if check_send_map[(src, dst)] != recv_map[dst][(src, dst)]:
                print(f"(src: {src}, dst: {dst}) pair num not match for {check_send_map[(src, dst)]} and {recv_map[dst][(src, dst)]}.")
        check_recv_map: Dict[Tuple, int] = recv_map[rank]
        for (src, dst) in check_recv_map.keys():
            if dst != rank:
                print(f"Wrong dst in rank {rank}, (src: {src}, dst: {dst}) pair.")
            if (src, dst) not in send_map[src]:
                print(f"(src: {src}, dst: {dst}) pair not found in send map {src}.")
            if check_recv_map[(src, dst)] != send_map[src][(src, dst)]:
                print(f"(src: {src}, dst: {dst}) pair num not match for {check_recv_map[(src, dst)]} and {send_map[src][(src, dst)]}.")