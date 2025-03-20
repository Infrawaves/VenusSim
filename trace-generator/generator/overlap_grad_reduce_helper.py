from typing import Any, Dict, List, Tuple
from generator.generator import PytorchPlusTrace, TraceNode
from generator.utils import (
    form_comp_node_custom,
    merge_branch_nodes
)
import generator.megatron_core.parallel_state as parallel_state
from argparse import Namespace

GRAD_REDUCE_SCATTER_NUM = None
EMBEDDING_GRAD_ALLREDUCE_NUM = None

# param in model
SHARE_EMBEDDINGS_AND_OUTPUT_WEIGHTS = None

def set_grad_reduce_scatter_num(param_num):
    global GRAD_REDUCE_SCATTER_NUM
    GRAD_REDUCE_SCATTER_NUM = param_num

def get_grad_reduce_scatter_num():
    global GRAD_REDUCE_SCATTER_NUM
    return GRAD_REDUCE_SCATTER_NUM

def set_embedding_grad_allreduce_num(param_num):
    global EMBEDDING_GRAD_ALLREDUCE_NUM
    EMBEDDING_GRAD_ALLREDUCE_NUM = param_num

def get_embedding_grad_allreduce_num():
    global EMBEDDING_GRAD_ALLREDUCE_NUM
    return EMBEDDING_GRAD_ALLREDUCE_NUM

def set_share_embeddings_and_output_weights(share: bool):
    global SHARE_EMBEDDINGS_AND_OUTPUT_WEIGHTS
    SHARE_EMBEDDINGS_AND_OUTPUT_WEIGHTS = share

def get_share_embeddings_and_output_weights():
    global SHARE_EMBEDDINGS_AND_OUTPUT_WEIGHTS
    return SHARE_EMBEDDINGS_AND_OUTPUT_WEIGHTS

class OverlapGradReduceHelper():
    def __init__(self, rank:int, args: Namespace, debug_mode: bool = False) -> None:
        self.rank = rank
        self.use_distributed_optimizer: bool = args.use_distributed_optimizer
        self.overlap_grad_reduce: bool = args.overlap_grad_reduce

        # {model_chunk_id: is_last_micro_batch}
        self.is_last_micro_batch: Dict[int, bool] = {}
        # {model_chunk_id: communication_issued}
        self.communication_issued: Dict[int, bool] = {}
        # {model_chunk_id: (ctrl_nodes, comm_node)}
        self.communication_handles: Dict[int, Tuple[str, TraceNode]] = {}

        num_model_chunk = (
            1 if args.virtual_pipeline_model_parallel_size is None 
            else args.virtual_pipeline_model_parallel_size
        )
        for model_chunk_id in range(num_model_chunk):
            self.is_last_micro_batch[model_chunk_id] = False
            self.communication_issued[model_chunk_id] = False
        self.debug_mode = debug_mode
        if self.debug_mode:
            print(f"OverlapGradReduceHelper {rank} init end")
        self.skip_grad_sync = (args.data_parallel_size == 1)
    
    def backward_hook(
        self, 
        model_chunk_id: int,
        trace: PytorchPlusTrace
    ):
        if self.overlap_grad_reduce:
            if self.is_last_micro_batch[model_chunk_id]:
                self.start_grad_sync(model_chunk_id, trace)

    def get_grad_reduce_scatter_node(
        self,
        model_chunk_id: int
    ) -> TraceNode:
        from generator.megatron_core.training import get_iteration
        if self.skip_grad_sync:
            node = TraceNode(
                "skip_grad_sync" + 
                "~iter~" + str(get_iteration()) +
                "~pp_rank~" + str(parallel_state.get_pipeline_model_parallel_rank()) + 
                "~tp_rank~" + str(parallel_state.get_tensor_model_parallel_rank()) + 
                "~model_chunk_id~" + str(model_chunk_id)
                # "~unique~" + str(grad_rs_unique_id[self.rank])
            )
            node.form_comp_node(1, 1)
            return node
        else:
            node = TraceNode(
                "nccl:grad_reducescatter" + 
                "~iter~" + str(get_iteration()) +
                "~pp_rank~" + str(parallel_state.get_pipeline_model_parallel_rank()) + 
                "~tp_rank~" + str(parallel_state.get_tensor_model_parallel_rank()) + 
                "~model_chunk_id~" + str(model_chunk_id)
                # "~unique~" + str(grad_rs_unique_id[self.rank])
            )
            node.form_comm_coll_node(int(get_grad_reduce_scatter_num()), 4, "dp_group")
            return node
    
    def get_embedding_grad_allreduce_node(
        self
    ) -> TraceNode:
        from generator.megatron_core.training import get_iteration
        node = TraceNode(
            "nccl:embedding_grad_allreduce" + 
            "~iter~" + str(get_iteration()) +
            "~pp_rank~" + str(parallel_state.get_pipeline_model_parallel_rank()) + 
            "~tp_rank~" + str(parallel_state.get_tensor_model_parallel_rank())
        )
        node.form_comm_coll_node(int(get_embedding_grad_allreduce_num()), 4, "embedding_group")
        return node

    def communication_handle_wait(
        self, 
        model_chunk_id: int, 
        trace: PytorchPlusTrace
    ):  
        merge_branch_nodes(
            ctrl_deps=[self.communication_handles[model_chunk_id][0]],
            branch_nodes=[self.communication_handles[model_chunk_id][1]],
            trace=trace
        )
        self.communication_issued[model_chunk_id] = False
        self.communication_handles.pop(model_chunk_id)
        if self.debug_mode:
            print(f"finish communication handle wait for rank {self.rank}, model chunk {model_chunk_id}")

    def start_grad_sync(
        self,
        model_chunk_id: int,
        trace: PytorchPlusTrace
    ) -> None:
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        # assert (
        #     self.communication_handle is None and not self.communication_issued
        # ), 'Should not have multiple communication calls in flight at once'
        assert (
            model_chunk_id not in self.communication_handles.keys() and
            not self.communication_issued[ model_chunk_id]
        ), f'Should not have multiple communication calls in flight at once in rank {self.rank}, model chunk: {model_chunk_id}.'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        # if self.check_for_nan_in_grad:
        #     global_rank = torch.distributed.get_rank()
        #     norm = self.grad_data.norm(p=2)
        #     assert not norm.isnan(), (
        #         f'Rank {global_rank}: found NaN in local grad norm in '
        #         f'backward pass before data-parallel communication collective. '
        #         f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        #     )

        # self.grad_data *= self.gradient_scaling_factor

        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
        #     local_data_view = shard_buffer(self.grad_data, self.data_parallel_world_size)[
        #         self.data_parallel_rank
        #     ]

        #     self.communication_handle = torch.distributed._reduce_scatter_base(
        #         local_data_view,
        #         self.grad_data,
        #         group=self.data_parallel_group,
        #         async_op=self.overlap_grad_reduce,
        #     )
            grad_rs_node = self.get_grad_reduce_scatter_node(model_chunk_id)
            if self.overlap_grad_reduce:
                ctrl_node = trace.get_last_node_name()
                self.communication_handles[model_chunk_id] = (ctrl_node, grad_rs_node)
                self.communication_issued[model_chunk_id] = True
            else:
                trace.add_node_with_ctrl_deps(grad_rs_node)
        else:
        #     self.communication_handle = torch.distributed.all_reduce(
        #         self.grad_data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
        #     )
            assert False, (
                "Only distributed optimzer is support yet."
            )
        if self.debug_mode:
            print(f"start grad sync for rank {self.rank}, model chunk {model_chunk_id}")
        # self.communication_issued = True

    def finish_grad_sync(
        self, 
        model_chunk_id: int,
        trace: PytorchPlusTrace
    ) -> None:
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        # if not self.overlap_grad_reduce:
        #     self.start_grad_sync()
        #     return
        if not self.overlap_grad_reduce:
            self.start_grad_sync(model_chunk_id, trace)
            return
        
        # assert self.communication_handle is not None and self.communication_issued, (
        #     f'Communication call has not been issued for this bucket '
        #     f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        # )
        assert (
            model_chunk_id in self.communication_handles.keys() and
            model_chunk_id in self.communication_issued.keys() and
            self.communication_issued[model_chunk_id]
        ), (
            f'Communication call has not been issued for model chunk: {model_chunk_id}.'
        )

        # self.communication_handle.wait()
        self.communication_handle_wait(model_chunk_id, trace)
        if self.debug_mode:
            print(f"finish grad sync for rank {self.rank}, model chunk {model_chunk_id}")

    def _allreduce_word_embedding_grads(
        # model: List[torch.nn.Module], 
        # config: TransformerConfig,
        self, 
        model_chunk_id_list: List[int],
        trace: PytorchPlusTrace
    ):
        """
        All-reduce word embedding grads.

        Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
        sync. This should only run for models that support pipelined model parallelism (BERT and GPT).
        """
        # print(parallel_state.get_global_rank(), parallel_state.is_rank_in_embedding_group(ignore_virtual=True))
        if (
            parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
            and parallel_state.get_pipeline_model_parallel_world_size() > 1
        ):
            # if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            #     model_module = model[0]
            # elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            #     model_module = model[-1]
            # else:  # We do not support the interleaved schedule for T5 yet.
            #     model_module = model[0]

            # Look for module with 'pre_process' attribute to get around the fact that DDP and
            # other wrapper classes inherit from non-core MegatronModule that has
            # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
            # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
            # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
            # model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
            
            # if model_module.share_embeddings_and_output_weights:
            if get_share_embeddings_and_output_weights():
                # weight = model_module.shared_embedding_or_output_weight()
                # grad = weight.main_grad
                # torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())
                comm_node = self.get_embedding_grad_allreduce_node()
                trace.add_node_with_ctrl_deps(comm_node)


    # def _allreduce_position_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    #     """
    #     All-reduce position_embeddings grad across first (encoder) and split (decoder) stages to
    #     ensure that position embeddings parameters stay in sync. This should only run for T5 models
    #     with pipeline parallelism.
    #     """
    #     if (
    #         parallel_state.is_rank_in_position_embedding_group()
    #         and parallel_state.get_pipeline_model_parallel_world_size() > 1
    #         and config.pipeline_model_parallel_split_rank is not None
    #     ):
    #         model_module = model[0]
    #         grad = get_attr_wrapped_model(
    #             model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
    #         )
    #         torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


    def _allreduce_embedding_grads(
        # model: List[torch.nn.Module], 
        # config: TransformerConfig,
        self, 
        model_chunk_id_list: List[int],
        trace: PytorchPlusTrace
    ):
        """
        All-reduce both word and position embeddings.
        """
        # _allreduce_word_embedding_grads(model, config)
        self._allreduce_word_embedding_grads(model_chunk_id_list, trace)

        # This should only run for T5 models.
        # _allreduce_position_embedding_grads(model, config)
        # self._allreduce_position_embedding_grads(model_chunk_id_list, trace)

    def finalize_model_grads(
        # model: List[torch.nn.Module],
        self, 
        model_chunk_id_list: List[int],
        trace: PytorchPlusTrace
    ) -> None:
        """
        All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
        embedding grads across first and last pipeline stages (if not tied).
        """

        # config = get_model_config(model[0])

        # All-reduce / reduce-scatter across DP replicas.
        # if config.timers is not None:
        #     config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)

        # for model_chunk in model:
        #     model_chunk.finish_grad_sync()
        for model_chunk_id in model_chunk_id_list:
            self.finish_grad_sync(model_chunk_id, trace=trace)
        
        # if config.timers is not None:
        #     config.timers('all-grads-sync').stop()

        # All-reduce layer-norm grads (for sequence parallelism).
        # if config.timers is not None:
        #     config.timers('layernorm-grads-all-reduce', log_level=1).start(
        #         barrier=config.barrier_with_L1_time
        #     )

        # TODO: _allreduce_layernorm_grads(model, config)

        # if config.timers is not None:
        #     config.timers('layernorm-grads-all-reduce').stop()

        # All-reduce embedding grads (for pipeline parallelism).
        # if config.timers is not None:
        #     config.timers('embedding-grads-all-reduce', log_level=1).start(
        #         barrier=config.barrier_with_L1_time
        #     )

        # _allreduce_embedding_grads(model, config)
        self._allreduce_embedding_grads(model_chunk_id_list, trace)

        # if config.timers is not None:
        #     config.timers('embedding-grads-all-reduce').stop()
        if self.debug_mode:
            print(f"finish finalize model grads for rank {self.rank}, model chunk {model_chunk_id}")

    def disable_grad_sync(
        self,
        model_chunk_id: int
    ):
        assert model_chunk_id in self.is_last_micro_batch.keys(), \
            f"{model_chunk_id} not found in self.is_last_micro_batch."
        self.is_last_micro_batch[model_chunk_id] = False
    
    def enable_grad_sync(
        self,
        model_chunk_id: int
    ):
        assert model_chunk_id in self.is_last_micro_batch.keys(), \
            f"{model_chunk_id} not found in self.is_last_micro_batch."
        self.is_last_micro_batch[model_chunk_id] = True
