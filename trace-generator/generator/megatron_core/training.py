from typing import Dict, Any
from generator.generator import PytorchPlusTrace
from generator.overlap_grad_reduce_helper import OverlapGradReduceHelper
from generator.megatron_core.distrib_optimizer import DistributedOptimizer
from generator.train_step_hook_helper import TrainStepHookHelper
from generator.megatron_core.schedules import get_forward_backward_func
from argparse import Namespace

ITERATION: int
def get_iteration():
    return ITERATION

def train_step(
    # forward_step_func, data_iterator,
    # model, optimizer, opt_param_scheduler, config
    trace: PytorchPlusTrace, 
    args: Namespace = Namespace(),
):
    """Single training step."""
    # args = get_args()
    # timers = get_timers()

    # Set grad to zero.
    # for model_chunk in model:
    #     model_chunk.zero_grad_buffer()
    # optimizer.zero_grad()
    distributed_optimizer: DistributedOptimizer = args.distributed_optimizer
    distributed_optimizer.zero_grad(trace=trace)


    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    # losses_reduced = forward_backward_func(
    #     forward_step_func=forward_step_func,
    #     data_iterator=data_iterator,
    #     model=model,
    #     num_microbatches=get_num_microbatches(),
    #     seq_length=args.seq_length,
    #     micro_batch_size=args.micro_batch_size,
    #     decoder_seq_length=args.decoder_seq_length,
    #     forward_only=False)
    forward_backward_func(
        num_microbatches    =   args.num_micro_batches, 
        seq_length          =   args.seq_length, 
        micro_batch_size    =   args.micro_batch_size, 
        decoder_seq_length  =   args.decoder_seq_length, 
        forward_only        =   False, 
        args                =   args,
        trace               =   trace
    )

    # Empty unused memory.
    # if args.empty_unused_memory_level >= 1:
    #     torch.cuda.empty_cache()

    # Vision gradients.
    # if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
    #     unwrapped_model = unwrap_model(model[0])
    #     unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    # timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
    # update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    distributed_optimizer.step(trace=trace)
    # timers('optimizer').stop()

    # Vision momentum.
    # if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
    #     unwrapped_model = unwrap_model(model[0])
    #     unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    # if update_successful:
    #     increment = get_num_microbatches() * \
    #                 args.micro_batch_size * \
    #                 args.data_parallel_size
    #     opt_param_scheduler.step(increment=increment)
    #     skipped_iter = 0
    # else:
    #     skipped_iter = 1

    # Empty unused memory.
    # if args.empty_unused_memory_level >= 2:
    #     torch.cuda.empty_cache()

    # if mpu.is_pipeline_last_stage(ignore_virtual=True):
    #     # Average loss across microbatches.
    #     loss_reduced = {}
    #     for key in losses_reduced[0]:
    #         losses_reduced_for_key = [x[key] for x in losses_reduced]
    #         loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
    #     return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
    # return {}, skipped_iter, grad_norm, num_zeros_in_grad

def train(
    # forward_step_func, model, optimizer, opt_param_scheduler,
    # train_data_iterator, valid_data_iterator,
    # process_non_loss_data_func, config,
    trace: PytorchPlusTrace, 
    rank: int, 
    args: Namespace
):
    """Train the model function."""
    # args = get_args()
    # timers = get_timers()

    # Write args to tensorboard
    # write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    # for model_module in model:
    #     model_module.train()

    # Tracking loss.
    # total_loss_dict = {}

    # Iterations.
    # iteration = args.iteration
    global ITERATION
    ITERATION = 0
    # one_logger = get_one_logger()
    # if one_logger:
    #     iteration_start = iteration
    #     train_samples_start = args.consumed_train_samples
    #     train_samples_target = args.train_samples
    #     one_logger.log_metrics({
    #         'train_samples_start': args.consumed_train_samples,
    #         'train_iterations_start': iteration,
    #         'train_samples_target': train_samples_target,
    #         'train_iterations_target': args.train_iters,
    #     })

    # num_floating_point_operations_so_far = args.num_floating_point_operations_so_far

    # Setup some training config params
    # new param in simulation
    args.train_step_hook_helper = TrainStepHookHelper()
    args.overlap_grad_reduce_helper = OverlapGradReduceHelper(rank, args, False)
    args.distributed_optimizer = DistributedOptimizer(rank, args, False)

    args.train_step_hook_helper.set_forward_pre_hook(
        hook_name="distributed_optimizer_fph",
        hook=args.distributed_optimizer.get_pre_hook()
    )

    args.train_step_hook_helper.set_backward_hook(
        hook_name="overlap_grad_reduce_helper_bh",
        hook=args.overlap_grad_reduce_helper.backward_hook
    )
    # config.grad_scale_func = optimizer.scale_loss
    # config.timers = timers
    # if isinstance(model[0], DDP) and args.overlap_grad_reduce:
    args.grad_sync_func = None
    if args.overlap_grad_reduce:
        # assert config.no_sync_func is None, \
        #     ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
        #         'a custom no_sync_func is not supported when overlapping grad-reduce')
        # config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        # if len(model) == 1:
        #     config.no_sync_func = config.no_sync_func[0]

        # if args.delay_grad_reduce:
        if args.delay_grad_reduce:
            # config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            # if len(model) == 1:
            #     config.grad_sync_func = config.grad_sync_func[0]
            args.grad_sync_func = args.overlap_grad_reduce_helper.start_grad_sync

    # TODO: not support yet
    # if args.overlap_param_gather and args.delay_param_gather:
    if args.overlap_param_gather and args.delay_param_gather:
    #     config.param_sync_func = [lambda x: optimizer.finish_param_sync(model_index, x)
    #                                 for model_index in range(len(model))]
        args.param_sync_func = args.distributed_optimizer.finish_param_sync
    #     if len(model) == 1:
    #         config.param_sync_func = config.param_sync_func[0]
    else:
        args.param_sync_func = None
    args.finalize_model_grads_func = args.overlap_grad_reduce_helper.finalize_model_grads

    # timers('interval-time', log_level=0).start(barrier=True)
    # print_datetime('before the start of training step')
    # report_memory_flag = True
    # exit = False

    # if args.manual_gc:
    #     # Disable the default garbage collector and perform the collection manually.
    #     # This is to align the timing of garbage collection across ranks.
    #     assert args.manual_gc_interval >= 0, \
    #         'Manual garbage collection interval should be laerger than or equal to 0.'
    #     gc.disable()
    #     gc.collect()

    # num_microbatches = get_num_microbatches()
    # eval_duration = 0.0
    # eval_iterations = 0
    # def track_e2e_metrics():
    #     # Nested function to track a bunch of E2E APP metrics
    #     if one_logger:
    #         train_duration = timers('interval-time').active_time()  # overall_elapsed
    #         train_samples = args.consumed_train_samples - train_samples_start
    #         train_iterations = iteration - iteration_start
    #         train_iterations_time_msecs_avg = (train_duration * 1000.0) / train_iterations
    #         if eval_iterations:
    #             validation_iterations_time_msecs_avg = (eval_duration * 1000.0) / eval_iterations
    #         else:
    #             validation_iterations_time_msecs_avg = None

    #         one_logger.log_metrics({
    #             'train_iterations_end': iteration,
    #             'train_samples_end': args.consumed_train_samples,
    #             'train_iterations': train_iterations,
    #             'train_samples': train_samples,
    #             'train_iterations_time_msecs_avg': train_iterations_time_msecs_avg,
    #             'validation_iterations_time_msecs_avg': validation_iterations_time_msecs_avg
    #         })

    # while iteration < args.train_iters:
    while ITERATION < args.train_iters:
        # if args.profile and \
        #    iteration == args.profile_step_start and \
        #    torch.distributed.get_rank() in args.profile_ranks:
        #     torch.cuda.cudart().cudaProfilerStart()
        #     torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        # update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        # if get_num_microbatches() != num_microbatches and iteration != 0:
        #     assert get_num_microbatches() > num_microbatches, \
        #         "number of microbatches should be increasing due to batch size rampup"
        #     save_checkpoint_and_time(iteration, model, optimizer,
        #                              opt_param_scheduler,
        #                              num_floating_point_operations_so_far)
        # num_microbatches = get_num_microbatches()
        # update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        # args.curr_iteration = iteration
        # loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
        #     train_step(forward_step_func,
        #                train_data_iterator,
        #                model,
        #                optimizer,
        #                opt_param_scheduler,
        #                config)
        train_step(trace=trace, args=args)
        # iteration += 1
        ITERATION += 1
        # batch_size = mpu.get_data_parallel_world_size() * \
        #              args.micro_batch_size * \
        #              get_num_microbatches()
        # args.consumed_train_samples += batch_size
        # num_floating_point_operations_so_far += num_floating_point_operations(args, batch_size)

        # Logging.
    #     loss_scale = optimizer.get_loss_scale().item()
    #     params_norm = None
    #     if args.log_params_norm:
    #         params_norm = calc_params_l2_norm(model)

    #     if iteration % args.log_interval == 0:
    #         track_e2e_metrics()

    #     report_memory_flag = training_log(loss_dict, total_loss_dict,
    #                                       optimizer.param_groups[0]['lr'],
    #                                       iteration, loss_scale,
    #                                       report_memory_flag, skipped_iter,
    #                                       grad_norm, params_norm, num_zeros_in_grad)

    #     # Autoresume
    #     if args.adlr_autoresume and \
    #        (iteration % args.adlr_autoresume_interval == 0):
    #         check_adlr_autoresume_termination(iteration, model, optimizer,
    #                                           opt_param_scheduler)

    #     # Evaluation
    #     if args.eval_interval and iteration % args.eval_interval == 0 and \
    #        args.do_valid:
    #         timers('interval-time').stop()
    #         if args.use_distributed_optimizer and args.overlap_param_gather:
    #             optimizer.disable_pre_hook()
    #         if args.manual_gc and args.manual_gc_eval:
    #             # Collect all objects.
    #             gc.collect()
    #         prefix = 'iteration {}'.format(iteration)
    #         timers('eval-time', log_level=0).start(barrier=True)
    #         evaluate_and_print_results(prefix, forward_step_func,
    #                                    valid_data_iterator, model,
    #                                    iteration, process_non_loss_data_func,
    #                                    config, False)
    #         eval_duration += timers('eval-time').elapsed()
    #         eval_iterations += args.eval_iters
    #         timers('eval-time').stop()
    #         if args.manual_gc and args.manual_gc_eval:
    #             # Collect only the objects created and used in evaluation.
    #             gc.collect(generation=0)
    #         if args.use_distributed_optimizer and args.overlap_param_gather:
    #             optimizer.enable_pre_hook()
    #         timers('interval-time', log_level=0).start(barrier=True)

    #     # Checkpointing
    #     saved_checkpoint = False
    #     if args.exit_signal_handler:
    #         signal_handler = get_signal_handler()
    #         if any(signal_handler.signals_received()):
    #             save_checkpoint_and_time(iteration, model, optimizer,
    #                                      opt_param_scheduler,
    #                                      num_floating_point_operations_so_far)
    #             print_datetime('exiting program after receiving SIGTERM.')
    #             exit = True
    #             break

    #     if args.save and args.save_interval and \
    #        iteration % args.save_interval == 0:
    #         timers('interval-time').stop()
    #         save_checkpoint_and_time(iteration, model, optimizer,
    #                                  opt_param_scheduler,
    #                                  num_floating_point_operations_so_far)
    #         saved_checkpoint = True
    #         timers('interval-time', log_level=0).start(barrier=True)

    #     # Exiting based on duration
    #     if args.exit_duration_in_mins:
    #         train_time = (time.time() - _TRAIN_START_TIME) / 60.0
    #         done_cuda = torch.tensor(
    #             [train_time > args.exit_duration_in_mins],
    #             dtype=torch.int, device='cuda')
    #         torch.distributed.all_reduce(
    #             done_cuda, op=torch.distributed.ReduceOp.MAX)
    #         done = done_cuda.item()
    #         if done:
    #             if not saved_checkpoint:
    #                 save_checkpoint_and_time(iteration, model, optimizer,
    #                                          opt_param_scheduler,
    #                                          num_floating_point_operations_so_far)
    #             print_datetime('exiting program after {} minutes'.format(train_time))
    #             exit = True
    #             break

    #     # Exiting based on iterations
    #     if args.exit_interval and iteration % args.exit_interval == 0:
    #         if args.save and not saved_checkpoint:
    #             save_checkpoint_and_time(iteration, model, optimizer,
    #                                      opt_param_scheduler,
    #                                      num_floating_point_operations_so_far)
    #         torch.distributed.barrier()
    #         print_datetime('exiting program at iteration {}'.format(iteration))
    #         exit = True
    #         break

    #     if args.profile and \
    #        iteration == args.profile_step_end and \
    #        torch.distributed.get_rank() in args.profile_ranks:
    #         torch.cuda.cudart().cudaProfilerStop()

    #     if args.manual_gc:
    #         if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
    #             gc.collect()

    # track_e2e_metrics()

    # # Flush TensorBoard and WandB writers.
    # writer = get_tensorboard_writer()
    # if writer:
    #     writer.flush()
    # wandb_writer = get_wandb_writer()
    # if wandb_writer:
    #     wandb_writer.finish()

    # # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    # if args.use_distributed_optimizer and args.overlap_param_gather:
    #     optimizer.disable_pre_hook()

    # # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    # if exit:
    #     sys.exit()

    # return iteration, num_floating_point_operations_so_far