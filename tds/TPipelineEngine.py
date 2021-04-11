# Copyright 2021 The Tsinghua AI Team
# The original version comes from deepspeed.runtime.pipe.engine

import time
import logging
import copy
import os

from types import MethodType

from numpy import prod

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import deepspeed

from deepspeed.utils.logging import logger
from deepspeed.utils.timer import SynchronizedWallClockTimer, ThroughputTimer

from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.utils import PartitionedTensor, ensure_directory_exists
from deepspeed.runtime.dataloader import RepeatingLoader

from deepspeed.runtime.pipe.module import PipelineModule, PipelineError, TiedLayerSpec
from deepspeed.runtime.pipe import p2p
from deepspeed.runtime.pipe import schedule

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

def is_even(number):
    return number % 2 == 0

def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    def __init__(self, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
            " with pipeline parallelism."

        #  pipeline settings for sending and receiving tensors between stages
        self.input_grad = None
        self.input_type = None
        self.input_pipe_partitioned = None

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert self.train_batch_size() == \
            self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(batch_size=self.micro_batch_size *
                                           self.micro_batches,
                                           num_workers=self.dp_world_size,
                                           logging_fn=self.tput_log,
                                           monitor_memory=False,
                                           steps_per_output=self.steps_per_print())

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'output_tensors' : [], # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

    def _build_data_iter(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.dp_world_size,
            rank=self.mpu.get_data_parallel_rank(),
            shuffle=False)
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()
        self.module.allreduce_tied_weight_gradients()

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.is_data_parallel:
            self.buffered_allreduce_fallback(
                elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)
            # self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        if data_iter:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None

        # Do the work
        self.timers('train_batch').start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True)
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        # Tensorboard
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/train_loss',
                                        self.agg_train_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                if self.global_steps % self.steps_per_print() == 0:
                    self.summary_writer.flush()

        if self.wall_clock_breakdown(
        ) and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad'
            ])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(self, data_iter):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()
        self.total_loss = None

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        with torch.no_grad():
            self._exec_schedule(sched)

        self.agg_eval_loss = self._aggregate_total_loss()
        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/eval_loss',
                                        self.agg_eval_loss.mean().item(),
                                        self.global_samples)]
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])
                self.summary_writer.flush()

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        # Reset any buffers that may have been populated during the forward passes.
        #ds_checkpointing.reset()

        return self.agg_eval_loss

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss(self.total_loss)
            self.dp_group_loss = loss.clone().detach()
            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=self.mpu.get_pipe_parallel_group())
        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()
        return agg_loss

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        mp_rank = self.grid.get_slice_parallel_rank() \
            if self.is_model_parallel else 0
        batch = None
        # Only MP rank 0 loads the data.
        if mp_rank == 0:
            if self.data_iterator is None:
                raise ValueError(f"RANK={self.global_rank} no data iterator provided.")
            batch = next(self.data_iterator)
        # All MP ranks participate in batch_fn, where they might broadcast the data.
        if self.batch_fn:
            batch = self.batch_fn(batch)
        return batch

    def _merge_partitioned_grads(self, grad_layer):
        last_idx = 0
        _grad_tensors = []
        for idx, flag in enumerate(self.input_grad):
            if flag:
                if self.input_pipe_partitioned[idx]:
                    part_grad = PartitionedTensor.from_meta(
                        meta=self.grad_layer[last_idx],
                        local_part=self.grad_layer[last_idx+1],
                        group=self.grid.get_slice_parallel_group())
                    _grad_tensors.append(part_grad.full())
                    last_idx+=2
                    part_grad = None
                else:
                    _grad_tensors.append(self.grad_layer[last_idx])
                    last_idx+=1
        return tuple(_grad_tensors)

    def _merge_partitioned_tensors(self, outputs):
        last_idx = 0
        _outputs = []
        for flag in self.input_pipe_partitioned:
            if flag:
                part = PartitionedTensor.from_meta(
                    meta=outputs[last_idx],
                    local_part=outputs[last_idx + 1],
                    group=self.grid.get_slice_parallel_group())
                _outputs.append(part.full())
                last_idx += 2
                part = None
            else:
                _outputs.append(outputs[last_idx])
                last_idx += 1
        return tuple(_outputs)
    
    def _split_to_partitioned_tensors(self, outputs):
        _outputs = []
        _output_tensors = []
        for idx, flag in enumerate(self.input_pipe_partitioned):
            if flag:
                part = PartitionedTensor(tensor=outputs[idx],
                    group=self.grid.get_slice_parallel_group())
                _outputs.append(part.to_meta())
                _outputs.append(part.data())
                if self.input_grad[idx] and outputs[idx].is_floating_point():
                    outputs[idx].data = torch.zeros(1)
                    _output_tensors.append(outputs[idx])
                part = None
            else:
                _outputs.append(outputs[idx])
        return tuple(_output_tensors), tuple(_outputs)

    def _half_to_bool(self, buffers):
        if self.input_type is None or not 'bool' in self.input_type:
            return buffers
        if isinstance(buffers, torch.Tensor):
            return buffers.bool()
        buffers, buffers_is_tuple = self._list_or_tuple_to_list(buffers)
        is_pipe_partitioned = self.is_pipe_partitioned and self.input_pipe_partitioned is not None
        last_idx = 0
        for idx, buffer_type in enumerate(self.input_type):
            if is_pipe_partitioned and self.input_pipe_partitioned[idx]:
                last_idx += 1
            if buffer_type == 'bool':
                buffers[last_idx] = buffers[last_idx].bool()
            last_idx += 1
        buffers = self._list_to_list_or_tuple(buffers, buffers_is_tuple)
        return buffers

    def _bool_to_half(self, buffers):
        if self.input_type is None or not 'bool' in self.input_type:
            return buffers
        if isinstance(buffers, torch.Tensor):
            return buffers.half()
        buffers, buffers_is_tuple = self._list_or_tuple_to_list(buffers)
        is_pipe_partitioned = self.is_pipe_partitioned and self.input_pipe_partitioned is not None
        last_idx = 0
        for idx, buffer_type in enumerate(self.input_type):
            if is_pipe_partitioned and self.input_pipe_partitioned[idx]:
                last_idx += 1
            if buffer_type == 'bool':
                buffers[last_idx] = buffers[last_idx].half()
            last_idx += 1
        buffers = self._list_to_list_or_tuple(buffers, buffers_is_tuple)
        return buffers

    def _set_requires_grad(self, buffers):
        if isinstance(buffers, torch.Tensor):
            buffers.requires_grad = buffers.is_floating_point() and \
                                    (self.input_grad is None or self.input_grad[0])
        elif isinstance(buffers, (list, tuple)):
            buffers, buffers_is_tuple = self._list_or_tuple_to_list(buffers)
            for idx, buffer in enumerate(buffers):
                buffer.requires_grad = buffer.is_floating_point() and \
                                    (self.input_grad is None or self.input_grad[idx])
            buffers = self._list_to_list_or_tuple(buffers, buffers_is_tuple)
        return buffers

    def _list_or_tuple_to_list(self, buffers):
        if isinstance(buffers, tuple):
            return (list)(buffers), True
        else:
            assert isinstance(buffers, list)
            return buffers, False

    def _list_to_list_or_tuple(self, buffers, buffers_is_tuple = True):
        assert isinstance(buffers, list)
        if buffers_is_tuple:
            return tuple(buffers)
        else:
            return buffers

    def _dtype_to_id(self, dtype):
        lists = [torch.bool, torch.float16, torch.float32, torch.float64, torch.int16, torch.int32, torch.int64]
        for idx, torch_type in enumerate(lists):
            if dtype is torch_type:
                return idx
        raise NotImplementedError(f'Could not support meta type {dtype}')

    def _id_to_dtype(self, id):
        lists = [torch.bool, torch.float16, torch.float32, torch.float64, torch.int16, torch.int32, torch.int64]
        assert id < len(lists), 'Could not support meta type of {id}'
        return lists[id]

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list, 2: tupe)
            * num_tensors if type = list or type = tuple 
            for each tensor in buffer:
                * (ndims, dtype_id)
                * shape
        """
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size()), self._dtype_to_id(buffer.dtype)]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
        elif isinstance(buffer, list) or isinstance(buffer, tuple):
            if isinstance(buffer, list):
                type_tensor = torch.LongTensor(data=[1]).to(self.device)
            else:
                type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size()), self._dtype_to_id(tensor.dtype)]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()
        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0,0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims, recv_dtype_id = recv_ndims[0].item(), recv_ndims[1].item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, recv_dtype_id, num_buffers=1)[0]
        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes = []
            recv_dtype_ids = []
            for idx in range(num_tensors):
                recv_ndims = torch.LongTensor(data=[0,0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims, recv_dtype_id = recv_ndims[0].item(), recv_ndims[1].item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes.append(recv_shape.tolist())
                recv_dtype_ids.append(recv_dtype_id)

            buffers = self._allocate_buffers(recv_shapes, recv_dtype_ids, num_buffers=1)[0]
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers
        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()

        inputs = self.pipe_buffers['inputs'][buffer_id]
        self._zero_grads(inputs)

        outputs = super().forward(inputs)
        if self.is_pipe_partitioned and not self.is_last_stage():
            self.pipe_buffers['output_tensors'][buffer_id], outputs = \
                self._split_to_partitioned_tensors(outputs)

        self.pipe_buffers['outputs'][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self.loss_model is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                self.loss = self.loss_model(outputs, labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs

            if isinstance(self.loss, torch.Tensor):
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        # The last stage just runs backward on the loss using DeepSpeed's typical mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            return

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]
        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            outputs = self._merge_partitioned_tensors(outputs)
            out_tensors = []
            last_idx = 0
            for idx, flag in enumerate(self.input_grad):
                if flag:
                    self.pipe_buffers['output_tensors'][buffer_id][last_idx].data = outputs[idx]
                    out_tensors.append(self.pipe_buffers['output_tensors'][buffer_id][last_idx])
                    last_idx += 1
        elif isinstance(outputs, (tuple, list)):
            out_tensors = [t for idx, t in enumerate(outputs)  
             if (self.input_grad is None or self.input_grad[idx]) and t.is_floating_point()]
        else:
            out_tensors = (outputs,)

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            grad_tensors = self._merge_partitioned_grads(grad_tensors)  

        torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        outputs = None
        out_tensors = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if isinstance(batch[0], torch.Tensor):
                loaded = batch[0].clone().to(self.device).detach()
            else:
                assert isinstance(batch[0], (tuple, list))
                loaded = []
                for buffer in batch[0]:
                    assert isinstance(buffer, torch.Tensor)
                    mine = buffer.clone().detach().to(self.device)
                    loaded.append(mine)
                loaded = tuple(loaded)
            # loaded = self._set_requires_grad(loaded)
            self.pipe_buffers['inputs'][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if isinstance(batch[1], torch.Tensor):
                loaded = batch[1].clone().to(self.device).detach()
            else:
                assert isinstance(batch[1], (tuple, list))
                loaded = []
                for buffer in batch[1]:
                    assert isinstance(buffer, torch.Tensor)
                    buffer = buffer.clone().detach().to(self.device)
                    loaded.append(buffer)
                loaded = tuple(loaded)
            self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        outputs = self._bool_to_half(outputs)
        
        if self.first_output_send:
            self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for buffer in outputs:
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type ' f'{type(outputs)}')

        outputs = self._half_to_bool(outputs)

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            if self.is_grad_partitioned:
                part = PartitionedTensor(tensor=inputs.grad,
                                     group=self.grid.get_slice_parallel_group())
                inputs = tuple([part.to_meta(), part.data()])
                part = None
            else:
                inputs = inputs.grad
        else:
            _inputs = []
            for idx, flag in enumerate(self.input_grad):
                if flag:
                    assert inputs[idx].grad is not None
                    if self.is_grad_partitioned and self.input_pipe_partitioned[idx]:
                        part = PartitionedTensor(tensor=inputs[idx].grad,
                                     group=self.grid.get_slice_parallel_group())
                        _inputs.append(part.to_meta())
                        _inputs.append(part.data())
                        part = None        
                    else:
                        _inputs.append(inputs[idx].grad)
                else:
                    assert inputs[idx].grad is None
            inputs = tuple(_inputs)

        if self.first_gradient_send:
            self.first_gradient_send = False
            self._send_tensor_meta(inputs, self.prev_stage)

        if isinstance(inputs, torch.Tensor):
            p2p.send(inputs, self.prev_stage)
        elif isinstance(inputs, tuple):
            for buffer in inputs:
                p2p.send(buffer, self.prev_stage)
        else:
            raise NotImplementedError('Could not send input of type ' f'{type(inputs)}')

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None
        inputs = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recvd = None
        # allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
        
        # receive data
        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert isinstance(buffer, torch.Tensor)
                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()
            recvd = tuple(self._half_to_bool(recvd))
        
        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            recvd = self._merge_partitioned_tensors(recvd)
        recvd = self._set_requires_grad(recvd)
        self.pipe_buffers['inputs'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        # Allocate the buffer if necessary
        if self.grad_layer is None:
            self.grad_layer = self._recv_tensor_meta(self.next_stage)

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(self.grad_layer, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        if self.tensorboard_enabled():
            if self.global_rank == 0:
                self.summary_events = [(f'Train/Samples/lr',
                                        self.get_lr()[0],
                                        self.global_samples)]
                if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                    self.summary_events.append((f'Train/Samples/loss_scale',
                                                self.optimizer.cur_scale,
                                                self.global_samples))
                for event in self.summary_events:  # write_summary_events
                    self.summary_writer.add_scalar(event[0], event[1], event[2])

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'backward_allreduce_microstep',
                    'backward_tied_allreduce_microstep',
                    'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'step'
                ])

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, dtype_id, fp16=None, **kwargs):
        if fp16 is None:
            fp16 = self.fp16_enabled()

        if dtype_id <= 3:
            if fp16:
                return torch.zeros(shape, dtype=torch.half, device=self.device, **kwargs)
            else:
                return torch.zeros(shape, device=self.device, **kwargs)
        else:
            return torch.zeros(shape, dtype=torch.long, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, dtype_id, num_buffers=-1, **kwargs):
        buffers = []

        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers

        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, dtype_id, **kwargs))

        return buffers

    def _allocate_buffers(self, shapes, dtype_ids, num_buffers=-1, **kwargs):
        buffers = []

        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers

        for count in range(num_buffers):
            buffer = []
            for shape, dtype_id in zip(shapes, dtype_ids):
                buffer.append(self._allocate_zeros(shape, dtype_id, **kwargs))
            buffers.append(buffer)

        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path)
        return None

    def load_module_state_dict(self, state_dict, strict=True):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path, strict=strict)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}'
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self._exec_instr(**cmd.kwargs)

    def set_dataloader(self, loader):
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        self.batch_fn = fn
    
    def set_input_grad(self, input_grad):
        self.input_grad = input_grad

    def set_input_type(self, input_type):
        self.input_type = input_type

    def set_input_pipe_partitioned(self, input_pipe_partitioned):
        self.input_pipe_partitioned = input_pipe_partitioned

    def is_gradient_accumulation_boundary(self):
        return self._force_grad_boundary

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1
