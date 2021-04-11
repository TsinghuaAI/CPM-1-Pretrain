# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utilities for Megatron."""

import sys

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_adlr_autoresume
from megatron import mpu
from megatron.checkpointing import save_checkpoint
from megatron.data.samplers import DistributedBatchSampler
from megatron.fp16 import FP16_Optimizer


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()

    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print_rank_0(string)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    if isinstance(optimizer, FP16_Optimizer):
        optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, lr_scheduler):
    """Check for autoresume signal and exit if it is received."""
    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_masks_and_position_ids_for_t5(args,
                               tokenizer,
                               contexts,
                               targets,
                               labels,
                               ctx_eod_mask,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.ones(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    # Enc Position ids.
    enc_pos_ids = torch.arange(
        enc_seq_length, dtype=torch.long, device=contexts.device)
    enc_pos_ids = enc_pos_ids.unsqueeze(0).expand_as(contexts)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        enc_pos_ids = enc_pos_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            eod_pos = ctx_eod_mask[b].nonzero(as_tuple=False)
            prev_index = 0
            for i in eod_pos:
                if i < enc_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                            enc_attn_mask[b, 0, i+1:, :i+1] = 0
                            enc_attn_mask[b, 0, :i+1, i+1:] = 0
                    # Reset positions.
                    if reset_position_ids:
                        enc_pos_ids[b, i+1:] -= (i + 1 - prev_index)
                        prev_index = i + 1

    
    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

    # Dec Position ids.
    dec_pos_ids = torch.arange(
        dec_seq_length, dtype=torch.long, device=targets.device)
    dec_pos_ids = dec_pos_ids.unsqueeze(0).expand_as(targets)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        dec_pos_ids = dec_pos_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            eod_pos = (targets[b] == tokenizer.eod_id).nonzero(as_tuple=False)
            prev_index = 0
            for i in eod_pos:
                if i < dec_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                        dec_attn_mask[b, 0, i+1:, :i+1] = 0
                        dec_attn_mask[b, 0, :i+1, i+1:] = 0
                    # Reset positions.
                    if reset_position_ids:
                        dec_pos_ids[b, i+1:] -= (i + 1 - prev_index)
                        prev_index = i + 1

    # Loss mask.
    loss_mask = torch.ones(targets.size(), dtype=torch.float, device=targets.device)
    loss_mask[targets == tokenizer.eod_id] = 0.0
    loss_mask[labels == tokenizer.pad_id] = 0.0

    # Cross Attention Mask
    cross_attn_mask = torch.ones(
        batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            enc_eod_pos = ctx_eod_mask[b].nonzero(as_tuple=False)
            dec_eod_pos = (targets[b] == tokenizer.eod_id).nonzero(as_tuple=False)
            assert len(enc_eod_pos) == len(dec_eod_pos), (enc_eod_pos, dec_eod_pos)
            for enc_i, dec_i in zip(enc_eod_pos, dec_eod_pos):
                if enc_i < enc_seq_length and dec_i < dec_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                        cross_attn_mask[b, 0, dec_i+1:, :enc_i+1] = 0
                        cross_attn_mask[b, 0, :dec_i+1, enc_i+1:] = 0
    # if args.fp16:
    #     enc_attn_mask = enc_attn_mask.half()
    #     dec_attn_mask = dec_attn_mask.half()
    #     cross_attn_mask = cross_attn_mask.half()
    # model_batch = {
    #     "enc_attention_mask": enc_attn_mask,
    #     "enc_position_ids": enc_pos_ids,
    #     "dec_attention_mask": dec_attn_mask,
    #     "dec_position_ids": dec_pos_ids,
    #     "cross_attention_mask": cross_attn_mask,
    # }
    # no_model_batch = {
    #     "loss_mask": loss_mask
    # }
    return enc_attn_mask, enc_pos_ids, dec_attn_mask, dec_pos_ids, cross_attn_mask, loss_mask

