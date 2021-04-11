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

"""Pretrain T5"""

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.T5_dataset import build_train_valid_test_datasets
from megatron.model import T5ModelPipe, T5Model
from megatron.training import pretrain
from megatron.utils import get_masks_and_position_ids_for_t5
from megatron.utils import reduce_losses
from megatron.fp16 import fp32_to_fp16

def model_provider():
    """Build the model."""
    args = get_args()
    print_rank_0('building T5 model ...')
    if args.pipe_parallel_size == 0 or args.pipe_parallel_size == 1:
        model = T5Model(num_tokentypes=0, parallel_output=True)
    else:
        model = T5ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())
        model._megatron_batch_fn = get_batch_pipe
        model._input_grad = [True, False, True, False, False]
        model._input_type = ['float', 'int', 'float', 'int', 'int']
        model._input_pipe_partitioned = [True, False, True, False, False]
    return model

def get_batch(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = [
        "contexts",
        "targets",
        "labels",
        "ctx_eod_mask",
    ]
    datatype = torch.int64

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    
    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    contexts = data_b['contexts'].long()
    targets = data_b['targets'].long()
    labels = data_b['labels'].long()
    ctx_eod_mask = data_b['ctx_eod_mask'].long()

    # Unpack.
    enc_token_ids = contexts
    dec_token_ids = targets

    # Get the masks and postition ids.
    enc_attn_mask, enc_pos_ids, dec_attn_mask, dec_pos_ids, cross_attn_mask, loss_mask = get_masks_and_position_ids_for_t5(
        args,
        tokenizer,
        contexts,
        targets,
        labels,
        ctx_eod_mask,
        args.reset_position_ids,
        args.reset_attention_mask)

    if args.fp16:
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((enc_token_ids, enc_pos_ids, enc_attn_mask, 
                dec_token_ids, dec_pos_ids, dec_attn_mask,
                cross_attn_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (enc_token_ids, enc_pos_ids, enc_attn_mask, 
                dec_token_ids, dec_pos_ids, dec_attn_mask,
                cross_attn_mask), (labels, loss_mask)

def get_batch_pipe(data):
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = [
        "contexts",
        "targets",
        "labels",
        "ctx_eod_mask",
    ]
    datatype = torch.int64
    
    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    contexts = data_b['contexts'].long()
    targets = data_b['targets'].long()
    labels = data_b['labels'].long()
    ctx_eod_mask = data_b['ctx_eod_mask'].long()

    # Unpack.
    enc_token_ids = contexts
    dec_token_ids = targets

    # Get the masks and postition ids.
    enc_attn_mask, enc_pos_ids, dec_attn_mask, dec_pos_ids, cross_attn_mask, loss_mask = get_masks_and_position_ids_for_t5(
        args,
        tokenizer,
        contexts,
        targets,
        labels,
        ctx_eod_mask,
        args.reset_position_ids,
        args.reset_attention_mask)

    if args.fp16:
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((enc_token_ids, enc_pos_ids, enc_attn_mask, 
                dec_token_ids, dec_pos_ids, dec_attn_mask,
                cross_attn_mask)), fp32_to_fp16((labels, loss_mask))
    else:
        return (enc_token_ids, enc_pos_ids, enc_attn_mask, 
                dec_token_ids, dec_pos_ids, dec_attn_mask,
                cross_attn_mask), (labels, loss_mask)

def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()

    (enc_token_ids, enc_pos_ids, enc_attn_mask, 
     dec_token_ids, dec_pos_ids, dec_attn_mask,
     cross_attn_mask), (labels, loss_mask) = get_batch(data_iterator)
    
    timers('batch generator').stop()

    # Forward model.
    losses = model(enc_token_ids, enc_pos_ids, enc_attn_mask, 
                   dec_token_ids, dec_pos_ids, dec_attn_mask, cross_attn_mask, 
                   labels=labels)

    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}



def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets '
                 'for Enc-Dec ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        tokenizer=tokenizer,
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        enc_seq_length=args.enc_seq_length,
        dec_seq_length=args.dec_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating Enc-Dec datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'T5Tokenizer'})
