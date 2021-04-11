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

"""Transformer."""

import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from megatron.mpu import LayerNorm
from megatron.module import MegatronModule
from megatron.checkpointing import get_checkpoint_version
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl, gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu

import tds as deepspeed

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size,
            args.ff_hidden_size,
            bias = False,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ff_hidden_size,
            args.hidden_size,
            bias = False,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)
        self.hidden_bias = args.hidden_bias

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            if self.hidden_bias:
                intermediate_parallel = \
                        bias_gelu_impl(intermediate_parallel, bias_parallel)
            else:
                intermediate_parallel = \
                        gelu_impl(intermediate_parallel)
        else:
            if self.hidden_bias:
                intermediate_parallel = \
                    self.activation_func(intermediate_parallel + bias_parallel)
            else:
                intermediate_parallel = \
                    self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelSelfAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number,
                 is_cross_flag:False):
        super(ParallelSelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.is_cross_flag = is_cross_flag

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        d_attn_out = args.kv_hidden_size * args.num_attention_heads

        self.hidden_size_per_partition = mpu.divide(d_attn_out, world_size)
        self.hidden_size_per_attention_head = mpu.divide(d_attn_out, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(args.num_attention_heads, world_size)

        # Strided linear layer.
        if self.is_cross_flag:
            self.only_query = mpu.ColumnParallelLinear(
                args.hidden_size,
                1 * d_attn_out,
                bias = args.hidden_bias,
                gather_output=False,
                init_method=init_method, stride=1)
            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * d_attn_out,
                bias = args.hidden_bias,
                gather_output=False,
                init_method=init_method, stride=2)
        else:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * d_attn_out,
                bias = args.hidden_bias,
                gather_output=False,
                init_method=init_method, stride=3)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            args.scaled_upper_triang_masked_softmax_fusion,
            args.scaled_masked_softmax_fusion,
            self.attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output.
        self.dense = mpu.RowParallelLinear(
            d_attn_out,
            args.hidden_size,
            bias = args.hidden_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        input_shape = mixed_layer.size();
        if num_splits_first:
            """[s, b, num_splits * np * hn] 
            -->(view) [s, b, num_splits, np, hn] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (num_splits, self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous()
        else:
            """[s, b, np * hn * num_splits] 
            -->(view) [s, b, np, hn, num_splits] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head, num_splits)

            mixed_layer = mixed_layer.view(*intermediate_shape)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous()
        mixed_layer = mixed_layer.view(*input_shape)
        
        return mixed_layer

    def forward(self, hidden_states, attention_mask, 
                key_value_states=None,
                layer_past=None,
                get_key_value=False):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        if self.is_cross_flag:
            mixed_x_layer, _ = self.key_value(key_value_states)
            query_layer, _ = self.only_query(hidden_states)
            checkpoint_version = get_checkpoint_version()
            if checkpoint_version is not None:
                if checkpoint_version == 0:
                    # [s, b, (3 * np * hn)] --> [s, b, (np * 3 * hn)]
                    mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 2, True)
                    query_layer = self._transpose_last_dim(query_layer, 1, True)
                elif checkpoint_version == 1.0:
                    # [s, b, (np * hn * 3)] --> [s, b, (np * 3 * hn)]
                    mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 2, False)
                    query_layer = self._transpose_last_dim(query_layer, 1, False)
            # [sq, b, (np * 2 * hn)] --> [sq, b, np, 2 * hn]
            mixed_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads_per_partition, 2 * self.hidden_size_per_attention_head)
            query_tensor_shape = query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
           
            mixed_x_layer = mixed_x_layer.view(*mixed_tensor_shape)
            query_layer = query_layer.view(*query_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (key_layer, value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 2)
        else:
            mixed_x_layer, _ = self.query_key_value(hidden_states)
            checkpoint_version = get_checkpoint_version()
            if checkpoint_version is not None:
                if checkpoint_version == 0:
                    # [s, b, (3 * np * hn)] --> [s, b, (np * 3 * hn)]
                    mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
                elif checkpoint_version == 1.0:
                    # [s, b, (np * hn * 3)] --> [s, b, (np * 3 * hn)]
                    mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, False)
            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
            key_layer,
            value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), 
                       key_layer.size(0))
        
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0,1).transpose(1, 2),  #[b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)


        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out

def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add

def new_dropout_add(x, residual, prob, training) :
    # type: (Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def get_dropout_add(training):
    def _dropout_add(x, residual, prob):
        return new_dropout_add(x, residual, prob, training)
    return _dropout_add

@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)

@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)

@torch.jit.script
def dropout_add_fused_train(x, residual, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return new_dropout_add(x, residual, prob, True)

@torch.jit.script
def dropout_add_fused_inference(x, residual, prob):
    # type: (Tensor, Tensor, float) -> Tensor
    return new_dropout_add(x, residual, prob, False)

class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number, is_cross_flag = False):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()

        self.is_cross_flag = is_cross_flag
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(attention_mask_func, init_method,
                                               output_layer_init_method,
                                               layer_number, is_cross_flag = False)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.hidden_bias = args.hidden_bias

        if self.is_cross_flag:
            self.cross_attention = ParallelSelfAttention(attention_mask_func, init_method,
                                                output_layer_init_method,
                                                layer_number, is_cross_flag = True)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)

    def forward(self, enc_hidden_states, encoder_attention_mask=None, 
                      dec_hidden_states=None, decoder_attention_mask=None, 
                      cross_attention_mask=None,
                      layer_past=None, get_key_value=False):

        # set attention mask for encoder and decoder
        if self.is_cross_flag:
            attention_mask = decoder_attention_mask
            cross_attention_mask = cross_attention_mask
            hidden_states = dec_hidden_states
            key_value_states = enc_hidden_states
        else:
            attention_mask = encoder_attention_mask
            cross_attention_mask = None
            hidden_states = enc_hidden_states
            key_value_states = None

        # hidden_states: [b, s, h]
        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        _layer_past = layer_past[0] if layer_past is not None else None
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask, key_value_states,
                           layer_past = _layer_past,
                           get_key_value=get_key_value)
        if get_key_value:
            attention_output, presents = attention_output
        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.

        if self.hidden_bias:
            if self.bias_dropout_fusion:
                if self.training:
                    common_bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    common_bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                common_bias_dropout_add_func = get_bias_dropout_add(self.training)
        else:
            if self.bias_dropout_fusion:
                if self.training:
                    common_bias_dropout_add_func = dropout_add_fused_train
                else:
                    common_bias_dropout_add_func = dropout_add_fused_inference
            else:
                common_bias_dropout_add_func = get_dropout_add(self.training)
        
        with torch.enable_grad():
            if self.hidden_bias:
                layernorm_input = common_bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
            else:
                layernorm_input = common_bias_dropout_add_func(
                    attention_output,
                    residual,
                    self.hidden_dropout)
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # cross attention
        if self.is_cross_flag:
            _layer_past = layer_past[-1] if layer_past is not None else None
            attention_output, attention_bias = \
                self.cross_attention(layernorm_output,
                            cross_attention_mask, key_value_states,
                            layer_past=_layer_past,
                            get_key_value=get_key_value)
            if get_key_value:
                attention_output, cross_presents = attention_output
            # Residual connection.
            residual = layernorm_input
            # jit scripting for a nn.module (with dropout) is not 
            # trigerring the fusion kernel. For now, we use two 
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            with torch.enable_grad():
                if self.hidden_bias:
                    layernorm_input = common_bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(residual),
                        residual,
                        self.hidden_dropout)
                else:
                    layernorm_input = common_bias_dropout_add_func(
                        attention_output,
                        residual,
                        self.hidden_dropout)
            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        # Second residual connection.
        residual = layernorm_input
        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            if self.hidden_bias:
                output = common_bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
            else:
                output = common_bias_dropout_add_func(
                    mlp_output,
                    residual,
                    self.hidden_dropout)

        if get_key_value:
            if self.is_cross_flag:
                output = [output, presents, cross_presents]
            else:
                output = [output, presents]
        return output

class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline. """
    def forward(self, args):
        enc_hidden_states, encoder_attention_mask, dec_hidden_states, decoder_attention_mask, cross_attention_mask = args[0], args[1], args[2], args[3], args[4]
        res = super().forward(*args)
        if self.is_cross_flag:
            return (enc_hidden_states, encoder_attention_mask, res, decoder_attention_mask, cross_attention_mask)
        else:
            return (res, encoder_attention_mask, dec_hidden_states, decoder_attention_mask, cross_attention_mask)

class Seq2SeqParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, attention_mask_func,
                 init_method, output_layer_init_method):
        super(Seq2SeqParallelTransformer, self).__init__()
        args = get_args()

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers:
        self.num_layers = args.num_layers
        self.num_unique_layers = args.num_unique_layers
        if self.num_unique_layers is None:
            self.num_unique_layers = self.num_layers
        assert self.num_layers % self.num_unique_layers == 0, \
            'number of layers should be divisible by number of unique layers'
        self.param_sharing_style = args.param_sharing_style

        # Transformer layers.
        def build_layer(layer_number, is_cross_flag = False):
            return ParallelTransformerLayer(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number, is_cross_flag)
        _layers =  [build_layer(i + 1, False) for i in range(self.num_unique_layers)]
        _layers += [build_layer(i + 1 + self.num_unique_layers, True) for i in range(self.num_unique_layers)]        
        self.layers = torch.nn.ModuleList(_layers)
        # Print layer ordering.
        if self.num_layers != self.num_unique_layers:
            if torch.distributed.get_rank() == 0:
                print('> will be using the following layer ordering:')
                for i in range(self.num_layers):
                    print('   layer id: {:3d} --> unique layer id: '
                          '{:3d}'.format(i, self._get_layer_index(i)),
                          flush=True)

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _get_layer_index(self, layer_number):
        if self.param_sharing_style == 'grouped':
            return layer_number % (self.num_unique_layers * 2)
        if self.param_sharing_style == 'spaced':
            return layer_number // (self.num_layers // self.num_unique_layers) 
        assert False, 'should not be here'

    def _get_layer(self, layer_number):
        return self.layers[self._get_layer_index(layer_number)]

    def _checkpointed_forward(self, 
                enc_hidden_states, encoder_attention_mask=None, 
                dec_hidden_states = None, decoder_attention_mask=None, cross_attention_mask=None, key_value_states=None):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, inputs[1], inputs[2], inputs[3], inputs[4])
                return x_
            return custom_forward

        # Make sure memory is freed.
        mpu.reset_checkpointed_activations_memory_buffer()

        if key_value_states is None:
            l = 0
            hidden_states = enc_hidden_states
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.checkpoint_num_layers),
                    hidden_states, encoder_attention_mask,
                    None, None, None)
                l += self.checkpoint_num_layers
            key_value_states = hidden_states

        hidden_states = dec_hidden_states
        while l < self.num_layers * 2:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                key_value_states, encoder_attention_mask,
                hidden_states, decoder_attention_mask, cross_attention_mask)
            l += self.checkpoint_num_layers
        return hidden_states, key_value_states

    def forward(self, 
                enc_hidden_states, encoder_attention_mask=None, 
                dec_hidden_states = None, decoder_attention_mask=None, cross_attention_mask=None, 
                key_value_states=None,
                layer_past=None, get_key_value=False):

        # Checks
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        # data format change to avoid explicit tranposes : [b s h] --> [s b h]
        enc_hidden_states = enc_hidden_states.transpose(0, 1).contiguous()
        dec_hidden_states = dec_hidden_states.transpose(0, 1).contiguous()

        if self.checkpoint_activations:
            hidden_states, key_value_states = self._checkpointed_forward(
                enc_hidden_states, encoder_attention_mask, 
                dec_hidden_states, decoder_attention_mask, cross_attention_mask, 
                key_value_states)
        else:
            if key_value_states is None:
                hidden_states = enc_hidden_states
                for index in range(self.num_layers):
                    layer = self._get_layer(index)
                    hidden_states = layer(enc_hidden_states=hidden_states,
                                          encoder_attention_mask=encoder_attention_mask, 
                                          dec_hidden_states = None,
                                          decoder_attention_mask=None, 
                                          cross_attention_mask=None,
                                          layer_past=None, 
                                          get_key_value=False)
                key_value_states = hidden_states

            hidden_states = dec_hidden_states
            if get_key_value:
                presents = []
                cross_presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(self.num_layers + index)
                past, cross_past = None, None
                if layer_past is not None:
                    past = layer_past[0][index]
                    cross_past = layer_past[1][index]
                hidden_states = layer(enc_hidden_states=key_value_states,
                                      encoder_attention_mask=None, 
                                      dec_hidden_states=hidden_states,
                                      decoder_attention_mask=decoder_attention_mask, 
                                      cross_attention_mask=cross_attention_mask,
                                      layer_past=[past, cross_past], 
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present, cross_present = hidden_states
                    presents.append(present)
                    cross_presents.append(cross_present)

        # reverting data format change [s b h] --> [b s h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if get_key_value:
            output = [output, presents, cross_presents, key_value_states]

        return output
