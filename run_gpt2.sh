#! /bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=8888
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

WORKING_DIR=${HOME}/Megatron-LM-3D
DATA_DIR=${HOME}/gpt2_data

DATA_PATH="${DATA_DIR}/CPM-train/data" 
# VOCAB_PATH=data/gpt2-vocab.json
# MERGE_PATH=data/gpt2-merges.txt
TOKENIZER_PATH="${DATA_DIR}/bpe_3w_new/vocab.json"
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds
config_json="${WORKING_DIR}/ds_config_gpt2.json"

# Megatron Model Parallelism
mp_size=2
# DeepSpeed Pipeline parallelism
pp_size=2

NLAYERS=2
NHIDDEN=256
BATCHSIZE=4
GAS=16
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"


#Actication Checkpointing and Contigious Memory
checkpoint_activations
chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads 16 \
        --kv-hidden-size 16 \
        --ff-hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --gas $GAS \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --data-impl mmap \
        --vocab-file $TOKENIZER_PATH\
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 500 \
        --eval-interval 100 \
        --eval-iters 10 \
        --fp16 \
        --hidden-bias \
        --tensorboard-dir ${LOGDIR}
"

deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

if ["${checkpoint_activations}" = "true"]; then

        chkp_opt=" \
        --checkpoint-activations \
        --checkpoint-num-layers ${chkp_layers}"

        if [ "${PA}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --partition-activations"
        fi

        if [ "${PA_CPU}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --checkpoint-in-cpu"
        fi

        if [ "${SYNCHRONIZE}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --synchronize-each-layer"
        fi

        if [ "${CC}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --contigious-checkpointing"
        fi

        if [ "${PROFILE}" = "true" ]; then
        chkp_opt="${chkp_opt} \
                --profile-backward"
        fi
else
        chkp_opt = " "
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

run_cmd="deepspeed --master_port ${MASTER_PORT} -i node1:4,5,6,7 --hostfile hostfile pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
