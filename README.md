# CPM-Pretrain
[版本更新记录](doc/release-note.md)

为了促进中文自然语言处理研究的发展，本项目提供了大规模预训练语言模型的预训练代码。项目主要基于DeepSpeed、Megatron实现，可以支持数据并行、模型加速、流水并行的代码。

## 安装

1、首先安装pytorch等基础依赖，再安装APEX以支持fp16。
    
    pip install torch
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

2、安装DeepSpeed，这部分更多细节请参照[DeepSpeed](https://github.com/microsoft/DeepSpeed#installation)项目安装说明。

    pip install deepspeed

3、本项目也需要TDS方可运行，此部分更多细节可以见[TDS](https://github.com/TsinghuaAI/TDS)。TDS插件只需要将代码拷贝到当前项目的根目录下，再使用下列代码载入DeepSpeed即可（当前项目里此步骤已经完成）。

    import tds as deepspeed


## 模型

当前项目主要支持GPT-2、T5两类模型，未来会有更多模型放出。

## 预训练与设置脚本

1、多机多卡设置
    
    #每台机器的显卡数量
    GPUS_PER_NODE=4

    #支持机器之间通信的IP地址
    MASTER_ADDR=localhost
   
    #支持机器之间通信的端口
    MASTER_PORT=8888
   
    #本次训练一共涉及几台机器
    NNODES=1
   
    #当前机器的序号，编号从0开始
    NODE_RANK=0

2、数据并行、模型并行、流水并行设置

    # 整体模型划分为几层，进行流水并行
    pp_size=2
   
    # 每层模型划分为几块，进行模型并行
    mp_size=2


3、设置模型尺寸
    
    #层数
    NLAYERS=2
    
    #输入层大小
    NHIDDEN=128
    
    #单张卡每次接触到的batch大小
    BATCHSIZE=4
    
    #梯度累积步数
    GAS=16

4、DeepSpeed脚本设置，为DeepSpeed设置训练的流水并行、混合精度等细节，只有在流水并行被启用时，此脚本方被使用。如果pp_size=0或pp_size=1，意味着流水并行不被激活，此时无论脚本里的设置为何，均不会发生作用，代码只会使用Megatron-LM进行模型并行。

    config_json="ds_config_t5.json"
    
    脚本形式：
    {
        "train_batch_size": 128, 
        "train_micro_batch_size_per_gpu": 4, 
        "steps_per_print": 10, 
        "gradient_clipping": 1.0, 
        "fp16": {
            "enabled": true, 
            "loss_scale": 0, 
            "loss_scale_window": 1000, 
            "hysteresis": 2, 
            "initial_scale_power": 16
        }, 
        "zero_allow_untested_optimizer": true, 
        "wall_clock_breakdown": true
    }

注意：脚本里的 train_batch_size = BATCHSIZE * GAS * GPUS_PER_NODE * NNODES / pp_size / mp_size

5、也可以自己自由设置训练参数，以run_t5.sh为例

        --model-parallel-size 2 \
        --pipe-parallel-size 2 \
        --num-layers 12\
        --hidden-size 128 \
        --kv-hidden-size 16 \
        --ff-hidden-size 256 \
        --num-attention-heads 8 \
        --enc-seq-length 1024\
        --dec-seq-length 384\
        --max-position-embeddings 1024 \
        --batch-size 4\
        --gas 16 \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \

6、运行脚本

    bash run_t5.sh 或 bash run_gpt2.sh

 
## 实现自己的模型

我们以pretrain_gpt2.py为例子，介绍下如何实现自己的模型

*   使用Megatron-LM实现模型并行

    Megatron-LM的较为核心的代码是 megatron/mpu/layers.py，这里面定义了
    *   VocabParallelEmbedding旨在将embedding层切割到不同的GPU上，
    *   ColumnParallelLinear、RowParallelLinear则是将矩阵按照行列切割到不同的GPU上，完整的输出结果需要通过一次all-reduce来从各个GPU上汇总。

   这里我们以GPT-2中的Transformer的线性层来举例。Transformer的线性层分为两部分：
   *    第一部分是将$\text{hidden_size}$维的向量映射为$4 \times \text{hidden_size}$维的向量，并且通过GeLU激活；
   *    第二部分是将激活后的$4 \times \text{hidden_size}$维的向量映射回$\text{hidden_size}$维，并且施加Dropout。
   
   这里，我们对第一部分的线性层矩阵按列切分，第二部分按行切分。
   *    对于第一部分$\mathbf{Y}=\text{GeLU}(\mathbf{X}\mathbf{A})$，我们将矩阵$\mathbf{A}$切分为$\mathbf{A}_1,\mathbf{A}_2$两个部分，那么整体的计算也被相应分为两部分$\mathbf{Y}_1=\text{GeLU}(\mathbf{X}\mathbf{A}_1)$和$\mathbf{Y}_2=\text{GeLU}(\mathbf{X}\mathbf{A}_2)$。此时$\mathbf{Y}_1$和$\mathbf{Y}_2$分别为$\mathbf{Y}$的前$2 \times \text{hidden_size}$结果和后$2 \times \text{hidden_size}$结果。

    *   对于$\mathbf{Z}=\mathbf{Y}\mathbf{B}$，我们将矩阵$\mathbf{B}$按行切分为$\mathbf{B}_1, \mathbf{B}_2$。将第一部分的结果$\mathbf{Y}_1$和$\mathbf{Y}_2$传入后，可计算$\mathbf{Z}_1=\mathbf{Y}_1\mathbf{B}_1$与$\mathbf{Z}_1=\mathbf{Y}_2\mathbf{B}_2$，此时有$\mathbf{Z}=\mathbf{Z}_1+\mathbf{Z}_1$。在Dropout之前，我们需要对$\mathbf{Z}_1$与$\mathbf{Z}_2$进行一次同步，加和后的结果再过Dropout。

    总的来看，整个线性层计算过程中，每张显卡都只负责了一半的计算量，整体结果需要一次额外的all_reduce来对输出结果进行汇总。这里也给出具体的代码实现，与我们上述的过程是一致的。此部分代码可见 megatron/model/transformer.py
    
    ```python
    class ParallelMLP(MegatronModule):
    
        def __init__(self, ...):
            super(ParallelMLP, self).__init__()
            args = get_args()
            # Project to 4h.
            self.dense_h_to_4h = mpu.ColumnParallelLinear(
                args.hidden_size,
                args.hidden_size * 4,...)
                
            ...
            # Project back to h.
            self.dense_4h_to_h = mpu.RowParallelLinear(
                args.args.hidden_size * 4,
                args.hidden_size,...)
    
        def forward(self, hidden_states):
            # [s, b, 4hp]
            intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)
            ...
            # [s, b, h]
            output, output_bias = self.dense_4h_to_h(intermediate_parallel)
            return output, output_bias
    
    ```

    同样的方法我们也可以实现Attention层、Embedding层的模型并行。

*   使用DeepSpeed实现流水并行

    使用DeepSpeed进行流水并行也并不复杂，只需要将模型的各个层加入队列中，并用队列实例化DeepSpeed的PipelineModule类即可。
    
    但是受限于DeepSpeed框架本身的设计，队列中的各个层输入的tensor数量和输出的tensor数量必须保持一致，以确保各层之间可以无缝衔接。
    
    例如实现一个流水的Transformer版本，模型的每一层都需要输出hidden_states和mask，与下一层的输入相衔接。

    ```python
    from deepspeed.pipe import PipelineModule, LayerSpec
    ...
    class TransformerBlockPipe(TransformerBlock)
        def forward(self, ...):
            hidden, mask = ...
            ...
            output = super().forward(hidden, mask)
            return (output, mask)
    ...
    
    class GPT2(PipelineModule):
        def __init__(self, num_layers, ...):
            ...
            specs = [TransformerBlockPipe() for _ in range(num_layers) ]
            ...
            super().__init__(layers=specs, ...)
    ```

    上述代码有一个LayerSpec的实现。LayerSpec有点类似一层懒惰标记的封装。
    
    具体而言，LayerSpec的参数分别为神经网络层的class以及class的初始化参数。LayerSpec将class和初始化参数记忆下来，但不会立刻实例化，而是在流水线构建过程中再将class和初始化参数结合起来进行实例化。
    
    使用DeepSpeed进行混合加速也十分方便，只需要让加入流水队列里的模型实现是基于模型并行的即可，如使用Megatron-LM来实现上述代码的TransformerBlockPipe和TransformerBlock。

*   使用TDS实现训练流程

    TDS重新实现了Deepspeed的流水线，通过适配器模式把DeepSpeed的其他功能封装进来，所以用起来也非常简单。在安装完DeepSpeed之后，只需要将TDS的代码拷贝到工程中，然后用以下的方式加载库即可。
    
    ```python
    import tds as deepspeed
    ```

    使用TDS的训练流程代码与之前的DeepSpeed和Megatron中的范例代码是一致的，首先需要一个获取数据iterator的模块。

    ```python
    def train_valid_test_datasets_provider(...):
        ...
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(...)
        ...
        return train_ds, valid_ds, test_ds
    ```

    然后，需要一个从data_iterator中获取数据，并且分发到各个GPU上的模块。
    
    ```python
    def get_batch(data_iterator):
        ...
        # 定义传输的数据类型
        keys = ['text']
        datatype = torch.int64
        # 将数据压缩之后广播到同一个数据并行group中的每个GPU上
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)
        # 将数据解压
        tokens_ = data_b['text'].long()
        # 从数据中提取token, label, mask信息
        tokens, labels, loss_mask, attention_mask, position_ids = (...)
        ...
        return tokens, labels, loss_mask, attention_mask, position_ids
    ```

    流水并行需要一个单独的数据获取分发模块，整体结构与get_batch是类似的，但有两个细微差别。一个是DeepSpeed中将流水并行的data_iterator交给了后台管理，所以get_batch_pipe只需要负责分发数据即可。第二个是DeepSpeed对给流水线使用的data_iterator是有格式限制的，必须是返回两个tuple，前一个是用来进行forward的输入，后一个是用来进行loss的计算的输入。
    
    ```python
    def get_batch_pipe(data):
        ...
        # 定义传输的数据类型
        keys = ['text']
        datatype = torch.int64
        # 将数据压缩之后广播到同一个数据并行group中的每个GPU上
        data_b = mpu.broadcast_data(keys, data, datatype)
        # 将数据解压
        tokens_ = data_b['text'].long()
        # 从数据中提取token, label, mask信息
        tokens, labels, loss_mask, attention_mask, position_ids = (...)
        ...
        return (tokens, position_ids, attention_mask), (labels, loss_mask)
    ```

    在实现了数据获取的各项模块后，模型获取模块就十分简单，根据是否使用流水并行返回不同模型。与DeepSpeed相比，TDS中需要制定流水线中间的输入输出类型，是否需要保存梯度，以及是否需要将中间结果切割到多个GPU来减少显存使用。
    
    ```python
    def model_provider():
        """Build the model."""
        args = get_args()
        if args.pipe_parallel_size == 0 :
            model = GPT2Model(...)
        else:
            model = GPT2ModelPipe(...)
            model._megatron_batch_fn = get_batch_pipe
            # TDS中需要制定流水线中间的输入输出类型，是否需要保存梯度，以及是否需要将中间结果切割到多个GPU来减少显存使用
            model._input_grad = [True, False]
            model._input_type = ['float', 'bool']
            model._input_pipe_partitioned = [True, False]
        return model
    ```
    
    最后只要基于上述模块，启动pretrain就能开始计算。
    
    ```python
    from megatron.training import pretrain
    
    if __name__ == "__main__":
        pretrain(train_valid_test_datasets_provider, model_provider, ...)
    ```
    
    
## 引用

如果您使用了我们的代码，请您引用下面的文章。

```[latex]
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```
