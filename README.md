# CPM-Pretrain

可以支持数据并行、模型加速、流水并行的代码。主要基于DeepSpeed和Megatron实现。

DeepSpeed中的流水代码存在问题，我们对此重新实现了DeepSpeed中的PipelineEngine，并且将其他的接口保留做成了一个DeepSpeed的插件TDS。更多细节可以见[TDS](https://github.com/TsinghuaAI/TDS)。

环境的安装和配置可以详见DeepSpeed和Megatron的文档，也可以按照[CPM-Finetune](https://github.com/TsinghuaAI/CPM-Finetune)来配置。

这是一个我们重构过后的版本，可以支持GPT-2和T5的训练。一个稳定的版本和模型参数我们会在2021年4月底发布。

也欢迎大家来给我们提issue和bug ：）
