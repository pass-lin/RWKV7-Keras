# 施工中，暂不可用

目前还在施工中，施工后再完善文档，目标是实现基于keras的多后端rwkv7，让最广泛的人都能使用到RWKV7
## RWKV 项目进度表

| 任务名称                         | 状态 |
| -------------------------------- | ---- |
| 实现 RWKV BackBone                   | √    |
| 简单的测试脚本                             | √    |
| Keras 原生实现多后端算子         | √    |
| Torch 后端使用 Triton 算子       | √    |
| Jax 后端使用 Triton 算子或 CUDA 算子 | x    |
| 转换现成的 RWKV 模型以及转换脚本 | x    |
| RWKV 推理模块                     | x    |
| RWKV 推理模块测试                 | x    |
| Keras_hub 风格的 RWKV tokenizer  | x    |
| 提交到 Keras 官方仓库             | x    |