import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERNEL_TYPE"] = "native"
import keras
import keras.ops as ops

from src.rwkv7_causal_lm_preprocessor import RWKV7CausalLMPreprocessor, RWKVTokenizer
from src.rwkv7_causal_lm import RWKV7CausalLM, RWKV7Backbone
from keras import ops
import json
import keras

keras.config.set_dtype_policy("bfloat16")
rwkv_path = "rwkv7_world_0.1B"
with open("%s/config.json" % rwkv_path) as f:
    config = json.load(f)
backbone = RWKV7Backbone(**config["config"])
backbone.load_weights("%s/model.weights.h5" % rwkv_path)
backbone.summary()
tokenizer = RWKVTokenizer()
tokenizer.load_assets(rwkv_path)
preprocessor = RWKV7CausalLMPreprocessor(tokenizer, sequence_length=16)
causal_lm = RWKV7CausalLM(backbone, preprocessor)

prompts = ["Bubble sort\n```python", "Hello World\n```python\n"]

causal_lm.compile(sampler="greedy")

outputs = causal_lm.generate(prompts, max_length=128)
for out in outputs:
    print(out)
    print("-" * 100)
