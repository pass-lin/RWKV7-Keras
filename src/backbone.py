from src.layer import *
from keras_hub.src.models.backbone import Backbone
import keras
def rwkv7_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)
class RWKV7Backbone(Backbone):
    def __init__(
        self,
        hidden_size,
        head_size,
        hidden_dim,
        num_layers,
        vocabulary_size,
        intermediate_dim,
        gate_lora=128,
        mv_lora=32,
        aaa_lora=64,
        decay_lora=64,
        dtype = None,
        dropout = 0,
        **kwargs,
    ):

        
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=rwkv7_kernel_initializer(),
            dtype=dtype,
            name="token_embedding",
        )

        self.output_layer_norm = keras.layers.LayerNormalization(
                epsilon=1e-5, name="output_norm"
            )
        
        self.dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="embeddings_dropout",
        )
        self.rwkv_layers = []
        for i in range(num_layers):
            layer = RWKV7_Block(
                hidden_size,
                head_size,
                intermediate_dim,
                gate_lora,
                mv_lora,
                aaa_lora,
                decay_lora,
                use_initial_norm= i==0,
                kernel_initializer=rwkv7_kernel_initializer(),
                dtype=dtype,
                name=f"rwkv_layer_{i}",
            )
            self.rwkv_layers.append(layer)

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        
        mask = ops.not_equal(token_id_input,0)

        x = self.token_embedding(token_id_input)
        v_first = ops.zeros_like(x)
        for rwkv_layer in self.rwkv_layers:
            x, v_first = rwkv_layer(x, v_first,mask)
            x = self.dropout(x)
        sequence_output = self.output_layer_norm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
            },
            outputs={
                "sequence_output": sequence_output,
            },
            dtype=dtype,
            **kwargs,
        )
        
        self.num_layers = num_layers
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size
        self.dropout = dropout
        self.intermediate_dim = intermediate_dim

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "head_size": self.head_size,
            "gate_lora": self.gate_lora,
            "mv_lora": self.mv_lora,
            "aaa_lora": self.aaa_lora,
            "decay_lora": self.decay_lora,
            "hidden_dim": self.hidden_dim,
            "vocabulary_size": self.vocabulary_size,
            "dropout":self.dropout,
            "intermediate_dim":self.intermediate_dim,
            "num_layers":self.num_layers
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
