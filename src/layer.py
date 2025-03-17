import keras
from keras import ops
from keras.layers import Dense,Layer
from standard_rwkv.rwkv7_layer import RWKV7_OP
class GroupNorm(Layer):
    def __init__(self,hidden_size,num_heads,epsilon=64*1e-5,name="group_norm",**kwargs):
        super(GroupNorm,self).__init__(name=name,**kwargs)

        self.hidden_size = hidden_size
        self.head_size = hidden_size //num_heads
        self.num_heads = num_heads
        self.epsilon =epsilon
        assert hidden_size % num_heads == 0

    def call(self,inputs,shape):
        B,T,C = shape
        x = ops.reshape(inputs,(B,T,self.num_heads,self.head_size))
        x =  ops.reshape(self.scale,(1,1,self.num_heads,self.head_size)) * x +  ops.reshape(self.offset,(1,1,self.num_heads,self.head_size))
        o = ops.reshape(x,(B,T,C))
        return o
    def compute_output_shape(self, input_shape):
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.scale = self.add_weight(shape=(self.num_heads,self.head_size))
        self.offset = self.add_weight(shape=(self.num_heads,self.head_size))
    def get_config(self):
        config = {
            'num_heads':self.num_heads,
            'hidden_size':self.hidden_size,
            'epsilon':self.epsilon
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class TimeShift(Layer):
    def __init__(self,name="time_shift"):
        super(TimeShift, self).__init__(name=name)
    def call(self, inputs,cache_x=None):
        x = ops.pad(inputs,[[0,0],[1,0],[0,0]],constant_values=0.)[:,:-1,:]
        if cache_x is not None:
            x = ops.slice_update(x,[0,0,0],cache_x)
        return x
    def compute_output_shape(self, input_shape):
        return input_shape


class RWKV7_ChannelMix(Layer):
    def __init__(self,dim_ffn,**kwargs):
        super().__init__(**kwargs)
        self.dim_ffn = dim_ffn
        

    def call(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = self.key(k) ** 2
        return self.value(k)
    def compute_output_shape(self, input_shape):
        if isinstance(input_shape,list):
            return input_shape[0]
        return input_shape
    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape,list):
            input_shape = input_shape[0]
        self.x_k  = self.add_weight(shape=(1,1,input_shape[-1]),name="time_mix_k")
        self.time_shift = TimeShift()
        self.key = Dense(self.dim_ffn,activation="relu",use_bias=False,name="dense_k")
        self.value = Dense(input_shape[-1],use_bias=False,name="dense_v")
        
    def get_config(self):
        config = {
            'dim_ffn':self.dim_ffn,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
class RWKV7_TimeMix(Layer):
    def __init__(self,hidden_size,
                 head_size,
                 gate_lora = 128,
                 mv_lora = 32,
                 aaa_lora = 64,
                 decay_lora = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.n_head = hidden_size // self.head_size
        self.gate_lora = gate_lora
        self.mv_lora = mv_lora
        self.aaa_lora = aaa_lora
        self.decay_lora = decay_lora
        assert self.hidden_size % self.n_head == 0
    def build(self, input_shape):
        super().build(input_shape)
        if isinstance(input_shape,list):
            input_shape = input_shape[0]
        H = self.n_head
        N = self.head_size
        C = input_shape[-1]

        self.x_r = self.add_weight(shape=(1,1,C), name="x_r")
        self.x_w = self.add_weight(shape=(1,1,C), name="x_w")
        self.x_k = self.add_weight(shape=(1,1,C), name="x_k")
        self.x_v = self.add_weight(shape=(1,1,C), name="x_v")
        self.x_a = self.add_weight(shape=(1,1,C), name="x_a")
        self.x_g = self.add_weight(shape=(1,1,C), name="x_g")

        self.w0 = self.add_weight(shape=(1,1,C), name="w0")
        self.w1 = self.add_weight(shape=(C, self.decay_lora), name="w1")
        self.w2 = self.add_weight(shape=(self.decay_lora, C), name="w2")

        self.a0 = self.add_weight(shape=(1,1,C), name="a0")
        self.a1 = self.add_weight(shape=(C, self.aaa_lora), name="a1")
        self.a2 = self.add_weight(shape=(self.aaa_lora, C), name="a2")

        self.v0 = self.add_weight(shape=(1,1,C), name="v0")
        self.v1 = self.add_weight(shape=(C, self.mv_lora), name="v1")
        self.v2 = self.add_weight(shape=(self.mv_lora, C), name="v2")

        self.g1 = self.add_weight(shape=(C, self.gate_lora), name="g1")
        self.g2 = self.add_weight(shape=(self.gate_lora, C), name="g2")

        self.k_k = self.add_weight(shape=(1,1,C), name="k_k")
        self.k_a = self.add_weight(shape=(1,1,C), name="k_a")
        self.r_k = self.add_weight(shape=(H,N), name="r_k")
                                
        self.time_shift = TimeShift()
        self.receptance = Dense(C, use_bias=False)
        self.key = Dense(C, use_bias=False)
        self.value = Dense(C, use_bias=False)
        self.output = Dense(C, use_bias=False)
        self.ln_x = GroupNorm(C, H,  epsilon=64e-5)

    def call(self,x,v_first=None,mask=None):
        B, T, C = ops.shape(x)
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -ops.softplus(-(self.w0 + ops.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if v_first==None:
            v_first = v
        else:
            v = v + (v_first - v) * ops.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        
        a = ops.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = ops.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = ops.normalize(ops.reshape(kk,(B,T,H,-1)), axis=-1, order=2, epsilon=1e-12)
        kk = ops.reshape(kk,(B,T,C))

        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(ops.reshape(x,(B * T, C)),shape = ops.shape(x))

        x = ops.reshape(x,(B, T, C))
        r = ops.reshape(r,(B,T,H,-1))
        k = ops.reshape(k,(B,T,H,-1))
        v = ops.reshape(v,(B, T, C))

        rwkv = ops.sum(r*k*self.r_k,axis=-1, keepdims=True) * ops.reshape(v,(B,T,H,-1))

        x = x + ops.reshape(rwkv,(B,T,C))
        x = self.output(x * g)
        return x, v_first
    
    def get_config(self):
        config = {
            'hidden_size':self.hidden_size,
            'head_size':self.head_size,
            'gate_lora':self.gate_lora,
            'mv_lora':self.mv_lora,
            'aaa_lora':self.aaa_lora,
            'decay_lora':self.decay_lora,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))