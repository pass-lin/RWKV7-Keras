from keras import ops
def RWKV7_OP(r, w, k, v, a, b):
    DTYPE = r.dtype
    B, T, H,N = ops.shape(r)
    C = H*N
    r = ops.cast(r,"float32")
    k = ops.cast(k,"float32")
    v = ops.cast(v,"float32")
    a = ops.cast(a,"float32")
    b = ops.cast(b,"float32")
    w = ops.cast(ops.reshape(w,(B, T, H, N)),"float32")
    w = ops.exp(-ops.exp(w))
    out = ops.zeros((B, T, H, N),  dtype="float32")
    state = ops.zeros((B, H, N, N),  dtype="float32")
    
    def  step(t,inputs):
        state,out = inputs
        kk = ops.reshape(k[:, t, :],(B, H, 1, N))
        rr = ops.reshape(r[:, t, :],(B, H, N, 1))
        vv = ops.reshape(v[:, t, :],(B, H, N, 1))
        aa = ops.reshape(a[:, t, :],(B, H, N, 1))
        bb = ops.reshape(b[:, t, :],(B, H, 1, N))
        state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
        out= ops.slice_update(out,
                              [0,t,0,0],
                              ops.reshape((state @ rr),(B, 1,H, N)))
        return state,out
    state,out = ops.fori_loop(0,T,step,(state,out))


    return ops.cast(ops.reshape(out,(B, T, C)),DTYPE)