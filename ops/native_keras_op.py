import keras
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def generalized_delta_rule(
    r,
    w,
    k,
    v,
    a,
    b,
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
    use_chunk=None,
):
    DTYPE = r.dtype
    B, T, H, N = ops.shape(r)
    C = H * N
    r = ops.cast(transpose_head(r, head_first), "float32")
    k = ops.cast(transpose_head(k, head_first), "float32")
    v = ops.cast(transpose_head(v, head_first), "float32")
    a = ops.cast(transpose_head(a, head_first), "float32")
    b = ops.cast(transpose_head(b, head_first), "float32")
    w = ops.cast(transpose_head(w, head_first), "float32")
    w = ops.exp(-ops.exp(w))
    out = ops.zeros((B, T, H, N), dtype="float32")
    if initial_state is not None:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.repeat(state,B,axis=0)
    else:
        state = ops.zeros((B, H, N, N), dtype="float32")

    def step(t, inputs):
        state, out = inputs
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = (
            state * ops.expand_dims(w[:, t], -2)
            + ops.matmul(state, ops.matmul(aa, bb))
            + ops.matmul(vv, kk)
        )
        out = ops.slice_update(
            out, [0, t, 0, 0], ops.reshape(ops.matmul(state, rr), (B, 1, H, N))
        )
        return [state, out]

    if keras.config.backend() == "openvino":
        for t in range(T):
            state, out = step(t, [state, out])
    else:
        state, out = ops.fori_loop(0, T, step, [state, out])
    out = ops.cast(out, DTYPE)
    if output_final_state:
        return out, state
    return out
