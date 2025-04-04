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
    if initial_state is not None:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N), dtype="float32")

    def step(state, xs):
        kk, rr, vv, aa, bb, w = xs
        kk = ops.reshape(kk, (B, H, 1, N))
        rr = ops.reshape(rr, (B, H, N, 1))
        vv = ops.reshape(vv, (B, H, N, 1))
        aa = ops.reshape(aa, (B, H, N, 1))
        bb = ops.reshape(bb, (B, H, 1, N))
        state = (
            state * ops.expand_dims(w, -2)
            + ops.matmul(state, ops.matmul(aa, bb))
            + ops.matmul(vv, kk)
        )
        out = ops.reshape(ops.matmul(state, rr), (B, H, N))
        return state, out

    state, out = ops.scan(
        step,
        init=state,
        xs=[
            ops.transpose(k, [1, 0, 2, 3]),
            ops.transpose(r, [1, 0, 2, 3]),
            ops.transpose(v, [1, 0, 2, 3]),
            ops.transpose(a, [1, 0, 2, 3]),
            ops.transpose(b, [1, 0, 2, 3]),
            ops.transpose(w, [1, 0, 2, 3]),
        ],
        length=T,
    )
    out = ops.cast(ops.transpose(out, [1, 0, 2, 3]), DTYPE)
    if output_final_state:
        return out, state
    return out
