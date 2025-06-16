import keras
from keras import ops


def transpose_head(x, head_first):
    x = ops.cast(x, "float32")
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
):
    DTYPE = r.dtype
    B, T, H, N = ops.shape(r)
    r = transpose_head(r, head_first)

    k = transpose_head(k, head_first)

    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N), dtype="float32")

    def step(state, xs):
        kk, rr, vv, aa, bb, w = xs
        kk = ops.expand_dims(kk, -2)
        rr = ops.expand_dims(rr, -1)
        vv = ops.expand_dims(vv, -1)
        aa = ops.expand_dims(aa, -1)
        bb = ops.expand_dims(bb, -2)
        state = state * w[:, :, None, :] + state @ aa @ bb + vv @ kk
        out = ops.reshape(state @ rr, (B, H, N))
        return state, out

    if keras.config.backend() == "jax":
        import jax

        step = jax.checkpoint(step)
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

    out = ops.transpose(out, [1, 0, 2, 3])
    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)
