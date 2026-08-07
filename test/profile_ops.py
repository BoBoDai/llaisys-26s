"""
Profile all LLAISYS operators on CPU and GPU, producing a comparison table.

Shapes match Qwen2 1.5B decode (seqlen=1) for realistic per-token cost.

Usage:
    python test/profile_ops.py                  # both CPU and GPU
    python test/profile_ops.py --device cpu     # CPU only
    python test/profile_ops.py --device nvidia  # GPU only
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__))

import llaisys
from test_utils import (
    random_tensor, random_int_tensor, arrange_tensor,
    llaisys_device,
)

# ---- config ----
WARMUP = 5
REPEAT = 50


def measure_op(llaisys_func, setup_func, device_name):
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    args = setup_func()
    for _ in range(WARMUP):
        llaisys_func(*args)
    api.device_synchronize()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys_func(*args)
    api.device_synchronize()
    return (time.perf_counter() - start) / REPEAT * 1000.0


# ---- Qwen2 1.5B decode shapes (seqlen=1, hs=1536, di=8192, nh=12, nkvh=2, dh=128, vocab=151936) ----

HS, DI, NH, NKVH, DH, VOCAB = 1536, 8192, 12, 2, 128, 151936


def setup_add(device_name, dtype_name):
    s = (1, HS)
    _, a_ = random_tensor(s, dtype_name, device_name)
    _, b_ = random_tensor(s, dtype_name, device_name)
    _, c_ = random_tensor(s, dtype_name, device_name)
    return (c_, a_, b_)


def setup_argmax(device_name, dtype_name):
    _, v_ = random_tensor((VOCAB,), dtype_name, device_name)
    _, i_ = random_int_tensor((1,), device_name, "i64")
    _, m_ = random_tensor((1,), dtype_name, device_name)
    return (i_, m_, v_)


def setup_embedding(device_name, dtype_name):
    _, embd_ = random_tensor((VOCAB, HS), dtype_name, device_name)
    _, idx_ = random_int_tensor((1,), device_name, high=VOCAB)
    _, out_ = random_tensor((1, HS), dtype_name, device_name)
    return (out_, idx_, embd_)


def setup_linear_proj(device_name, dtype_name):
    """Q/K/V/O projection: (1,HS) @ (HS,HS) + bias"""
    _, x_ = random_tensor((1, HS), dtype_name, device_name, scale=0.1)
    _, w_ = random_tensor((HS, HS), dtype_name, device_name, scale=0.01)
    _, b_ = random_tensor((HS,), dtype_name, device_name)
    _, o_ = random_tensor((1, HS), dtype_name, device_name)
    return (o_, x_, w_, b_)


def setup_linear_head(device_name, dtype_name):
    """lm_head: (1,HS) @ (VOCAB,HS)^T, bias omitted (zero-filled)"""
    _, x_ = random_tensor((1, HS), dtype_name, device_name, scale=0.1)
    _, w_ = random_tensor((VOCAB, HS), dtype_name, device_name, scale=0.01)
    # linear op requires a bias tensor; pass zero when unused
    _, b_ = random_tensor((VOCAB,), dtype_name, device_name)
    _, o_ = random_tensor((1, VOCAB), dtype_name, device_name)
    return (o_, x_, w_, b_)


def setup_linear_mlp_gate_up(device_name, dtype_name):
    """gate/up projection: (1,HS) @ (DI,HS)^T -> (1,DI)"""
    _, x_ = random_tensor((1, HS), dtype_name, device_name, scale=0.1)
    _, w_ = random_tensor((DI, HS), dtype_name, device_name, scale=0.01)
    _, b_ = random_tensor((DI,), dtype_name, device_name)
    _, o_ = random_tensor((1, DI), dtype_name, device_name)
    return (o_, x_, w_, b_)


def setup_linear_mlp_down(device_name, dtype_name):
    """down projection: (1,DI) @ (HS,DI)^T, bias omitted (zero-filled)"""
    _, x_ = random_tensor((1, DI), dtype_name, device_name, scale=0.1)
    _, w_ = random_tensor((HS, DI), dtype_name, device_name, scale=0.01)
    # linear op requires a bias tensor; pass zero when unused
    _, b_ = random_tensor((HS,), dtype_name, device_name)
    _, o_ = random_tensor((1, HS), dtype_name, device_name)
    return (o_, x_, w_, b_)


def setup_rms_norm(device_name, dtype_name):
    _, x_ = random_tensor((1, HS), dtype_name, device_name)
    _, w_ = random_tensor((HS,), dtype_name, device_name)
    _, o_ = random_tensor((1, HS), dtype_name, device_name)
    return (o_, x_, w_, 1e-6)


def setup_rope(device_name, dtype_name):
    _, x_ = random_tensor((1, NH, DH), dtype_name, device_name)
    _, o_ = random_tensor((1, NH, DH), dtype_name, device_name)
    _, p_ = arrange_tensor(0, 1, device_name)
    return (o_, x_, p_, 10000.0)


def setup_rope_kv(device_name, dtype_name):
    """RoPE for K: (1, NKVH, DH)"""
    _, x_ = random_tensor((1, NKVH, DH), dtype_name, device_name)
    _, o_ = random_tensor((1, NKVH, DH), dtype_name, device_name)
    _, p_ = arrange_tensor(0, 1, device_name)
    return (o_, x_, p_, 10000.0)


def setup_self_attention(device_name, dtype_name):
    """decode attn: q=(1,NH,DH), kv=(kvlen,NKVH,DH)"""
    kvlen = 128  # short context, fast
    _, q_ = random_tensor((1, NH, DH), dtype_name, device_name)
    _, k_ = random_tensor((kvlen, NKVH, DH), dtype_name, device_name)
    _, v_ = random_tensor((kvlen, NKVH, DH), dtype_name, device_name)
    _, a_ = random_tensor((1, NH, DH), dtype_name, device_name)
    return (a_, q_, k_, v_, 1.0 / (DH ** 0.5))


def setup_swiglu(device_name, dtype_name):
    _, g_ = random_tensor((1, DI), dtype_name, device_name)
    _, u_ = random_tensor((1, DI), dtype_name, device_name)
    _, o_ = random_tensor((1, DI), dtype_name, device_name)
    return (o_, g_, u_)


OPS = [
    # (name,       setup_func,              op_func)
    ("add",             setup_add,              llaisys.Ops.add),
    ("argmax",          setup_argmax,           llaisys.Ops.argmax),
    ("embedding",       setup_embedding,        llaisys.Ops.embedding),
    ("linear (proj)",   setup_linear_proj,      llaisys.Ops.linear),
    ("linear (head)",   setup_linear_head,      llaisys.Ops.linear),
    ("linear (gate/up)",setup_linear_mlp_gate_up, llaisys.Ops.linear),
    ("linear (down)",   setup_linear_mlp_down,  llaisys.Ops.linear),
    ("rms_norm",        setup_rms_norm,         llaisys.Ops.rms_norm),
    ("rope (q)",        setup_rope,             llaisys.Ops.rope),
    ("rope (k)",        setup_rope_kv,          llaisys.Ops.rope),
    ("self_attention",  setup_self_attention,   llaisys.Ops.self_attention),
    ("swiglu",          setup_swiglu,           llaisys.Ops.swiglu),
]


def profile(device_name):
    results = {}
    for name, setup, op_func in OPS:
        try:
            ms = measure_op(op_func, lambda s=setup: s(device_name, "f32"), device_name)
            results[name] = ms
            print(f"  {name:22s}  {ms:10.4f} ms")
        except Exception as e:
            results[name] = None
            print(f"  {name:22s}  {'FAILED':>10s}  ({e})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"])
    args = parser.parse_args()

    devices = ["cpu", "nvidia", "metax"] if args.device == "all" else [args.device]
    all_results = {}

    for dev in devices:
        print(f"\n{'='*55}")
        print(f"Profiling on {dev.upper()}  (Qwen2 1.5B decode shapes, f32)")
        print(f"{'='*55}")
        all_results[dev] = profile(dev)

    if len(devices) == 3:
        cpu = all_results["cpu"]
        gpu = all_results["nvidia"]
        metax = all_results["metax"]
        cpu_total = sum(v for v in cpu.values() if v is not None)
        gpu_total = sum(v for v in gpu.values() if v is not None)
        metax_total = sum(v for v in metax.values() if v is not None)

        print(f"\n{'='*85}")
        print("CPU vs GPU vs Metax Comparison  (Qwen2 1.5B decode, f32, time in ms)")
        print(f"{'='*85}")
        print(f"{'Operator':<22s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'Metax (ms)':>10s} {'Speedup':>8s} {'CPU %':>8s}")
        print(f"{'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

        for name, _, _ in OPS:
            c, g, m = cpu.get(name), gpu.get(name), metax.get(name)
            if c is None or g is None or m is None:
                print(f"{name:<22s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>8s}")
                continue
            speedup = c / g if g > 0 else float("inf")
            pct = c / cpu_total * 100 if cpu_total > 0 else 0
            print(f"{name:<22s} {c:9.4f}  {g:9.4f}  {m:9.4f}  {speedup:7.1f}x  {pct:7.1f}%")

        print(f"{'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
        ts = cpu_total / gpu_total if gpu_total > 0 else float("inf")
        print(f"{'Total':<22s} {cpu_total:9.4f}  {gpu_total:9.4f}  {metax_total:9.4f}  {ts:7.1f}x  {100:7.1f}%")

        # CPU time breakdown bar chart
        print(f"\nCPU time breakdown:")
        print(f"{'Operator':<22s} {'Time (ms)':>10s} {'%':>7s}  Distribution")
        print(f"{'-'*22} {'-'*10} {'-'*7}  {'-'*40}")
        for name, _, _ in OPS:
            ms = cpu.get(name)
            if ms is None:
                continue
            pct = ms / cpu_total * 100
            bar = "#" * int(pct)
            print(f"{name:<22s} {ms:9.4f}  {pct:6.1f}%  {bar}")
        print(f"{'Total':<22s} {cpu_total:9.4f}  {100:6.1f}%")


if __name__ == "__main__":
    main()
