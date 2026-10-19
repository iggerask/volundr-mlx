"""Shared timing helpers for the small-M kernel benchmarks."""
import atexit, os, subprocess, time
import mlx.core as mx

# Other sessions benchmark on this GPU too. Hold a shared lock file while timing, and also wait
# while any other venv python job runs, so neither side measures under contention.
LOCK = os.environ.get("GPU_LOCK") or os.path.expanduser("~/Library/Application Support/Claude/side-session-notes/"
                          "local_4b1c7a56-3804-4800-b519-29795572f128/gpu.lock")


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _others_running():
    # Match on the executable, not the command line: a shell that launched python also has the
    # interpreter path in its arguments.
    out = subprocess.run(["ps", "-axo", "pid=,comm=,args="], capture_output=True, text=True).stdout
    me = os.getpid()
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if (os.path.basename(comm).startswith("python") and "personal-dev/llms" in args
                and "multiprocessing" not in args and int(pid) != me):
            return True
    return False


def wait_gpu_idle(poll=2.0):
    """Block until no other venv python job runs and nobody else holds the lock; then hold it."""
    waited = 0.0
    while True:
        holder = None
        try:
            holder = int(open(LOCK).read().strip() or 0)
        except (OSError, ValueError):
            pass
        busy = (holder and holder != os.getpid() and _alive(holder)) or _others_running()
        if not busy:
            break
        if waited == 0.0:
            print("  [waiting for the GPU to go idle]", flush=True)
        time.sleep(poll)
        waited += poll
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def _release():
    try:
        if int(open(LOCK).read().strip()) == os.getpid():
            os.remove(LOCK)
    except (OSError, ValueError):
        pass


atexit.register(_release)

BF = mx.bfloat16
SHAPES = {"gate_up": (34816, 5120), "down": (5120, 17408), "kda_qkv": (10240, 5120), "out": (5120, 6144)}


def bench(f, K=1, iters=10, warmup=2, reps=3):
    """Seconds per call (best of reps), with K calls queued per eval."""
    def run():
        mx.eval([f() for _ in range(K)])
    for _ in range(warmup):
        run()
    best = 1e9
    for _ in range(reps):
        wait_gpu_idle()
        mx.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            run()
        mx.synchronize()
        best = min(best, (time.perf_counter() - t) / (iters * K))
    return best


def spin_up(seconds=2.0):
    """Keep the GPU busy so clocks are up before the first measurement."""
    a = mx.random.normal((4096, 4096)).astype(BF)
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        mx.eval([a @ a for _ in range(8)])


def qweight(n, k, bits=4, group=128):
    w = mx.random.randint(0, 2**31 - 1, (n, k * bits // 32)).astype(mx.uint32)
    s = (mx.random.uniform(shape=(n, k // group)) * 0.01).astype(BF)
    b = (-s * 8).astype(BF)
    mx.eval(w, s, b)
    return w, s, b


def qbytes(n, k, bits=4, group=128):
    return n * k * bits / 8 + 2 * (n * k // group) * 2


def compare(fns, K=1, reps=7, iters=5, warmup=2):
    """Interleaved best-of-reps seconds per call for each f in fns (name -> f).

    Interleaving and taking the minimum keeps a GPU job from another process from biasing one
    candidate against another."""
    def run(f):
        mx.eval([f() for _ in range(K)])
    for f in fns.values():
        for _ in range(warmup):
            run(f)
    best = {n: 1e9 for n in fns}
    for _ in range(reps):
        wait_gpu_idle()
        for n, f in fns.items():
            mx.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                run(f)
            mx.synchronize()
            best[n] = min(best[n], (time.perf_counter() - t) / (iters * K))
    return best
