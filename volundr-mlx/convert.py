"""Convert an Agens Volundr compressed-tensors checkpoint to MLX safetensors.

Input: the HF release folder (config.json + model.safetensors.index.json + shards), e.g.
Blockway/Agens-Volundr-32B-v1-INT4 (W4 g128) or the Engine-16GB build (W2/W3/W4 inside the int4 container, g64 with
g128 scales repeated, per-linear types in quantization_audit.json).

Linear formats (compressed-tensors `pack-quantized`, symmetric):
  weight_packed I32 [n, k/8]: 8 nibbles per word, lowest nibble = lowest input index, nibble = q + 8
  weight_scale BF16 [n, k/g]:  w = q * scale
MLX affine b-bit: w = scale * u + bias with u in [0, 2^b), packed as a little-endian bitstream per row. So
  W4: the packed words are already MLX 4-bit (u = q + 8), bias = -8 * scale
  W3: q in [-4, 3], u = q + 4, repacked to 3 bits, bias = -4 * scale
  W2: q in [-2, 1], u = q + 2, repacked to 2 bits, bias = -2 * scale
-8/-4/-2 * scale is exact in bf16 (a power-of-two multiply), so dequantized values equal q * scale bit for bit.

Kept unquantized: bf16 linears of the release (lm_head, in_proj_b, mhc.dyn, engram value/gate, BCSA indexer), norms,
mHC statics, Engram tables, embeddings (bf16); conv1d, A_log, dt_bias (fp32, as the engine runs them).
Text model only: the vision tower and any MTP head are dropped.

usage: python convert.py --hf-path Blockway/Agens-Volundr-32B-v1-INT4 --mlx-path models/volundr-int4-mlx
       [--audit quantization_audit.json]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "agens_b_tokens.json",
]
SHARD_BYTES = 5 << 30


def resolve(hf_path: str) -> Path:
    p = Path(hf_path)
    if p.is_dir():
        return p
    from huggingface_hub import try_to_load_from_cache

    # never downloads here: fetch the checkpoint explicitly first (hf download ...; README assets may be excluded)
    cfg = try_to_load_from_cache(hf_path, "config.json")
    if not isinstance(cfg, str):
        raise FileNotFoundError(f"{hf_path} is not in the HF cache: hf download {hf_path} --exclude 'assets/*'")
    return Path(cfg).parent


def load_types(path) -> dict:
    """Per-linear (bits, group) from quantization_audit.json ({"quantized_matrices": [{"name", "gsq_type"}]}) or an
    RCO assignment ({name: "w3g128"} or [{"name", "type"|"gsq_type"}]). Names are checkpoint names without .weight."""
    if path is None:
        return {}
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict) and "quantized_matrices" in obj:
        items = [(m["name"], m["gsq_type"]) for m in obj["quantized_matrices"]]
    elif isinstance(obj, dict) and "assignment" in obj:
        a = obj["assignment"]
        items = list(a.items()) if isinstance(a, dict) else [(m["name"], m.get("gsq_type", m.get("type"))) for m in a]
    elif isinstance(obj, dict):
        items = [(k, v) for k, v in obj.items() if isinstance(v, str)]
    else:
        items = [(m["name"], m.get("gsq_type", m.get("type"))) for m in obj]
    out = {}
    for name, t in items:
        m = re.fullmatch(r"w(\d)g(\d+)", t)
        if not m:
            raise ValueError(f"{name}: unknown type {t!r}")
        out[name.removesuffix(".weight")] = (int(m.group(1)), int(m.group(2)))
    return out


def unpack_int4(packed_u32: np.ndarray) -> np.ndarray:
    """[n, k/8] uint32 -> [n, k] int8 signed values (nibble - 8)."""
    n, w = packed_u32.shape
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    nib = (packed_u32[:, :, None] >> shifts) & np.uint32(0xF)
    return (nib.astype(np.int16) - 8).astype(np.int8).reshape(n, w * 8)


def pack_bits(u: np.ndarray, bits: int) -> np.ndarray:
    """[n, k] unsigned values < 2^bits -> [n, k*bits/32] uint32, MLX layout (little-endian bitstream per row)."""
    n, k = u.shape
    assert (k * bits) % 32 == 0, (k, bits)
    b = ((u.astype(np.uint8)[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1).astype(np.uint8)
    by = np.packbits(b.reshape(n, k * bits), axis=1, bitorder="little")
    return np.ascontiguousarray(by).view("<u4").reshape(n, k * bits // 32)


def collapse_groups(scale: mx.array, k: int, group: int):
    """Scales [n, k/g_stored] -> [n, k/group] when every `group` block repeats one scale; else None."""
    n, cols = scale.shape
    g_stored = k // cols
    if group == g_stored:
        return scale
    if group % g_stored:
        return None
    r = group // g_stored
    s = scale.reshape(n, cols // r, r)
    if not mx.all(s == s[..., :1]).item():
        return None
    return s[..., 0]


def convert_linear(base, packed, scale, shape, bits, group):
    """-> (weight uint32, scales bf16, biases bf16, bits, group) in MLX affine layout."""
    n, k = shape
    assert packed.shape == (n, k // 8), (base, packed.shape, shape)
    g_stored = k // scale.shape[1]
    s = collapse_groups(scale, k, group)
    if s is None:
        print(f"  {base}: scales do not repeat over g{group}; keeping the stored g{g_stored}")
        s, group = scale, g_stored
    if group not in (32, 64, 128):
        raise ValueError(f"{base}: group {group} not supported by MLX")
    u32 = packed.view(mx.uint32)
    if bits == 4:
        w = u32
    else:
        q = unpack_int4(np.array(u32))
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        qmin, qmax = int(q.min()), int(q.max())
        if qmin < lo or qmax > hi:
            raise ValueError(f"{base}: W{bits} values span [{qmin}, {qmax}], outside [{lo}, {hi}]")
        w = mx.array(pack_bits((q.astype(np.int16) - lo).astype(np.uint8), bits))
    s = s.astype(mx.bfloat16)
    biases = s * (-(1 << (bits - 1)))
    return w, s, biases, bits, group


def check_linear(base, packed, scale, w, s, b, bits, group, rows=4):
    """Dequantize a few rows through MLX and compare with q * scale from the original container."""
    n, k = packed.shape[0], packed.shape[1] * 8
    q = unpack_int4(np.array(packed[:rows].view(mx.uint32)))
    g_stored = k // scale.shape[1]
    sc = np.array(scale[:rows].astype(mx.float32))
    ref = q.astype(np.float32) * np.repeat(sc, g_stored, axis=1)
    got = np.array(mx.dequantize(w[:rows], s[:rows], b[:rows], group_size=group, bits=bits).astype(mx.float32))
    # both sides are q * scale rounded to bf16 by MLX; compare against the bf16 rounding of the exact product
    ref_bf = np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))
    if not np.array_equal(got, ref_bf):
        bad = np.abs(got - ref_bf).max()
        raise AssertionError(f"{base}: MLX dequant differs from q * scale (max abs {bad})")


def mlx_name(name: str) -> str:
    return name.replace("model.language_model.", "model.", 1)


def keep(name: str) -> bool:
    return name.startswith("model.language_model.") or name == "lm_head.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-path", required=True)
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--audit", default=None, help="quantization_audit.json / rco_assignment.json (per-linear types)")
    args = ap.parse_args()

    src = resolve(args.hf_path)
    dst = Path(args.mlx_path)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["text_config"])
    qc = cfg.get("quantization_config") or {}
    w0 = qc.get("config_groups", {}).get("group_0", {}).get("weights", {})
    default = (int(w0.get("num_bits", 4)), int(w0.get("group_size", 128)))
    audit = args.audit
    if audit is None and (src / "quantization_audit.json").exists():
        audit = src / "quantization_audit.json"
    types = load_types(audit)
    print(f"source {src}\ndefault linear type w{default[0]}g{default[1]}, {len(types)} per-linear types from {audit}")

    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {f: mx.load(str(src / f)) for f in sorted(set(wmap.values()))}
    get = lambda n: shards[wmap[n]][n]

    names = sorted(n for n in wmap if keep(n))
    bases = sorted({n[: -len(".weight_packed")] for n in names if n.endswith(".weight_packed")})
    plain = [n for n in names if not re.search(r"\.weight_(packed|scale|shape)$", n)]

    out, out_bytes, shard_files, index = {}, 0, [], {}
    quant = {}
    stats = {}

    def flush(final=False):
        nonlocal out, out_bytes
        if not out or (not final and out_bytes < SHARD_BYTES):
            return
        fn = f"model-{len(shard_files):05d}.safetensors"
        mx.save_safetensors(str(dst / fn), out, metadata={"format": "mlx"})
        shard_files.append(fn)
        index.update({k: fn for k in out})
        print(f"  wrote {fn} ({out_bytes / 2**30:.2f} GiB, {len(out)} tensors)")
        out, out_bytes = {}, 0

    def put(name, arr):
        nonlocal out_bytes
        mx.eval(arr)
        out[name] = arr
        out_bytes += arr.nbytes
        flush()

    checked = set()
    for base in bases:
        packed, scale = get(base + ".weight_packed"), get(base + ".weight_scale")
        shp = base + ".weight_shape"
        shape = tuple(int(x) for x in np.array(get(shp))) if shp in wmap else (packed.shape[0], packed.shape[1] * 8)
        bits, group = types.get(base, default)
        w, s, b, bits, group = convert_linear(base, packed, scale, shape, bits, group)
        if (bits, group) not in checked:
            check_linear(base, packed, scale, w, s, b, bits, group)
            checked.add((bits, group))
            print(f"  dequant check w{bits}g{group}: {base} ok")
        m = mlx_name(base)
        put(m + ".weight", w)
        put(m + ".scales", s)
        put(m + ".biases", b)
        if (bits, group) != default:
            quant[m] = {"group_size": group, "bits": bits}
        stats[(bits, group)] = stats.get((bits, group), 0) + 1

    for n in plain:
        a = get(n)
        if n.endswith("conv1d.weight"):  # [C, 1, K] torch depthwise -> MLX [C, K, 1], fp32
            a = a.moveaxis(2, 1).astype(mx.float32)
        elif n.endswith(".A_log") or n.endswith(".dt_bias"):
            a = a.astype(mx.float32)
        put(mlx_name(n), a)
    flush(final=True)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum((dst / f).stat().st_size for f in shard_files)},
                    "weight_map": dict(sorted(index.items()))}, indent=2))

    gen = json.loads((src / "generation_config.json").read_text()) if (src / "generation_config.json").exists() else {}
    sup = gen.get("suppress_tokens") or []
    rope = tc.get("rope_parameters") or {}
    out_cfg = {k: v for k, v in tc.items() if k not in ("layer_types", "architectures", "rope_parameters")}
    out_cfg.update(
        model_type="volundr",
        architectures=["VolundrForCausalLM"],
        rope_theta=float(rope.get("rope_theta", 1e7)),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
        eos_token_id=gen.get("eos_token_id", tc.get("eos_token_id")),
        vocab_valid=min(sup) if sup else tc["vocab_size"],
        quantization={"group_size": default[1], "bits": default[0], "mode": "affine", **quant},
        source={"repo": str(args.hf_path), "linear_types": {f"w{b}g{g}": c for (b, g), c in sorted(stats.items())}},
    )
    out_cfg["quantization_config"] = out_cfg["quantization"]
    # BCSA indexer shape is not in config.json (bcsa.py defaults 4 x 128): read it off the weights
    iq = next((n for n in wmap if n.endswith("self_attn.idx_q.weight")), None)
    ik = next((n for n in wmap if n.endswith("self_attn.idx_k.weight")), None)
    if iq and ik:
        idx_dim = get(ik).shape[0]
        out_cfg.update(bcsa_index_dim=idx_dim, bcsa_index_heads=get(iq).shape[0] // idx_dim)
    (dst / "config.json").write_text(json.dumps(out_cfg, indent=2))
    for f in COPY_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    print(f"done: {len(index)} tensors, linears {dict((f'w{b}g{g}', c) for (b, g), c in stats.items())} -> {dst}")


if __name__ == "__main__":
    main()
