"""CGBlockAttn.forward 逐步形状追踪（用于讲解，不参与推理）。."""

import math

import torch

from ultralytics.nn.modules import CGBlockAttn


def trace(c1, h, w, block=8, heads=4):
    m = CGBlockAttn(c1, c1, block=block, num_heads=heads).eval()
    x = torch.randn(1, c1, h, w)
    print(f"\n{'=' * 70}\n输入 x = {tuple(x.shape)}  |  block={block} heads={heads} hidden={m.hidden}\n{'=' * 70}")

    b = 1
    # --- Q/K/V 投影 ---
    qmap, kmap, vmap = m.q(x), m.k(x), m.v(x)
    print(f"[1] Q/K/V 1x1卷积       : {tuple(qmap.shape)}   (c1={c1} -> hidden={m.hidden})")

    # --- 块池化成 token ---
    q, (hp, wp) = m._to_tokens(qmap)
    k, _ = m._to_tokens(kmap)
    v, _ = m._to_tokens(vmap)
    N = q.shape[-1]
    print(f"[2] _to_tokens 分块平均  : {tuple(q.shape)}")
    print(f"    pad 后尺寸 {hp}x{wp} -> 网格 {hp // block}x{wp // block} = N={N} 个 token")
    print(f"    token 数 {N} vs 像素数 {h * w}  ->  压缩 {h * w / N:.0f} 倍")

    # --- 多头注意力 ---
    n = q.shape[-1]
    hd = q.shape[1] // m.num_heads
    print(f"[3] 拆多头               : hd = hidden/heads = {q.shape[1]}/{m.num_heads} = {hd}")
    qh = q.reshape(b, m.num_heads, hd, n).permute(0, 1, 3, 2)
    kh = k.reshape(b, m.num_heads, hd, n)
    vh = v.reshape(b, m.num_heads, hd, n).permute(0, 1, 3, 2)
    print(f"    q {tuple(qh.shape)}  k {tuple(kh.shape)}  v {tuple(vh.shape)}")

    attn = torch.softmax(qh @ kh / math.sqrt(hd), dim=-1)
    print(f"[4] attn = softmax(qk/√hd): {tuple(attn.shape)}   (每个 token 对 N 个 token 的权重)")

    out = (attn @ vh).permute(0, 1, 3, 2).reshape(b, m.hidden, n)
    print(f"[5] out  = attn @ v      : {tuple(out.shape)}   -> reshape 回 {tuple(out.shape)}")

    # --- 广播回像素 ---
    outp = out.reshape(b, m.hidden, hp // block, 1, wp // block, 1).expand(-1, -1, -1, block, -1, block)
    print(f"[6] expand 广播回块       : {tuple(outp.shape)}")
    outp = outp.reshape(b, m.hidden, hp, wp)[:, :, :h, :w]
    print(f"[7] reshape + 裁剪到原始  : {tuple(outp.shape)}")

    # --- proj / BN / SiLU ---
    y = m.act(m.norm(m.proj(outp)))
    print(f"[8] y = SiLU(BN(proj(out))): {tuple(y.shape)}   (hidden={m.hidden} -> c2={c1})")

    # --- 门控残差 ---
    g = torch.sigmoid(m.gate(x))
    gb = m.gate.bias.detach().float().mean().item()
    print(
        f"[9] gate = sigmoid(conv(x)): {tuple(g.shape)}   bias均值={gb:.1f} -> sigmoid={torch.sigmoid(torch.tensor(gb)).item():.3f}"
    )
    out_final = x + y * g
    print(f"[10] 输出 = x + y * gate  : {tuple(out_final.shape)}")

    # --- 校验与原始模块一致 ---
    with torch.no_grad():
        ref = m(x)
    print(f"\n✔ 追踪结果与模块真实输出一致: {torch.allclose(out_final, ref, atol=1e-5)}")


if __name__ == "__main__":
    trace(128, 80, 80)  # P3 级（yolo26-cgba.yaml 层 5）
    trace(128, 40, 40)  # P4 级（层 8）
