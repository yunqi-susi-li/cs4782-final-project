import torch
from .diffusion import DPLMDiffusion, DPLMDiffusionConfig
from .model import DPLM, DPLMConfig
from .tokenizer import DPLMTokenizer

def main():
    cfg = DPLMConfig(
        vocab_size=25, max_len=64,
        dim=128, num_heads=4, num_layers=4, ffn_mult=2,
        dropout=0.0,
        num_isotypes=4, num_v_families=4, num_light_loci=3,
    )
    model = DPLM(cfg, cond_drop_prob=0.1)
    diff = DPLMDiffusion(
        model=model,
        cfg=DPLMDiffusionConfig(cfg_weight=2.0, num_sampling_steps=10),
    )

    B, L = 3, 64
    tok = DPLMTokenizer()
    tokens = torch.randint(4, 24, (B, L))   # only real AAs
    tokens[:, 0]  = tok.bos_id
    tokens[:, -1] = tok.eos_id
    iso  = torch.tensor([0, 1, 2])
    vfam = torch.tensor([0, 1, 2])
    loc  = torch.tensor([0, 1, 0])

    loss = diff.loss(tokens, iso, vfam, loc)
    print(f"[loss] {loss.item():.4f}")
    loss.backward()

    model.eval()
    with torch.no_grad():
        out = diff.sample(B, L, iso, vfam, loc, torch.device("cpu"),
                          num_steps=10, cfg_weight=2.0)
    print(f"[sample] shape={tuple(out.shape)}  first row: {out[0, :15].tolist()}")
    print(f"[decode] sample (untrained, garbage): {tok.decode(out[0].tolist())[:40]}...")

    print("\nDPLM smoke test OK.")


if __name__ == "__main__":
    main()