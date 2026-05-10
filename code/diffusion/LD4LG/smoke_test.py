import torch
from .autoencoder import AutoencoderConfig, LanguageAutoencoder
from .denoiser import Denoiser, DenoiserConfig
from .diffusion import DiffusionConfig, GaussianDiffusion
from .tokenizer import AATokenizer


def main():
    # tiny config so the test runs fast
    tok = AATokenizer()
    ae_cfg = AutoencoderConfig(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id,
        max_source_len=64, max_target_len=64,
        dim=128, num_heads=4, ffn_mult=2,
        encoder_layers=2, decoder_layers=2,
        compress_layers=2, reconstruct_layers=2,
        latent_len=8, latent_dim=16,
        dropout=0.0,
    )
    ae = LanguageAutoencoder(ae_cfg)

    B, L = 3, 64
    src = torch.randint(4, tok.vocab_size, (B, L))   # only real AA tokens
    src[:, 0]  = tok.bos_id
    src[:, -1] = tok.eos_id
    dec_in = src.clone(); dec_in[:, 1:] = src[:, :-1]; dec_in[:, 0] = tok.bos_id
    tgt = src.clone()

    loss, logits = ae(src, dec_in, tgt)
    print(f"[autoencoder] loss={loss.item():.4f}  logits shape={tuple(logits.shape)}")
    loss.backward()

    d_cfg = DenoiserConfig(
        latent_len=ae_cfg.latent_len,
        latent_dim=ae_cfg.latent_dim,
        dim=128, num_heads=4, ffn_mult=2,
        num_layers=4, dense_connections=1,
        dropout=0.0,
        num_isotypes=4, num_v_families=4, num_light_loci=3,
    )
    denoiser = Denoiser(d_cfg, cond_drop_prob=0.1)
    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        cfg=DiffusionConfig(
            latent_dim=ae_cfg.latent_dim,
            self_cond_prob=0.5,
            num_sampling_steps=8,
            cfg_weight=2.0,
        ),
    )

    with torch.no_grad():
        x = ae.encode(src)
    iso  = torch.tensor([0, 1, 2])
    vfam = torch.tensor([0, 1, 2])
    loc  = torch.tensor([0, 1, 0])

    loss = diffusion.loss(x, iso, vfam, loc)
    print(f"[diffusion training loss] = {loss.item():.4f}")
    loss.backward()

    # sample + decode (untrained, so the output is garbage; we just check shapes)
    denoiser.eval(); ae.eval()
    with torch.no_grad():
        x_samp = diffusion.sample(
            batch_size=B, latent_len=ae_cfg.latent_len, latent_dim=ae_cfg.latent_dim,
            iso=iso, vfam=vfam, loc=loc, device=torch.device("cpu"),
            num_steps=8, cfg_weight=2.0,
        )
        ids = ae.generate_from_latent(x_samp, max_len=64)
    print(f"[sample] x shape={tuple(x_samp.shape)}  decoded ids shape={tuple(ids.shape)}")
    print(f"[decode] sample seq (untrained, garbage): {tok.decode(ids[0].tolist())[:40]}...")

    print("\nAll forward + backward passes OK.")


if __name__ == "__main__":
    main()