"""Model factory: builds the right diffusion model from a Config (also for old checkpoints)."""


def build_model(config, load_weights: bool = True):
    """load_weights=False skips reading the lc0 .pb.gz (use when a checkpoint state dict follows)."""
    model_type = getattr(config, "model_type", "masked_diffusion")
    if model_type == "masked_diffusion":
        from MaskedDiffusion.model import MaskedDiffusion
        return MaskedDiffusion(config)
    if model_type == "bt4":
        from BT4.diffusion_model import BT4Diffusion
        return BT4Diffusion(config, load_weights=load_weights)
    raise ValueError(f"unknown model_type {model_type!r}")
