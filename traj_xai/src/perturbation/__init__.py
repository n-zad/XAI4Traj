"""
Perturbation package for trajectory manipulation.
"""

from .perturbation import _perturbation
import numpy as np

from .gan.models import TrajectoryGenerator
import torch

# Convenience functions for backward compatibility
def gaussian_perturbation(segment, mean=0, std=3, scale=1.5):
    """Legacy function for Gaussian perturbation."""
    return _perturbation.apply(
        segment, ["gaussian"], {"gaussian": {"mean": mean, "std": std, "scale": scale}}
    )

def scaling_perturbation(segment, scale_factor=1.2):
    """Legacy function for scaling perturbation."""
    return _perturbation.apply(
        segment, ["scaling"], {"scaling": {"scale_factor": scale_factor}}
    )

def rotation_perturbation(segment, angle=np.pi / 18):
    """Legacy function for rotation perturbation."""
    return _perturbation.apply(segment, ["rotation"], {"rotation": {"angle": angle}})

def gan_perturbation(segment, scale=0.5, preserve_endpoints=False):
    """GAN perturbation"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load("../src/perturbation/gan/eth_8_model.pt", weights_only=True, map_location=device)
    args = checkpoint["args"]
    G = TrajectoryGenerator(
        obs_len=len(segment),#args.get("obs_len", 8),
        pred_len=len(segment),#args.get("pred_len", 8),
        embedding_dim=args.get("embedding_dim", 16),
        encoder_h_dim=args.get("encoder_h_dim_g", 32),
        decoder_h_dim=args.get("decoder_h_dim_g", 32),
        mlp_dim=args.get("mlp_dim", 64),
        bottleneck_dim=args.get("bottleneck_dim", 8),
        num_layers=args.get("num_layers", 1),
        noise_dim=tuple(args.get("noise_dim", (8,))) if isinstance(args.get("noise_dim", (8,)), (list, tuple)) else (args.get("noise_dim", 8),),
        noise_type=args.get("noise_type", "gaussian"),
        noise_mix_type=args.get("noise_mix_type", "global"),
        pooling_type=None if args.get("pooling_type", "none") in ("none", None) else args.get("pooling_type"),
        pool_every_timestep=bool(args.get("pool_every_timestep", False)),
        dropout=float(args.get("dropout", 0.0)),
        neighborhood_size=float(args.get("neighborhood_size", 2.0)),
        grid_size=int(args.get("grid_size", 8)),
        batch_norm=bool(args.get("batch_norm", False)),
    )
    G.load_state_dict(checkpoint["g_state"], strict=True)
    G.eval()
    return _perturbation.apply(segment, ["gan"], {"gan": {"G":G, "device":device, "preserve_endpoints":preserve_endpoints, "scale":scale}})

__all__ = [
    "gaussian_perturbation",
    "scaling_perturbation",
    "rotation_perturbation",
]
