import copy

import ml_collections


def model_config(name: str) -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG)
    return cfg


CONFIG = ml_collections.ConfigDict(
    {
        "data": {},
        "model": {
            "global_config": {
                "deterministic": True,
                "subcollocation_size": 128,
                "w_init": "glorot_uniform",
                "residual_weight": 1.0,
                "bc_weight": 1.0,
                "ic_weights": 1.0,
            },
            "density_function": {"num_layer": 5, "latent_dim": 128},
            "equation": {"kn": 1.0},
        },
    }
)
