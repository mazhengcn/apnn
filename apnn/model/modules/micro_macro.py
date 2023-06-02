import dataclasses
from typing import Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

from apnn.model.integrate import quad
from apnn.model.tf import project_features
from apnn.utils import mean_squared_loss_fn

FEATURES = project_features.FEATURES
VMAP_AXES = {k: 0 if True else None for k in FEATURES}


@dataclasses.dataclass
class MicroMacro(hk.Module):
    config: ConfigDict
    name: Optional[str] = "pinn"

    def __call__(self, batch, is_training, compute_loss=False, compute_metrics=False):
        c = self.config
        gc = self.config.global_config
        ret = {}

        def rho(t, x):
            tx = jnp.concatenate([t, x], axis=-1)
            return hk.nets.MLP(c.density.output_sizes, name="macro")(tx)

        def g(t, x, v):
            txv = jnp.concatenate([t, x, v], axis=-1)
            return hk.nets.MLP(c.density.ouput_sizes, name="micro")(txv)

        t = batch["time_stampes"]
        x, v = jnp.split(batch["phase_coords"], axis=-1)

        predicted_rho = rho(t, x)
        predicted_g = g(t, x)
        predictions = rho + self.kn * g
        ret.update(
            {
                "predictions": predictions,
                "predicted_rho": predicted_rho,
                "predicted_g": predicted_g,
            }
        )

        if compute_loss:
            quad_v = batch["velocity_coords"]
            quad_w = batch["velocity_weights"]

            def residual(t, x, v):
                grad_t, grad_x = hk.grad(rho, argnums=(0, 1))(t, x)

                residual = grad_t + v * grad_x - (rho - f)
                return residual

            total_loss = 0.0

            residual_loss = mean_squared_loss_fn(hk.vmap(residual)(t, x, v), 0.0)
            total_loss += residual_loss
            ret["loss"] = {"residual_loss": residual_loss}

            boundary_x, boundary_v = jnp.split(batch["boundary_coords"], axis=-1)
            bc = f(t, boundary_x, boundary_v) - batch["boundary_labels"]
            bc_loss = mean_squared_loss_fn(bc)
            total_loss += bc_loss
            ret["loss"] = {"bc_loss": bc_loss}

            initial_x, initial_v = jnp.split(batch["initial_coords"], axis=-1)
            initial = f(0.0, initial_x, initial_v) - batch["initial_labels"]
            initial_loss = mean_squared_loss_fn(initial)
            total_loss += initial_loss
            ret["loss"] = {"initial_loss": initial_loss}

        if compute_metrics:
            labels = batch["labels"]
            mse = mean_squared_loss_fn(predictions, labels, axis=-1)
            relative_mse = mse / jnp.mean(labels**2)
            ret.update({"metrics": {"mse": mse, "rmspe": relative_mse}})

        if compute_loss:
            return total_loss, ret

        return ret
