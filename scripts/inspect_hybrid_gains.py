#!/usr/bin/env python3

"""Print per-layer FoHSTU hybrid tail gains from training checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint has no state dictionary: {path}")
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.checkpoints:
        state_dict = _load_state_dict(path)
        rho_items = sorted(
            (key, value)
            for key, value in state_dict.items()
            if key.endswith("._hybrid_tail_rho")
        )
        if not rho_items:
            raise ValueError(f"no hybrid tail gains found in {path}")
        for key, rho in rho_items:
            rho_values = rho.detach().float()
            alpha_values = 2.0 * torch.tanh(rho_values / 2.0)
            print(
                json.dumps(
                    {
                        "checkpoint": str(path),
                        "parameter": key,
                        "rho": rho_values.tolist(),
                        "alpha": alpha_values.tolist(),
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
