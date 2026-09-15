import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402


SPLIT_NAMES = {
    -99999: "spinup",
    -1: "train",
    0: "selection",
    1: "testing",
}


def safe_value(value):
    return -99999 if np.isnan(value) else float(value)


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, kge_a, kge_b, kgess = KGE(sim, obs)
    return {
        "NSE": float(NS(sim, obs)),
        "KGE": safe_value(kge),
        "KGE-A": float(kge_a),
        "KGE-B": float(kge_b),
        "Corr": safe_value(corr),
        "MSE": float(mean_squared_error(obs, sim)),
        "KGEss": safe_value(kgess),
    }


def resolve_path(value, default_path):
    path = Path(value) if value else default_path
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def load_data(data_dir, device):
    forcing_file = data_dir / "LeafRiverDaily_43YR.txt"
    flag_file = data_dir / "LeafRiverDaily_43YR_Flag.txt"

    data = pd.read_csv(
        forcing_file,
        header=None,
        sep=r"\s+",
        names=["P", "PET", "Q"],
    )
    flags = pd.read_csv(
        flag_file,
        header=None,
        sep=r"\s+",
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError("Forcing/flow data and skill flags have different lengths.")

    x = torch.tensor(
        data[["P", "PET"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    y = torch.tensor(
        data[["Q"]].to_numpy(),
        dtype=torch.float32,
        device=device,
    )
    flag_t = torch.tensor(flags.to_numpy(), device=device)

    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }
    return data, flags, x, y, masks


def evaluate_splits(predictions, y, masks):
    rows = []
    for split in ("train", "selection", "testing", "spinup"):
        sim = torch.masked_select(predictions, masks[split]).unsqueeze(1)
        obs = torch.masked_select(y, masks[split]).unsqueeze(1)
        row = {"split": split, "n": int(sim.numel())}
        row.update(calculate_metrics(sim, obs))
        rows.append(row)
    return pd.DataFrame(rows)


def save_common_outputs(output_dir, data, flags, predictions, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries = data.copy()
    timeseries["Qsim"] = predictions.detach().cpu().numpy().reshape(-1)
    timeseries["Flag"] = flags.to_numpy()
    timeseries["Split"] = timeseries["Flag"].map(SPLIT_NAMES)
    timeseries.to_csv(output_dir / "evaluation_timeseries.csv", index=False)

    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    print(metrics.to_string(index=False))


from MCPBRNN_lib_tools.MCNZoo import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate,
    MCPBRNN_GWVariant_Routing,
    MCPBRNN_SW_Variant_Routing,
)


class Model(nn.Module):
    def __init__(self, spin_len, train_time_len):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(
            input_size=1, hidden_size=1, gate_dim=1,
            spinLen=spin_len, traintimeLen=train_time_len,
            initial_forget_bias=0,
        )
        self.MCPBRNNNode_SW = MCPBRNN_SW_Variant_Routing(
            input_size=1, hidden_size=1, gate_dim=1,
            spinLen=spin_len, traintimeLen=train_time_len,
            initial_forget_bias=0,
        )
        self.MCPBRNNNode_GW = MCPBRNN_GWVariant_Routing(
            input_size=1, hidden_size=1, gate_dim=1,
            spinLen=spin_len, traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(
        self, x, time_lag, y_obs, c_mean, c_std,
        sw_mean, sw_std, gw_mean, gw_std, initial_gw_storage,
    ):
        (
            main_q, main_storage, loss, loss_constrained, bypass, recharge,
            gate_i, gate_o, gate_l, gate_l_constrained, gate_f, gate_ogw,
            _, _,
        ) = self.MCPBRNNNode(x, 0, time_lag, y_obs, c_mean, c_std)

        sw_q, sw_storage, gate_o_sw, gate_f_sw = self.MCPBRNNNode_SW(
            main_q, 0, time_lag, y_obs, sw_mean, sw_std
        )
        gw_q, gw_storage, gate_o_gw, gate_f_gw = self.MCPBRNNNode_GW(
            recharge, 0, time_lag, y_obs, gw_mean, gw_std, initial_gw_storage
        )

        total_q = sw_q + gw_q
        return (
            total_q, main_q, sw_q, gw_q,
            main_storage, sw_storage, gw_storage,
            loss, loss_constrained, recharge, bypass,
            gate_i, gate_o, gate_o_sw, gate_o_gw,
            gate_l, gate_l_constrained,
            gate_f, gate_f_sw, gate_f_gw, gate_ogw,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained MCP MCA5 model only; no training.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument("--SW_mean", type=float, default=3.051248249)
    parser.add_argument("--SW_std", type=float, default=3.365261828)
    parser.add_argument("--GW_mean", type=float, default=1.87176748)
    parser.add_argument("--GW_std", type=float, default=2.085571043)
    parser.add_argument("--Ini_C", type=float, default=1.6871063)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    checkpoint = resolve_path(args.checkpoint, SCRIPT_DIR / "model_epoch619.pt")
    data_dir = resolve_path(
        args.data_dir,
        PROJECT_DIR / "20220527-MDUPLEX-LeafRiver",
    )
    output_dir = resolve_path(
        args.output_dir,
        SCRIPT_DIR / "evaluation_model3",
    )

    spin_len = 1095 - args.time_lag
    train_time_len = 8400 - args.time_lag
    data, flags, x, y, masks = load_data(data_dir, device)

    model = Model(spin_len, train_time_len).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()

    with torch.no_grad():
        result = model(
            x, args.time_lag, y,
            args.c_mean, args.c_std,
            args.SW_mean, args.SW_std,
            args.GW_mean, args.GW_std,
            args.Ini_C,
        )

    predictions = result[0]
    metrics = evaluate_splits(predictions, y, masks)
    save_common_outputs(output_dir, data, flags, predictions, metrics)

    diagnostic_names = [
        "Qsim_total", "Q_main", "Q_surface", "Q_groundwater",
        "storage_main", "storage_surface", "storage_groundwater",
        "loss", "loss_constrained", "recharge", "bypass",
        "gate_i", "gate_o_main", "gate_o_surface", "gate_o_groundwater",
        "gate_l", "gate_l_constrained",
        "gate_f_main", "gate_f_surface", "gate_f_groundwater", "gate_ogw",
    ]
    for name, tensor in zip(diagnostic_names, result):
        pd.DataFrame(tensor.detach().cpu().numpy()).to_csv(
            output_dir / f"{name}.csv", index=False
        )

    print(f"\nCheckpoint: {checkpoint}")
    print(f"Outputs:    {output_dir}")


if __name__ == "__main__":
    main()
