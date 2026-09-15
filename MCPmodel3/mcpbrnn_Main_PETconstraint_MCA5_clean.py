import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

# MCPBRNN_lib_tools is one directory above this script's working directory.
sys.path.insert(0, str(Path.cwd().parent))

from MCPBRNN_lib_tools.MCNZoo import (  # noqa: E402
    MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate,
    MCPBRNN_GWVariant_Routing,
    MCPBRNN_SW_Variant_Routing,
)
from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.Loss_Function import KGELoss  # noqa: E402


PARAMETER_COLUMNS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_yom_gw",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.bias_b0_yom_gw",
    "MCPBRNNNode.weight_b1_yom_gw",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
    "MCPBRNNNode_SW.weight_r_yom",
    "MCPBRNNNode_SW.weight_r_yfm",
    "MCPBRNNNode_SW.bias_b0_yom",
    "MCPBRNNNode_SW.weight_b1_yom",
    "MCPBRNNNode_GW.weight_r_yom",
    "MCPBRNNNode_GW.weight_r_yfm",
    "MCPBRNNNode_GW.bias_b0_yom",
    "MCPBRNNNode_GW.weight_b1_yom",
]

METRIC_COLUMNS = [
    "NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss",
    "NSE_selection", "KGE_selection", "KGE-A_selection", "KGE-B_selection",
    "Corr_selection", "mse_selection", "KGEss_selection",
    "NSE_testing", "KGE_testing", "KGE-A_testing", "KGE-B_testing",
    "Corr_testing", "mse_testing", "KGEss_testing",
    "NSE_spinup", "KGE_spinup", "KGE-A_spinup", "KGE-B_spinup",
    "Corr_spinup", "mse_spinup", "KGEss_spinup",
]

LAG_COLUMNS = ["KGEtimelag_1", "KGEtimelag_2", "KGEtimelag_3"]
SUMMARY_COLUMNS = PARAMETER_COLUMNS + METRIC_COLUMNS + LAG_COLUMNS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_no", type=int, default=0)
    parser.add_argument("--epoch_no", type=int, default=1)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--seed_no", type=int, default=2925)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
    parser.add_argument("--SW_mean", type=float, default=3.051248249)
    parser.add_argument("--SW_std", type=float, default=3.365261828)
    parser.add_argument("--GW_mean", type=float, default=1.87176748)
    parser.add_argument("--GW_std", type=float, default=2.085571043)
    parser.add_argument("--Ini_C", type=float, default=1.6871063)
    return parser.parse_args()


def safe_value(value):
    return -99999 if np.isnan(value) else value


def calculate_metrics(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()

    kge, corr, kge_a, kge_b, kgess = KGE(sim, obs)
    return [
        NS(sim, obs),
        safe_value(kge),
        kge_a,
        kge_b,
        safe_value(corr),
        mean_squared_error(obs, sim),
        safe_value(kgess),
    ]


def calculate_lag_kge(sim_t, obs_t):
    sim = sim_t.detach().cpu().numpy()
    obs = obs_t.detach().cpu().numpy()
    n = obs.shape[0]

    values = []
    for lag in (1, 2, 3):
        kge = KGE(sim[lag:n], obs[: n - lag])[0]
        values.append(kge)
    return values


class Model(nn.Module):
    """MCA5 wrapper; module names are retained for checkpoint compatibility."""

    def __init__(self, spin_len, train_time_len):
        super().__init__()

        self.MCPBRNNNode = MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )
        self.MCPBRNNNode_SW = MCPBRNN_SW_Variant_Routing(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )
        self.MCPBRNNNode_GW = MCPBRNN_GWVariant_Routing(
            input_size=1,
            hidden_size=1,
            gate_dim=1,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(
        self,
        x,
        epoch,
        time_lag,
        y_obs,
        c_mean,
        c_std,
        sw_mean,
        sw_std,
        gw_mean,
        gw_std,
        initial_gw_storage,
    ):
        (
            main_q,
            main_storage,
            loss,
            loss_constrained,
            bypass,
            recharge,
            gate_i,
            gate_o,
            gate_l,
            gate_l_constrained,
            gate_f,
            gate_ogw,
            _,
            _,
        ) = self.MCPBRNNNode(
            x, epoch, time_lag, y_obs, c_mean, c_std
        )

        sw_q, sw_storage, gate_o_sw, gate_f_sw = self.MCPBRNNNode_SW(
            main_q, epoch, time_lag, y_obs, sw_mean, sw_std
        )
        gw_q, gw_storage, gate_o_gw, gate_f_gw = self.MCPBRNNNode_GW(
            recharge,
            epoch,
            time_lag,
            y_obs,
            gw_mean,
            gw_std,
            initial_gw_storage,
        )

        total_q = sw_q + gw_q

        # Preserve the original wrapper return order exactly.
        return (
            total_q,
            main_q,
            sw_q,
            gw_q,
            main_storage,
            sw_storage,
            gw_storage,
            loss,
            loss_constrained,
            recharge,
            bypass,
            gate_i,
            gate_o,
            gate_o_sw,
            gate_o_gw,
            gate_l,
            gate_l_constrained,
            gate_f,
            gate_f_sw,
            gate_f_gw,
            gate_ogw,
        )


def main():
    args = parse_args()

    np.random.seed(args.seed_no)
    torch.manual_seed(args.seed_no)

    time_lag = args.time_lag
    spin_len = 1095 - time_lag
    train_time_len = 8400 - time_lag

    learning_rate = 0.025
    learning_rates = {300: 0.0125, 600: 0.0125}

    case_name = f"MCPBRNN_Generic_PETconstraint_MCA5_1Layer_1node_{args.case_no}"
    case_dir = Path(case_name)
    case_dir.mkdir(exist_ok=True)

    initial_checkpoint = Path("model_epoch619.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data = pd.read_csv(
        "../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR.txt",
        header=None,
        sep=r"\s+",
        names=["P", "PET", "Q"],
    )
    flags = pd.read_csv(
        "../20220527-MDUPLEX-LeafRiver/LeafRiverDaily_43YR_Flag.txt",
        header=None,
        sep=r"\s+",
        names=["Flag"],
    )["Flag"]

    if len(data) != len(flags):
        raise ValueError("Forcing/flow data and skill flags have different lengths.")

    # seq_length is always 1, so the original rolling-window loop is unnecessary.
    x = torch.tensor(data[["P", "PET"]].to_numpy(), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(data[["Q"]].to_numpy(), dtype=torch.float32)

    x = x.to(device)
    y = y.to(device)

    flag_t = torch.tensor(flags.to_numpy(), device=device)
    masks = {
        "train": flag_t.eq(-1).unsqueeze(1),
        "selection": flag_t.eq(0).unsqueeze(1),
        "testing": flag_t.eq(1).unsqueeze(1),
        "spinup": flag_t.eq(-99999).unsqueeze(1),
    }

    # ------------------------------------------------------------------
    # Model / training
    # ------------------------------------------------------------------
    model = Model(spin_len, train_time_len).to(device)
    model.load_state_dict(torch.load(initial_checkpoint, map_location=device))

    loss_func = KGELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Save the starting checkpoint, matching the original workflow.
    torch.save(model.state_dict(), case_dir / "model_epoch0.pt")

    out_matrix = np.zeros((args.epoch_no, len(SUMMARY_COLUMNS)))

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        predictions = model(
            x,
            epoch,
            time_lag,
            y,
            args.c_mean,
            args.c_std,
            args.SW_mean,
            args.SW_std,
            args.GW_mean,
            args.GW_std,
            args.Ini_C,
        )[0]

        sim_train = torch.masked_select(predictions, masks["train"]).unsqueeze(1)
        obs_train = torch.masked_select(y, masks["train"]).unsqueeze(1)

        loss = loss_func(sim_train, obs_train)
        loss.backward()
        optimizer.step()

        # Re-run after the update for skill calculation, matching the original code.
        with torch.no_grad():
            predictions = model(
                x,
                epoch,
                time_lag,
                y,
                args.c_mean,
                args.c_std,
                args.SW_mean,
                args.SW_std,
                args.GW_mean,
                args.GW_std,
                args.Ini_C,
            )[0]

            split_tensors = {}
            split_metrics = []
            for split in ("train", "selection", "testing", "spinup"):
                sim = torch.masked_select(predictions, masks[split]).unsqueeze(1)
                obs = torch.masked_select(y, masks[split]).unsqueeze(1)
                split_tensors[split] = (sim, obs)
                split_metrics.extend(calculate_metrics(sim, obs))

        state = model.state_dict()
        parameter_values = [
            state[name].detach().cpu().reshape(-1)[0].item()
            for name in PARAMETER_COLUMNS
        ]

        lag_metrics = calculate_lag_kge(*split_tensors["train"])
        out_matrix[epoch - 1, :] = parameter_values + split_metrics + lag_metrics

        kge_selection_col = len(PARAMETER_COLUMNS) + 8
        print(
            f"Epoch {epoch}: "
            f"KGE_selection = {out_matrix[epoch - 1, kge_selection_col]:.6f}"
        )

        torch.save(model.state_dict(), case_dir / f"model_epoch{epoch}.pt")

    # ------------------------------------------------------------------
    # Save summary and evaluate the best selection-KGE checkpoint
    # ------------------------------------------------------------------
    pd.DataFrame(out_matrix, columns=SUMMARY_COLUMNS).to_csv(
        case_dir / f"IC_caseno_{args.case_no}_summary.csv"
    )

    kge_selection_col = len(PARAMETER_COLUMNS) + 8
    best_epoch = int(np.argmax(out_matrix[:, kge_selection_col]) + 1)
    best_checkpoint = case_dir / f"model_epoch{best_epoch}.pt"

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        result = model(
            x,
            best_epoch,
            time_lag,
            y,
            args.c_mean,
            args.c_std,
            args.SW_mean,
            args.SW_std,
            args.GW_mean,
            args.GW_std,
            args.Ini_C,
        )

    # Export every physically useful state/flux/gate from the selected model.
    outputs = {
        f"Outhidden_{args.case_no}_summary.csv": result[0],
        f"Outhidden_Main_{args.case_no}_summary.csv": result[1],
        f"Outhidden_SW_{args.case_no}_summary.csv": result[2],
        f"Outhidden_GW_{args.case_no}_summary.csv": result[3],
        f"Outcell_{args.case_no}_summary.csv": result[4],
        f"Outcell_SW_{args.case_no}_summary.csv": result[5],
        f"Outcell_GW_{args.case_no}_summary.csv": result[6],
        f"Outloss_{args.case_no}_summary.csv": result[7],
        f"Outlossc_{args.case_no}_summary.csv": result[8],
        f"Outrecharge_{args.case_no}_summary.csv": result[9],
        f"Outbypass_{args.case_no}_summary.csv": result[10],
        f"Out_gatei_{args.case_no}_summary.csv": result[11],
        f"Out_gateo_{args.case_no}_summary.csv": result[12],
        f"Out_gateo_SW_{args.case_no}_summary.csv": result[13],
        f"Out_gateo_GW_{args.case_no}_summary.csv": result[14],
        f"Out_gatel_{args.case_no}_summary.csv": result[15],
        f"Out_gatelc_{args.case_no}_summary.csv": result[16],
        f"Out_gatef_{args.case_no}_summary.csv": result[17],
        f"Out_gatef_SW_{args.case_no}_summary.csv": result[18],
        f"Out_gatef_GW_{args.case_no}_summary.csv": result[19],
        f"Out_gateogw_{args.case_no}_summary.csv": result[20],
    }

    for filename, tensor in outputs.items():
        pd.DataFrame(tensor.detach().cpu().numpy()).to_csv(case_dir / filename)

    pd.DataFrame([best_epoch], columns=["Best_Epoch"]).to_csv(
        case_dir / f"Best_Epoch_Caseno{args.case_no}_summary.csv"
    )

    final_checkpoint = case_dir / f"best_model_epoch{best_epoch}.pt"
    best_checkpoint.replace(final_checkpoint)

    # Remove all temporary epoch checkpoints, including model_epoch0.pt.
    for checkpoint in case_dir.glob("model_epoch*.pt"):
        checkpoint.unlink()

    print(f"Best epoch = {best_epoch}")
    print(
        "Best KGE_selection = "
        f"{out_matrix[best_epoch - 1, kge_selection_col]:.6f}"
    )
    print(f"Saved best model to: {final_checkpoint}")


if __name__ == "__main__":
    main()
