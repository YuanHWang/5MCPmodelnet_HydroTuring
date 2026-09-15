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
    MCPBRNN_PETconstraint_MassRelax_Regular,
)
from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.Loss_Function import KGELoss  # noqa: E402


PARAMETER_COLUMNS = [
    "MCPBRNNNode.weight_r_yom",
    "MCPBRNNNode.weight_r_ylm",
    "MCPBRNNNode.weight_r_yfm",
    "MCPBRNNNode.weight_r_yvm",
    "MCPBRNNNode.bias_b0_yom",
    "MCPBRNNNode.weight_b1_yom",
    "MCPBRNNNode.bias_b0_ylm",
    "MCPBRNNNode.weight_b2_ylm",
    "MCPBRNNNode.weight_s_yvm",
    "MCPBRNNNode.bias_b0_yrm",
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
    parser.add_argument("--hidden_size", type=int, default=1)
    parser.add_argument("--c_mean", type=float, default=475.0253683)
    parser.add_argument("--c_std", type=float, default=89.86101728)
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

    scores = []
    for lag in (1, 2, 3):
        kge, *_ = KGE(sim[lag:n], obs[: n - lag])
        scores.append(safe_value(kge))
    return scores


class Model(nn.Module):
    """Thin wrapper retained so existing checkpoint keys remain compatible."""

    def __init__(self, hidden_size, spin_len, train_time_len):
        super().__init__()
        self.MCPBRNNNode = MCPBRNN_PETconstraint_MassRelax_Regular(
            input_size=1,
            hidden_size=hidden_size,
            gate_dim=1,
            gate_dim_ucorr=3,
            spinLen=spin_len,
            traintimeLen=train_time_len,
            initial_forget_bias=0,
        )

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std):
        result = self.MCPBRNNNode(
            x, epoch, time_lag, y_obs, c_mean, c_std
        )
        # Preserve the original wrapper ordering.
        return (
            result[0],   # out
            result[0],   # h_t
            result[1],   # c_t
            result[2],   # l_t
            result[3],   # lc_t
            result[4],   # bypass
            result[5],   # input gate
            result[6],   # output gate
            result[7],   # loss gate
            result[8],   # constrained loss gate
            result[9],   # forget gate
            result[10],  # h_nout
            result[11],  # obs_std
            result[12],  # mass-relaxation gate
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

    case_name = (
        "MCPBRNN_PETconstraint_MCRelaxed_Generic_"
        f"1Layer_1node_{args.case_no}_Relaxed"
    )
    case_dir = Path(case_name)
    case_dir.mkdir(exist_ok=True)

    initial_checkpoint = Path("model_epoch3.pt")
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

    # seq_length is always 1, so no rolling-window construction is necessary.
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
    model = Model(args.hidden_size, spin_len, train_time_len).to(device)
    model.load_state_dict(torch.load(initial_checkpoint, map_location=device))

    loss_func = KGELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    torch.save(model.state_dict(), case_dir / "model_epoch0.pt")

    out_matrix = np.zeros((args.epoch_no, len(SUMMARY_COLUMNS)))

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        # Match the original script: objective uses wrapper return index 1.
        predictions = model(
            x, epoch, time_lag, y, args.c_mean, args.c_std
        )[1]

        sim_train = torch.masked_select(predictions, masks["train"]).unsqueeze(1)
        obs_train = torch.masked_select(y, masks["train"]).unsqueeze(1)

        loss = loss_func(sim_train, obs_train)
        loss.backward()
        optimizer.step()

        # Re-evaluate after updating the parameters.
        with torch.no_grad():
            predictions = model(
                x, epoch, time_lag, y, args.c_mean, args.c_std
            )[1]

            split_tensors = {}
            metric_values = []
            for split in ("train", "selection", "testing", "spinup"):
                sim = torch.masked_select(predictions, masks[split]).unsqueeze(1)
                obs = torch.masked_select(y, masks[split]).unsqueeze(1)
                split_tensors[split] = (sim, obs)
                metric_values.extend(calculate_metrics(sim, obs))

        state = model.state_dict()
        parameter_values = [
            state[name].detach().cpu().reshape(-1)[0].item()
            for name in PARAMETER_COLUMNS
        ]
        lag_values = calculate_lag_kge(*split_tensors["train"])

        out_matrix[epoch - 1, :] = (
            parameter_values + metric_values + lag_values
        )

        kge_selection_col = len(PARAMETER_COLUMNS) + 8
        print(
            f"Epoch {epoch}: "
            f"KGE_selection = {out_matrix[epoch - 1, kge_selection_col]:.6f}"
        )

        torch.save(model.state_dict(), case_dir / f"model_epoch{epoch}.pt")

    # ------------------------------------------------------------------
    # Save summary and select best checkpoint by KGE_selection.
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
            x, best_epoch, time_lag, y, args.c_mean, args.c_std
        )

    outputs = {
        f"Outhidden_{args.case_no}_summary.csv": result[1],
        f"Outcell_{args.case_no}_summary.csv": result[2],
        f"Outloss_{args.case_no}_summary.csv": result[3],
        f"Outlossc_{args.case_no}_summary.csv": result[4],
        f"Outbypass_{args.case_no}_summary.csv": result[5],
        f"Out_gatei_{args.case_no}_summary.csv": result[6],
        f"Out_gateo_{args.case_no}_summary.csv": result[7],
        f"Out_gatel_{args.case_no}_summary.csv": result[8],
        f"Out_gatelc_{args.case_no}_summary.csv": result[9],
        f"Out_gatef_{args.case_no}_summary.csv": result[10],
        f"Out_gatemr_{args.case_no}_summary.csv": result[13],
    }

    for filename, tensor in outputs.items():
        pd.DataFrame(tensor.detach().cpu().numpy()).to_csv(case_dir / filename)

    pd.DataFrame([best_epoch], columns=["Best_Epoch"]).to_csv(
        case_dir / f"Best_Epoch_Caseno{args.case_no}_summary.csv"
    )

    final_checkpoint = case_dir / f"best_model_epoch{best_epoch}.pt"
    best_checkpoint.replace(final_checkpoint)

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
