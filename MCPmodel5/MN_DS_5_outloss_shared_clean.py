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
    MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate,
)
from MCPBRNN_lib_tools.Eval_Metric import KGE, NS  # noqa: E402
from MCPBRNN_lib_tools.Loss_Function import KGELoss  # noqa: E402


METRIC_COLUMNS = [
    "NSE", "KGE", "KGE-A", "KGE-B", "Corr", "mse", "KGEss",
    "NSE_selection", "KGE_selection", "KGE-A_selection", "KGE-B_selection",
    "Corr_selection", "mse_selection", "KGEss_selection",
    "NSE_testing", "KGE_testing", "KGE-A_testing", "KGE-B_testing",
    "Corr_testing", "mse_testing", "KGEss_testing",
    "NSE_spinup", "KGE_spinup", "KGE-A_spinup", "KGE-B_spinup",
    "Corr_spinup", "mse_spinup", "KGEss_spinup",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_no", type=int, default=0)
    parser.add_argument("--epoch_no", type=int, default=1)
    parser.add_argument("--time_lag", type=int, default=0)
    parser.add_argument("--seed_no", type=int, default=2925)
    parser.add_argument("--hidden_size", type=int, default=5)
    parser.add_argument("--c_mean", type=float, default=412.9139085)
    parser.add_argument("--c_std", type=float, default=77.49943867)
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


class Model(nn.Module):
    """Thin wrapper retained so existing checkpoint keys remain compatible."""

    def __init__(self, hidden_size, spin_len, train_time_len):
        super().__init__()
        self.MCPBRNNNode = (
            MCPBRNN_Generic_PETconstraint_MultiParallel_MCA1_shareoutputlossGate_LossGateUpdate(
                input_size=1,
                hidden_size=hidden_size,
                gate_dim=1,
                gate_dim_ucorr=1,
                spinLen=spin_len,
                traintimeLen=train_time_len,
                initial_forget_bias=0,
            )
        )

    def forward(self, x, epoch, time_lag, y_obs, cmean, cstd):
        (
            out,
            hidden,
            cell,
            loss,
            loss_constrained,
            bypass,
            gate_i,
            gate_o,
            gate_l,
            gate_l_constrained,
            gate_f,
            obs_std,
        ) = self.MCPBRNNNode(x, epoch, time_lag, y_obs, cmean, cstd)

        # Preserve the original wrapper return order exactly.
        return (
            out,
            hidden,
            cell,
            loss,
            bypass,
            gate_i,
            gate_o,
            gate_l,
            gate_l_constrained,
            gate_f,
            obs_std,
            loss_constrained,
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
        f"MCN_Parallel_Node_{args.hidden_size}_case_{args.case_no}_"
        "outputlosscellshared_LossMultiINput"
    )
    case_dir = Path(case_name)
    case_dir.mkdir(exist_ok=True)

    initial_checkpoint = Path("best_model_epoch2707.pt")
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

    # seq_length is always 1 in this model, so no rolling-window loop is needed.
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
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    # Save the starting checkpoint, matching the original workflow.
    torch.save(model.state_dict(), case_dir / "model_epoch0.pt")

    out_matrix = np.zeros((args.epoch_no, len(METRIC_COLUMNS)))

    for epoch in range(1, args.epoch_no + 1):
        if epoch in learning_rates:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch]

        model.train()
        optimizer.zero_grad()

        predictions = model(x, epoch, time_lag, y, args.c_mean, args.c_std)[0]
        sim_train = torch.masked_select(predictions, masks["train"]).unsqueeze(1)
        obs_train = torch.masked_select(y, masks["train"]).unsqueeze(1)

        loss = loss_func(sim_train, obs_train)
        loss.backward()
        optimizer.step()

        # Re-run after the update for skill calculation, as in the original code.
        with torch.no_grad():
            predictions = model(x, epoch, time_lag, y, args.c_mean, args.c_std)[0]

            split_metrics = []
            for split in ("train", "selection", "testing", "spinup"):
                sim = torch.masked_select(predictions, masks[split]).unsqueeze(1)
                obs = torch.masked_select(y, masks[split]).unsqueeze(1)
                split_metrics.extend(calculate_metrics(sim, obs))

        out_matrix[epoch - 1, :] = split_metrics
        print(f"Epoch {epoch}: KGE_selection = {out_matrix[epoch - 1, 8]:.6f}")

        torch.save(model.state_dict(), case_dir / f"model_epoch{epoch}.pt")

    # ------------------------------------------------------------------
    # Save summary and evaluate the best selection-KGE checkpoint
    # ------------------------------------------------------------------
    pd.DataFrame(out_matrix, columns=METRIC_COLUMNS).to_csv(
        case_dir / f"IC_caseno_{args.case_no}_summary.csv"
    )

    best_epoch = int(np.argmax(out_matrix[:, 8]) + 1)
    best_checkpoint = case_dir / f"model_epoch{best_epoch}.pt"
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    model.eval()

    with torch.no_grad():
        result = model(x, best_epoch, time_lag, y, args.c_mean, args.c_std)

    outputs = {
        f"Outhidden_{args.case_no}_summary.csv": result[0],
        f"Outhidden_ALL_{args.case_no}_summary.csv": result[1],
        f"Outcell_{args.case_no}_summary.csv": result[2],
        f"Outloss_{args.case_no}_summary.csv": result[3],
        f"Outlossc_{args.case_no}_summary.csv": result[11],
        f"Out_gateo_{args.case_no}_summary.csv": result[6],
        f"Out_gatel_{args.case_no}_summary.csv": result[7],
        f"Out_gatelc_{args.case_no}_summary.csv": result[8],
        f"Out_gatef_{args.case_no}_summary.csv": result[9],
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


if __name__ == "__main__":
    main()
