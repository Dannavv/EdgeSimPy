import pandas as pd
import matplotlib.pyplot as plt


REQ_CSV = "experiment_markov_fog_requests.csv"
SRV_CSV = "experiment_markov_fog_servers.csv"
USR_CSV = "experiment_markov_fog_users.csv"


def load_data():
    df_req = pd.read_csv(REQ_CSV)
    df_srv = pd.read_csv(SRV_CSV)
    df_usr = pd.read_csv(USR_CSV)

    # Make sure types are clean
    df_req["t"] = df_req["t"].astype(int)
    df_srv["Time Step"] = df_srv["Time Step"].astype(int)
    df_usr["Time Step"] = df_usr["Time Step"].astype(int)

    # sla_violated might be stored as bool, int, or string; normalize to 0/1 float
    if df_req["sla_violated"].dtype == bool:
        df_req["sla_violated"] = df_req["sla_violated"].astype(float)
    else:
        df_req["sla_violated"] = df_req["sla_violated"].astype(float)

    return df_req, df_srv, df_usr




if __name__ == "__main__":
    df_req, df_srv, df_usr = load_data()

    # plot_avg_delay_over_time(df_req)
    # plot_sla_violation_over_time(df_req)
    # plot_fog_vs_cloud_share(df_req)
    # plot_server_cpu_util(df_srv)
    # plot_server_energy(df_srv)
    # plot_delay_hist_by_layer(df_req)
    # plot_user_positions_last_step(df_usr)

    
# --- 1) Delay over time (built using per-request log) ---

    if not df_req.empty:
        delay_time = (
            df_req.groupby("t")["total_delay"]
            .mean()
            .reset_index(name="avg_total_delay")
        )

        plt.figure(figsize=(10, 6))
        plt.plot(delay_time["t"], delay_time["avg_total_delay"])
        plt.xlabel("Step")
        plt.ylabel("Average total delay")
        plt.title("Average request delay over steps")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Requests per step
        req_count = (
            df_req.groupby("t")["user_id"]
            .count()
            .reset_index(name="num_requests")
        )

        plt.figure(figsize=(10, 6))
        plt.plot(req_count["t"], req_count["num_requests"])
        plt.xlabel("Step")
        plt.ylabel("Number of requests")
        plt.title("Request count per step")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # SLA violation rate over time
        sla_time = (
            df_req.groupby("t")["sla_violated"]
            .mean()
            .reset_index(name="sla_violation_rate")
        )

        plt.figure(figsize=(10, 6))
        plt.plot(sla_time["t"], sla_time["sla_violation_rate"])
        plt.xlabel("Step")
        plt.ylabel("SLA violation rate")
        plt.title("SLA violation rate over steps")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # --- 2) Server metrics over time (from built-in agent_metrics snapshots) ---

    if not df_srv.empty:
        # EdgeSimPy usually logs a time/step column; handle common names
        if "Step" in df_srv.columns:
            step_col = "Step"
        elif "t" in df_srv.columns:
            step_col = "t"
        elif "time" in df_srv.columns:
            step_col = "time"
        else:
            # If there is no explicit step column, assume snapshots are in order
            df_srv = df_srv.copy()
            df_srv["Step"] = df_srv.groupby("Instance ID").cumcount()
            step_col = "Step"

        # CPU Util per server over time
        if "CPU Util" in df_srv.columns:
            plt.figure(figsize=(10, 6))
            for name, g in df_srv.groupby("Name"):
                plt.plot(g[step_col], g["CPU Util"], label=name)
            plt.xlabel("Step")
            plt.ylabel("CPU Utilisation")
            plt.title("Server CPU utilisation over steps")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Energy per server over time
        if "Energy" in df_srv.columns:
            plt.figure(figsize=(10, 6))
            for name, g in df_srv.groupby("Name"):
                plt.plot(g[step_col], g["Energy"], label=name)
            plt.xlabel("Step")
            plt.ylabel("Energy (arbitrary units)")
            plt.title("Server energy accumulation over steps")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


    # --- 3) User mobility over time (built-in coordinates from agent_metrics) ---

    if not df_usr.empty:
        # Again, detect step column
        if "Step" in df_usr.columns:
            u_step_col = "Step"
        elif "t" in df_usr.columns:
            u_step_col = "t"
        elif "time" in df_usr.columns:
            u_step_col = "time"
        else:
            df_usr = df_usr.copy()
            df_usr["Step"] = df_usr.groupby("Instance ID").cumcount()
            u_step_col = "Step"

        # Extract x, y from built-in Coordinates
        if "Coordinates" in df_usr.columns:
            df_usr = df_usr.copy()
            df_usr[["x", "y"]] = df_usr["Coordinates"].apply(
                lambda c: pd.Series(c if c is not None else [None, None])
            )

            plt.figure(figsize=(10, 6))
            for uid, g in df_usr.groupby("Instance ID"):
                plt.plot(g["x"], g["y"], alpha=0.2)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("User trajectories (built from agent coordinates over steps)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

