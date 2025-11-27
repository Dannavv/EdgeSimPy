import pandas as pd

from test2 import run_experiment_markov_edgesimpy
from visualize import generate_all_visualizations


def summarize_run(params, df_srv, df_usr, df_bs):
    row = {}

    for k, v in params.items():
        row[f"params.{k}"] = v

    if df_srv is not None and not df_srv.empty:
        cloud = df_srv[df_srv["Layer"] == "cloud"] 
        fog = df_srv[df_srv["Layer"] == "fog"]

        row["avg_delay_cloud"] = cloud["Avg Delay"].mean() if not cloud.empty else 0.0
        row["avg_delay_fog"] = fog["Avg Delay"].mean() if not fog.empty else 0.0

        row["cloud_cpu_max"] = cloud["CPU Util"].max() if not cloud.empty else 0.0
        row["fog_cpu_max"] = fog["CPU Util"].max() if not fog.empty else 0.0

        row["cloud_energy"] = cloud["Energy"].max() if not cloud.empty else 0.0
        row["fog_energy"] = fog["Energy"].max() if not fog.empty else 0.0
    else:
        row["avg_delay_cloud"] = 0.0
        row["avg_delay_fog"] = 0.0
        row["cloud_cpu_max"] = 0.0
        row["fog_cpu_max"] = 0.0
        row["cloud_energy"] = 0.0
        row["fog_energy"] = 0.0

    if df_usr is not None and not df_usr.empty:
        last_sla = (
            df_usr.sort_values("Time Step")
                  .groupby("Instance ID")["SLA Violation Rate"]
                  .last()
        )
        row["sla_violation"] = last_sla.mean()

        tmp = df_usr.sort_values(["Instance ID", "Time Step"]).copy()
        tmp["PrevBS"] = tmp.groupby("Instance ID")["Base Station"].shift(1)
        handoffs = (tmp["Base Station"] != tmp["PrevBS"]).sum()
        row["handoffs"] = int(handoffs)
    else:
        row["sla_violation"] = 0.0
        row["handoffs"] = 0

    return row


def run_sweep():
    configs = []

    n_users_list = [200, 500, 1000, 1500, 2000]
    move_speed_list = [5, 10, 15]
    max_fog_distance_list = [100, 150, 200]

    max_steps = 400
    delay_sla = 15
    area_width = 800
    area_height = 200
    data_size_min = 100
    data_size_max = 1000
    fog_mips = 8000

    seed = 0

    for n_users in n_users_list:
        for move_speed in move_speed_list:
            for max_fog_distance in max_fog_distance_list:
                params = {
                    "n_users": n_users,
                    "move_speed": move_speed,
                    "max_fog_distance": max_fog_distance,
                    "max_steps": max_steps,
                }
                print("Running config:", params)

                sim, df_srv, df_usr, df_bs = run_experiment_markov_edgesimpy(
                    n_users=n_users,
                    max_steps=max_steps,
                    max_fog_distance=max_fog_distance,
                    delay_sla=delay_sla,
                    area_width=area_width,
                    area_height=area_height,
                    move_speed=move_speed,
                    data_size_min=data_size_min,
                    data_size_max=data_size_max,
                    fog_mips=fog_mips,
                    seed=seed,
                )

                row = summarize_run(params, df_srv, df_usr, df_bs)
                configs.append(row)

    df_all = pd.DataFrame(configs)
    df_all.to_csv("experiment_markov_sweep_summary.csv", index=False)
    print("Saved sweep summary → experiment_markov_sweep_summary.csv")

    generate_all_visualizations(df_all)
    print("Generated all visualizations from visualize.py")


if __name__ == "__main__":
    run_sweep()
