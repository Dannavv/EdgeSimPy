import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from math import sqrt


# ==============================================================
#                 HELPER: extract parameters
# ==============================================================

def extract_param(df, param):
    return df[f"params.{param}"] if f"params.{param}" in df.columns else None


# ==============================================================
#                        HEATMAP 1:
#     SLA Violation Rate vs n_users vs move_speed
# ==============================================================

def heatmap_sla_users_speed(df, save="heatmap_sla_users_speed.png"):
    df["n_users"]     = extract_param(df, "n_users")
    df["move_speed"]  = extract_param(df, "move_speed")

    pivot = df.pivot_table(
        index="move_speed",
        columns="n_users",
        values="sla_violation",
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds")
    plt.title("SLA Violation Heatmap (Users vs Speed)")
    plt.xlabel("Number of Users")
    plt.ylabel("Movement Speed")
    plt.savefig(save)
    plt.close()


# ==============================================================
#                        HEATMAP 2:
#        Delay vs Fog Distance vs Movement Speed
# ==============================================================

def heatmap_delay_fogdist_speed(df, save="heatmap_delay_fogdist_speed.png"):
    df["max_fog_distance"] = extract_param(df, "max_fog_distance")
    df["move_speed"]       = extract_param(df, "move_speed")

    pivot = df.pivot_table(
        index="move_speed",
        columns="max_fog_distance",
        values="avg_delay_cloud",  # cloud delay is more sensitive
        aggfunc="mean"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Cloud Delay Heatmap (Fog Distance vs Speed)")
    plt.xlabel("Fog Distance")
    plt.ylabel("Movement Speed")
    plt.savefig(save)
    plt.close()



# ==============================================================
#                          3D SURFACE:
#       Delay vs Users vs Movement Speed (Fog or Cloud)
# ==============================================================

def plot_3d_surface_delay(df, mode="cloud", save="3d_delay_surface.png"):
    delay_col = "avg_delay_cloud" if mode=="cloud" else "avg_delay_fog"

    df["n_users"]     = extract_param(df, "n_users")
    df["move_speed"]  = extract_param(df, "move_speed")

    users = sorted(df["n_users"].unique())
    speeds = sorted(df["move_speed"].unique())

    Z = np.zeros((len(speeds), len(users)))

    for i, sp in enumerate(speeds):
        for j, u in enumerate(users):
            Z[i, j] = df[
                (df["n_users"]==u) & (df["move_speed"]==sp)
            ][delay_col].mean()

    X, Y = np.meshgrid(users, speeds)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis)

    ax.set_title(f"3D Delay Surface ({mode.title()} Layer)")
    ax.set_xlabel("Users")
    ax.set_ylabel("Movement Speed")
    ax.set_zlabel("Delay (ms)")

    plt.savefig(save)
    plt.close()



# ==============================================================
#                     SENSITIVITY CURVE:
#               Fog Distance vs Delay/SLA/Cloud Load
# ==============================================================

def sensitivity_fog_distance(df, save="fogdistance_sensitivity.png"):
    df["max_fog_distance"] = extract_param(df, "max_fog_distance")

    fogdist = sorted(df["max_fog_distance"].unique())
    delay = []
    cloud_load = []
    sla = []

    for d in fogdist:
        subset = df[df["max_fog_distance"] == d]
        delay.append(subset["avg_delay_cloud"].mean())
        cloud_load.append(subset["cloud_cpu_max"].mean())
        sla.append(subset["sla_violation"].mean())

    plt.figure(figsize=(12, 7))
    plt.plot(fogdist, delay, label="Cloud Delay", linewidth=2)
    plt.plot(fogdist, cloud_load, label="Cloud CPU Load", linewidth=2)
    plt.plot(fogdist, sla, label="SLA Violations", linewidth=2)

    plt.title("Fog Distance Sensitivity Curve")
    plt.xlabel("Fog Coverage Distance")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(save)
    plt.close()



# ==============================================================
#                         STRESS CURVE:
#                   Delay vs Number of Users
# ==============================================================

def stress_delay_vs_users(df, save="stress_delay_vs_users.png"):
    df["n_users"] = extract_param(df, "n_users")

    users = sorted(df["n_users"].unique())
    delay_cloud = []
    delay_fog = []

    for u in users:
        subset = df[df["n_users"] == u]
        delay_cloud.append(subset["avg_delay_cloud"].mean())
        delay_fog.append(subset["avg_delay_fog"].mean())

    plt.figure(figsize=(12, 7))
    plt.plot(users, delay_cloud, label="Cloud Delay", linewidth=3)
    plt.plot(users, delay_fog, label="Fog Delay", linewidth=3)

    plt.title("Stress Curve: Delay vs Users")
    plt.xlabel("Number of Users")
    plt.ylabel("Average Delay (ms)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save)
    plt.close()



# ==============================================================
#                           RADAR CHART:
#            Full System State Summary for a Single Config
# ==============================================================

def radar_chart(df, index=0, save="radar_chart.png"):
    row = df.iloc[index]

    metrics = {
        "Fog CPU": row["fog_cpu_max"],
        "Cloud CPU": row["cloud_cpu_max"],
        "Fog Energy": row["fog_energy"],
        "Cloud Energy": row["cloud_energy"],
        "SLA Violations": row["sla_violation"],
        "Handoffs": row["handoffs"]
    }

    labels = list(metrics.keys())
    values = list(metrics.values())

    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title("System State Radar Chart")
    plt.savefig(save)
    plt.close()



# ==============================================================
#                 MARKOV vs NON-MARKOV COMPARISON
# ==============================================================

def comparison_markov_nonmarkov(df_markov, df_nomarkov,
                                save="markov_comparison.png"):
    df_markov["n_users"] = extract_param(df_markov, "n_users")
    df_nomarkov["n_users"] = extract_param(df_nomarkov, "n_users")

    users = sorted(df_markov["n_users"].unique())

    mk_delay = []
    nm_delay = []

    for u in users:
        mk_delay.append(df_markov[df_markov["n_users"] == u]["avg_delay_cloud"].mean())
        nm_delay.append(df_nomarkov[df_nomarkov["n_users"] == u]["avg_delay_cloud"].mean())

    plt.figure(figsize=(12, 7))
    plt.plot(users, mk_delay, label="Markov Delay", linewidth=3)
    plt.plot(users, nm_delay, label="Non-Markov Delay", linewidth=3)

    plt.title("Markov vs Non-Markov Delay Comparison")
    plt.xlabel("Number of Users")
    plt.ylabel("Cloud Delay")
    plt.legend()
    plt.grid(True)
    plt.savefig(save)
    plt.close()



# ==============================================================
#                     RUN ALL VISUALIZATIONS
# ==============================================================

def generate_all_visualizations(df):
    heatmap_sla_users_speed(df)
    heatmap_delay_fogdist_speed(df)
    plot_3d_surface_delay(df, mode="cloud")
    plot_3d_surface_delay(df, mode="fog", save="3d_delay_fog_surface.png")
    sensitivity_fog_distance(df)
    stress_delay_vs_users(df)
    radar_chart(df)
