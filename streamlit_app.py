# =========================
#   IMPORTS (FULL SET)
# =========================
import os
import sys
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Polygon, Ellipse, Arc  # used in plotting functions
from tqdm import tqdm
import streamlit as st
import joblib


# Ensure scikit-learn available for joblib models that depend on it
try:
    import sklearn  # noqa
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn  # noqa


# =========================
#   MODELS & ENGINEERING
# =========================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features from TrackMan-style columns for model scoring.
    Expects lowercased columns: pitcher, relside, relspeed, spinrate,
    extension, relheight, horzbreak, inducedvertbreak, autopitchtype, pitchuid.
    """
    need = [
        "pitcher","relside","relspeed","spinrate","extension",
        "relheight","horzbreak","inducedvertbreak","autopitchtype","pitchuid"
    ]
    keep = [c for c in need if c in df.columns]
    df = df[keep].copy()

    # Handedness from average relside
    hand = (
        df.groupby("pitcher", as_index=False)["relside"]
          .mean().rename(columns={"relside": "avg_side"})
    )
    hand["pitcher_hand"] = np.where(hand["avg_side"] > 0, "R", "L")
    df = df.merge(hand[["pitcher","pitcher_hand"]], on="pitcher", how="left")

    # Standard names
    df = df.rename(columns={
        "relspeed": "start_speed",
        "spinrate": "spin_rate",
        "extension": "extension",
        "relheight": "z0",
        "relside": "x0",
        "horzbreak": "ax",
        "inducedvertbreak": "az",
        "autopitchtype": "pitch_type"
    })

    # Mirror horizontal components for LHP
    df["ax"] = np.where(df["pitcher_hand"] == "L", -df["ax"], df["ax"])
    df["x0"] = np.where(df["pitcher_hand"] == "L", -df["x0"], df["x0"])

    # Most-used fastball baseline (Four-Seam/Sinker)
    fastball_types = ["Four-Seam", "Sinker"]
    fb = df[df["pitch_type"].isin(fastball_types)].copy()
    if not fb.empty:
        agg = (
            fb.groupby(["pitcher","pitch_type"], as_index=False)
              .agg(avg_fastball_speed=("start_speed","mean"),
                   avg_fastball_az=("az","mean"),
                   avg_fastball_ax=("ax","mean"),
                   count=("start_speed","count"))
              .sort_values(["count","avg_fastball_speed"], ascending=[False, False])
              .drop_duplicates(subset=["pitcher"], keep="first")
        )
        df = df.merge(
            agg[["pitcher","avg_fastball_speed","avg_fastball_az","avg_fastball_ax"]],
            on="pitcher", how="left"
        )
        df["speed_diff"] = df["start_speed"] - df["avg_fastball_speed"]
        df["az_diff"]    = df["az"]          - df["avg_fastball_az"]
        df["ax_diff"]    = df["ax"]          - df["avg_fastball_ax"]
    else:
        df["speed_diff"] = np.nan
        df["az_diff"]    = np.nan
        df["ax_diff"]    = np.nan

    df["is_fastball"] = df["pitch_type"].isin(fastball_types)

    # Convert feet → inches for release metrics
    df["z0"] = df["z0"] * 12
    df["x0"] = df["x0"] * 12
    return df


def run_model_and_scale(df_for_model: pd.DataFrame) -> pd.DataFrame:
    """
    Load local joblib models, score df, scale to tj_stuff_plus and xWhiff.
    Expects columns produced by feature_engineering.
    """
    repo_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(repo_path, "NCAA_STUFF_PLUS_ALL.joblib")
    whiff_path = os.path.join(repo_path, "whiff_model_grouped_training.joblib")

    model = joblib.load(model_path)
    whiff_model = joblib.load(whiff_path)

    features = ["start_speed","spin_rate","extension","az","ax","x0","z0",
                "speed_diff","az_diff","ax_diff","is_fastball"]
    for c in features:
        if c not in df_for_model.columns:
            df_for_model[c] = np.nan
    df_for_model[features] = df_for_model[features].apply(pd.to_numeric, errors="coerce")

    df_for_model["target"] = model.predict(df_for_model[features])

    # 2023 baseline for scaling
    target_mean_2023 = 0.011532333993710725
    target_std_2023  = 0.009399038486978739
    df_for_model["target_zscore"] = (df_for_model["target"] - target_mean_2023) / target_std_2023
    df_for_model["tj_stuff_plus"] = 100 - (df_for_model["target_zscore"] * 10)

    df_for_model["xWhiff"] = whiff_model.predict(df_for_model[features])
    return df_for_model


# =========================
#      LOAD BASE DATA
# =========================
TM_URL = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/usd_baseball_TM_master_file.csv"
df = pd.read_csv(TM_URL)

# Tag source and dedupe
df["Source"] = "Preseason"
df.drop_duplicates(subset=["PitchUID"], inplace=True)

# Normalize to TitleCase so downstream plotting code works with:
#   'Platelocside','Platelocheight','Relspeed','Spinrate', etc.
df.columns = [col.strip().capitalize() for col in df.columns]

# Ensure Pitcherabbrevname is present and filled with Pitcher where missing
if "Pitcherabbrevname" not in df.columns:
    df["Pitcherabbrevname"] = df.get("Pitcher", "")
else:
    df["Pitcherabbrevname"] = df["Pitcherabbrevname"].fillna(df.get("Pitcher", ""))
    


# Clean spaces and ensure string type
df["Pitcherabbrevname"] = df["Pitcherabbrevname"].astype(str).str.strip()

if "Pitchtype" not in df.columns:
    if "Autopitchtype" in df.columns:
        df["Pitchtype"] = df["Autopitchtype"]
    elif "Taggedpitchtype" in df.columns:
        df["Pitchtype"] = df["Taggedpitchtype"]
    else:
        df["Pitchtype"] = "Undefined"
else:
    df["Pitchtype"] = df["Pitchtype"].fillna(
        df.get("Autopitchtype", df.get("Taggedpitchtype", "Undefined"))
    )

# Clean spacing and lowercase normalization for consistency
df["Pitchtype"] = df["Pitchtype"].astype(str).str.strip().str.lower()
✅ What this does
# --- Skipping unchanged preprocessing for brevity ---

# =========================
#   SIDEBAR FILTERS
# =========================
st.sidebar.header("Filter Options")

pitchers = df.get("Pitcherabbrevname", pd.Series(dtype="object")).dropna().unique().tolist()
pitchers = [p for p in pitchers if p != "Bunnell, Jack"]
pitchers.insert(0, "All Pitchers")
selected_pitcher = st.sidebar.selectbox("Select Pitcher", pitchers)

sources_available = df.get("Source", pd.Series(dtype="object")).dropna().unique().tolist()
selected_sources = st.sidebar.multiselect("Select Sources", sources_available, default=sources_available)

if selected_pitcher == "All Pitchers":
    dates_available = df[df["Source"].isin(selected_sources)]["Date"].dropna().unique().tolist()
else:
    dates_available = df[
        (df["Pitcherabbrevname"] == selected_pitcher) & (df["Source"].isin(selected_sources))
    ]["Date"].dropna().unique().tolist()

selected_dates = st.sidebar.multiselect("Select Dates", dates_available, default=dates_available)

# ============== 7) FILTER THE MAIN DATAFRAME ACCORDINGLY (STOP HERE) ==============
filtered_data = df[df["Date"].isin(selected_dates) & df["Source"].isin(selected_sources)]
if selected_pitcher != "All Pitchers":
    filtered_data = filtered_data[filtered_data["Pitcherabbrevname"] == selected_pitcher]

# --- Fix pitch name normalization and color map alignment ---
pitch_map = {
    "FourSeamFastBall": "fastball",
    "TwoSeamFastBall": "twoseamfastball",
    "Fastball": "fastball",
    "Sinker": "sinker",
    "Slider": "slider",
    "ChangeUp": "changeup",
    "Cutter": "cutter",
    "Curveball": "curveball",
    "Splitter": "splitter",
    "Undefined": "undefined"
}

for col in ["Pitchtype", "Taggedpitchtype"]:
    if col in filtered_data.columns:
        filtered_data[col] = (
            filtered_data[col]
            .astype(str)
            .str.strip()
            .map(pitch_map)
            .fillna(filtered_data[col].str.lower())
        )

# Base color map for all pitch types
color_map = {
    "Fastball": "#1f77b4",
    "TwoSeamFastBall": "#1f77b4",
    "FourSeamFastBall": "#1f77b4",
    "Riding Fastball": "#1f77b4",
    "Cutter": "#9467bd",
    "Slider": "#ff7f0e",
    "Gyro Slider": "#ff7f0e",
    "Two-Plane Slider": "#ff7f0e",
    "Sweeper": "#ff7f0e",
    "Curveball": "#2ca02c",
    "Slurve": "#2ca02c",
    "Slow Curve": "#2ca02c",
    "Sinker": "#8c564b",
    "Changeup": "#d62728",
    "Movement-Based Changeup": "#d62728",
    "Velo-Based Changeup": "#d62728",
    "Splitter": "#e377c2",
    "Knuckleball": "#7f7f7f",
    "Other": "#bcbd22",
    "Undefined": "#17becf"
}

# Normalize color_map keys to lowercase to match pitch values
color_map = {k.lower(): v for k, v in color_map.items()}



# Function to create scatter plot for pitch locations
def plot_pitch_locations():
    # Define the color map for pitch types
    # color_map = {
    #     'fastball': '#1f77b4',   # Blue
    #     'twoseamfastball': '#1f77b4',  # Blue
    #     'slider': '#ff7f0e',     # Orange
    #     'curveball': '#2ca02c',  # Green
    #     'changeup': '#d62728',   # Red
    #     'cutter': '#9467bd',     # Purple
    #     'sinker': '#8c564b',     # Brown
    #     'splitter': '#e377c2',   # Pink
    #     'knuckleball': '#7f7f7f' # Gray
    # }

    # # Normalize pitch types to lowercase for consistency
    # filtered_data['pitch_type'] = filtered_data['pitch_type'].str.strip().str.lower()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    batter_sides = ['Right', 'Left']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    
    for i, batter_side in enumerate(batter_sides):
        side_data = filtered_data[filtered_data['Batterside'] == batter_side]
        
        sns.scatterplot(
            data=side_data, 
            x='Platelocside', 
            y='Platelocheight', 
            hue='Taggedpitchtype',
            palette=color_map, 
            s=100, 
            edgecolor='black', 
            ax=axes[i]
        )
        
        # Add the strike zone rectangle
        axes[i].add_patch(Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none'))
        
        # Add home plate polygon
        plate = Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
        axes[i].add_patch(plate)
        
        # Set the plot title
        axes[i].set_title(f'{selected_pitcher} vs {batter_side} Handed Batters')
        axes[i].set_xlim(-2.5, 2.5)
        axes[i].set_ylim(0, 5)
    
    st.pyplot(fig)

# Function to create polar plots in Streamlit
def create_polar_plots():
    if 'Tilt_float' not in filtered_data or filtered_data['Tilt_float'].isnull().all():
        st.write("No tilt data available for selected pitcher and dates.")
        return
    
    tilt_radians = np.radians(filtered_data['Tilt_float'] * 30)
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))
    
    # Set the configuration for the first polar plot (Velocity vs. Tilt)
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    sc1 = axs[0].scatter(tilt_radians, filtered_data['Relspeed'], 
                         c=filtered_data['Pitchtype'].map(color_map), s=100)
    axs[0].set_title('Velocity vs. Tilt', fontsize=14)
    axs[0].set_ylim(filtered_data['Relspeed'].min() - 5, filtered_data['Relspeed'].max() + 5)
    axs[0].set_xticks(np.radians(np.arange(0, 360, 30)))
    axs[0].set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

    # Set the configuration for the second polar plot (Spin Rate vs. Tilt)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    sc2 = axs[1].scatter(tilt_radians, filtered_data['Spinrate'], 
                         c=filtered_data['Pitchtype'].map(color_map), s=100)
    axs[1].set_title('Spin Rate vs. Tilt', fontsize=14)
    axs[1].set_ylim(filtered_data['Spinrate'].min() - 500, filtered_data['Spinrate'].max() + 500)
    axs[1].set_xticks(np.radians(np.arange(0, 360, 30)))
    axs[1].set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

    # Add a single legend to the figure
    handles = [plt.Line2D([0], [0], marker='o', color=color_map[pt], markersize=10, linestyle='') 
               for pt in pitch_types]
    fig.legend(handles, pitch_types, title='Pitch Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add a main title to the figure
    plt.suptitle(f'Polar Plots for {selected_pitcher}', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)


# Function to create release plot with mirrored x-axis
def create_release_plot():
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=filtered_data, x='Relside', y='Relheight', hue='Pitchtype', palette=color_map, s=100, edgecolor='black')
    
    # Set x-axis limits and invert them to mirror the plot
    ax.set_xlim(4, -4)  # This reverses the x-axis direction
    ax.set_ylim(0, 7)
    
    st.pyplot(fig)

# Function to create break plot
def create_break_plot():
    # Define a custom color palette

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        data=filtered_data, 
        x='Horzbreak', 
        y='Inducedvertbreak', 
        hue='Pitchtype', 
        palette=color_map,  # Use custom color map
        s=100, 
        edgecolor='black'
    )
    ax.axvline(0, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    
    pitcher_throws = filtered_data['Pitcherthrows'].iloc[0]
    arm_side_x, glove_side_x = (20, -20) if pitcher_throws == 'Right' else (-20, 20)
    
    ax.text(arm_side_x, -23, 'Arm Side', fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(glove_side_x, -23, 'Glove Side', fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    if not filtered_data.empty and 'armangle_prediction' in filtered_data.columns and filtered_data['armangle_prediction'].notna().all():
        avg_horz_break = filtered_data['Horzbreak'].mean()
        avg_arm_angle = filtered_data['armangle_prediction'].mean()
    
        # Convert angle to radians for line direction
        angle_rad = np.radians(avg_arm_angle)
    
        # Determine direction sign based on avg_horz_break
        direction_sign = 1 if avg_horz_break >= 0 else -1
    
        # Use a fixed length for the line
        length = 25  # Adjust as desired
    
        # Calculate end coordinates of the line
        x_end = direction_sign * length * np.cos(angle_rad)
        y_end = length * np.sin(angle_rad)
    
        # Draw the blue dashed line with 50% transparency
        ax.plot([0, x_end], [0, y_end], color='blue', linestyle='--', alpha=0.5)

        if direction_sign > 0:
            x_label = 1
            ha_label = 'left'
        else:
            x_label = -1
            ha_label = 'right'
        
        y_label = -0.5
        ax.text(x_label, y_label, f'Arm Angle = ~ {avg_arm_angle:.1f}°', ha=ha_label, va='top', fontsize=12)


        # Draw a small arc to illustrate the angle
        if direction_sign > 0:
            # Line going right: arc from 0 to avg_arm_angle
            theta1 = 0
            theta2 = avg_arm_angle
        else:
            # Line going left: start from 180° and go backwards by avg_arm_angle
            theta1 = 180 - avg_arm_angle
            theta2 = 180

        # Add an arc at the origin with a small radius to show the angle visually
        arc = Arc((0, 0), width=10, height=10, angle=0, theta1=theta1, theta2=theta2, color='blue', alpha=0.5)
        ax.add_patch(arc)

    st.pyplot(fig)

def create_break_stuff_plot():
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 10))
    # Normalize tj_stuff_plus for color mapping
    norm = Normalize(vmin=70, vmax=130)
    cmap = plt.get_cmap('RdBu_r')

    # Scatter plot colored by tj_stuff_plus
    sc = ax.scatter(
        filtered_data['Horzbreak'],
        filtered_data['Inducedvertbreak'],
        c=filtered_data['tj_stuff_plus'],
        cmap=cmap,
        norm=norm,
        s=100,
        edgecolor='black'
    )

    ax.axvline(0, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    
    pitcher_throws = filtered_data['Pitcherthrows'].iloc[0]
    arm_side_x, glove_side_x = (20, -20) if pitcher_throws == 'Right' else (-20, 20)
    
    ax.text(arm_side_x, -23, 'Arm Side', fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(glove_side_x, -23, 'Glove Side', fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    if not filtered_data.empty and 'armangle_prediction' in filtered_data.columns and filtered_data['armangle_prediction'].notna().all():
        avg_horz_break = filtered_data['Horzbreak'].mean()
        avg_arm_angle = filtered_data['armangle_prediction'].mean()
    
        # Convert angle to radians for line direction
        angle_rad = np.radians(avg_arm_angle)
    
        # Determine direction sign based on avg_horz_break
        direction_sign = 1 if avg_horz_break >= 0 else -1
    
        # Use a fixed length for the line
        length = 25  # Adjust as desired
    
        # Calculate end coordinates of the line
        x_end = direction_sign * length * np.cos(angle_rad)
        y_end = length * np.sin(angle_rad)
    
        # Draw the blue dashed line with 50% transparency
        ax.plot([0, x_end], [0, y_end], color='blue', linestyle='--', alpha=0.5)

        if direction_sign > 0:
            x_label = 1
            ha_label = 'left'
        else:
            x_label = -1
            ha_label = 'right'
        
        y_label = -0.5
        ax.text(x_label, y_label, f'Arm Angle = ~ {avg_arm_angle:.1f}°', ha=ha_label, va='top', fontsize=12)

        # Draw a small arc to illustrate the angle
        if direction_sign > 0:
            # Line going right: arc from 0 to avg_arm_angle
            theta1 = 0
            theta2 = avg_arm_angle
        else:
            # Line going left: start from 180° and go backwards by avg_arm_angle
            theta1 = 180 - avg_arm_angle
            theta2 = 180

        # Add an arc at the origin with a small radius to show the angle visually
        arc = Arc((0, 0), width=10, height=10, angle=0, theta1=theta1, theta2=theta2, color='blue', alpha=0.5)
        ax.add_patch(arc)

    # Add colorbar for tj_stuff_plus
    cbar = plt.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label('TJ Stuff+', fontsize=12)
    cbar.set_ticks([70, 80, 90, 100, 110, 120, 130])

    st.pyplot(fig)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Arc
from tqdm import tqdm
import streamlit as st

# Function to create break plot with in-place pitch type classification
def create_break_plot2():
    global filtered_data
    data = filtered_data.copy()

    # Mirror Horzbreak for lefties
    if 'Pitcherthrows' in data.columns and 'Horzbreak' in data.columns:
        data['Horzbreak_adj'] = np.where(
            data['Pitcherthrows'] == 'Left',
            -data['Horzbreak'],
            data['Horzbreak']
        )
    else:
        st.error("Missing Pitcherthrows or Horzbreak column.")
        return

    # Define platoon
    data['platoon'] = np.where(data['Pitcherthrows'] == data['Batterside'], 0, 1)

    # Group by pitch type with adjusted Horzbreak
    pitch_type_df = data.groupby('Pitch_type').agg(
        Count=('Relspeed', 'count'),
        velo=('Relspeed', 'mean'),
        hb=('Horzbreak_adj', 'mean'),
        ivb=('Inducedvertbreak', 'mean'),
        platoon=('platoon', 'mean')
    ).sort_values(by='Count', ascending=False)

    # Establish baselines
    try:
        ff_baseline = pitch_type_df.loc['FF']
    except KeyError:
        try:
            ff_baseline = pitch_type_df.loc['FA']
        except KeyError:
            try:
                si_baseline = pitch_type_df.loc['SI']
                ff_baseline = si_baseline + [0, 1, -5, 8, 0]
            except KeyError:
                try:
                    ct_baseline = pitch_type_df.loc['FC']
                    if ct_baseline['ivb'] > 6:
                        ff_baseline = ct_baseline + [0, 3, 8, 5, 0]
                    else:
                        ff_baseline = ct_baseline + [0, 6, 8, 10, 0]
                except KeyError:
                    data['true_pitch_type'] = np.nan
                    filtered_data['true_pitch_type'] = np.nan
                    return

    try:
        si_baseline = pitch_type_df.loc['SI']
    except KeyError:
        si_baseline = ff_baseline + [0, -1, 5, -8, 0]

    _, ffvel, ffh, ffv, _ = ff_baseline
    _, sivel, sih, siv, _ = si_baseline

    baseline_debug = f"FF baseline = HB:{ffh:.1f}, IVB:{ffv:.1f}, Velo:{ffvel:.1f} | SI baseline = HB:{sih:.1f}, IVB:{siv:.1f}, Velo:{sivel:.1f}"

    # Pitch archetypes [hb, ivb, velo]
    pitch_archetypes = np.array([
        [ffh, 18, ffvel],           # Riding Fastball
        [ffh, 11, ffvel],           # Fastball
        [sih, siv, sivel],          # Sinker
        [-3, 8, ffvel - 3],         # Cutter
        [-3, 0, ffvel - 9],         # Gyro Slider
        [-8, 0, ffvel - 11],        # Two-Plane Slider
        [-16, 1, ffvel - 14],       # Sweeper
        [-16, -6, ffvel - 15],      # Slurve
        [-8, -12, ffvel - 16],      # Curveball
        [-8, -12, ffvel - 22],      # Slow Curve
        [sih, siv - 2, sivel - 4],  # Movement-Based Changeup
        [sih, siv - 2, sivel - 10], # Velo-Based Changeup
    ])

    pitch_names = np.array([
        'Riding Fastball', 'Fastball', 'Sinker', 'Cutter', 'Gyro Slider', 'Two-Plane Slider',
        'Sweeper', 'Slurve', 'Curveball', 'Slow Curve', 'Movement-Based Changeup',
        'Velo-Based Changeup', 'Knuckleball'
    ])

    for pitch_type, group in pitch_type_df.iterrows():
        if pitch_type == 'KN':
            data.loc[data['Pitch_type'] == pitch_type, 'true_pitch_type'] = 'Knuckleball'
            continue

        # Movement only for primary classification
        shape = np.array([group['hb'], group['ivb']])
        movement_archetypes = pitch_archetypes[:, :2]
        # Exclude movement-based and velo-based changeup candidates if HB is glove side
        exclude_changeup = group['hb'] < 0
        valid_indexes = np.arange(len(pitch_archetypes))
        
        if exclude_changeup:
            # Indices 10 and 11 are changeups
            valid_indexes = valid_indexes[~np.isin(valid_indexes, [10, 11])]
        
        # Compute movement-only distances
        movement_archetypes = pitch_archetypes[valid_indexes][:, :2]
        movement_dists = np.linalg.norm(movement_archetypes - shape, axis=1)
        
        # Get index of closest
        min_indexes = np.where(np.isclose(movement_dists, movement_dists.min(), atol=0.1))[0]
        
        if len(min_indexes) == 1:
            chosen_index = valid_indexes[min_indexes[0]]
        else:
            tied = pitch_archetypes[valid_indexes[min_indexes]]
            vel_dists = np.abs(tied[:, 2] - group['velo'])
            chosen_index = valid_indexes[min_indexes[np.argmin(vel_dists)]]


        if len(min_indexes) == 1:
            chosen_index = min_indexes[0]
        else:
            # Tie: break using velocity
            vel = group['velo']
            tied = pitch_archetypes[min_indexes]
            tied_vels = tied[:, 2]
            vel_dists = np.abs(tied_vels - vel)
            chosen_index = min_indexes[np.argmin(vel_dists)]

        pitch_name = pitch_names[chosen_index]

        # Special logic for changeups
        if pitch_name in ['Movement-Based Changeup', 'Velo-Based Changeup']:
            if sivel - group['velo'] > 6:
                pitch_name = 'Velo-Based Changeup'
            else:
                pitch_name = 'Movement-Based Changeup'

        # Optional debug print
        if pitch_name == 'Curveball' and group['hb'] > 5 and group['ivb'] > 5:
            print(f"[CHECK] {pitch_type} → Curveball? HB={group['hb']:.1f}, IVB={group['ivb']:.1f}, Velo={group['velo']:.1f}")

        data.loc[data['Pitch_type'] == pitch_type, 'true_pitch_type'] = pitch_name

    # Update filtered_data
    filtered_data['true_pitch_type'] = data['true_pitch_type']
    filtered_data['Horzbreak_adj'] = data['Horzbreak_adj']

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        data=filtered_data,
        x='Horzbreak_adj',
        y='Inducedvertbreak',
        hue='true_pitch_type',
        palette=color_map,
        s=100,
        edgecolor='black'
    )

    ax.axvline(0, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    pitcher_throws = filtered_data['Pitcherthrows'].iloc[0]
    arm_side_x, glove_side_x = (20, -20) if pitcher_throws == 'Right' else (-20, 20)

    ax.text(arm_side_x, -23, 'Arm Side', fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(glove_side_x, -23, 'Glove Side', fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_title(f'Pitch Shape Classification\n{baseline_debug}', fontsize=12)

    if ('armangle_prediction' in filtered_data.columns and
        filtered_data['armangle_prediction'].notna().all()):
        avg_horz_break = filtered_data['Horzbreak_adj'].mean()
        avg_arm_angle = filtered_data['armangle_prediction'].mean()
        angle_rad = np.radians(avg_arm_angle)
        direction_sign = 1 if avg_horz_break >= 0 else -1
        length = 25
        x_end = direction_sign * length * np.cos(angle_rad)
        y_end = length * np.sin(angle_rad)
        ax.plot([0, x_end], [0, y_end], color='blue', linestyle='--', alpha=0.5)
        x_label = 1 if direction_sign > 0 else -1
        ha_label = 'left' if direction_sign > 0 else 'right'
        ax.text(x_label, -0.5, f'Arm Angle = ~ {avg_arm_angle:.1f}°', ha=ha_label, va='top', fontsize=12)

        if direction_sign > 0:
            theta1 = 0
            theta2 = avg_arm_angle
        else:
            theta1 = 180 - avg_arm_angle
            theta2 = 180
        arc = Arc((0, 0), width=10, height=10, angle=0, theta1=theta1, theta2=theta2, color='blue', alpha=0.5)
        ax.add_patch(arc)

    st.pyplot(fig)







def plot_rolling_stuff_plus():
    """
    Plots a ing average of tj_stuff_plus over time, by pitch, for the selected pitcher.
    Expects 'Date', 'Time', and 'tj_stuff_plus' in the DataFrame, plus 'Pitcher'.
    """
    # If there's no data, bail out early
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
        return

    # Check if 'tj_stuff_plus' exists
    if "tj_stuff_plus" not in filtered_data.columns:
        st.write("tj_stuff_plus column not found. Make sure you ran the model/scaling steps.")
        return

 # Create a copy and filter for the selected pitcher
    df_temp = filtered_data.copy()
    # df_temp = df_temp[df_temp["Pitcher"] == selected_pitcher]
    # if df_temp.empty:
    #     st.write(f"No data available for {selected_pitcher}.")
    #     return

    # Convert 'Date' to a proper datetime if it's just a string
    df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce")

    # Group by Date and compute the average tj_stuff_plus
    df_grouped = (
        df_temp
        .groupby(["Date","Pitchtype"], as_index=False)["tj_stuff_plus"]
        .mean()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df_grouped,
        x="Date",
        y="tj_stuff_plus",
        hue='Pitchtype',
        ax=ax
    )
    ax.set_title(f"Average TJStuff+ by Date for {selected_pitcher}")
    ax.set_xlabel("Date")
    ax.set_ylabel("TJStuff+")
    # Set static y-axis limits
    ax.set_ylim(60, 125)
    plt.xticks(rotation=45)

    st.pyplot(fig)




def create_confidence_ellipse():
    st.write("### Pitch Command Confidence Ellipse by Pitch Type")
    
    if filtered_data.empty or 'Horzrelangle' not in filtered_data.columns or 'Vertrelangle' not in filtered_data.columns:
        st.warning("Not enough data for Horizontal and Vertical Release Angles to plot confidence ellipse.")
        return
    
    # Filter and group by pitch type
    pitch_types = filtered_data['Pitchtype'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for pitch_type in pitch_types:
        pitch_data = filtered_data[filtered_data['Pitchtype'] == pitch_type]
        horz = pitch_data['Horzrelangle'].dropna()
        vert = pitch_data['Vertrelangle'].dropna()
        
        if len(horz) < 2 or len(vert) < 2:
            continue  # Skip if insufficient data for this pitch type
        
        # Calculate Mean and Covariance
        mean = [horz.mean(), vert.mean()]
        cov = np.cov(horz, vert)
        
        # Eigen decomposition for ellipse parameters
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)
        
        # Get the color from the pitch type mapping
        color = color_map.get(pitch_type, 'gray')  # Fallback to gray if not in color_map
        
        # Plot each ellipse with matching color and transparency
        ellipse = Ellipse(
            xy=mean, 
            width=width, 
            height=height, 
            angle=theta, 
            edgecolor=color, 
            facecolor=color, 
            lw=2.5, 
            alpha=0.3
        )
        ax.add_patch(ellipse)
        
        # Plot scatter points for pitch type
        ax.scatter(horz, vert, s=30, label=f'{pitch_type}', color=color, alpha=0.7, edgecolor='black')
        ax.scatter(*mean, color='blue', marker='x')
    
    ax.set_xlabel('Horizontal Release Angle')
    ax.set_ylabel('Vertical Release Angle')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_title(f'Confidence Ellipse by Pitch Type for {selected_pitcher}')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)

#############################################
# Ideal Pitch Locations Page
#############################################
def plot_ideal_pitch_locations():
    """
    Simulate and display ideal pitch locations using the rv_with_plateloc.joblib model.
    The model expects these features:
      ["start_speed", "spin_rate", "extension", "az", "ax", "x0", "z0", "PX", "PZ"]

    This function:
      1. Renames raw columns to the model-ready names.
      2. Maps plate location columns (PlateLocSide/PlateLocHeight) to PX/PZ.
      3. Lets the user select one pitch type from filtered_data.
      4. Builds a grid over (PX, PZ) using median values (from filtered_data for the selected pitch type)
         for the other features.
      5. Predicts outcomes on this grid using the model and displays a heatmap.
    """
    st.write("## Ideal Pitch Locations")

    # Compute the average 'relside' per pitcher and assign pitcher_hand
    filtered_data["avg_relside"] = filtered_data.groupby("Pitcher")["Relside"].transform("mean")
    filtered_data["pitcher_hand"] = np.where(filtered_data["avg_relside"] < 0, "L", "R")

    
    
    # --- STEP 1: Map Raw Columns to Engineered Feature Names (if not already mapped) ---
    if "start_speed" not in filtered_data.columns:
        filtered_data.rename(columns={
            "Relspeed":           "start_speed",
            "Spinrate":           "spin_rate",
            "Extension":          "extension",
            "Relheight":          "z0",
            "Relside":            "x0",
            "Horzbreak":          "ax",
            "Inducedvertbreak":   "az"
        }, inplace=True)
    
    # Map plate location columns.
    if "PX" not in filtered_data.columns and "Platelocside" in filtered_data.columns:
        filtered_data["PX"] = filtered_data["Platelocside"]
    if "PZ" not in filtered_data.columns and "Platelocheight" in filtered_data.columns:
        filtered_data["PZ"] = filtered_data["Platelocheight"]
    
    # --- STEP 2: Check for Data Availability ---
    if filtered_data.empty:
        st.write("No data available for simulation.")
        return

    # Multiply x0, ax, and PX by -1 for rows where pitcher_hand is "L"
    filtered_data.loc[filtered_data["pitcher_hand"] == "L", ["x0", "ax", "PX"]] *= -1

    filtered_data["z0"] = filtered_data["z0"] * 12
    filtered_data["x0"] = filtered_data["x0"] * 12

    # --- STEP 3: Let the User Select a Single Pitch Type ---
    available_pitch_types = filtered_data["Pitchtype"].dropna().unique().tolist()
    if not available_pitch_types:
        st.write("No pitchtype data available.")
        return

    selected_pitch_type = st.selectbox(
        "Select Pitch Type to Simulate",
        options=available_pitch_types
    )

    # --- STEP 4: Define Plate Location Grid Parameters ---
    st.write("#### Adjust Plate Location Grid")
    px_min  = -2
    px_max  = 2
    px_step = 0.05
    pz_min  = 0.5
    pz_max  = 4.5
    pz_step = 0.05

    # --- STEP 5: Load the Model ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rv_model_path = os.path.join(BASE_DIR, "rv_with_plateloc.joblib")
    rv_model = joblib.load(rv_model_path)

    # The model requires these features:
    # Make sure required_features includes "same_side"
    # Ensure required_features includes "same_side"
    required_features = [
        "start_speed", "spin_rate", "extension",
        "az", "ax", "x0", "z0", "PX", "PZ", "same_side"
    ]
    
    def simulate_pitch_grid(pitch_type):
        """
        For the given pitch type, compute median values (from filtered_data)
        for all features (except PX, PZ, and same_side), build a grid over PX and PZ,
        then split the grid into two versions (same_side=1 and same_side=0) and predict.
        """
        sub = filtered_data[filtered_data["Pitchtype"] == pitch_type]
        if sub.empty:
            return pd.DataFrame(), pd.DataFrame()
        # Exclude "same_side" for computing medians
        features_for_medians = [f for f in required_features if f != "same_side"]
        medians = sub[features_for_medians].median(numeric_only=True).to_dict()
    
        # Create grid arrays for PX and PZ.
        px_values = np.arange(px_min, px_max + px_step, px_step)
        pz_values = np.arange(pz_min, pz_max + pz_step, pz_step)
        grid_rows = []
        for px in px_values:
            for pz in pz_values:
                row = medians.copy()
                row["PX"] = px
                row["PZ"] = pz
                row["Pitchtype"] = pitch_type  # for reference
                grid_rows.append(row)
        grid_df = pd.DataFrame(grid_rows)
    
        # Split into same-side and opposite-side versions
        df_same = grid_df.copy()
        df_opposite = grid_df.copy()
        df_same["same_side"] = 1
        df_opposite["same_side"] = 0
    
        # Predict on both versions
        df_same["run_value"] = rv_model.predict(df_same[required_features])
        df_opposite["run_value"] = rv_model.predict(df_opposite[required_features])
        return df_same, df_opposite
    
    # --- Calling Code ---
    df_same, df_opposite = simulate_pitch_grid(selected_pitch_type)
    if df_same.empty or df_opposite.empty:
        st.write("No data available for the selected pitch type.")
        return
    
    # Assign pitcher_hand from filtered_data to each simulation grid
    pitcher_hand_value = filtered_data["pitcher_hand"].iloc[0]
    df_same["pitcher_hand"] = pitcher_hand_value
    df_opposite["pitcher_hand"] = pitcher_hand_value
    
 
    # Flip PX for left-handed pitchers in both grids
    df_same.loc[df_same["pitcher_hand"] == "L", "PX"] *= -1
    df_opposite.loc[df_opposite["pitcher_hand"] == "L", "PX"] *= -1

    
    # Retrieve pitcher's hand and compute its opposite.
    pitcher_hand = filtered_data["pitcher_hand"].iloc[0]
    opposite_hand = "R" if pitcher_hand == "L" else "L"
    
        # Set the continuous color range from -0.1 to 0.1.
    norm = Normalize(vmin=-0.1, vmax=0.1)
    
    # Plot both heatmaps side by side using scatterplots.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot same-side data
    sns.scatterplot(data=df_same, x="PX", y="PZ", hue="run_value",
                    palette="RdBu_r", ax=axes[0], hue_norm=norm)
    axes[0].invert_yaxis()  # so higher PZ values are at the top
    axes[0].set_title(f"Vs. {pitcher_hand}HH")
    
    # Plot opposite-side data
    sns.scatterplot(data=df_opposite, x="PX", y="PZ", hue="run_value",
                    palette="RdBu_r", ax=axes[1], hue_norm=norm)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Vs. {opposite_hand}HH")
    
    # Add the strike zone rectangle and home plate polygon.
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    for ax in axes:
        ax.add_patch(Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none'))
        plate = Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(plate)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
    
    plt.tight_layout()
    st.pyplot(fig)



# ------------------------------------------------------------------
# 1) FUNCTION TO CALCULATE PITCH METRICS (Grouped by Pitcher,Pitchtype)
# ------------------------------------------------------------------
def calculate_pitch_metrics(pitcher_data):
    """
    Calculate pitch metrics so each row is (Pitcher, Pitchtype).
    We keep the same logic, but replace .value_counts() with a groupby
    on ['Pitcher','Pitchtype'].
    """

    # We now group on both Pitcher and Pitchtype, so each row in the final DF
    # corresponds to that unique combination.
    grouped = pitcher_data.groupby(['Pitcherabbrevname', 'Pitchtype'])

    # Count total pitches in each (Pitcher,Pitchtype)
    pitch_type_counts = grouped.size().rename('Count')

    # List of columns to average
    avg_cols = [
        'Relspeed', 'Spinrate', 'Inducedvertbreak', 'Horzbreak',
        'Relheight', 'Relside', 'Extension', 'Vertapprangle', 'Horzapprangle'
    ]

    # If we have 'tj_stuff_plus', compute the average; otherwise create an empty Series
    if 'tj_stuff_plus' in pitcher_data.columns:
        stuff_plus_mean = grouped['tj_stuff_plus'].mean().round(1).rename('TJStuff+')
    else:
        stuff_plus_mean = pd.Series(dtype=float, name='TJStuff+')

    # Mean of the listed columns, grouped by (Pitcher,Pitchtype)
    pitch_type_averages = grouped[avg_cols].mean().round(1)

    # Ensure the relevant columns are Boolean
    for col in ['Whiff', 'Swing', 'Strike', 'Inzone']:
        pitcher_data[col] = pitcher_data[col].astype(bool)

    # -------------------
    # STRIKE %
    # -------------------
    # 1) Identify "strikes"
    # Identify strikes using the 'Strike' column
    strikes = pitcher_data[pitcher_data['Strike']]
    # Count strikes by (Pitcher, Pitchtype)
    strikes_count = strikes.groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    # Compute strike percentage
    strike_percentages = (strikes_count / pitch_type_counts * 100).rename('Strike %').round(1)

    # -------------------
    # WHIFF %
    # -------------------
    # Compute whiff percentage as: (Number of whiffs) / (Number of swings) * 100
    swinging_strikes = pitcher_data[pitcher_data['Whiff']].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    total_swings = pitcher_data[pitcher_data['Swing']].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    whiff_percentages = (swinging_strikes / total_swings * 100).rename('Whiff %').fillna(0).round(1)
    
    # -------------------
    # INZONE %
    # -------------------
    # Compute in-zone percentage as: (Number of in-zone pitches) / (Total pitches) * 100
    in_zone_counts = pitcher_data[pitcher_data['Inzone']].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    in_zone_percentage = (in_zone_counts / pitch_type_counts * 100).rename('InZone %').fillna(0).round(1)
    
    # -------------------
    # INZONE WHIFF %
    # -------------------
    # Compute in-zone whiff percentage as: (Number of whiffs in-zone) / (Number of swings in-zone) * 100
    in_zone_swinging_strikes = pitcher_data[(pitcher_data['Inzone']) & (pitcher_data['Whiff'])].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    in_zone_swings = pitcher_data[(pitcher_data['Inzone']) & (pitcher_data['Swing'])].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    in_zone_whiff_percentages = (in_zone_swinging_strikes / in_zone_swings * 100).rename('InZone Whiff %').fillna(0).round(1)
    
    # -------------------
    # CHASE %
    # -------------------
    # Here we define 'Chase %' as the percentage of out-of-zone swings.
    out_zone_swings = pitcher_data[(~pitcher_data['Inzone']) & (pitcher_data['Swing'])].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    total_out_zone_pitches = pitcher_data[~pitcher_data['Inzone']].groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    chase_percentage = (out_zone_swings / total_out_zone_pitches * 100).rename('Chase %').fillna(0).round(1)
    # -------------------
    # xWHIFF %
    # -------------------
    if 'xWhiff' in pitcher_data.columns:
        # Multiply by 100 to get a percentage
        xwhiff_percentages = (
            grouped['xWhiff'].mean() * 100
        ).rename('xWhiff %').fillna(0).round(1)
    else:
        xwhiff_percentages = pd.Series(dtype=float, name='xWhiff %')

    # -------------------
    # MAX VELO
    # -------------------
    max_velocity = grouped['Relspeed'].max().rename('Max velo').round(1)

  
    # -------------------
    # COMPLOC %
    # -------------------
    comp_data = pitcher_data[pitcher_data['Comploc']]
    comp_counts = comp_data.groupby(['Pitcherabbrevname', 'Pitchtype']).size()
    comploc_percentage = (
        comp_counts / pitch_type_counts * 100
    ).rename('CompLoc %').fillna(0).round(1)



    # -----------------------
    # FINAL DATAFRAME JOIN
    # -----------------------
    metrics_df = (
        pitch_type_counts.to_frame()
        .join(stuff_plus_mean)
        .join(max_velocity)
        .join(pitch_type_averages)
        .join(strike_percentages)
        .join(xwhiff_percentages)
        .join(whiff_percentages)
        .join(in_zone_percentage)
        .join(comploc_percentage)
        .join(in_zone_whiff_percentages)
        .join(chase_percentage)
        .fillna(0)
    )

    # Rename columns for clarity
    metrics_df.columns = [
        'P',           # # of Pitches
        'TJStuff+',    # Mean of tj_stuff_plus
        'Max velo',
        'AVG velo',
        'Spinrate',
        'IVB',
        'HB',
        'yRel',
        'xRel',
        'Ext.',
        'VAA',
        'HAA',
        'Strike %',
        'xWhiff%',
        'Whiff %',
        'InZone %',
        'CompLoc%',
        'InZone Whiff %',
        'Chase %'
    ]

    return metrics_df


# ------------------------------------------------------------------
# 2) FUNCTION TO DISPLAY PITCH METRICS IN STREAMLIT
# ------------------------------------------------------------------
def display_pitch_metrics():
    """
    Displays a table with the grouped-by (Pitcher, Pitchtype) metrics
    for whatever subset of data is in 'filtered_data'.
    """
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
        return

    # Calculate the metrics on the subset data
    metrics_df = calculate_pitch_metrics(filtered_data)

    # Style the DataFrame for better readability
    styled_df = (
        metrics_df
        .style
        .format(precision=1, na_rep="-")
        .set_properties(**{'width': '10px'})  # Adjust as needed
    )

    st.write(f"### Pitch Metrics for {selected_pitcher}")
    st.dataframe(styled_df, use_container_width=True)

   

# Function to display raw data
def display_raw_data():
    st.write(f"### Raw Data for {selected_pitcher}")
    st.dataframe(filtered_data)

def generate_pitch_reports_page():
    st.write("#### Pitch Reports")
    from pitcher_reports import generate_reports
    generate_reports(filtered_data)  # Call the function with the filtered dataset


# -------------------------------
# NEW PLOTTING FUNCTION: Rolling 3-Pitch Averages of TJStuff+
# -------------------------------
def plot_rolling_3_pitch_averages(df):
    """
    Produces two separate plots, averaging metrics by groups of 10 pitches (PitcherPitchNo intervals).

    Process:
    1. Group pitches into bins of 10: <=10, 11-20, 21-30, etc.
    2. Aggregate by computing mean tj_stuff_plus and Relspeed per pitch type for each bin.
    3. Generate plots for each aggregated metric.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    required_cols = ["tj_stuff_plus", "PitcherPitchNo", "Pitchtype", "Pitcher", "Relspeed"]
    if df.empty or not all(col in df.columns for col in required_cols):
        st.write("Required columns not found or no data available.")
        return

    df_temp = df.copy()

    # Define pitch group intervals (10-pitch increments)
    df_temp["PitchGroup"] = ((df_temp["PitcherPitchNo"] - 1) // 10 + 1) * 10

    # Aggregate means per group
    df_agg = df_temp.groupby(["Pitcher", "Pitchtype", "PitchGroup"], as_index=False).agg({
        "tj_stuff_plus": "mean",
        "Relspeed": "mean"
    })

    pitcher_name = df["Pitcher"].iloc[0] if "Pitcher" in df.columns else "Selected Pitcher"

    # Plot tj_stuff_plus
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for pitch_type in df_agg['Pitchtype'].dropna().unique():
        df_plot = df_agg[df_agg['Pitchtype'] == pitch_type].sort_values("PitchGroup")
        ax1.plot(df_plot["PitchGroup"], df_plot["tj_stuff_plus"], marker="o", label=pitch_type)
    ax1.set_title(f"Avg Stuff+ by Pitch Count for {pitcher_name}")
    ax1.set_xlabel("Pitch Count")
    ax1.set_ylabel("Average Stuff+")
    ax1.legend(title="Pitch Type")

    # Plot Relspeed
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for pitch_type in df_agg['Pitchtype'].dropna().unique():
        df_plot = df_agg[df_agg['Pitchtype'] == pitch_type].sort_values("PitchGroup")
        ax2.plot(df_plot["PitchGroup"], df_plot["Relspeed"], marker="o", label=pitch_type)
    ax2.set_title(f"Avg Relspeed by Pitch Count for {pitcher_name}")
    ax2.set_xlabel("Pitch Count")
    ax2.set_ylabel("Average Relspeed")
    ax2.legend(title="Pitch Type")

    st.pyplot(fig1)
    st.pyplot(fig2)











# Streamlit Page Navigation
st.sidebar.title("Navigation")
pages = {
    "Post-Series Report": generate_pitch_reports_page,  # Add this,
    "NEW Stuff and Velo throughout game": lambda: plot_rolling_3_pitch_averages(filtered_data),
    "Pitch Locations - RHH/LHH": plot_pitch_locations,
    "Release Plot - Tipping pitches?": create_release_plot,
    "Break Plot - Movement Profile": create_break_plot,
    "Break Plot - Stuff+": create_break_stuff_plot,  # ← ADD THIS 
    "TEST Break Plot": create_break_plot2,
    "Pitch Metric AVG Table": display_pitch_metrics,
    "Confidence Ellipse - Command Analysis": create_confidence_ellipse,
    "Stuff+ Over Time": plot_rolling_stuff_plus,  # ← ADD THIS
    "Polar Plots - Understanding Tilt": create_polar_plots,
    "Ideal Pitch Locations": plot_ideal_pitch_locations,
    "Raw Data": display_raw_data
}
selected_page = st.sidebar.radio("Select Plot", list(pages.keys()))

# Display selected plot
st.title(f"{selected_page}")
pages[selected_page]()
