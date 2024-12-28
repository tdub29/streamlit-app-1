import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import seaborn as sns
from datetime import datetime
import math
from matplotlib.patches import Arc
from matplotlib.patches import Ellipse
import joblib
import os
import subprocess
import sys
import lightgbm

try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

import sklearn

#############################################
# 1) DEFINE HELPER FUNCTIONS
#############################################
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for a baseball dataset with columns:
      - relspeed      : pitch velocity
      - spinrate      : spin rate
      - extension     : release extension
      - relheight     : release height
      - relside       : release side (+ => typically R, - => L)
      - ax0           : horizontal pitch break
      - az0           : vertical pitch break
      - autopitchtype : pitch type (e.g., "Four-Seam", "Sinker", etc.)
      - pitcher       : pitcher identifier
      ... other columns as needed
    Steps:
      1) Determine pitcher handedness from average 'relside' (R if > 0, else L).
      2) Rename columns to standard references (start_speed, ax, az, etc.).
      3) Mirror horizontal release & break for left-handed pitchers.
      4) From fastball types ["Four-Seam","Sinker"], find the most-used fastball
         per (pitcher). If there's a tie, pick the one with the highest average speed.
      5) Merge those metrics back & compute diffs:
         - speed_diff = start_speed - avg_fastball_speed
         - az_diff    = az - avg_fastball_az
         - ax_diff    = ax - avg_fastball_ax
      6) Flip x0 sign (df["x0"] = df["x0"] * -1) at the end.
    """
    # 1) Keep only the columns we need
    needed_cols = [
        "pitcher",
        "relside",
        "relspeed",
        "spinrate",
        "extension",
        "relheight",
        "ax0",
        "az0",
        "autopitchtype",
        "pitchuid"
    ]
    df = df[needed_cols].copy()
    
    # 1) DETERMINE PITCHER HANDEDNESS
    df_hand = (
        df.groupby("pitcher", as_index=False)["relside"].mean()
          .rename(columns={"relside": "avg_side"})
    )
    df_hand["pitcher_hand"] = np.where(df_hand["avg_side"] > 0, "R", "L")

    # Merge handedness info back
    df = pd.merge(df, df_hand[["pitcher", "pitcher_hand"]], on="pitcher", how="left")

    # 2) RENAME COLUMNS
    df = df.rename(columns={
        "relspeed":      "start_speed",
        "spinrate":      "spin_rate",
        "extension":     "extension",
        "relheight":     "z0",
        "relside":       "x0",
        "ax0":           "ax",         
        "az0":           "az",         
        "autopitchtype": "pitch_type"
    })

    # 3) MIRROR FOR LEFT-HANDED PITCHERS
    df["ax"] = np.where(df["pitcher_hand"] == "L", -df["ax"], df["ax"])
    print(df["x0"].iloc[0])
    df["x0"] = np.where(df["pitcher_hand"] == "L", -df["x0"], df["x0"])

    # 4) MOST-USED FASTBALL LOGIC
    fastball_types = ["Four-Seam", "Sinker"]
    df_fb = df[df["pitch_type"].isin(fastball_types)].copy()

    df_agg = (
        df_fb.groupby(["pitcher", "pitch_type"], as_index=False)
             .agg(
                 avg_fastball_speed=("start_speed", "mean"),
                 avg_fastball_az=("az", "mean"),
                 avg_fastball_ax=("ax", "mean"),
                 count=("start_speed", "count")
             )
    )
    df_agg = df_agg.sort_values(["count", "avg_fastball_speed"], ascending=[False, False])
    df_agg = df_agg.drop_duplicates(subset=["pitcher"], keep="first")

    df = pd.merge(
        df,
        df_agg[["pitcher", "avg_fastball_speed", "avg_fastball_az", "avg_fastball_ax"]],
        on=["pitcher"],
        how="left"
    )

    df["speed_diff"] = df["start_speed"] - df["avg_fastball_speed"]
    df["az_diff"]    = df["az"] - df["avg_fastball_az"]
    df["ax_diff"]    = df["ax"] - df["avg_fastball_ax"]

    # 5) Flip x0 sign
    df["x0"] = df["x0"] * -1

    return df


def run_model_and_scale(df_for_model: pd.DataFrame) -> pd.DataFrame:
    """
    1) Load the trained model from disk.
    2) Predict using the engineered features.
    3) Add 'target' column.
    4) Apply z-score and tj_stuff_plus using pre-known baseline stats.
    Returns a new df with 'target', 'target_zscore', 'tj_stuff_plus'.
    """

    # -- LOAD MODEL
    # Get the directory of the currently running .py file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to your joblib file
    model_path = os.path.join(BASE_DIR, "lgbm_model_2020_2023.joblib")
    
    # Load the model
    model = joblib.load(model_path)

    # -- DEFINE FEATURES
    features = [
        "start_speed",
        "spin_rate",
        "extension",
        "az",
        "ax",
        "x0",
        "z0",
        "speed_diff",
        "az_diff",
        "ax_diff"
    ]

    # -- MAKE PREDICTIONS
    predictions = model.predict(df_for_model[features])
    df_for_model["target"] = predictions

    # -- APPLY z-score & stuff-plus scaling
    target_mean_2023 = 0.003621590946415154   
    target_std_2023  = 0.006897066586011802

    df_for_model["target_zscore"] = (
        (df_for_model["target"] - target_mean_2023) / target_std_2023
    )
    df_for_model["tj_stuff_plus"] = (
        100 - (df_for_model["target_zscore"] * 10)
    )

    return df_for_model


# Load the CSV file
file_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/usd_baseball_TM_master_file.csv"
df = pd.read_csv(file_path)

df.drop_duplicates(subset=['PitchUID'], inplace=True)

# Standardize column capitalization
df.columns = [col.strip().capitalize() for col in df.columns]

df_for_model = df.copy()

# Rename columns to lowercase for the feature_engineering function
df_for_model.columns = [c.lower() for c in df_for_model.columns]

# Ensure the columns needed by feature_engineering exist
# (RelSpeed, RelHeight, RelSide, ax0, az0, AutoPitchType, Pitcher, SpinRate, Extension)
# If any are missing, you may need to handle that or rename them properly.

# 1) FEATURE ENGINEERING
df_for_model = feature_engineering(df_for_model)

# 2) RUN MODEL + SCALING
df_for_model = run_model_and_scale(df_for_model)

# OPTIONAL: Merge the new columns (target, tj_stuff_plus) back into the original "df"
# so that you can reference them in your existing plots/tables if desired.
# We'll merge on a unique identifier you have (e.g., Pitchuid), if it exists in both.
# For demonstration, let's assume "pitchuid" (lowercase in df_for_model).
if "pitchuid" in df_for_model.columns and "Pitchuid" in df.columns:
    # We select only the new columns from df_for_model we want to bring back
    merged_cols = ["pitchuid", "target", "target_zscore", "tj_stuff_plus"]
    df = pd.merge(
        df, 
        df_for_model[merged_cols], 
        left_on="Pitchuid", right_on="pitchuid", 
        how="left"
    )
    # You might drop the duplicate "pitchuid" column from df
    df.drop(columns=["pitchuid"], inplace=True, errors="ignore")
    
# Load arm angle CSV
armangle_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/armangle_final_fall_usd.csv"
armangle_df = pd.read_csv(armangle_path)

# Merge arm angle data into df on 'Pitcher'
df = df.merge(armangle_df[['Pitcher', 'armangle_prediction']], on='Pitcher', how='left')

df["datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"  # invalid parses -> NaT
)


# Convert 'Tilt' column from HH:MM format to float (1:45 -> 1.75)
def convert_tilt_to_float(tilt_value):
    try:
        if isinstance(tilt_value, str) and ':' in tilt_value:
            hours, minutes = tilt_value.split(':')
            return float(hours) + float(minutes) / 60
        return np.nan
    except Exception as e:
        st.write(f"Error converting Tilt: {e}")
        return np.nan

# Apply Tilt conversion
df['Tilt_float'] = df['Tilt'].apply(convert_tilt_to_float)

# Identify in-zone pitches based on PlateLocSide and PlateLocHeight
df['Inzone'] = df.apply(
    lambda row: -0.83 <= row['Platelocside'] <= 0.83 and 1.5 <= row['Platelocheight'] <= 3.6, axis=1)

# Define pitch categories based on initial pitch types
pitch_categories = {
    "Breaking Ball": ["Slider", "Curveball"],
    "Fastball": ["Fastball", "Four-Seam", "Sinker", "Cutter"],
    "Offspeed": ["ChangeUp", "Splitter"]
}

# Function to categorize pitch types into broader groups
def categorize_pitch_type(pitch_type):
    for category, pitches in pitch_categories.items():
        if pitch_type in pitches:
            return category
    return None

# Create a new column 'Pitchcategory' to categorize pitches
df['Pitchcategory'] = df['Autopitchtype'].apply(categorize_pitch_type)

# Set up the color palette based on pitch type
pitch_types = df['Autopitchtype'].unique()
palette = sns.color_palette('Set2', len(pitch_types))
color_map = dict(zip(pitch_types, palette))

# Streamlit Sidebar Filters
st.sidebar.header("Filter Options")
selected_pitcher = st.sidebar.selectbox("Select Pitcher", df['Pitcher'].unique())
dates_available = df[df['Pitcher'] == selected_pitcher]['Date'].unique()
selected_dates = st.sidebar.multiselect("Select Dates", dates_available, default=dates_available)

# Filter data based on selection
filtered_data = df[(df['Pitcher'] == selected_pitcher) & (df['Date'].isin(selected_dates))]

# Function to create scatter plot for pitch locations
def plot_pitch_locations():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    batter_sides = ['Right', 'Left']
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    
    for i, batter_side in enumerate(batter_sides):
        side_data = filtered_data[filtered_data['Batterside'] == batter_side]
        sns.scatterplot(data=side_data, x='Platelocside', y='Platelocheight', hue='Autopitchtype',
                        palette=color_map, s=100, edgecolor='black', ax=axes[i])
        axes[i].add_patch(Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none'))
        plate = Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='k', facecolor='none')
        axes[i].add_patch(plate)
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
                         c=filtered_data['Autopitchtype'].map(color_map), s=100)
    axs[0].set_title('Velocity vs. Tilt', fontsize=14)
    axs[0].set_ylim(filtered_data['Relspeed'].min() - 5, filtered_data['Relspeed'].max() + 5)
    axs[0].set_xticks(np.radians(np.arange(0, 360, 30)))
    axs[0].set_xticklabels(['12', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

    # Set the configuration for the second polar plot (Spin Rate vs. Tilt)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    sc2 = axs[1].scatter(tilt_radians, filtered_data['Spinrate'], 
                         c=filtered_data['Autopitchtype'].map(color_map), s=100)
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
    sns.scatterplot(data=filtered_data, x='Relside', y='Relheight', hue='Autopitchtype', palette=color_map, s=100, edgecolor='black')
    
    # Set x-axis limits and invert them to mirror the plot
    ax.set_xlim(4, -4)  # This reverses the x-axis direction
    ax.set_ylim(0, 7)
    
    st.pyplot(fig)

# Function to create break plot
def create_break_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=filtered_data, x='Horzbreak', y='Inducedvertbreak', hue='Autopitchtype', palette=color_map, s=100, edgecolor='black')
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

    if not filtered_data.empty and 'armangle_prediction' in filtered_data.columns:
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
        # If direction_sign is positive, arc from 0° to avg_arm_angle°
        # If direction_sign is negative, arc from 180° to 180° + avg_arm_angle
        # Draw a small arc to illustrate the angle
        if direction_sign > 0:
            # Line going right: arc from 0 to avg_arm_angle
            theta1 = 0
            theta2 = avg_arm_angle
        else:
            # Line going left: start from 180° and go backwards by avg_arm_angle
            # This means the line is at 180 - avg_arm_angle, which is above the horizontal.
            theta1 = 180 - avg_arm_angle
            theta2 = 180

        # Add an arc at the origin with a small radius to show the angle visually
        arc = Arc((0, 0), width=10, height=10, angle=0, theta1=theta1, theta2=theta2, color='blue', alpha=0.5)
        ax.add_patch(arc)

    
    st.pyplot(fig)


def plot_rolling_stuff_plus():
    """
    Plots a rolling average of tj_stuff_plus over time, by pitch, for the selected pitcher.
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

    # 1) Create a combined 'datetime' column if you haven't already
    #    (If you already have a 'Datetime' column, skip this step)
    df_temp = filtered_data.copy()
    

    # 2) Sort by datetime for correct chronological order
    df_temp = df_temp.sort_values("datetime")

    # 3) Group by Pitcher & compute rolling average of tj_stuff_plus
    #    (Here we pick a rolling window of 10 pitches, min_periods=1 so it starts immediately)


    # 4) Filter again by the selected pitcher (optional, if you want only one line)
    #    If you want to see all pitchers, skip this line
    pitcher_data = df_temp[df_temp["Pitcher"] == selected_pitcher]
    if pitcher_data.empty:
        st.write(f"No data available for {selected_pitcher} after building datetime.")
        return

    # 5) Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=pitcher_data,
        x="datetime",
        y="tj_stuff_plus",
        ax=ax
    )
    ax.set_title(f"Rolling TJStuff+  for {selected_pitcher}")
    ax.set_xlabel("Date-Time")
    ax.set_ylabel("TJStuff+")
    plt.xticks(rotation=45)

    st.pyplot(fig)




def create_confidence_ellipse():
    st.write("### Pitch Command Confidence Ellipse by Pitch Type")
    
    if filtered_data.empty or 'Horzrelangle' not in filtered_data.columns or 'Vertrelangle' not in filtered_data.columns:
        st.warning("Not enough data for Horizontal and Vertical Release Angles to plot confidence ellipse.")
        return
    
    # Filter and group by pitch type
    pitch_types = filtered_data['Autopitchtype'].unique()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for pitch_type in pitch_types:
        pitch_data = filtered_data[filtered_data['Autopitchtype'] == pitch_type]
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
        ax.scatter(*mean, color='red', marker='x')
    
    ax.set_xlabel('Horizontal Release Angle')
    ax.set_ylabel('Vertical Release Angle')
    ax.set_title(f'Confidence Ellipse by Pitch Type for {selected_pitcher}')
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)



# Function to calculate pitch metrics for each pitch type
def calculate_pitch_metrics(pitcher_data):
    pitch_type_counts = pitcher_data['Autopitchtype'].value_counts().rename('Count')
    avg_cols = ['Relspeed', 'Spinrate', 'Inducedvertbreak', 'Horzbreak',
                'Relheight', 'Relside', 'Extension', 'Vertapprangle', 'Horzapprangle']
    pitch_type_averages = pitcher_data.groupby('Autopitchtype')[avg_cols].mean().round(1)

    if 'tj_stuff_plus' in pitcher_data.columns:
        stuff_plus_mean = (
            pitcher_data.groupby('Autopitchtype')['tj_stuff_plus']
                       .mean()
                       .round(1)
                       .rename('TJStuff+')
        )
    else:
    # If the column doesn't exist, create an empty series
        stuff_plus_mean = pd.Series(dtype=float, name='TJStuff+')
    
    strikes = pitcher_data[pitcher_data['Pitchcall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay'])]
    strike_percentages = (strikes.groupby('Autopitchtype').size() / pitch_type_counts * 100).rename('Strike %').round(1)
    
    swinging_strikes = pitcher_data[pitcher_data['Pitchcall'] == 'StrikeSwinging'].groupby('Autopitchtype').size()
    total_swings = pitcher_data[pitcher_data['Pitchcall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])].groupby('Autopitchtype').size()
    whiff_percentages = (swinging_strikes / total_swings * 100).rename('Whiff %').fillna(0).round(1)
    
    max_velocity = pitcher_data.groupby('Autopitchtype')['Relspeed'].max().rename('Max velo').round(1)
    
    in_zone_total = pitcher_data[pitcher_data['Inzone']].groupby('Autopitchtype').size()
    in_zone_percentage = (in_zone_total / pitch_type_counts * 100).rename('InZone %').fillna(0).round(1)
    
    in_zone_swinging_strikes = pitcher_data[(pitcher_data['Inzone']) & (pitcher_data['Pitchcall'] == 'StrikeSwinging')].groupby('Autopitchtype').size()
    in_zone_swings = pitcher_data[pitcher_data['Inzone'] & pitcher_data['Pitchcall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])].groupby('Autopitchtype').size()
    in_zone_whiff_percentages = (in_zone_swinging_strikes / in_zone_swings * 100).rename('InZone Whiff %').fillna(0).round(1)
    
    out_zone_swings = pitcher_data[(~pitcher_data['Inzone']) & pitcher_data['Pitchcall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])].groupby('Autopitchtype').size()
    total_out_zone_pitches = pitcher_data[~pitcher_data['Inzone']].groupby('Autopitchtype').size()
    chase_percentage = (out_zone_swings / total_out_zone_pitches * 100).rename('Chase %').fillna(0).round(1)

    metrics_df = (pitch_type_counts.to_frame()
                  .join(stuff_plus_mean) 
                  .join(max_velocity)
                  .join(pitch_type_averages)
                  .join(strike_percentages)
                  .join(whiff_percentages)
                  .join(in_zone_percentage)
                  .join(in_zone_whiff_percentages)
                  .join(chase_percentage)
                  .fillna(0))
    metrics_df.columns = ['P', 'TJStuff+' ,'Max velo', 'AVG velo', 'Spinrate', 'IVB', 'HB', 'yRel', 'xRel', 'Ext.', 'VAA', 'HAA', 'Strike %', 'Whiff %', 'InZone %', 'InZone Whiff %', 'Chase %']
    return metrics_df

# Function to display pitch metrics table in Streamlit
def display_pitch_metrics():
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
        return
    
    metrics_df = calculate_pitch_metrics(filtered_data)
    
    # Adjust column width by setting a max width on the dataframe
    # Format values to avoid additional decimal places
    styled_df = metrics_df.style.format(precision=1, na_rep="-").set_properties(**{'width': '15px'})  # Adjust width as needed

    st.write(f"### Pitch Metrics for {selected_pitcher}")
    st.dataframe(styled_df, use_container_width=True)  # Ensures it fits the Streamlit container

   

# Function to display raw data
def display_raw_data():
    st.write(f"### Raw Data for {selected_pitcher} on {', '.join(selected_dates)}")
    st.dataframe(filtered_data)



# Streamlit Page Navigation
st.sidebar.title("Navigation")
pages = {
    "Pitch Locations - RHH/LHH": plot_pitch_locations,
    "Polar Plots - Understanding Tilt": create_polar_plots,
    "Release Plot - Tipping pitches?": create_release_plot,
    "Break Plot - Movement Profile": create_break_plot,
    "Confidence Ellipse - Command Analysis": create_confidence_ellipse,
    "Stuff+ Over Time": plot_rolling_stuff_plus,  # ← ADD THIS
    "Pitch Metric AVG Table": display_pitch_metrics,
    "Raw Data": display_raw_data
}
selected_page = st.sidebar.radio("Select Plot", list(pages.keys()))

# Display selected plot
st.title(f"{selected_page}")
pages[selected_page]()
