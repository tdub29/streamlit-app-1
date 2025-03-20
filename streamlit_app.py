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
import xgboost as xgb
import catboost
from matplotlib.colors import Normalize
# from pitcher_reports import generate_report



try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    import sklearn

import sklearn

def Trumedia_feature_engineering(df):
    import pandas as pd
    import numpy as np

    # -------------------------------
    # 1. Define pitch mapping
    # -------------------------------
    pitch_mapping = {
        'FA': 'Fast',
        'CU': 'Break',
        'CH': 'Slow',
        'SL': 'Break',
        'SI': 'Fast',
        'FC': 'Fast',
        'UN': None,  # filter out
        'IN': None,  # filter out
        'FF': 'Fast',
        'FS': 'Slow',
        'KN': 'Slow'
    }

    # Filter out rows where pitchType maps to None
    df = df[df['pitchType'].isin([pt for pt, group in pitch_mapping.items() if group is not None])].copy()
    # Map pitch types to a new column 'pitchgroup'
    df['pitchgroup'] = df['pitchType'].map(pitch_mapping)

    df['Autopitchtype'] =  df['Taggedpitchtype']

    # -------------------------------
    # 2. Categorize the pitchResult
    # -------------------------------
    def categorize_event(event):
        """
        Categorize the pitchResult into a standardized event label.
        Bunt events are ignored (return None).
        """
        if not isinstance(event, str):
            return None
    
        event = event.lower()
        
        if "bunt" in event or "unknown" in event:
            return None
        elif "single" in event:
            return "single"
        elif "double play" in event:
            return "field_out"
        elif "double" in event:
            return "double"
        elif "triple" in event:
            return "triple"
        elif "home run" in event:
            return "home_run"
        elif "looking" in event:
            return "called_strike"
        elif "swinging" in event:
            return "swinging_strike"
        elif "hit by pitch" in event:
            return "hit_by_pitch"
        elif "walk" in event or "ball" in event:
            return "ball"
        elif "foul" in event:
            return "foul"
        elif ("line out" in event or "fly out" in event or "ground out" in event or
              "pop out" in event or "double play" in event or "reached on error" in event or
              "in play out" in event or "sac fly" in event or "fielder's choice" in event):
            return "field_out"
        elif "ball in the dirt" in event:
            return "ball"
        else:
            return "unknown"


    df['event_category'] = df['pitchResult'].apply(categorize_event)
    # Drop rows where event_category is None or unknown
    df = df[(df['event_category'].notna()) & (df['event_category'] != 'unknown')].copy()

    # -------------------------------
    # 3. Split 'count' into balls and strikes
    # -------------------------------
    df[['balls', 'strikes']] = df['count'].str.split('-', expand=True)
    df['balls'] = pd.to_numeric(df['balls'], errors='coerce')
    df['strikes'] = pd.to_numeric(df['strikes'], errors='coerce')

    # -------------------------------
    # 4. Merge with run_values
    # -------------------------------
    run_values = pd.read_csv("https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/run_values.csv")
    run_values = run_values.rename(columns={"event": "event_category"})
    df_joined = pd.merge(
        df,
        run_values,
        on=["balls", "strikes", "event_category"],
        how="left"
    )



    # -------------------------------
    # 5. Create Binary 'win' Column
    # -------------------------------
    win_events = {'foul', 'called_strike', 'swinging_strike', 'field_out', 'strikeout'}
    df_joined['win'] = df_joined['event_category'].apply(lambda x: 1 if x in win_events else 0)

    # -------------------------------
    # 6. Adjust PX Orientation
    # -------------------------------
    df_joined['PX'] = df_joined['PX'] * -1

    # -------------------------------
    # 7. Create Binary Count Categories
    # -------------------------------
    df_joined['count_0_0'] = ((df_joined['balls'] == 0) & (df_joined['strikes'] == 0)).astype(int)
    df_joined['count_hitters'] = df_joined[['balls', 'strikes']].apply(
        lambda x: 1 if (x['balls'], x['strikes']) in [(1,0), (2,0), (3,0), (3,1)] else 0, axis=1
    )
    df_joined['count_pitchers'] = df_joined[['balls', 'strikes']].apply(
        lambda x: 1 if (x['balls'], x['strikes']) in [(0,2), (0,1), (1,2)] else 0, axis=1
    )
    df_joined['count_2k'] = ((df_joined['strikes'] == 2) & (df_joined['balls'] != 3)).astype(int)

    # -------------------------------
    # 8. Additional Binary Features
    # -------------------------------
    # 8.1 Strike
    strike_events = {"foul", "called_strike", "swinging_strike", "field_out", "strikeout",
                     "home_run", "triple", "double", "single"}
    df_joined['Strike'] = df_joined['event_category'].isin(strike_events)

    # # 8.2 Comploc
    # df_joined['Comploc'] = df_joined.apply(
    #     lambda row: -1.15 <= row['PX'] <= 1.15 and 1.1 <= row['PZ'] <= 3.9, axis=1
    # )

    # # 8.3 Inzone
    # df_joined['Inzone'] = df_joined.apply(
    #     lambda row: -0.83 <= row['PX'] <= 0.83 and 1.5 <= row['PZ'] <= 3.5, axis=1
    # )

    # 8.4 Swing
    swing_events = {"foul", "swinging_strike", "field_out", "home_run", "triple", "double", "single"}
    df_joined['Swing'] = df_joined['event_category'].isin(swing_events)

    # 8.5 Whiff
    df_joined['Whiff'] = df_joined['pitchResult'].str.lower().str.contains("swinging", na=False).astype(int)

    # -------------------------------
    # 9. Create delta_run_exp_squared
    # -------------------------------
    df_joined['delta_run_exp_squared'] = df_joined['delta_run_exp'].apply(lambda x:
        0.5 + (x - 0.5) * 0.5 if x > 0.5 else
        -0.5 + (x + 0.5) * 0.5 if x < -0.5 else
        0.2 + (x - 0.2) * 0.75 if x > 0.2 else
        -0.2 + (x + 0.2) * 0.75 if x < -0.2 else
        x * 2.5
    )

    # -------------------------------
    # 10. Convert Numeric Columns
    # -------------------------------
    numeric_cols = ["Vel", "delta_run_exp", "Extension", "HorzApprAngle", "VertApprAngle", 
                    "IndVertBrk", "HorzBrk", "RelZ", "RelX","HorzRelAngle", "VertRelAngle"]
    for col in numeric_cols:
        df_joined[col] = pd.to_numeric(df_joined[col], errors='coerce')

    # -------------------------------
    # 11. Calculate Runs Scored
    # -------------------------------
    df_joined['Runs Scored'] = np.maximum(
        0,
        np.maximum(
            df_joined['opponentCurrentRuns'].shift(-1) - df_joined['opponentCurrentRuns'],
            df_joined['currentRuns'].shift(-1) - df_joined['currentRuns'].fillna(0)
        )
    )

    # -------------------------------
    # 12. Convert gameDate to datetime
    # -------------------------------
    df_joined['gameDate'] = pd.to_datetime(df_joined['gameDate'])

    # -------------------------------
    # 13. Clean pitchResult, abbreviate
    # -------------------------------
    df_joined['clean_pitchResult'] = df_joined['pitchResult'].str.split(' on a').str[0].str.strip()
    event_abbreviations = {
        "Single": "1B",
        "Foul": "Foul",
        "Hit By Pitch": "HBP",
        "Strike Swinging": "SS",
        "Strikeout (Swinging)": "K",   # forward K for swinging
        "Strikeout (Looking)": "ꓘ",   # backward K for looking
        "Ball": "B",
        "Walk": "BB",
        "Home Run": "HR",
        "Double": "2B",
        "Fielder's Choice": "FC",
        "Triple": "3B",
        "Reached on Error": "ROE",
        "Sac Fly": "SF"
    }
    df_joined['clean_pitchResult'] = df_joined['clean_pitchResult'].map(event_abbreviations).fillna(df_joined['clean_pitchResult'])

    # -------------------------------
    # 14. Create Event_Desc
    # -------------------------------
    df_joined['Event_Desc'] = df_joined.apply(lambda row: (
        f"{row['balls']}-{row['strikes']} "
        f"{row['pitchTypeFull']}, "
        + (f"{int(row['Runs Scored'])} Run " if row['Runs Scored'] > 0 else '')
        + f"{row['clean_pitchResult']}, "
        + (
            "Bases Empty"
            if not (row['ManOn1st'] == 1 or row['ManOn2nd'] == 1 or row['ManOn3rd'] == 1)
            else "Runners on "
                 + " ".join(filter(None, [
                     "1st" if row['ManOn1st'] == 1 else '',
                     "2nd" if row['ManOn2nd'] == 1 else '',
                     "3rd" if row['ManOn3rd'] == 1 else ''
                 ]))
        )
        + f", {row['inn']} "
        f'{row["outs"]} Out'
    ), axis=1)

    # -------------------------------
    # 15. Mark Leadoff Batters
    # -------------------------------
    valid_leadoff_events = {"single", "double", "triple", "home_run", "walk", "hit_by_pitch"}

    # Step 1: Identify leadoff batters
    df_joined["inning_leadoff"] = df_joined.groupby(["gameDate", "inn"])["abNumInGame"].transform("min") == df_joined["abNumInGame"]

    # Step 2: Identify successful leadoff batters
    df_joined["inning_leadoff"] = df_joined["inning_leadoff"] & df_joined["event_category"].isin(valid_leadoff_events)

    # Step 3: If a leadoff batter succeeded in an inning, mark success for all rows in that inning
    df_joined["inning_leadoff_success"] = df_joined.groupby(["gameDate", "inn"])["inning_leadoff"].transform("max")

    # Convert True/False → 1/0
    df_joined["inning_leadoff"] = df_joined["inning_leadoff"].astype(int)
    df_joined["inning_leadoff_success"] = df_joined["inning_leadoff_success"].astype(int)

    df_joined['Taggedpitchtype'] = df_joined['pitchTypeFull']
    df_joined['Autopitchtype'] = df_joined['pitchTypeFull']

    # print("Feature engineering complete. Here's a preview:")
    # print(df_joined.head())
    # Ensure relevant columns are numeric
    numeric_columns = ["RelX", "HorzBrk", "IndVertBrk", "Vel"]
    for col in numeric_columns:
        if col in df_joined.columns:
            df_joined[col] = pd.to_numeric(df_joined[col], errors="coerce")

    # Drop rows with NaN in critical columns to avoid aggregation errors
    df_joined = df_joined.dropna(subset=numeric_columns)

    # Determine pitcher handedness
    # df_hand = (
    #     df_joined.groupby("pitcherId", as_index=False)["RelX"].mean()
    #     .rename(columns={"RelX": "avg_RelX"})
    # )
    # df_hand["pitcher_hand"] = np.where(df_hand["avg_RelX"] > 0, "R", "L")

    # # Merge handedness info back
    # df_joined = pd.merge(df_joined, df_hand[["pitcherId", "pitcher_hand"]], on="pitcherId", how="left")

    # Rename columns to standard references
    df_joined = df_joined.rename(columns={
        "Vel": "start_speed",
        "Spin": "spin_rate",
        "Extension": "extension",
        "RelZ": "z0",           # Release height
        "RelX": "x0",           # Release side
        "HorzBrk": "ax",        # Horizontal break
        "IndVertBrk": "az",     # Vertical break
        "pitchType": "pitch_type"
    })

    # Mirror for left-handed pitchers
    df_joined["pitcher_hand"] = df_joined["pitcherHand"]
    df_joined["ax"] = np.where(df_joined["pitcher_hand"] == "L", -df_joined["ax"], df_joined["ax"])
    df_joined["x0"] = np.where(df_joined["pitcher_hand"] == "L", -df_joined["x0"], df_joined["x0"])
    df_joined["is_fastball"] = df_joined["pitch_type"].isin(["FF", "FA", "SI"])
    # Most-used fastball logic
    fastball_types = ["FF", "SI", "FA"]
    df_joined["is_fastball"] = df_joined["pitch_type"].isin(["FF", "FA", "SI"])
    df_fb = df_joined[df_joined["pitch_type"].isin(fastball_types)].copy()

    # Group by (pitcherId, pitch_type), compute means & usage count
    df_agg = (
        df_fb.groupby(["pitcherId", "pitch_type"], as_index=False)
        .agg(
            avg_fastball_speed=("start_speed", "mean"),
            avg_fastball_az=("az", "mean"),
            avg_fastball_ax=("ax", "mean"),
            count=("start_speed", "count")
        )
    )

    # Sort by usage count, then avg_fastball_speed, descending
    df_agg = df_agg.sort_values(["count", "avg_fastball_speed"], ascending=[False, False])

    # Keep only the top row (most-used & fastest) per pitcherId
    df_agg = df_agg.drop_duplicates(subset=["pitcherId"], keep="first")

    # Merge back & compute diffs
    df_joined = pd.merge(
        df_joined,
        df_agg[["pitcherId", "avg_fastball_speed", "avg_fastball_az", "avg_fastball_ax"]],
        on="pitcherId",
        how="left"
    )

    df_joined["speed_diff"] = df_joined["start_speed"] - df_joined["avg_fastball_speed"]
    df_joined["az_diff"] = df_joined["az"] - df_joined["avg_fastball_az"]
    df_joined["ax_diff"] = df_joined["ax"] - df_joined["avg_fastball_ax"]

    return df_joined

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
        "horzbreak",
        "inducedvertbreak",
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
        "horzbreak":           "ax",         
        "inducedvertbreak":           "az",         
        "autopitchtype": "pitch_type"
    })

    # 3) MIRROR FOR LEFT-HANDED PITCHERS
    df["ax"] = np.where(df["pitcher_hand"] == "L", -df["ax"], df["ax"])
    # print(df["x0"].iloc[0])
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

    df["is_fastball"] = df["pitch_type"].isin(fastball_types)

    df["z0"] = df["z0"] * 12
    df["x0"] = df["x0"] * 12

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
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Import joblib from the local repository
    repo_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
    sys.path.append(repo_path)  # Ensure the repo is in the Python path
    
    import joblib  # Now import the local joblib
    
    # Construct the path to your joblib file
    model_path = os.path.join(repo_path, "NCAA_STUFF_PLUS_ALL.joblib")
    
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
        "ax_diff",
        "is_fastball"
    ]

    df_for_model[features] = df_for_model[features].apply(pd.to_numeric, errors='coerce')

    # -- MAKE PREDICTIONS
    predictions = model.predict(df_for_model[features])
    df_for_model["target"] = predictions

    # -- APPLY z-score & stuff-plus scaling
    target_mean_2023 = 0.011532333993710725
    target_std_2023  = 0.009399038486978739

    df_for_model["target_zscore"] = (
        (df_for_model["target"] - target_mean_2023) / target_std_2023
    )
    df_for_model["tj_stuff_plus"] = (
        100 - (df_for_model["target_zscore"] * 10)
    )

    whiff_model = joblib.load("whiff_model_grouped_training.joblib")

    # --- PREDICT WHIFF (xWhiff) ---
    df_for_model["xWhiff"] = whiff_model.predict(df_for_model[features])

    return df_for_model


# Load the CSV file
file_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/usd_baseball_TM_master_file.csv"
df = pd.read_csv(file_path)


df['Source'] = 'Preseason'

df.drop_duplicates(subset=['PitchUID'], inplace=True)

# Standardize column capitalization
df.columns = [col.strip().capitalize() for col in df.columns]

trufilepath = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/USDPITCHINGYTD.csv"

Trumediadf = pd.read_csv(trufilepath)


Trumediadf['Source'] = 'InSeason'




def process_relheight(df):
    # Define pitchers to exclude
    excluded_pitchers = ["Bunnell, Jack"]
    
    # Compute overall average Relheight per player
    player_overall_avg_relheight = (
        df.groupby('Pitcher')['Relheight']
        .mean()
        .reset_index()
        .rename(columns={'Relheight': 'player_overall_avg_relheight'})
    )
    
    # Filter out excluded pitchers
    df_non_excluded = df[~df['Pitcher'].isin(excluded_pitchers)].copy()
    
    # Compute average Relheight per player per date
    player_date_avg_relheight = (
        df_non_excluded.groupby(['Pitcher', 'Date'])['Relheight']
        .mean()
        .reset_index()
        .rename(columns={'Relheight': 'player_date_avg_relheight'})
    )
    
    # Merge overall and daily averages
    player_diff = pd.merge(
        player_date_avg_relheight, player_overall_avg_relheight, on='Pitcher', how='left'
    )
    
    # Compute the difference
    player_diff['diff_relheight'] = (
        player_diff['player_overall_avg_relheight'] - player_diff['player_date_avg_relheight']
    )
    
    # Compute average difference per date
    avg_diff_per_date = (
        player_diff.groupby('Date')['diff_relheight']
        .mean()
        .reset_index()
        .rename(columns={'diff_relheight': 'avg_diff_relheight'})
    )
    
    # Merge average difference back to main DataFrame
    df = pd.merge(df, avg_diff_per_date, on='Date', how='left')
    
    # Rename original 'Relheight' and create scaled 'Relheight'
    df.rename(columns={'Relheight': 'relheight_uncleaned'}, inplace=True)
    df['avg_diff_relheight'] = df['avg_diff_relheight'].fillna(0)
    df['Relheight'] = df['relheight_uncleaned'] + df['avg_diff_relheight']
    
    # Handle missing values
    df['Relheight'] = df['Relheight'].fillna(df['relheight_uncleaned'])
    
    return df

# Apply transformation only if Source is 'Trumedia'
if 'Source' in df.columns and (df['Source'] == 'Preseason').any():
    # Process Relheight
    df = process_relheight(df)
    
    # Create 'Swing' column based on Exitspeed and Pitchcall
    df['Swing'] = np.where(
        (df['Exitspeed'] > 0) | (df['Pitchcall'].str.contains('Swing|Foul', case=False, na=False)),
        'Swing',
        'Take'
    )

    # Create 'Contact' column based on Exitspeed
    df['Contact'] = np.where(df['Exitspeed'] > 0, 'Yes', 'No')

    # Create 'Whiff' column
    df['Whiff'] = np.where(
        (df['Swing'] == 'Swing') & (df['Contact'] == 'No'),
        1,  # Mark as 1 if both conditions are met
        0    # Otherwise, mark as 0
    ).astype(int)

    # Create 'Count' column combining Balls and Strikes
    df['Count'] = df['Balls'].astype(str) + '-' + df['Strikes'].astype(str)


df_for_model = df.copy()

# Rename columns to lowercase for the feature_engineering function
df_for_model.columns = [c.lower() for c in df_for_model.columns]

trudf_for_model = Trumediadf.copy()

trudf_for_model['Taggedpitchtype'] = trudf_for_model['pitchTypeFull']
trudf_for_model['Autopitchtype'] = trudf_for_model['pitchTypeFull']


# Ensure the columns needed by feature_engineering exist
# (RelSpeed, RelHeight, RelSide, ax0, az0, AutoPitchType, Pitcher, SpinRate, Extension)
# If any are missing, you may need to handle that or rename them properly.

# 1) FEATURE ENGINEERING
df_for_model = feature_engineering(df_for_model)

# 2) RUN MODEL + SCALING
df_for_model = run_model_and_scale(df_for_model)

trudf_for_model = Trumedia_feature_engineering(trudf_for_model)

trudf_for_model.columns = [c.lower() for c in trudf_for_model.columns]

trudf_for_model = run_model_and_scale(trudf_for_model)



if "pitchuid" in df_for_model.columns and "Pitchuid" in df.columns:
    # We select only the new columns from df_for_model we want to bring back
    merged_cols = ["pitchuid", "target", "target_zscore", "tj_stuff_plus", "xWhiff"]
    df = pd.merge(
        df, 
        df_for_model[merged_cols], 
        left_on="Pitchuid", right_on="pitchuid", 
        how="left"
    )
    # You might drop the duplicate "pitchuid" column from df
    df.drop(columns=["pitchuid"], inplace=True, errors="ignore")


import pandas as pd

# 1) Load CSV file
trumedia_df = pd.read_csv("https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/trumediatotrackmannamejoin.csv")

# 2) Rename pitcherAbbrevName to pitcherabbrevname in 'trumedia_df'
trumedia_df.rename(columns={"pitcherAbbrevName": "pitcherabbrevname"}, inplace=True)

# 3) Drop 'Pitcher' from 'trudf_for_model' if it already exists
if "pitcher" in trudf_for_model.columns:
    trudf_for_model.drop(columns=["pitcher"], inplace=True)


trudf_for_model['Pitcherthrows'] = trudf_for_model['pitcher_hand'].apply(lambda x: 'Left' if x == 'L' else 'Right')

# 4) Merge 'trudf_for_model' with 'trumedia_df' to pull in the 'Pitcher' column
trudf_for_model = trudf_for_model.merge(
    trumedia_df[['pitcherabbrevname', 'Pitcher']], 
    on='pitcherabbrevname', 
    how='left'
)
trudf_for_model.columns = [col.strip().capitalize() for col in trudf_for_model.columns]
# 5) Rename columns in 'trudf_for_model'
trudf_for_model = trudf_for_model.rename(columns={
    "Start_speed": "Relspeed",
    "Spin_rate": "Spinrate",
    "extension": "Extension",
    "Z0": "Relheight",         # Release height
    "X0": "Relside",           # Release side
    "Ax": "Horzbreak",         # Horizontal break
    "Az": "Inducedvertbreak",  # Vertical break
    "pitchTypeFull": "Taggedpitchtype",
    "horzapprangle": "Horzapprangle",
    "vertapprangle": "Vertapprangle",
    "horzrelangle": "Horzrelangle",
    "vertrelangle": "Vertrelangle",
    "uniqPitchId": "pitchuid",
    "Pz": "Platelocheight",
    "Px": "Platelocside",
    "Batterhand": "Batterside",
    "Spindir": "Spinaxis",
    "balls": "Balls",
    "strikes": "Strikes",
    "Tj_stuff_plus": "tj_stuff_plus",
    "Launchang": "Angle",
    "Exitdir": "Direction",
    "Exitvelocity": "Exitspeed",
    "Xwhiff": "xWhiff"

})
trudf_for_model['Date'] = pd.to_datetime(trudf_for_model['Date']).dt.date
trudf_for_model[['Relheight', 'Relside']] /= 12
trudf_for_model['Batterside'] = trudf_for_model['Batterside'].map({'R': 'Right', 'L': 'Left'})



# 6) If 'df' doesn't exist yet, define it as an empty DataFrame
if 'df' not in globals():
    df = pd.DataFrame()

# 7) Remove any duplicate columns in both DataFrames
df = df.loc[:, ~df.columns.duplicated()].copy()
trudf_for_model = trudf_for_model.loc[:, ~trudf_for_model.columns.duplicated()].copy()

# 8) Concatenate the two DataFrames
df = pd.concat([df, trudf_for_model], ignore_index=True, sort=False)

# 'df' now contains the appended data with the updated 'Pitcher' column




# OPTIONAL: Merge the new columns (target, tj_stuff_plus) back into the original "df"
# so that you can reference them in your existing plots/tables if desired.
# We'll merge on a unique identifier you have (e.g., Pitchuid), if it exists in both.
# For demonstration, let's assume "pitchuid" (lowercase in df_for_model).

    
# Load arm angle CSV
armangle_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/armangle_final_fall_usd.csv"
armangle_df = pd.read_csv(armangle_path)

# Merge arm angle data into df on 'Pitcher'
df = df.merge(armangle_df[['Pitcher', 'armangle_prediction']], on='Pitcher', how='left')

# df.dropna(subset=['Date'], inplace=True)
# df["datetime"] = pd.to_datetime(df["Date"], errors="coerce")
# df["Pitchno"] = pd.to_numeric(df["Pitchno"], errors="coerce")

# # 2) Add 12 hours (to shift from midnight to noon) 
# #    plus the minutes indicated by 'Time'
# df["datetime"] = (
#     df["datetime"]
#     + pd.to_timedelta(12, unit="h")         # shift to noon
#     + pd.to_timedelta(df["Pitchno"], unit="m")  # add the minutes from noon
# )


df['Pitchtype'] = df['Taggedpitchtype'].replace('Undefined', np.nan).fillna(df['Autopitchtype'])
df['Pitchtype'] = df['Pitchtype'].replace(['Four-Seam', 'FourSeamFastBall'], 'Fastball')
df['Pitchtype'] = df['Pitchtype'].replace(['ChangeUp'], 'Changeup')

# df.to_csv('streamlit_2024_fall_data.csv', index=False)


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
    lambda row: -0.83 <= row['Platelocside'] <= 0.83 and 1.5 <= row['Platelocheight'] <= 3.5, axis=1)

df['Comploc'] = df.apply(
    lambda row: -1.15 <= row['Platelocside'] <= 1.15 and 1.1 <= row['Platelocheight'] <= 3.9, axis=1)

# Define pitch categories based on initial pitch types
pitch_categories = {
    "Breaking Ball": ["Slider", "Curveball"],
    "Fastball": ["Fastball", "Four-Seam", "Sinker", "Cutter", "TwoSeamFastBall"],
    "Offspeed": ["ChangeUp", "Splitter"]
}

# Function to categorize pitch types into broader groups
def categorize_pitch_type(pitch_type):
    for category, pitches in pitch_categories.items():
        if pitch_type in pitches:
            return category
    return None

# Create a new column 'Pitchcategory' to categorize pitches
df['Pitchcategory'] = df['Pitchtype'].apply(categorize_pitch_type)

df = df.copy()
df['PitcherPitchNo'] = df.groupby(['Pitcher', 'Date']).cumcount() + 1

# Set up the color palette based on pitch type
pitch_types = df['Pitchtype'].unique()
palette = sns.color_palette('Set2', len(pitch_types))
color_map = {
        'Fastball': '#1f77b4',  # Blue
        'TwoSeamFastBall': '#1f77b4',  # Blue
        'Slider': '#ff7f0e',    # Orange
        'Curveball': '#2ca02c', # Green
        'ChangeUp': '#d62728',  # Red
        'Changeup': '#d62728',  # Red
        'Cutter': '#9467bd',    # Purple
        'Sinker': '#8c564b',    # Brown
        'Splitter': '#e377c2',  # Pink
        'Knuckleball': '#7f7f7f', # Gray
        'Other': '#7f7f7f', # Gray
        'Undefined': '#7f7f7f',
        'FourSeamFastBall': '#1f77b4'
    }

# ------------------------------------
#  STREAMLIT SIDEBAR FILTERS (REPLACE)
# ------------------------------------
st.sidebar.header("Filter Options")

# 1) Get unique pitchers and exclude "Bunnell, Jack"
filtered_pitchers = df['Pitcher'].unique()
filtered_pitchers = [pitcher for pitcher in filtered_pitchers if pitcher != "Bunnell, Jack"]
# 2) Insert "All Pitchers" at the top (not as default selection)
filtered_pitchers.insert(0, "All Pitchers")

# 3) Selectbox for pitcher
selected_pitcher = st.sidebar.selectbox("Select Pitcher", filtered_pitchers)

# 6) Get unique sources and add filter
sources_available = df['Source'].unique()
selected_sources = st.sidebar.multiselect("Select Sources", sources_available, default=sources_available)

# 4) Determine which dates to show:
if selected_pitcher == "All Pitchers":
    # If "All Pitchers" is selected, filter dates based on selected sources
    dates_available = df[df['Source'].isin(selected_sources)]['Date'].unique()
else:
    # Otherwise, filter dates for the selected pitcher and selected sources
    dates_available = df[(df['Pitcher'] == selected_pitcher) & (df['Source'].isin(selected_sources))]['Date'].unique()

# 5) Multiselect for dates
selected_dates = st.sidebar.multiselect("Select Dates", dates_available, default=dates_available)

# 7) Filter the main DataFrame accordingly
filtered_data = df[df['Date'].isin(selected_dates) & df['Source'].isin(selected_sources)]
if selected_pitcher != "All Pitchers":
    filtered_data = filtered_data[filtered_data['Pitcher'] == selected_pitcher]

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
    grouped = pitcher_data.groupby(['Pitcher', 'Pitchtype'])

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
    strikes_count = strikes.groupby(['Pitcher', 'Pitchtype']).size()
    # Compute strike percentage
    strike_percentages = (strikes_count / pitch_type_counts * 100).rename('Strike %').round(1)

    # -------------------
    # WHIFF %
    # -------------------
    # Compute whiff percentage as: (Number of whiffs) / (Number of swings) * 100
    swinging_strikes = pitcher_data[pitcher_data['Whiff']].groupby(['Pitcher', 'Pitchtype']).size()
    total_swings = pitcher_data[pitcher_data['Swing']].groupby(['Pitcher', 'Pitchtype']).size()
    whiff_percentages = (swinging_strikes / total_swings * 100).rename('Whiff %').fillna(0).round(1)
    
    # -------------------
    # INZONE %
    # -------------------
    # Compute in-zone percentage as: (Number of in-zone pitches) / (Total pitches) * 100
    in_zone_counts = pitcher_data[pitcher_data['Inzone']].groupby(['Pitcher', 'Pitchtype']).size()
    in_zone_percentage = (in_zone_counts / pitch_type_counts * 100).rename('InZone %').fillna(0).round(1)
    
    # -------------------
    # INZONE WHIFF %
    # -------------------
    # Compute in-zone whiff percentage as: (Number of whiffs in-zone) / (Number of swings in-zone) * 100
    in_zone_swinging_strikes = pitcher_data[(pitcher_data['Inzone']) & (pitcher_data['Whiff'])].groupby(['Pitcher', 'Pitchtype']).size()
    in_zone_swings = pitcher_data[(pitcher_data['Inzone']) & (pitcher_data['Swing'])].groupby(['Pitcher', 'Pitchtype']).size()
    in_zone_whiff_percentages = (in_zone_swinging_strikes / in_zone_swings * 100).rename('InZone Whiff %').fillna(0).round(1)
    
    # -------------------
    # CHASE %
    # -------------------
    # Here we define 'Chase %' as the percentage of out-of-zone swings.
    out_zone_swings = pitcher_data[(~pitcher_data['Inzone']) & (pitcher_data['Swing'])].groupby(['Pitcher', 'Pitchtype']).size()
    total_out_zone_pitches = pitcher_data[~pitcher_data['Inzone']].groupby(['Pitcher', 'Pitchtype']).size()
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
    comp_counts = comp_data.groupby(['Pitcher', 'Pitchtype']).size()
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
    ax1.set_title(f"Avg TJStuff+ per 10-Pitch Group for {pitcher_name}")
    ax1.set_xlabel("PitcherPitchNo Interval")
    ax1.set_ylabel("Average TJStuff+")
    ax1.legend(title="Pitch Type")

    # Plot Relspeed
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for pitch_type in df_agg['Pitchtype'].dropna().unique():
        df_plot = df_agg[df_agg['Pitchtype'] == pitch_type].sort_values("PitchGroup")
        ax2.plot(df_plot["PitchGroup"], df_plot["Relspeed"], marker="o", label=pitch_type)
    ax2.set_title(f"Avg Relspeed per 10-Pitch Group for {pitcher_name}")
    ax2.set_xlabel("PitcherPitchNo Interval")
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
