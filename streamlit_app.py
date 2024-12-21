import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import seaborn as sns
from datetime import datetime

# Load the CSV file
file_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/usd_baseball_TM_master_file.csv"
df = pd.read_csv(file_path)

df.drop_duplicates(subset=['PitchUID'], inplace=True)

# Standardize column capitalization
df.columns = [col.strip().capitalize() for col in df.columns]

# Load arm angle CSV
armangle_path = "https://raw.githubusercontent.com/tdub29/streamlit-app-1/refs/heads/main/armangle_final_fall_usd.csv"
armangle_df = pd.read_csv(armangle_path)

# Merge arm angle data into df on 'Pitcher'
df = df.merge(armangle_df[['Pitcher', 'armangle_prediction']], on='Pitcher', how='left')


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

        # Add a label for the arm angle under the line
        ax.text(0, -3, f'Arm Angle = {avg_arm_angle:.1f}°', ha='center', va='top', fontsize=12)

        # Draw a small arc to illustrate the angle
        # If direction_sign is positive, arc from 0° to avg_arm_angle°
        # If direction_sign is negative, arc from 180° to 180° + avg_arm_angle
        start_angle = 0 if direction_sign > 0 else 180
        end_angle = start_angle + avg_arm_angle
        
        # Add an arc at the origin with a small radius (10 units) to show the angle visually
        arc = Arc((0, 0), width=10, height=10, angle=0, theta1=start_angle, theta2=end_angle, color='blue', alpha=0.5)
        ax.add_patch(arc)

    
    st.pyplot(fig)

# Function to calculate pitch metrics for each pitch type
def calculate_pitch_metrics(pitcher_data):
    pitch_type_counts = pitcher_data['Autopitchtype'].value_counts().rename('Count')
    avg_cols = ['Relspeed', 'Spinrate', 'Inducedvertbreak', 'Horzbreak',
                'Relheight', 'Relside', 'Extension', 'Vertapprangle', 'Horzapprangle']
    pitch_type_averages = pitcher_data.groupby('Autopitchtype')[avg_cols].mean().round(1)
    
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
                  .join(max_velocity)
                  .join(pitch_type_averages)
                  .join(strike_percentages)
                  .join(whiff_percentages)
                  .join(in_zone_percentage)
                  .join(in_zone_whiff_percentages)
                  .join(chase_percentage)
                  .fillna(0))
    metrics_df.columns = ['P', 'Max velo', 'AVG velo', 'Spinrate', 'IVB', 'HB', 'yRel', 'xRel', 'Ext.', 'VAA', 'HAA', 'Strike %', 'Whiff %', 'InZone %', 'InZone Whiff %', 'Chase %']
    return metrics_df

# Function to display pitch metrics table in Streamlit
def display_pitch_metrics():
    if filtered_data.empty:
        st.write("No data available for the selected filters.")
        return
    
    metrics_df = calculate_pitch_metrics(filtered_data)
    
    # Adjust column width by setting a max width on the dataframe
    # Format values to avoid additional decimal places
    styled_df = metrics_df.style.format(precision=1, na_rep="-").set_properties(**{'width': '80px'})  # Adjust width as needed

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
    "Pitch Metric AVG Table": display_pitch_metrics,
    "Raw Data": display_raw_data
}
selected_page = st.sidebar.radio("Select Plot", list(pages.keys()))

# Display selected plot
st.title(f"{selected_page}")
pages[selected_page]()
