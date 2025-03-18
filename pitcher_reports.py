import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta
import os
import streamlit as st
# ------------------------------------------------------------------
# Reset and force white backgrounds and navy text/lines
# ------------------------------------------------------------------
plt.style.use('default')
sns.set_theme(style=None)
plt.rcdefaults()
mpl.rcParams['figure.facecolor'] = '#FFFFFF'
mpl.rcParams['axes.facecolor']   = '#FFFFFF'
mpl.rcParams['savefig.facecolor'] = '#FFFFFF'
for param in ('text.color','axes.edgecolor','axes.labelcolor',
              'xtick.color','ytick.color','grid.color'):
    mpl.rcParams[param] = 'navy'

# ------------------------------------------------------------------
# Helper function to lighten a color
# ------------------------------------------------------------------
def lighten_color(color, amount=0.7):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = mc.to_rgb(c)
    c = colorsys.rgb_to_hls(*c)
    lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return lightened

# ------------------------------------------------------------------
# Load the percentile reference CSV (ensure it's in your working directory)
# ------------------------------------------------------------------
percentile_table = pd.read_csv("pitch_metric_percentiles.csv")

def get_dynamic_norm(pitch_type, metric):
    rows = percentile_table[percentile_table["pitch_type"] == pitch_type]
    if rows.empty:
        return TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    try:
        vmin = float(rows.loc[rows["percentile"] == 0.10, metric].values[0])
        vcenter = float(rows.loc[rows["percentile"] == 0.50, metric].values[0])
        vmax = float(rows.loc[rows["percentile"] == 0.90, metric].values[0])
    except IndexError:
        return TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    return TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

def get_strike_zone():
    return Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none')

def get_home_plate():
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    return Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='black', facecolor='none')

# ------------------------------------------------------------------
# Load the count summary CSV for the table
# ------------------------------------------------------------------
df_count_summary = pd.read_csv("count_summary_table.csv")

def plot_count_summary_table(ax, df):
    """
    Plots a table in 'ax' showing RV/100, Win %, and Total Pitches (P)
    for the count categories: '0-0', 'Hitters', 'Pitchers', '2K'
    """
    ax.axis('off')
    categories = {
        "0-0": "Count_0_0",
        "Hitters": "Count_hitters",
        "Pitchers": "Count_pitchers",
        "2K": "Count_2k"
    }
    table_data = []
    for cat_name, cat_col in categories.items():
        sub = df[df[cat_col] == 1]
        total_pitches = len(sub)
        if total_pitches == 0:
            rv_100 = 0.0
            win_percent = 0.0
        else:
            total_rv = sub['Delta_run_exp'].sum()
            total_wins = sub['Win'].sum()
            rv_100 = total_rv
            win_percent = (total_wins / total_pitches) * 100
        table_data.append([cat_name, f"{rv_100:.1f}", f"{win_percent:.0f}", total_pitches])
    col_labels = ["Count", "RV", "Win %", "P"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 1.2)
    rv_col_idx = col_labels.index("RV")
    win_col_idx = col_labels.index("Win %")
    for r, row in enumerate(table_data):
        row_data = df_count_summary[df_count_summary["Count"] == row[0]]
        if row_data.empty:
            continue
        rv_vcenter = float(row_data["RV"].values[0])
        win_vcenter = float(row_data["Win %"].values[0])
        rv_norm = TwoSlopeNorm(vmin=-5, vcenter=rv_vcenter, vmax=10)
        win_norm = TwoSlopeNorm(vmin=0, vcenter=win_vcenter, vmax=100)
        for c, cell in table.get_celld().items():
            if c[0] == r+1:
                try:
                    value = float(cell.get_text().get_text())
                    if c[1] == rv_col_idx:
                        color = plt.cm.RdYlGn_r(rv_norm(value))
                        cell.set_facecolor(lighten_color(color, amount=0.7))
                    elif c[1] == win_col_idx:
                        color = plt.cm.RdYlGn(win_norm(value))
                        cell.set_facecolor(lighten_color(color, amount=0.7))
                except ValueError:
                    pass
    ax.set_title("RV, Win% by Count", fontsize=12, pad=10)

def plot_logo(ax, logo_path):
    ax.axis('off')
    try:
        logo = mpimg.imread(logo_path)
        ax.imshow(logo, extent=(0, 0.5, 0, 1), aspect='auto')
    except FileNotFoundError:
        ax.text(0.25, 0.5, "Logo not found", ha='center', va='center', fontsize=12)

def plot_header(ax, text_content):
    ax.axis('off')
    ax.text(0.75, 0.5, text_content, ha='left', va='center', fontsize=12)

def plot_blank(ax):
    ax.axis('off')

def plot_color_bar(ax, cmap):
    gradient = np.linspace(-0.5, 1.5, 256).reshape(256, 1)
    norm = plt.Normalize(-0.5, 1.5)
    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm, origin='lower')
    ax.set_xticks([])
    ax.set_yticks([0, 255])
    ax.set_yticklabels(["-0.5\n3-0 out", "1.5\n0-2 HR"], fontsize=8)
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_title("Run Value per Pitch", fontsize=14, rotation=90, x=-0.5, y=0.5, va='center')

def plot_pitch_scatter(ax, data, cmap, norm, title=None, overall_top_5=None):
    """
    Plots a PX/PZ scatter for the given data, with an optional title.
    Maintains the aspect ratio and hides ticks.

    Additionally, if overall_top_5 is provided (the overall top 5 pitches for the pitcher),
    it labels those points that are in the current subset using their overall ranking.
    """
    if title:
        ax.set_title(title, fontsize=16, pad=10)

    # Check for column name inconsistencies and adjust accordingly
    px_col = "Platelocside"
    pz_col = "Platelocheight"
    exp_col = "delta_run_exp_squared" if "delta_run_exp_squared" in data.columns else "Delta_run_exp_squared"

    # Scatter plot for pitch locations
    ax.scatter(
        data[px_col], data[pz_col],
        c=data[exp_col], cmap=cmap, norm=norm,
        s=60, edgecolor='black'
    )

    # Add strike zone and home plate visualization
    ax.add_patch(get_strike_zone())
    ax.add_patch(get_home_plate())

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(5 / 4)

    # If overall_top_5 is provided, label points with their overall ranking
    if overall_top_5 is not None:
        top_subset = overall_top_5[overall_top_5.index.isin(data.index)]

        for _, row in top_subset.iterrows():
            x_val = row[px_col]
            y_val = row[pz_col]
            overall_rank = row.get('overall_rank', None)  # Check if column exists
            if overall_rank is not None:
                ax.text(
                    x_val + 0.05,
                    y_val + 0.05,
                    str(overall_rank),
                    color='blue',
                    fontsize=10,
                    fontweight='bold'
                )
              
def compute_bottom_row_summary(df):
    format_map = {
        "P": "{:.0f}", "Usage%": "{:.0f}", "Vel": "{:.1f}", "MaxVel": "{:.1f}",
        "Stuff+": "{:.1f}", "RV/100": "{:.1f}", "Str%": "{:.0f}", "Comp%": "{:.0f}",
        "zWhiff%": "{:.0f}", "Chase%": "{:.0f}", "Ext": "{:.1f}", "HAA": "{:.1f}",
        "VAA": "{:.1f}", "IVB": "{:.1f}", "HB": "{:.1f}", "RelZ": "{:.1f}", "RelX": "{:.1f}"
    }
    metrics = ["P", "Usage%", "Vel", "MaxVel", "Stuff+", "RV/100", "Str%", "Comp%",
               "zWhiff%", "Chase%", "Ext", "HAA", "VAA", "IVB", "HB", "RelZ", "RelX"]
    grouped_data = {}
    abbrev_map = {}
    for pitch_full, group in df.groupby('Pitchtype'):
        total_pitches = len(group)
        pitch_abbrev = group["Pitchtype"].iloc[0] if not group.empty else pitch_full
        abbrev_map[pitch_full] = pitch_abbrev
        if total_pitches == 0:
            grouped_data[pitch_full] = {m: 0.0 for m in metrics}
            continue
        # CHANGED Tj_stuff_plus → tj_stuff_plus
        avg_stuff_plus = group["tj_stuff_plus"].mean()
        rv_100 = group["Delta_run_exp"].sum()
        strike_pct = group["Strike"].mean() * 100
        comploc_pct = group["Comploc"].mean() * 100
        in_zone_swings = group[group["Inzone"]]
        whiffs_in_zone = in_zone_swings["Whiff"].sum()
        z_whiff_pct = (whiffs_in_zone / len(in_zone_swings) * 100) if len(in_zone_swings) > 0 else 0.0
        out_of_zone_pitches = group[~group["Inzone"]]
        out_of_zone_swings = out_of_zone_pitches["Swing"].sum()
        chase_pct = (out_of_zone_swings / len(out_of_zone_pitches) * 100) if len(out_of_zone_pitches) > 0 else 0.0
        grouped_data[pitch_full] = {
            "P": total_pitches,
            "Usage%": (total_pitches / len(df)) * 100,
            "Vel": group["Relspeed"].mean(),
            "MaxVel": group["Relspeed"].max(),
            "Stuff+": avg_stuff_plus,
            "RV": rv_100,
            "Str%": strike_pct,
            "Comp%": comploc_pct,
            "zWhiff%": z_whiff_pct,
            "Chase%": chase_pct,
            "Ext": group["Extension"].mean(),
            "HAA": group["Horzapprangle"].mean(),
            "VAA": group["Vertapprangle"].mean(),
            "IVB": group["Inducedvertbreak"].mean(),
            "HB": group["Horzbreak"].mean(),
            "RelZ": group["Relheight"].mean(),
            "RelX": group["Relside"].mean()
        }
    row_labels = sorted(grouped_data.keys(), key=lambda x: grouped_data[x]["P"], reverse=True)
    cellText = []
    for pitch_full in row_labels:
        row_dict = grouped_data[pitch_full]
        row_formatted = [format_map[m].format(row_dict[m]) if row_dict[m] is not None else "-" for m in metrics]
        cellText.append(row_formatted)
    return row_labels, metrics, cellText, abbrev_map

def plot_bottom_row_table(ax, df):
    ax.axis('off')
    row_labels, col_labels, cellText, abbrev_map = compute_bottom_row_summary(df)
    table = ax.table(cellText=cellText, rowLabels=row_labels, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.8, 1.8)
    rv100_col_idx = col_labels.index("RV")
    stuff_col_idx = col_labels.index("Stuff+")
    dynamic_metrics = {"Str%": "strike_pct", "Comp%": "comploc_pct", "zWhiff%": "z_whiff_pct", "Chase%": "chase_pct"}
    dynamic_cols = {name: col_labels.index(name) for name in dynamic_metrics}
    rv_norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=10)
    stuff_norm = TwoSlopeNorm(vmin=80, vcenter=100, vmax=120)
    for (r, c), cell in table.get_celld().items():
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")
        cell.set_linewidth(0.5)
        if r > 0:
            pitch_full = row_labels[r - 1]
            pitch_type = abbrev_map.get(pitch_full, pitch_full)
            try:
                value = float(cellText[r - 1][c])
                if c == rv100_col_idx:
                    norm_obj = rv_norm
                    color = plt.cm.RdYlGn_r(norm_obj(value))
                elif c == stuff_col_idx:
                    norm_obj = stuff_norm
                    color = plt.cm.RdYlGn(norm_obj(value))
                elif c in dynamic_cols.values():
                    col_name = [name for name, idx in dynamic_cols.items() if idx == c][0]
                    norm_obj = get_dynamic_norm(pitch_type, dynamic_metrics[col_name])
                    color = plt.cm.RdYlGn(norm_obj(value))
                else:
                    continue
                cell.set_facecolor(lighten_color(color, amount=0.7))
            except ValueError:
                pass
    for col_index in range(len(col_labels)):
        table.auto_set_column_width(col_index)
    ax.set_title("Pitch Metrics", fontsize=14, pad=2, y=0.90)

def plot_top_5_events(ax, df):
    ax.axis('off')
    # CHANGED 'Event_Desc' → 'Event_desc'
    top_events = df[['Event_desc', 'Delta_run_exp']].copy()
    top_events['abs_delta_run_exp'] = top_events['Delta_run_exp'].abs()
    top_events = top_events.sort_values(by='abs_delta_run_exp', ascending=False).head(5)
    lines = []
    for i, (_, row) in enumerate(top_events.iterrows(), 1):
        color = 'green' if row['Delta_run_exp'] < 0 else 'darkred'
        event_text = f"#{i}. {row['Event_desc']}: {row['Delta_run_exp']:.2f} RV"
        lines.append((event_text, color))
    y_pos = 0.8
    for text, color in lines:
        ax.text(0.025, y_pos, text, ha='left', va='center', fontsize=9, color=color)
        y_pos -= 0.13
    ax.set_title("5 Biggest Pitches - See Location Plots", fontsize=12, pad=5, x=0.85)

def compute_game_summary(df):
    total_pitches = len(df)
    total_strikes = df['Strike'].sum()
    total_balls = df['Event_category'].isin(["ball", "walk", "hit_by_pitch"]).sum()
    strikeouts = df['Pitchresult'].str.contains("Strikeout", na=False).sum()
    walks = df['Pitchresult'].isin(["Walk", "Hit By Pitch"]).sum()
    hits = df['Event_category'].isin(["single", "double", "triple", "home_run"]).sum()
    runs_allowed = int(df['Runs scored'].sum())
    first_pitch_total = df[df["Pitchnuminab"] == 1]
    first_pitch_strike_pct = (first_pitch_total["Strike"].mean() * 100) if not first_pitch_total.empty else 0.0
    total_whiffs = df["Whiff"].sum()
    summary = {
        "P": total_pitches,
        "K-B": f"{total_strikes}-{total_balls}",
        "K": strikeouts,
        "BB": walks,
        "Hits": hits,
        "Runs": runs_allowed,
        "FPStr%": f"{first_pitch_strike_pct:.1f}%",
        "Whiffs": total_whiffs
    }
    return summary

def plot_game_summary(ax, df):
    ax.axis('off')
    summary = compute_game_summary(df)
    col_labels = list(summary.keys())
    cellText = [[str(value) for value in summary.values()]]
    table = ax.table(cellText=cellText, colLabels=col_labels, loc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(2.6, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")
        cell.set_linewidth(0.5)

# ---------------------------------------------------------------
# Generate a report for the filtered data (single report)
# ---------------------------------------------------------------
def generate_reports(filtered_df):
    """
    Generate a pitching report for the filtered data.
    Uses the first available pitcher name and prints a report.
    """
    if filtered_df is None or filtered_df.empty:
        print("No data available to generate a report.")
        return
    
    pitcher = filtered_df['Pitcher'].iloc[0] if 'Pitcher' in filtered_df.columns else "UnknownPitcher"
    batting_team = filtered_df['Battingteam'].iloc[0] if 'Battingteam' in filtered_df.columns else "UnknownTeam"
    report_date = pd.to_datetime(filtered_df['Date']).min().strftime('%Y-%m-%d') if 'Date' in filtered_df.columns else "UnknownDate"

    overall_top_5 = filtered_df.loc[filtered_df['Delta_run_exp'].abs().nlargest(5).index].copy()
    overall_top_5.sort_values('Delta_run_exp', ascending=False, inplace=True)
    overall_top_5['overall_rank'] = range(1, len(overall_top_5) + 1)

    fig = plt.figure(figsize=(12, 14))
    fig.patch.set_facecolor('white')
    height_ratios = [0.13, 0.15, 0.3, 0.3, 0.25]
    width_ratios = [0.7, 0.8, 2.55, 2.55, 2.55]
    gs = GridSpec(nrows=5, ncols=5, figure=fig, height_ratios=height_ratios, width_ratios=width_ratios)

    # Row 1: Logo
    ax_logo = fig.add_subplot(gs[0, 0:2])
    ax_logo.set_facecolor('white')
    plot_logo(ax_logo, "San_Diego_Toreros_logo.svg.png")

    # Row 2: Header text
    ax_header = fig.add_subplot(gs[1, 0:2])
    ax_header.set_facecolor('white')
    header_text = f"{pitcher}\nPost-Series Pitching Report\n{report_date}"
    ax_header.text(0, 0.5, header_text, ha='left', va='center', fontsize=18)
    ax_header.axis('off')

    # Row 1 Top-right: Count summary table
    ax_count = fig.add_subplot(gs[0, 4])
    ax_count.set_facecolor('white')
    plot_count_summary_table(ax_count, filtered_df)

    # Row 1 Top-center: Top 5 events
    ax_events = fig.add_subplot(gs[0, 2])
    ax_events.set_facecolor('white')
    plot_top_5_events(ax_events, filtered_df)

    # Row 2 Center: Game Summary
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.set_facecolor('white')
    plot_game_summary(ax_summary, filtered_df)

    # Rows 3 and 4: Color bar
    ax_color = fig.add_subplot(gs[2:4, 0])
    ax_color.set_facecolor('white')
    plot_color_bar(ax_color, plt.cm.RdYlGn_r)

    ax_vs_rhh = fig.add_subplot(gs[2, 1])
    ax_vs_rhh.set_facecolor('white')
    ax_vs_rhh.text(0.5, 0.5, "vs RHH", ha='center', va='center', fontsize=14)
    ax_vs_rhh.axis('off')

    ax_vs_lhh = fig.add_subplot(gs[3, 1])
    ax_vs_lhh.set_facecolor('white')
    ax_vs_lhh.text(0.5, 0.5, "vs LHH", ha='center', va='center', fontsize=14)
    ax_vs_lhh.axis('off')

        # Define pitch groups and visualization parameters.
    pitch_groups = {"Fast": "Fast", "Break": "Break", "Slow": "Slow"}
    custom_cmap = plt.cm.RdYlGn_r
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.05, vmax=1.5)

    # Row 3 & 4: Scatter plots for each pitch group, separated by batter side.
    for i, (pitch_label, pitch_group) in enumerate(pitch_groups.items()):
        # For Right-handed batters (Row 3)
        ax_pitch_rhh = fig.add_subplot(gs[2, i+2])
        ax_pitch_rhh.set_facecolor('white')
        data_rhh = filtered_df[(filtered_df['Pitchgroup'] == pitch_group) & (filtered_df['Batterside'] == 'Right')]
        plot_pitch_scatter(ax_pitch_rhh, data_rhh, custom_cmap, norm, title=pitch_label, overall_top_5=overall_top_5)
        
        # For Left-handed batters (Row 4)
        ax_pitch_lhh = fig.add_subplot(gs[3, i+2])
        ax_pitch_lhh.set_facecolor('white')
        data_lhh = filtered_df[(filtered_df['Pitchgroup'] == pitch_group) & (filtered_df['Batterside'] == 'Left')]
        plot_pitch_scatter(ax_pitch_lhh, data_lhh, custom_cmap, norm, overall_top_5=overall_top_5)

    # Row 5: Bottom row table of aggregated pitch metrics
    ax_bottom = fig.add_subplot(gs[4, :])
    ax_bottom.set_facecolor('white')
    plot_blank(ax_bottom)
    plot_bottom_row_table(ax_bottom, filtered_df)

    

    

    for ax in fig.get_axes():
        ax.set_facecolor('white')
    plt.subplots_adjust(hspace=0.05)

    st.pyplot(fig)

    plt.close(fig)
