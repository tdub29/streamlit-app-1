import matplotlib as mpl 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Polygon, Arc, Ellipse
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta
import os

# ------------------------------------------------------------------
# Reset and force white backgrounds and navy text/lines
# ------------------------------------------------------------------
plt.style.use('default')
sns.set_theme(style=None)
plt.rcdefaults()
plt.style.use('default')
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
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
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
    """
    Given a pitch_type and a metric key (from the percentile CSV),
    return a TwoSlopeNorm using the 10th, 50th, and 90th percentile values.
    """
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
    """Returns a new Rectangle patch for the strike zone."""
    return Rectangle((-0.83, 1.5), 1.66, 2.1, edgecolor='black', facecolor='none')

def get_home_plate():
    """Returns a new Polygon patch for home plate."""
    plate_vertices = [(-0.83, 0.1), (0.83, 0.1), (0.65, 0.25), (0, 0.5), (-0.65, 0.25)]
    return Polygon(plate_vertices, closed=True, linewidth=1, edgecolor='black', facecolor='none')

# ------------------------------------------------------------------
# Load the count summary CSV for the table
# ------------------------------------------------------------------
df_count_summary = pd.read_csv("count_summary_table.csv")

def plot_count_summary_table(ax, df):
    """
    Plots a clean table in 'ax' showing 
    RV/100, Win %, and Total Pitches (P) for these four count categories:
      '0-0', 'Hitters', 'Pitchers', '2K'
    The vcenter values for RV/100 and Win % are dynamically loaded from df_count_summary.
    """
    ax.axis('off')
    categories = {
        "0-0": "Count_0_0",
        "Hitters": "Count_Hitters",
        "Pitchers": "Count_Pitchers",
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
            rv_100 = (total_rv / total_pitches) * 100
            win_percent = (total_wins / total_pitches) * 100
        table_data.append([cat_name, f"{rv_100:.1f}", f"{win_percent:.0f}", total_pitches])
    col_labels = ["Count", "RV/100", "Win %", "P"]
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.3, 1.2)

    rv_col_idx = col_labels.index("RV/100")
    win_col_idx = col_labels.index("Win %")

    for r, row in enumerate(table_data):
        count_type = row[0]
        row_data = df_count_summary[df_count_summary["Count"] == count_type]
        if row_data.empty:
            continue
        rv_vcenter = float(row_data["RV/100"].values[0])
        win_vcenter = float(row_data["Win %"].values[0])
        rv_norm = TwoSlopeNorm(vmin=-5, vcenter=rv_vcenter, vmax=10)
        win_norm = TwoSlopeNorm(vmin=0, vcenter=win_vcenter, vmax=100)
        for c, cell in table.get_celld().items():
            if c[0] == r + 1:  # Skip header row
                try:
                    value = float(cell.get_text().get_text())
                    if c[1] == rv_col_idx:
                        color = plt.cm.RdYlGn_r(rv_norm(value))
                        color = lighten_color(color, amount=0.7)
                        cell.set_facecolor(color)
                    elif c[1] == win_col_idx:
                        color = plt.cm.RdYlGn(win_norm(value))
                        color = lighten_color(color, amount=0.7)
                        cell.set_facecolor(color)
                except ValueError:
                    pass
    ax.set_title("RV/100, Win% by Count", fontsize=12, pad=10)

def plot_logo(ax, logo_path):
    """Displays the logo on the left side of the Axes."""
    ax.axis('off')
    try:
        logo = mpimg.imread(logo_path)
        ax.imshow(logo, extent=(0, 0.5, 0, 1), aspect='auto')
    except FileNotFoundError:
        ax.text(0.25, 0.5, "Logo not found", ha='center', va='center', fontsize=12)

def plot_header(ax, text_content):
    """Displays the header text on the right side of the Axes."""
    ax.axis('off')
    ax.text(0.75, 0.5, text_content, ha='left', va='center', fontsize=12)

def plot_blank(ax):
    """Leaves the Axes blank."""
    ax.axis('off')

def plot_color_bar(ax, cmap):
    """
    Plots the vertical gradient (from -0.5 to 1.5) in the specified Axes.
    The labels have been swapped: -0.5\n3-0 out, 1.5\n0-2 HR
    """
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
    Plots a scatter of Px/Pz for the given data with color based on Delta_run_exp_squared.
    If a title is provided it is set. Additionally, if overall_top_5 is provided,
    those points are labeled with their overall ranking.
    """
    if title:
        ax.set_title(title, fontsize=16, pad=10)
    # Use "Px" and "Pz" (assumed to be precomputed in the data)
    ax.scatter(
        data['Px'], data['Pz'],
        c=data['Delta_run_exp_squared'], cmap=cmap, norm=norm,
        s=60, edgecolor='black'
    )
    ax.add_patch(get_strike_zone())
    ax.add_patch(get_home_plate())
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(5/4)
    if overall_top_5 is not None:
        top_subset = overall_top_5[overall_top_5.index.isin(data.index)]
        for _, row in top_subset.iterrows():
            x_val = row['Px']
            y_val = row['Pz']
            overall_rank = row['overall_rank']
            ax.text(x_val + 0.05, y_val + 0.05, str(overall_rank),
                    color='blue', fontsize=10, fontweight='bold')

def compute_bottom_row_summary(df):
    """
    Computes average values for selected pitch metrics grouped by Pitchtype,
    while storing the abbreviated pitch type for color mapping.
    """
    format_map = {
        "P": "{:.0f}",
        "Usage%": "{:.0f}",
        "Vel": "{:.1f}",
        "MaxVel": "{:.1f}",
        "Stuff+": "{:.1f}",
        "RV/100": "{:.1f}",
        "Str%": "{:.0f}",
        "Comp%": "{:.0f}",
        "zWhiff%": "{:.0f}",
        "Chase%": "{:.0f}",
        "Ext": "{:.1f}",
        "HAA": "{:.1f}",
        "VAA": "{:.1f}",
        "IVB": "{:.1f}",
        "HB": "{:.1f}",
        "RelZ": "{:.1f}",
        "RelX": "{:.1f}"
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
        avg_stuff_plus = group["Tj_stuff_plus"].mean()
        rv_100 = (group["Delta_run_exp"].sum() / total_pitches) * 100
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
            "RV/100": rv_100,
            "Str%": strike_pct,
            "Comp%": comploc_pct,
            "zWhiff%": z_whiff_pct,
            "Chase%": chase_pct,
            "Ext": group["Extension"].mean(),
            "HAA": group["Horzapprangle"].mean(),
            "VAA": group["Vertapprangle"].mean(),
            "IVB": group["Az"].mean(),
            "HB": group["Ax"].mean(),
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
    """Plots the bottom-row table grouped by Pitchtype with dynamic coloring."""
    ax.axis('off')
    row_labels, col_labels, cellText, abbrev_map = compute_bottom_row_summary(df)
    table = ax.table(cellText=cellText, rowLabels=row_labels, colLabels=col_labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.8, 1.8)
    rv100_col_idx = col_labels.index("RV/100")
    stuff_col_idx = col_labels.index("Stuff+")
    dynamic_metrics = {
        "Str%": "strike_pct",
        "Comp%": "comploc_pct",
        "zWhiff%": "z_whiff_pct",
        "Chase%": "chase_pct"
    }
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
                    color = plt.cm.PiYG(norm_obj(value))
                else:
                    continue
                color = lighten_color(color, amount=0.7)
                cell.set_facecolor(color)
            except ValueError:
                pass
    for col_index in range(len(col_labels)):
        table.auto_set_column_width(col_index)
    ax.set_title("Pitch Metrics", fontsize=14, pad=2, y=0.90)

def plot_top_5_events(ax, df):
    """
    Plots the top 5 most significant events by absolute Delta_run_exp in descending order.
    Negative values (good for pitcher) are green; positive values are dark red.
    """
    ax.axis('off')
    top_events = df[['Event_Desc', 'Delta_run_exp']].copy()
    top_events['abs_delta_run_exp'] = top_events['Delta_run_exp'].abs()
    top_events = top_events.sort_values(by='abs_delta_run_exp', ascending=False).head(5)
    lines = []
    for i, (_, row) in enumerate(top_events.iterrows(), 1):
        color = 'green' if row['Delta_run_exp'] < 0 else 'darkred'
        event_text = f"#{i}. {row['Event_Desc']}: {row['Delta_run_exp']:.2f} RV"
        lines.append((event_text, color))
    y_pos = 0.8
    for text, color in lines:
        ax.text(0.025, y_pos, text, ha='left', va='center', fontsize=9, color=color)
        y_pos -= 0.13
    ax.set_title("5 Biggest Pitches - See Location Plots", fontsize=12, pad=5, x=0.85)

def compute_game_summary(df):
    """
    Computes a summary of key pitching stats:
      - Total Pitches (P)
      - Strike-Ball Count (K-B)
      - Strikeouts (K)
      - Walks (BB)
      - Hits
      - Runs Allowed
      - First-Pitch Strike % (FPStr%)
      - Total Whiffs
    """
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
    """Plots the computed game summary in a table inside the given Axes."""
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

# ------------------------------------------------------------------
# Loop through each unique pitcher and generate a report
# ------------------------------------------------------------------

# Here we assume the final merged DataFrame is called df (with columns capitalized).
# It should include at least the following columns:
# "Pitcher", "Date", "Battingteam", "Delta_run_exp", "Delta_run_exp_squared", 
# "Pitchgroup", "Batterside", "Event_Desc", "Pitchresult", "Pitchnuminab", "Runs scored", etc.
for pitcher in df['Pitcher'].unique():
    # Define date range: last 5 days
    five_days_ago = datetime.now() - timedelta(days=5)
    # Filter data for current pitcher, excluding rows where Battingteam is 'USD'
    df_transformed = df[(df['Pitcher'] == pitcher) &
                        (df['Battingteam'] != 'USD') &
                        (pd.to_datetime(df['Date']) >= five_days_ago)]
    if df_transformed.empty:
        print(f"Skipping {pitcher} as no valid data is available.")
        continue
    # Compute overall top 5 pitches based on abs(Delta_run_exp)
    overall_top_5 = df_transformed.loc[df_transformed['Delta_run_exp'].abs().nlargest(5).index].copy()
    overall_top_5.sort_values('Delta_run_exp', ascending=False, inplace=True)
    overall_top_5['overall_rank'] = range(1, len(overall_top_5) + 1)

    # Set up figure and GridSpec layout
    fig = plt.figure(figsize=(12, 14))
    fig.patch.set_facecolor('white')
    height_ratios = [0.13, 0.15, 0.3, 0.3, 0.25]
    width_ratios = [0.7, 0.8, 2.55, 2.55, 2.55]
    gs = GridSpec(nrows=5, ncols=5, figure=fig,
                  height_ratios=height_ratios, width_ratios=width_ratios)

    # Row 1: Logo (cols 0-1)
    ax_logo = fig.add_subplot(gs[0, 0:2])
    ax_logo.set_facecolor('white')
    plot_logo(ax_logo, r"C:\Users\TrevorWhite\Downloads\San_Diego_Toreros_logo.svg.png")

    # Row 2: Header text (cols 0-1)
    ax_header = fig.add_subplot(gs[1, 0:2])
    ax_header.set_facecolor('white')
    header_text = f"{df_transformed['Pitcher'].iloc[0]}\nPost-Series Pitching Report\n" + \
                  ", ".join(pd.to_datetime(df_transformed['Date']).dt.strftime('%Y-%m-%d').unique())
    ax_header.text(0, 0.5, header_text, ha='left', va='center', fontsize=18)
    ax_header.axis('off')

    # Top-right count summary table
    ax3 = fig.add_subplot(gs[0, 4])
    ax3.set_facecolor('white')
    plot_count_summary_table(ax3, df_transformed)

    # Top-center: Top 5 events
    ax35 = fig.add_subplot(gs[0, 2])
    ax35.set_facecolor('white')
    plot_top_5_events(ax35, df_transformed)

    # Row 2: Game Summary in center (cols 2)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('white')
    plot_game_summary(ax4, df_transformed)

    # Rows 3 and 4: Color Bar (col 0) + Scatter Plots
    ax5 = fig.add_subplot(gs[2:4, 0])
    ax5.set_facecolor('white')
    plot_color_bar(ax5, plt.cm.RdYlGn_r)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor('white')
    ax6.text(0.5, 0.5, "vs RHH", ha='center', va='center', fontsize=14)
    ax6.axis('off')

    ax10 = fig.add_subplot(gs[3, 1])
    ax10.set_facecolor('white')
    ax10.text(0.5, 0.5, "vs LHH", ha='center', va='center', fontsize=14)
    ax10.axis('off')

    pitch_groups = {"Fast": "Fast", "Break": "Break", "Slow": "Slow"}
    custom_cmap = plt.cm.RdYlGn_r
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.05, vmax=1.5)

    # Rows 3 & 4: Scatter plots for each pitch group vs RHH and LHH
    for i, (pitch_label, pitch_group) in enumerate(pitch_groups.items()):
        ax_pitch_rhh = fig.add_subplot(gs[2, i+2])
        ax_pitch_rhh.set_facecolor('white')
        data_rhh = df_transformed[(df_transformed['Pitchgroup'] == pitch_group) & (df_transformed['Batterside'] == 'R')]
        plot_pitch_scatter(ax_pitch_rhh, data_rhh, custom_cmap, norm, title=pitch_label, overall_top_5=overall_top_5)

        ax_pitch_lhh = fig.add_subplot(gs[3, i+2])
        ax_pitch_lhh.set_facecolor('white')
        data_lhh = df_transformed[(df_transformed['Pitchgroup'] == pitch_group) & (df_transformed['Batterside'] == 'L')]
        plot_pitch_scatter(ax_pitch_lhh, data_lhh, custom_cmap, norm, overall_top_5=overall_top_5)

    # Row 5: Blank area for table background
    ax14 = fig.add_subplot(gs[4, :])
    ax14.set_facecolor('white')
    plot_blank(ax14)
    # Bottom row table of aggregated pitch metrics
    plot_bottom_row_table(ax14, df_transformed)

    for ax in fig.get_axes():
        ax.set_facecolor('white')
    plt.subplots_adjust(hspace=0.05)

    batting_team = df_transformed['Battingteam'].iloc[0] if 'Battingteam' in df_transformed.columns else "UnknownTeam"
    min_date = pd.to_datetime(df_transformed['Date']).min().strftime('%Y-%m-%d')
    file_name = f"{pitcher}_{batting_team}_{min_date}.pdf"
    save_path = os.path.join(r"C:\Users\TrevorWhite\OneDrive - Good360\Documents\bsb\scorecards", file_name)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(file_name)
    plt.close(fig)
