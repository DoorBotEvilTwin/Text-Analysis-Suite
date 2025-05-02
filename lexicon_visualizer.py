import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import math
import os
import sys
import base64
from io import BytesIO
import textwrap
import logging
import time # For timestamp in HTML

# --- Dependency Check ---
try:
    from pandas.plotting import parallel_coordinates
except ImportError:
    print("ERROR: Missing dependency for parallel_coordinates. Ensure pandas is installed correctly.")
    sys.exit(1)
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: 'scikit-learn' library not found. Install using: py -m pip install scikit-learn")
    print("         Some plot normalization/imputation might be affected.")
    MinMaxScaler = None
    SimpleImputer = None
    SKLEARN_AVAILABLE = False
try:
    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: 'scipy' library not found. Install using: py -m pip install scipy")
    print("         Clustered Heatmap will not be generated.")
    SCIPY_AVAILABLE = False


# --- Configuration ---
CSV_FILENAME = "_LEXICON.csv"
HTML_FILENAME = "_LEXICON.html"
LOG_FILENAME = "_LEXICON_errors.log"
OUTPUT_PLOT_DIR = "plots"
RADAR_N_GROUPS = 4

# --- Basic Logging Setup ---
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'), # Append mode 'a'
        logging.StreamHandler()
    ]
)
logging.info(f"\n{'='*10} Starting Visualization Run {'='*10}")


# --- Descriptions (Copied/Adapted from README) ---
METRIC_DESCRIPTIONS = {
    'Filename': "The name of the input text file.",
    'Total Words': "Total number of tokens (words) identified in the file using NLTK's tokenizer.",
    'Total Characters': "Total number of characters (including spaces and punctuation) in the raw file content.",
    'Unique Words': "Number of distinct word types (case-insensitive) found in the file.",
    'Average Sentence Length': "The average number of words per sentence (Total Words / Number of Sentences). Provides a basic measure of syntactic complexity.",
    'Flesch Reading Ease': "A standard score indicating text difficulty (0-100 scale). Higher scores indicate easier readability, typically based on average sentence length and syllables per word.",
    'TTR': "The simplest diversity measure: `Unique Words / Total Words`. Highly sensitive to text length. Values closer to 1 indicate higher diversity within that specific text length.",
    'RTTR': "An attempt to correct TTR for length: `Unique Words / sqrt(Total Words)`. Less sensitive to length than TTR, but less robust than VOCD/MTLD.",
    "Herdan's C (LogTTR)": "Another length normalization attempt: `log(Unique Words) / log(Total Words)`. Assumes a specific logarithmic relationship. Higher values suggest greater diversity relative to length under this model.",
    f'MATTR (W=100)': "Calculates TTR over sliding windows (default 100 words) and averages the results. Measures *local* lexical diversity and is less sensitive to overall text length than TTR.",
    f'MSTTR (W=100)': "Calculates TTR over sequential, *non-overlapping* segments (default 100 words) and averages them. Provides a different view of segmental diversity compared to MATTR.",
    'VOCD': "A robust measure designed to be independent of text length. Models the relationship between TTR and text length using hypergeometric distribution probabilities (specifically, the HD-D variant). Higher scores indicate greater underlying vocabulary richness.",
    'MTLD': "Another robust, length-independent measure. Calculates the average number of sequential words needed for the TTR to fall below a threshold (0.72). Higher scores indicate greater diversity (more words needed before repetition dominates).",
    "Yule's K": "Measures vocabulary richness based on word repetition patterns (frequency distribution). Lower values indicate higher repetition of frequent words (lower diversity); higher values indicate more even word usage (higher diversity). Calculated manually.",
    "Simpson's D": "Measures the probability that two randomly selected words from the text will be the same type. It reflects vocabulary *concentration* or the dominance of frequent words. Higher values (closer to 1) indicate *higher* concentration / *lower* lexical diversity. Calculated manually.",
    'Lexical Density': "The proportion of content words (nouns, verbs, adjectives, adverbs) to the total number of words. Higher density often suggests more informational or descriptive text.",
    'Distinct-2': "The proportion of unique bigrams (2-word sequences) relative to the total number of bigrams. Higher values indicate greater variety in word pairs.",
    'Repetition-2': "The proportion of bigram *tokens* that are repetitions of bigram *types* already seen earlier in the text. Higher values indicate more immediate phrase repetition.",
    'Distinct-3': "The proportion of unique trigrams (3-word sequences) relative to the total number of trigrams. Higher values indicate greater variety in short phrases.",
    'Repetition-3': "The proportion of trigram *tokens* that are repetitions of trigram *types* already seen earlier. Higher values indicate more immediate phrase repetition.",
    'Self-BLEU': "The average pairwise BLEU score calculated between all pairs of documents in the set. Measures surface-level (N-gram) similarity *across* documents. Higher scores indicate the documents are textually very similar to each other (low diversity in the set).",
    'Avg Pairwise Cosine Similarity': "Calculates a vector embedding for each document and finds the average cosine similarity between all pairs of embeddings. Measures *semantic* similarity across documents. Higher scores (closer to 1) indicate the documents discuss very similar topics or convey similar meanings (low semantic diversity).",
    'Avg Distance to Centroid': "Calculates the average distance (using cosine distance) of each document's embedding from the mean embedding (centroid) of the entire set. Measures the semantic *spread* or dispersion of the documents. Higher values indicate the documents are more spread out semantically (high semantic diversity)."
}

PLOT_INFO = {
    'normalized_diversity_bar': {
        'title': 'Normalized Diversity Metrics Comparison',
        'desc': "Compares key diversity metrics (TTR, MTLD, VOCD, Distinct-2) across all files.",
        'read': "Values are normalized (0-1 scale) to allow visual comparison despite different original scales. Higher bars generally indicate higher diversity for that specific metric relative to the other files in the set. Helps quickly identify files that are consistently high or low on these common measures."
    },
    'mtld_distribution': {
        'title': 'MTLD Score Distribution',
        'desc': "Shows the distribution (histogram and density curve) of MTLD scores across all analyzed files.",
        'read': "The histogram bars show how many files fall into specific MTLD score ranges. The curve estimates the underlying probability distribution. This helps understand the overall range, central tendency (peak), and spread of MTLD scores in your dataset. Is the diversity generally high or low? Is it consistent or highly variable?"
    },
    'mtld_vs_sentlen_scatter': {
        'title': 'MTLD vs. Average Sentence Length',
        'desc': "Plots each file as a point based on its MTLD score (Y-axis) and its Average Sentence Length (X-axis).",
        'read': "Look for patterns or trends. Is there a positive correlation (longer sentences associated with higher MTLD), negative correlation, or no clear relationship? Clusters of points might indicate groups of files with similar lexical diversity *and* sentence complexity characteristics."
    },
    'vocd_vs_unique_scatter': {
        'title': 'VOCD vs. Unique Words',
        'desc': "Plots each file based on its VOCD score (Y-axis) and its total number of Unique Words (X-axis).",
        'read': "Since VOCD is designed to be length-independent, ideally there shouldn't be a strong correlation with raw unique word count (which *is* length-dependent). This plot helps visually verify that relationship and see if files with similar unique word counts still exhibit a range of VOCD scores."
    },
    'grouped_profile_radar': {
        'title': 'Grouped Profile Comparison (Normalized)',
        'desc': "Creates average 'fingerprints' for groups of files. Files are grouped into quartiles (Low, Mid-Low, Mid-High, High) based on their MTLD scores. Each axis represents a different key metric (normalized and sometimes inverted so outward means 'more diverse/complex/readable'). Shows the *average* profile for files within each MTLD diversity tier.",
        'read': "Compare the shapes of the polygons for the different MTLD groups. Does the 'High MTLD' group consistently score higher on other diversity/complexity metrics? Are there trade-offs (e.g., does higher diversity correlate with lower readability in this dataset)? This shows average tendencies for different diversity levels."
    },
    'metrics_heatmap': {
        'title': 'Metrics Heatmap (Normalized)',
        'desc': "Provides a grid view where rows are files and columns are metrics. The color intensity of each cell represents the normalized value (0-1) of that metric for that file (typically, brighter colors like yellow mean higher normalized values, darker colors like purple mean lower).",
        'read': "Look for rows (files) or columns (metrics) with consistently bright or dark colors. Identify blocks of similar colors, which might indicate groups of files with similar metric profiles. Useful for spotting overall patterns and relationships visually at a glance."
    },
    'correlation_matrix': {
        'title': 'Metrics Correlation Matrix',
        'desc': "Shows the pairwise Pearson correlation coefficient (-1 to +1) between all numeric metrics calculated across all files.",
        'read': "Colors indicate the strength and direction of the correlation (e.g., warm colors like red for strong positive correlation, cool colors like blue for strong negative). The number in each cell is the correlation coefficient. Look for strongly correlated metrics (measuring similar underlying properties) or strongly anti-correlated metrics (measuring opposite properties). Helps understand redundancy and relationships between measures in *your specific dataset*."
    },
    'parallel_coordinates': {
        'title': 'Parallel Coordinates Profile (Normalized & Grouped)',
        'desc': "A 'super-chart' visualizing multiple key metrics simultaneously. Each file is represented by a line that connects points on parallel vertical axes. Each axis represents a different metric, normalized to a 0-1 scale (with Repetition/Simpson's D inverted for consistent interpretation where higher=better/more diverse). Lines are color-coded based on the file's MTLD quartile group.",
        'read': "Files with similar overall profiles across the selected metrics will have lines that follow similar paths and cluster together visually. The color-coding helps see if files within the same MTLD group exhibit similar patterns across other metrics (e.g., do 'High MTLD' lines generally stay high on other diversity axes?). Outliers will have lines that deviate significantly. Observe group trends rather than trying to trace every individual line."
    },
    # Removed 'collective_metrics_bar' entry
    'clustered_heatmap': {
        'title': 'Clustered Heatmap of All Metrics (Normalized)',
        'desc': "An enhanced heatmap where both the rows (files) and columns (metrics) have been reordered using hierarchical clustering based on their similarity. Similar files are placed near each other, and metrics that behave similarly across the files are placed near each other. Dendrograms (tree diagrams) alongside show the clustering hierarchy.",
        'read': "This chart is excellent for identifying distinct groups (clusters) of files that share similar linguistic profiles across *all* measured dimensions. Look for blocks of color indicating these groups. Also, observe which metrics cluster together – this reinforces findings from the correlation matrix about related measures. **Interpret the characteristics of a file cluster by examining the typical color patterns (high/low normalized values) across the metric columns for that block.** The dendrograms show how clusters are related but don't contain descriptive labels themselves."
    }
}


# --- Helper to find data end ---
def find_separator_index(filename):
    """Finds the 0-based index of the separator line itself."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if "--- Collective Metrics (E) ---" in line:
                    return i
    except FileNotFoundError: logging.error(f"CSV file not found at {filename} for separator check."); return None
    except Exception as e: logging.error(f"Error reading CSV to find separator: {e}"); return None
    logging.warning(f"Collective metrics separator not found in {filename}.")
    return None

# --- Helper to parse collective metrics ---
def parse_collective_metrics(filename, separator_index):
    """Parses collective metrics from the end of the CSV."""
    metrics = {}
    if separator_index is None:
        logging.warning("Separator index not found, cannot parse collective metrics.")
        return metrics
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Start reading from 2 lines after separator (skip blank line and header)
            start_line = separator_index + 2
            logging.info(f"Parsing collective metrics starting from line {start_line+1} (0-based index {start_line})")
            if start_line < len(lines):
                 for line_num, line in enumerate(lines[start_line:], start=start_line):
                      line = line.strip()
                      if not line: continue # Skip empty lines
                      parts = line.split(',')
                      if len(parts) == 2:
                           key = parts[0].strip()
                           value_str = parts[1].strip()
                           logging.debug(f"Parsing collective metric: Key='{key}', Value='{value_str}'")
                           try:
                               if value_str.lower() == 'n/a (<=1 file)' or value_str == '':
                                    metrics[key] = None
                                    logging.debug(f"  -> Parsed as None")
                               else:
                                    metrics[key] = float(value_str)
                                    logging.debug(f"  -> Parsed as float: {metrics[key]}")
                           except ValueError:
                                metrics[key] = value_str # Keep as string if not float
                                logging.debug(f"  -> Parsed as string: {metrics[key]}")
                      else:
                           logging.warning(f"Unexpected format in collective metrics section (line {line_num+1}): {line}")
            else:
                 logging.warning("No lines found after collective metrics separator.")
    except FileNotFoundError:
        logging.error(f"CSV file not found at {filename} for parsing collective metrics.")
    except Exception as e:
        logging.error(f"Error parsing collective metrics from CSV: {e}", exc_info=True)
    logging.info(f"Parsed collective metrics: {metrics}")
    return metrics


# --- Get Folder Path from User ---
target_folder = input(f"Enter the path to the folder containing '{CSV_FILENAME}' and '{HTML_FILENAME}': ")

if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
     logging.error(f"Invalid folder path '{target_folder}'"); sys.exit(1)
csv_path = os.path.join(target_folder, CSV_FILENAME)
html_path = os.path.join(target_folder, HTML_FILENAME)
if not os.path.isfile(csv_path): logging.error(f"CSV file '{CSV_FILENAME}' not found in '{target_folder}'"); sys.exit(1)
if not os.path.isfile(html_path): logging.error(f"HTML file '{HTML_FILENAME}' not found in '{target_folder}'. Run lexicon_analyzer.py first."); sys.exit(1)

# --- Create output directory ---
output_dir_path = os.path.join(target_folder, OUTPUT_PLOT_DIR)
os.makedirs(output_dir_path, exist_ok=True)
logging.info(f"Plots will be saved in: {output_dir_path}")
logging.info(f"Plots and descriptions will also be appended to: {html_path}")

# --- Load Data ---
separator_idx = find_separator_index(csv_path)
collective_metrics_data = parse_collective_metrics(csv_path, separator_idx)
try:
    if separator_idx is None:
        logging.warning(f"Could not find collective metrics separator in {csv_path}. Reading entire file for main data.")
        df = pd.read_csv(csv_path)
    else:
        rows_to_read = separator_idx
        logging.info(f"Separator line found at index {separator_idx}. Reading {rows_to_read} data rows.")
        df = pd.read_csv(csv_path, nrows=rows_to_read)
except Exception as e: logging.error(f"Could not read CSV file: {e}"); sys.exit(1)

# --- Data Cleaning ---
logging.info(f"Loaded {df.shape[0]} rows initially from CSV (excluding header).")
df.dropna(how='all', inplace=True) # Drop rows where ALL values are NaN
logging.info(f"DataFrame shape after dropping empty rows: {df.shape}")

numeric_cols = [col for col in df.columns if col != 'Filename']
df_numeric = df.copy()
for col in numeric_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

filename_col_present = 'Filename' in df_numeric.columns
if filename_col_present:
    if df_numeric.index.name != 'Filename':
        try: df_numeric.set_index('Filename', inplace=True)
        except KeyError: logging.warning("Could not set 'Filename' as index."); filename_col_present = False
else:
    logging.warning("'Filename' column not found after loading/cleaning. Plots will use default index.")
    df_numeric.index.name = 'File Index'


# --- Function to generate plot, save, and encode ---
def generate_encode_plot(plot_func, filename_base, *args, **kwargs):
    """Generates plot using plot_func, saves it, encodes to base64."""
    plot_filepath = os.path.join(output_dir_path, f"{filename_base}.png")
    base64_encoded = None
    logging.info(f"Generating {filename_base}.png...")
    fig = None
    try:
        plot_func(*args, **kwargs)
        fig = plt.gcf()
        if fig.get_axes():
            fig.savefig(plot_filepath, bbox_inches='tight', dpi=150)
            with open(plot_filepath, "rb") as image_file:
                base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            logging.info(f"-> Saved and encoded {filename_base}.png.")
        else:
            logging.warning(f"-> Skipping save/encode for {filename_base}: No content plotted.")
    except Exception as e:
        logging.error(f"-> ERROR generating/encoding plot {filename_base}: {e}", exc_info=True)
    finally:
        if fig: plt.close(fig)
        else: plt.close()
    return base64_encoded

# --- Plotting Functions ---
def plot_normalized_bar(df_plot, metrics_present):
    fig, ax = plt.subplots(figsize=(12, 7))
    df_plot.plot(kind='bar', ax=ax)
    ax.set_title('Comparison of Key Diversity Metrics (Normalized 0-1)')
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('File' if filename_col_present else 'Index')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(title='Metrics')
    plt.tight_layout()

def plot_distribution(series, metric_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=min(15, max(5, len(series)//2)), ax=ax)
    ax.set_title(f'Distribution of {metric_name}')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Frequency')
    plt.tight_layout()

def plot_scatter(df_plot, x_metric, y_metric):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x=x_metric, y=y_metric, ax=ax, alpha=0.7)
    ax.set_title(f'{y_metric} vs. {x_metric}')
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_grouped_radar(df_grouped_normalized, labels):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
    colors = plt.cm.get_cmap('Dark2', len(df_grouped_normalized))
    for i, (index, row) in enumerate(df_grouped_normalized.iterrows()):
        values = row.tolist(); values += values[:1]
        ax.plot(angles, values, color=colors(i), linewidth=2, linestyle='solid', label=str(index))
        ax.fill(angles, values, color=colors(i), alpha=0.25)
    ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_ylim(0, 1)
    ax.set_title('Average Group Profiles (Normalized Metrics)', size=16, y=1.1)
    ax.legend(title="MTLD Quartile Groups", loc='upper right', bbox_to_anchor=(1.4, 1.1));
    plt.tight_layout()

def plot_heatmap_simple(df_norm_heat):
    fig, ax = plt.subplots(figsize=(15, max(10, len(df_norm_heat.columns) * 0.5)))
    sns.heatmap(df_norm_heat.transpose(), annot=False, cmap="viridis", linewidths=.5, cbar=True, ax=ax)
    ax.set_title('Heatmap of Normalized Metrics Across Files')
    ax.set_xlabel('File' if filename_col_present else 'Index')
    ax.set_ylabel('Metric')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()

def plot_correlation_matrix(df_corr):
    # Reverted to creating figure inside
    plt.figure(figsize=(18, 14))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar=True, annot_kws={"size": 8})
    plt.title('Correlation Matrix of Numeric Metrics')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    # No tight_layout here, handled by wrapper

def plot_parallel_coordinates(df_plot, class_column='MTLD_Group', metrics=None, title='Parallel Coordinates Plot'):
    fig, ax = plt.subplots(figsize=(18, 8))
    if metrics:
        cols_to_select = metrics + ([class_column] if class_column in df_plot.columns else [])
        df_plot_selected = df_plot[cols_to_select]
    else:
        df_plot_selected = df_plot
    if class_column not in df_plot_selected.columns:
         logging.warning(f"Class column '{class_column}' not found for parallel coordinates. Using default color.")
         df_plot_selected['DummyClass'] = 'All Files'
         class_column = 'DummyClass'

    colormap = plt.cm.get_cmap('viridis', max(1, len(df_plot_selected[class_column].unique())))
    parallel_coordinates(df_plot_selected, class_column=class_column, colormap=colormap, alpha=0.7, ax=ax, axvlines_kwds={"color": "grey", "linestyle": ":", "alpha": 0.5})
    ax.set_title(title)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Normalized Value (Inverted for Repetition/Simpson D)')
    ax.tick_params(axis='x', rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    filtered_labels_handles = [(h, l) for h, l in zip(handles, labels) if l != 'DummyClass']
    if filtered_labels_handles:
        handles, labels = zip(*filtered_labels_handles)
        ax.legend(handles, labels, title=class_column.replace('_', ' '), loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        ax.legend().remove()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

# Removed plot_collective_bars_normalized function definition

# --- UPDATED Clustered Heatmap Plot ---
def plot_clustered_heatmap(df_norm_cluster):
    """Plots a clustered heatmap using seaborn with file labels."""
    if not SCIPY_AVAILABLE:
         logging.error("Cannot generate clustered heatmap: scipy library not found.")
         fig, ax = plt.subplots(figsize=(10, 8))
         ax.text(0.5, 0.5, 'Clustered Heatmap requires SciPy\nInstall: pip install scipy', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=12)
         ax.set_xticks([]); ax.set_yticks([])
         return

    try:
        height = max(10, len(df_norm_cluster) * 0.4)
        width = max(12, len(df_norm_cluster.columns) * 0.6)

        # Use seaborn's clustermap function
        # Standardize columns (z-score) before clustering for better distance calculation? Optional.
        # scaler = StandardScaler()
        # df_scaled_cluster = pd.DataFrame(scaler.fit_transform(df_norm_cluster), index=df_norm_cluster.index, columns=df_norm_cluster.columns)

        g = sns.clustermap(df_norm_cluster, # Use normalized (0-1) data for color mapping
                           cmap="viridis",
                           figsize=(width, height),
                           linewidths=.5,
                           annot=False,
                           # Ensure row labels (filenames/indices) are shown
                           yticklabels=True,
                           xticklabels=True,
                           cbar_pos=(0.02, 0.8, 0.03, 0.15), # Position color bar on left
                           cbar_kws={'label': 'Normalized Value (0-1)'}, # Add label to color bar
                           dendrogram_ratio=(.15, .2))

        g.ax_heatmap.set_title('Clustered Heatmap of Normalized Metrics', y=1.05)
        g.ax_heatmap.set_xlabel('Metrics')
        g.ax_heatmap.set_ylabel('Files' if filename_col_present else 'Index')
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8) # Ensure y-labels are readable

    except Exception as e:
         logging.error(f"Error during clustermap generation: {e}", exc_info=True)
         fig, ax = plt.subplots(figsize=(10, 8))
         ax.text(0.5, 0.5, f'Error generating Clustered Heatmap:\n{e}', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
         ax.set_xticks([]); ax.set_yticks([])


# --- Data Preparation for Grouped Plots ---
mtld_group_col = 'MTLD_Group'
grouping_possible = False
if 'MTLD' in df_numeric.columns and df_numeric['MTLD'].notna().sum() >= RADAR_N_GROUPS:
    try:
        df_numeric[mtld_group_col] = pd.qcut(df_numeric['MTLD'], q=RADAR_N_GROUPS,
                                             labels=[f"Q{i+1} (Low)" if i==0 else f"Q{i+1}" if i < RADAR_N_GROUPS-1 else f"Q{i+1} (High)" for i in range(RADAR_N_GROUPS)],
                                             duplicates='drop')
        if df_numeric[mtld_group_col].nunique() >= 2:
            logging.info(f"Created {df_numeric[mtld_group_col].nunique()} groups based on MTLD quantiles.")
            grouping_possible = True
        else:
             logging.warning(f"Could only create {df_numeric[mtld_group_col].nunique()} distinct MTLD groups due to duplicate values. Skipping grouped plots.")
    except ValueError as e:
        logging.warning(f"Could not create {RADAR_N_GROUPS} distinct MTLD groups ({e}). Grouped plots might be affected.")
else:
    logging.warning("Skipping grouping: MTLD column missing, has too few values, or RADAR_N_GROUPS is too high.")


# --- Generate, Save, and Encode Plots ---
plot_data_base64 = {}

# Plot 1: Normalized Bar Chart
metrics_to_compare = ['TTR', 'MTLD', 'VOCD', 'Distinct-2']
metrics_present = [m for m in metrics_to_compare if m in df_numeric.columns and df_numeric[m].notna().any()]
if metrics_present and not df_numeric.empty:
    df_compare_norm = df_numeric[metrics_present].copy()
    for col in metrics_present:
        min_val = df_compare_norm[col].min(); max_val = df_compare_norm[col].max()
        if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val == 0): df_compare_norm[col] = 0.5
        else: df_compare_norm[col] = (df_compare_norm[col] - min_val) / (max_val - min_val)
    plot_data_base64['normalized_diversity_bar'] = generate_encode_plot(
        plot_normalized_bar, "normalized_diversity_metrics_bar_chart", df_compare_norm, metrics_present
    )
else: logging.warning("Skipping Normalized Bar Chart: Not enough valid data or required columns missing.")

# Plot 2: Distribution Plot
metric_for_dist = 'MTLD'
if metric_for_dist in df_numeric.columns and df_numeric[metric_for_dist].notna().any():
    plot_data_base64['mtld_distribution'] = generate_encode_plot(
        plot_distribution, "mtld_distribution", df_numeric[metric_for_dist].dropna(), metric_for_dist
    )
else: logging.warning(f"Skipping Distribution Plot: Metric '{metric_for_dist}' not found or has no valid data.")

# Plot 3: Scatter Plot (MTLD vs Sent Len)
x_metric_scatter = 'Average Sentence Length'
y_metric_scatter = 'MTLD'
if x_metric_scatter in df_numeric.columns and y_metric_scatter in df_numeric.columns and df_numeric[x_metric_scatter].notna().any() and df_numeric[y_metric_scatter].notna().any():
     df_scatter = df_numeric[[x_metric_scatter, y_metric_scatter]].dropna()
     if not df_scatter.empty:
        plot_data_base64['mtld_vs_sentlen_scatter'] = generate_encode_plot(
            plot_scatter, "mtld_vs_avg_sent_len_scatter", df_scatter, x_metric_scatter, y_metric_scatter
        )
     else: logging.warning(f"Skipping Scatter Plot: No overlapping valid data for '{x_metric_scatter}' and '{y_metric_scatter}'.")
else: logging.warning(f"Skipping Scatter Plot: One or both metrics ('{x_metric_scatter}', '{y_metric_scatter}') not found or have no valid data.")

# Plot 4: Scatter Plot (VOCD vs Unique Words)
x_metric_vocd = 'Unique Words'
y_metric_vocd = 'VOCD'
if x_metric_vocd in df_numeric.columns and y_metric_vocd in df_numeric.columns and df_numeric[x_metric_vocd].notna().any() and df_numeric[y_metric_vocd].notna().any():
     df_vocd_scatter = df_numeric[[x_metric_vocd, y_metric_vocd]].dropna()
     if not df_vocd_scatter.empty:
        plot_data_base64['vocd_vs_unique_scatter'] = generate_encode_plot(
            plot_scatter, "vocd_vs_unique_words_scatter", df_vocd_scatter, x_metric_vocd, y_metric_vocd
        )
     else: logging.warning(f"Skipping Scatter Plot: No overlapping valid data for '{x_metric_vocd}' and '{y_metric_vocd}'.")
else: logging.warning(f"Skipping Scatter Plot: One or both metrics ('{x_metric_vocd}', '{y_metric_vocd}') not found or have no valid data.")

# Plot 5: Grouped Radar Chart
metrics_for_radar = [
    'MTLD', 'VOCD', 'Yule\'s K', 'Simpson\'s D',
    'Distinct-2', 'Repetition-2',
    'Average Sentence Length', 'Flesch Reading Ease'
]
metrics_present_radar = [m for m in metrics_for_radar if m in df_numeric.columns and df_numeric[m].notna().any()]
if grouping_possible and len(metrics_present_radar) >= 3:
    logging.info(f"Generating Grouped Radar Chart using metrics: {', '.join(metrics_present_radar)}")
    grouped_data = df_numeric.groupby(mtld_group_col, observed=False)[metrics_present_radar].mean()
    df_grouped_normalized = grouped_data.copy()
    if MinMaxScaler:
        scaler = MinMaxScaler(); scaled_vals = scaler.fit_transform(df_grouped_normalized)
        df_grouped_normalized = pd.DataFrame(scaled_vals, index=df_grouped_normalized.index, columns=df_grouped_normalized.columns)
        for col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']:
            if col in df_grouped_normalized.columns: df_grouped_normalized[col] = 1.0 - df_grouped_normalized[col]
    else: # Manual fallback
         for col in metrics_present_radar:
            min_val = df_grouped_normalized[col].min(); max_val = df_grouped_normalized[col].max()
            if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val == 0): df_grouped_normalized[col] = 0.5
            else:
                 norm_val = (df_grouped_normalized[col] - min_val) / (max_val - min_val)
                 if col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']: df_grouped_normalized[col] = 1.0 - norm_val
                 else: df_grouped_normalized[col] = norm_val

    if not df_grouped_normalized.empty:
        radar_plot_filepath = os.path.join(output_dir_path, "grouped_profile_radar_chart.png")
        try:
            plot_grouped_radar(df_grouped_normalized, df_grouped_normalized.columns.tolist())
            plt.savefig(radar_plot_filepath, bbox_inches='tight')
            with open(radar_plot_filepath, "rb") as image_file:
                plot_data_base64['grouped_profile_radar'] = base64.b64encode(image_file.read()).decode('utf-8')
            logging.info("-> grouped_profile_radar_chart.png saved and encoded.")
        except Exception as e: logging.error(f"-> ERROR generating/encoding plot grouped_profile_radar_chart: {e}", exc_info=True)
        finally: plt.close()
    else: logging.warning("Skipping Grouped Radar Chart: No valid data after grouping/normalization.")
else: logging.warning("Skipping Grouped Radar Chart: Grouping not possible or < 3 metrics.")


# Plot 6: Heatmap (Simple Version)
if not df_numeric.empty and len(df_numeric.columns) > 1:
    logging.info("Generating Simple Heatmap of Metrics...")
    df_norm_heat = df_numeric.select_dtypes(include=np.number).copy()
    if MinMaxScaler:
        scaler = MinMaxScaler()
        if df_norm_heat.isnull().any().any():
            if SimpleImputer: imputer = SimpleImputer(strategy='mean'); df_imputed = imputer.fit_transform(df_norm_heat); df_norm_heat = pd.DataFrame(df_imputed, index=df_norm_heat.index, columns=df_norm_heat.columns)
            else: logging.warning("  Heatmap: NaNs found but scikit-learn not fully available for imputation. Filling with 0.5."); df_norm_heat.fillna(0.5, inplace=True)
        scaled_vals = scaler.fit_transform(df_norm_heat)
        df_norm_heat = pd.DataFrame(scaled_vals, index=df_norm_heat.index, columns=df_norm_heat.columns)
    else: # Manual fallback
        for col in df_norm_heat.columns:
             min_val = df_norm_heat[col].min(); max_val = df_norm_heat[col].max()
             if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val != 0): df_norm_heat[col] = (df_norm_heat[col] - min_val) / (max_val - min_val)
             else: df_norm_heat[col] = 0.5
    df_norm_heat.dropna(axis=1, how='all', inplace=True)
    if not df_norm_heat.empty:
        plot_data_base64['metrics_heatmap'] = generate_encode_plot(
            plot_heatmap_simple, "metrics_heatmap", df_norm_heat
        )
    else: logging.warning("Skipping Heatmap: No valid numeric data after normalization.")
else: logging.warning("Skipping Heatmap: No valid data.")


# Plot 7: Correlation Matrix
logging.info("Generating Correlation Matrix...")
df_numeric_only = df_numeric.select_dtypes(include=np.number).dropna(axis=1, how='all')
if not df_numeric_only.empty and len(df_numeric_only.columns) > 1:
    df_numeric_only = df_numeric_only.loc[:, df_numeric_only.apply(pd.Series.nunique) > 1]
    if len(df_numeric_only.columns) > 1:
        df_corr = df_numeric_only.corr()
        plot_data_base64['correlation_matrix'] = generate_encode_plot(
            plot_correlation_matrix, "metrics_correlation_heatmap", df_corr
        )
    else: logging.warning("Skipping Correlation Matrix: Not enough columns with variance.")
else: logging.warning("Skipping Correlation Matrix: Not enough numeric data.")


# Plot 8: Parallel Coordinates Plot
logging.info("Generating Parallel Coordinates Plot...")
metrics_for_parallel = [
    'MTLD', 'VOCD', 'Yule\'s K', 'Simpson\'s D', 'Distinct-2', 'Repetition-2',
    'Average Sentence Length', 'Flesch Reading Ease', 'Lexical Density'
]
metrics_present_parallel = [m for m in metrics_for_parallel if m in df_numeric.columns and df_numeric[m].notna().any()]
class_col_parallel = mtld_group_col if grouping_possible else ('Filename' if filename_col_present else 'File Index')

if len(metrics_present_parallel) > 1 and not df_numeric.empty:
    cols_for_parallel = metrics_present_parallel + ([class_col_parallel] if class_col_parallel in df_numeric.columns else [])
    df_parallel_input = df_numeric[cols_for_parallel].copy()
    df_parallel_norm = df_parallel_input.copy()
    if MinMaxScaler:
        scaler = MinMaxScaler()
        if df_parallel_norm[metrics_present_parallel].isnull().any().any():
             if SimpleImputer: imputer = SimpleImputer(strategy='mean'); imputed_vals = imputer.fit_transform(df_parallel_norm[metrics_present_parallel]); df_parallel_norm[metrics_present_parallel] = imputed_vals
             else: logging.warning("  Parallel Coords: NaNs found but scikit-learn not fully available for imputation. Filling with 0.5."); df_parallel_norm[metrics_present_parallel] = df_parallel_norm[metrics_present_parallel].fillna(0.5)
        scaled_vals = scaler.fit_transform(df_parallel_norm[metrics_present_parallel])
        df_parallel_norm[metrics_present_parallel] = scaled_vals
        for col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']:
            if col in df_parallel_norm.columns: df_parallel_norm[col] = 1.0 - df_parallel_norm[col]
    else: # Manual fallback
        for col in metrics_present_parallel:
            min_val = df_parallel_norm[col].min(); max_val = df_parallel_norm[col].max()
            if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val == 0): df_parallel_norm[col] = 0.5
            else:
                norm_val = (df_parallel_norm[col] - min_val) / (max_val - min_val)
                if col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']: df_parallel_norm[col] = 1.0 - norm_val
                else: df_parallel_norm[col] = norm_val

    # Add back the original index/grouping column if needed
    if class_col_parallel == 'Filename' and filename_col_present: df_parallel_norm['Filename'] = df_numeric.index
    elif class_col_parallel == mtld_group_col and grouping_possible: df_parallel_norm[mtld_group_col] = df_numeric[mtld_group_col]
    elif class_col_parallel == 'File Index': df_parallel_norm['File Index'] = df_numeric.index

    df_parallel_norm.dropna(subset=metrics_present_parallel, inplace=True)

    if not df_parallel_norm.empty and len(df_parallel_norm.columns) > 1:
        parallel_plot_filepath = os.path.join(output_dir_path, "parallel_coordinates_plot.png")
        try:
            plot_parallel_coordinates(df_parallel_norm, class_column=class_col_parallel, metrics=metrics_present_parallel, title='File Profiles on Normalized Metrics (Color by MTLD Group)')
            plt.savefig(parallel_plot_filepath, bbox_inches='tight')
            with open(parallel_plot_filepath, "rb") as image_file:
                plot_data_base64['parallel_coordinates'] = base64.b64encode(image_file.read()).decode('utf-8')
            logging.info(" -> parallel_coordinates_plot.png saved and encoded.")
        except Exception as e: logging.error(f"   -> ERROR generating/encoding plot parallel_coordinates_plot: {e}", exc_info=True)
        finally: plt.close()
    else: logging.warning("Skipping Parallel Coordinates Plot: No valid data after normalization/NaN drop.")
else: logging.warning("Skipping Parallel Coordinates Plot: Not enough valid metrics or data.")

# Removed the call block for Collective Metrics Bar Chart

# Plot 9: Clustered Heatmap ("Super-Chart")
if not df_numeric.empty and len(df_numeric.columns) > 1 and SCIPY_AVAILABLE:
    logging.info("Generating Clustered Heatmap of Metrics...")
    df_cluster_input = df_numeric.select_dtypes(include=np.number).copy()
    # Impute NaNs before clustering/scaling
    if df_cluster_input.isnull().any().any():
        if SimpleImputer:
            imputer = SimpleImputer(strategy='mean')
            df_imputed = imputer.fit_transform(df_cluster_input)
            df_cluster_input = pd.DataFrame(df_imputed, index=df_cluster_input.index, columns=df_cluster_input.columns)
        else:
            logging.warning("  Clustered Heatmap: NaNs found but scikit-learn not fully available for imputation. Filling with 0.5.")
            df_cluster_input.fillna(0.5, inplace=True)

    # Normalize data 0-1 for visualization consistency
    if MinMaxScaler:
        scaler = MinMaxScaler()
        scaled_vals = scaler.fit_transform(df_cluster_input)
        df_cluster_norm = pd.DataFrame(scaled_vals, index=df_cluster_input.index, columns=df_cluster_input.columns)
    else: # Manual fallback
        df_cluster_norm = df_cluster_input.copy()
        for col in df_cluster_norm.columns:
             min_val = df_cluster_norm[col].min(); max_val = df_cluster_norm[col].max()
             if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val != 0):
                  df_cluster_norm[col] = (df_cluster_norm[col] - min_val) / (max_val - min_val)
             else: df_cluster_norm[col] = 0.5

    df_cluster_norm.dropna(axis=1, how='all', inplace=True) # Drop fully NaN columns *after* imputation/scaling

    if not df_cluster_norm.empty and len(df_cluster_norm.columns) > 1:
        # Need to call clustermap differently as it returns a ClusterGrid object
        clustermap_plot_filepath = os.path.join(output_dir_path, "clustered_metrics_heatmap.png")
        try:
            # Clustermap creates its own figure
            plot_clustered_heatmap(df_cluster_norm) # Call the function which handles figure creation
            plt.savefig(clustermap_plot_filepath, bbox_inches='tight', dpi=150) # Save the current figure
            with open(clustermap_plot_filepath, "rb") as image_file:
                plot_data_base64['clustered_heatmap'] = base64.b64encode(image_file.read()).decode('utf-8')
            logging.info(" -> clustered_metrics_heatmap.png saved and encoded.")
        except Exception as e:
            logging.error(f"   -> ERROR generating/encoding plot clustered_metrics_heatmap: {e}", exc_info=True)
        finally:
            plt.close('all') # Close all figures just in case
    else: logging.warning("Skipping Clustered Heatmap: No valid numeric data after normalization.")
elif not SCIPY_AVAILABLE:
     logging.warning("Skipping Clustered Heatmap: SciPy library not available.")
else: logging.warning("Skipping Clustered Heatmap: No valid data.")


# --- Append Descriptions, Plots, and JS to HTML ---
logging.info(f"Updating {html_path} with descriptions, plots, and sorting JS...")
try:
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # --- Construct Metric Descriptions HTML ---
    metrics_html = "\n<hr>\n<h2 id='metric_descriptions'>Metric Descriptions</h2>\n<dl style='margin-left: 20px;'>\n"
    all_metric_keys = df.columns.tolist()
    if filename_col_present: all_metric_keys.insert(0, df_numeric.index.name if df_numeric.index.name else 'Filename')
    all_metric_keys.extend(collective_metrics_data.keys())
    processed_keys = set()
    for header in all_metric_keys:
        if header in processed_keys: continue
        processed_keys.add(header)
        lookup_key = header
        if header.startswith("MATTR"): lookup_key = f'MATTR (W=100)'
        if header.startswith("MSTTR"): lookup_key = f'MSTTR (W=100)'
        description = METRIC_DESCRIPTIONS.get(lookup_key, "No description available.")
        wrapped_desc = "<br>".join(textwrap.wrap(description, width=120))
        metrics_html += f"  <dt style='font-weight: bold; margin-top: 10px;'>{header}</dt>\n"
        metrics_html += f"  <dd style='margin-left: 20px; margin-bottom: 8px; font-size: 0.95em;'>{wrapped_desc}</dd>\n"
    metrics_html += "</dl>\n"

    # --- Construct Plots HTML ---
    plots_html = "\n<hr>\n<h2 id='generated_plots'>Generated Plots</h2>\n"
    plot_order = [ # Updated order
        'normalized_diversity_bar', 'mtld_distribution', 'mtld_vs_sentlen_scatter',
        'vocd_vs_unique_scatter',
        'grouped_profile_radar',
        'metrics_heatmap',
        'correlation_matrix',
        'parallel_coordinates',
        # Removed 'collective_metrics_bar'
        'clustered_heatmap'       # Now #9
    ]
    for key in plot_order:
        plot_info = PLOT_INFO.get(key)
        if plot_info:
            title = plot_info['title']; desc = plot_info['desc']; read = plot_info['read']
            plots_html += f"<div style='margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #eee;'>\n"
            plots_html += f"<h3>{title}</h3>\n<p><b>Description:</b> {desc}</p>\n<p><b>How To Read:</b> {read}</p>\n"
            if key in plot_data_base64 and plot_data_base64[key]:
                plots_html += f'<img src="data:image/png;base64,{plot_data_base64[key]}" alt="{title}" style="max-width: 90%; height: auto; border: 1px solid #ccc; margin-top: 10px; display: block; margin-left: auto; margin-right: auto; background-color: white; padding: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">\n'
            else: plots_html += f"<p style='color: red;'><i>Plot generation failed or skipped for '{title}'. Check logs/data.</i></p>\n"
            plots_html += f"</div>\n"
        else: # Fallback
             title = key.replace('_', ' ').title()
             plots_html += f"<div style='margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #eee;'>\n<h3>{title}</h3>\n<p><i>Plot info missing.</i></p>\n"
             if key in plot_data_base64 and plot_data_base64[key]: plots_html += f'<img src="data:image/png;base64,{plot_data_base64[key]}" alt="{title}" style="max-width: 90%; height: auto; border: 1px solid #ccc; margin-top: 10px; display: block; margin-left: auto; margin-right: auto; background-color: white; padding: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">\n'
             else: plots_html += f"<p style='color: red;'><i>Plot generation failed or skipped for '{title}'. Check logs/data.</i></p>\n"
             plots_html += f"</div>\n"

    # --- JavaScript for Table Sorting ---
    js_code = """
<script>
function sortTable(tableId, columnIndex, isNumeric) {
    const table = document.getElementById(tableId);
    if (!table) { console.error("Table with ID '" + tableId + "' not found."); return; }
    const tbody = table.tBodies[0];
    if (!tbody) { console.error("Table body not found."); return; }
    const rows = Array.from(tbody.rows);
    const headerCell = table.tHead.rows[0].cells[columnIndex];
    const currentSort = headerCell.getAttribute('data-sort-direction') || 'none';
    let direction = 'asc';

    if (currentSort === 'asc') { direction = 'desc'; }
    else { direction = 'asc'; }

    Array.from(table.tHead.rows[0].cells).forEach((th, index) => {
        if (index !== columnIndex) {
            th.classList.remove('sort-asc', 'sort-desc');
            th.removeAttribute('data-sort-direction');
        }
    });

    headerCell.setAttribute('data-sort-direction', direction);
    headerCell.classList.remove('sort-asc', 'sort-desc');
    headerCell.classList.add(direction === 'asc' ? 'sort-asc' : 'sort-desc');

    rows.sort((rowA, rowB) => {
        const cellA = rowA.cells[columnIndex]?.textContent.trim() ?? '';
        const cellB = rowB.cells[columnIndex]?.textContent.trim() ?? '';
        let valA = cellA;
        let valB = cellB;

        if (isNumeric) {
            valA = cellA === 'FAIL' || cellA === 'NaN' || cellA === 'INF' ? -Infinity : parseFloat(cellA);
            valB = cellB === 'FAIL' || cellB === 'NaN' || cellB === 'INF' ? -Infinity : parseFloat(cellB);
            if (isNaN(valA)) valA = -Infinity;
            if (isNaN(valB)) valB = -Infinity;
        } else {
             valA = cellA.toLowerCase();
             valB = cellB.toLowerCase();
        }

        if (valA < valB) { return direction === 'asc' ? -1 : 1; }
        if (valA > valB) { return direction === 'asc' ? 1 : -1; }
        // Secondary sort by filename if primary values are equal (assuming filename is col 0)
        const fileA = rowA.cells[0]?.textContent.trim().toLowerCase() ?? '';
        const fileB = rowB.cells[0]?.textContent.trim().toLowerCase() ?? '';
        if (fileA < fileB) { return -1; } // Keep original alpha order on tie
        if (fileA > fileB) { return 1; }
        return 0;
    });

    rows.forEach(row => tbody.appendChild(row));
}

document.addEventListener('DOMContentLoaded', () => {
    const table = document.getElementById('resultsTable');
    if (table && table.tHead && table.tHead.rows.length > 0) {
        const headers = table.tHead.rows[0].cells;
        const numericHeadersBase = ["Total Words", "Total Characters", "Unique Words", "Average Sentence Length", "Flesch Reading Ease", "TTR", "RTTR", "Herdan's C (LogTTR)", "MATTR", "MSTTR", "VOCD", "MTLD", "Yule's K", "Simpson's D", "Lexical Density", "Distinct-2", "Repetition-2", "Distinct-3", "Repetition-3"];

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const baseHeaderText = header.textContent.split(' (W=')[0].trim();
            const isNumeric = numericHeadersBase.includes(baseHeaderText);
            header.classList.add('sortable-header');
            header.addEventListener('click', ((index, numeric) => {
                return () => sortTable('resultsTable', index, numeric);
            })(i, isNumeric));
        }
        console.log("Table sorting enabled.");
    } else {
        console.error("Could not find table or table header to enable sorting.");
    }
});
</script>
"""

    # --- Modify HTML Content ---
    final_html = html_content

    # Find where to insert the appended content (before closing body tag)
    insertion_point = final_html.lower().rfind('</body>')
    if insertion_point != -1:
        # Check if sections already exist to avoid duplication
        if "<h2 id='metric_descriptions'>Metric Descriptions</h2>" not in final_html:
             final_html = final_html[:insertion_point] + metrics_html + plots_html + js_code + final_html[insertion_point:]
             logging.info("Appending descriptions, plots, and JS to HTML.")
        else:
             logging.warning("Descriptions/plots section already found in HTML. Skipping append.")
             if "<script>\nfunction sortTable(tableId" not in final_html:
                  final_html = final_html[:insertion_point] + js_code + final_html[insertion_point:]
                  logging.info("Appending sorting JS to HTML as it was missing.")
             else:
                  logging.info("Sorting JS already present in HTML.")
    else:
        final_html = final_html + metrics_html + plots_html + js_code
        logging.warning("Closing </body> tag not found in HTML. Appending all content to the end.")

    # Write the modified HTML back
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    logging.info("Successfully updated HTML file.")


except FileNotFoundError:
    logging.error(f"HTML file '{html_path}' not found. Cannot append content.")
except Exception as e:
    logging.error(f"Failed to update HTML file: {e}", exc_info=True)


logging.info("Visualization script finished.")