# lexicon_visualizer.py
# Updated to save linkage matrix and reordered indices as .npy files
# Updated to save plot metadata to _plot_metadata.json
# Updated to add TOC and REMOVE collapsible sections from HTML output
# Includes Self-BLEU workaround
# MODIFIED: TOC insertion point changed to be AFTER summary paragraphs
# MODIFIED: Explicitly move Collective Metrics section after results table
# MODIFIED: Added 5-second countdown timer for folder input
# PATCHED: Fixed imputation shape mismatch error.
# PATCHED: Added 3 new plots for Sentiment, Emotion, Readability distributions.
# PATCHED: Updated Clustered Heatmap to use correctly imputed data.
# PATCHED: Updated JavaScript to sort both HTML tables.
# PATCHED: Fixed ValueError in plot_parallel_coordinates function.
# PATCHED: Replaced Sentiment Distribution plot with Aggregated Emotion Scores plot.
# PATCHED: Replaced Dominant Emotion plot with Category Profiles by Topic plot. Reordered plots.
# PATCHED: Fixed NameError when saving .npy/.json files from clustermap.
# PATCHED: Changed scatter plots (MTLD vs VOCD, Unique Words vs Avg Sent Len).
# PATCHED: Replaced Readability box plot with grouped bar chart of normalized scores.
# PATCHED: Corrected filename used for saving clustered heatmap image.
# PATCHED: Added hue/legend to scatter plots.
# PATCHED: Added filename lists below Radar/Parallel Coords plots in HTML.
# PATCHED: Replaced Category Profiles plot with Grouped Emotion Bars plot.
# PATCHED: Reordered plots to align better with table structure.
# PATCHED: Corrected plot keys saved to metadata JSON.
# PATCHED: Added new grouped emotion bar plot.
# PATCHED: Excluded collective metrics rows from plotting dataframes.
# PATCHED: Fixed missing grouped_emotion_bars plot generation.
# PATCHED: Fixed indentation for grouped plot filename lists.
# PATCHED: Removed sidebar box for scatter plot legends.
# PATCHED: Ensure all plot keys are added to metadata JSON, even if plot fails/skips.
# PATCHED: Corrected emotion data parsing and merging logic.
# PATCHED: Adjusted clustered heatmap image styling in HTML for fixed 50% width.
# PATCHED: Inverted heatmap color schemes (viridis -> viridis_r). # <<< THIS FIX

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np # Ensure numpy is imported
import math
import os
import sys
import base64
from io import BytesIO
import textwrap
import logging
import time # For timestamp in HTML AND countdown
import re # For searching HTML
import json # For saving plot metadata
import threading # Added for timed input
from collections import defaultdict # Added for aggregating emotion scores

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
    from scipy.cluster import hierarchy # Needed for linkage calculation within clustermap
    from scipy.spatial import distance
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: 'scipy' library not found. Install using: py -m pip install scipy")
    print("         Clustered Heatmap will not be generated, and .npy/.json files cannot be saved.")
    SCIPY_AVAILABLE = False # Set to False if not available
# --- Added BS4 check here ---
try:
    from bs4 import BeautifulSoup, NavigableString # Import NavigableString
    BS4_AVAILABLE = True
except ImportError:
    print("ERROR: 'beautifulsoup4' library not found. Install using: py -m pip install beautifulsoup4")
    print("       HTML modification (including Self-BLEU workaround) will fail.")
    BeautifulSoup = None
    NavigableString = None
    BS4_AVAILABLE = False
    # Exit if BS4 is needed for the workaround but not available
    sys.exit(1)
# --- Try importing lxml for potentially better parsing ---
try:
    import lxml
    LXML_AVAILABLE = True
    # logging.debug("lxml parser is available.") # Logging might not be set up yet
except ImportError:
    lxml = None
    LXML_AVAILABLE = False
    # logging.debug("lxml parser not found, falling back to html.parser.")


# --- START: Added Helper Function for Timed Input ---
def get_input_with_timeout(prompt, timeout):
    """Gets input from the user with a timeout and countdown."""
    user_input = [None] # Use a list to allow modification from thread
    interrupted = [False] # Flag to signal if user provided input

    def _get_input_target(prompt_str, result_list, interrupt_flag):
        try:
            result_list[0] = input(prompt_str)
            interrupt_flag[0] = True # Signal that input was received
        except EOFError: # Handle case where input stream is closed unexpectedly
            pass

    thread = threading.Thread(target=_get_input_target, args=(prompt, user_input, interrupted))
    thread.daemon = True # Allow program to exit even if thread is stuck
    thread.start()

    # Countdown loop
    for i in range(timeout, 0, -1):
        if interrupted[0]: break
        print(f"\rNo path provided. Using current directory in {i}s... Press Enter or type path to override: ", end="", flush=True)
        time.sleep(1)
        if interrupted[0]: break
    print("\r" + " " * 100 + "\r", end="", flush=True) # Overwrite with spaces
    if interrupted[0] and user_input[0] is not None: return user_input[0]
    else: return None
# --- END: Added Helper Function ---


# --- Configuration ---
CSV_FILENAME = "_LEXICON.csv"
HTML_FILENAME = "_LEXICON.html"
LOG_FILENAME = "_LEXICON_errors.log"
OUTPUT_PLOT_DIR = "plots"
RADAR_N_GROUPS = 4
# --- NEW: Filenames for saving dendrogram data & plot metadata ---
LINKAGE_MATRIX_FILENAME = "_linkage_matrix.npy"
REORDERED_INDICES_FILENAME = "_reordered_indices.npy"
ORIGINAL_FILENAMES_FILENAME = "_original_filenames.json" # Also save this mapping
PLOT_METADATA_FILENAME = "_plot_metadata.json" # For transformer script

# --- Basic Logging Setup ---
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
# Remove existing handlers to prevent duplicates if script is re-run
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'), # Append mode 'a'
        logging.StreamHandler()
    ]
)
logging.info(f"\n{'='*10} Starting Visualization Run {'='*10}")


# --- Descriptions (Copied/Adapted from README - Needs Update for New Metrics) ---
METRIC_DESCRIPTIONS = {
    # Basic Info
    'Filename': "The name of the input text file.",
    'Total Words': "Total number of tokens (words) identified in the file using NLTK's tokenizer.",
    'Total Characters': "Total number of characters (including spaces and punctuation) in the raw file content.",
    'Unique Words': "Number of distinct word types (case-insensitive) found in the file.",
    'Average Sentence Length': "The average number of words per sentence (Total Words / Number of Sentences). Provides a basic measure of syntactic complexity.",
    # Readability
    'Flesch Reading Ease': "A standard score indicating text difficulty (0-100 scale). Higher scores indicate easier readability, typically based on average sentence length and syllables per word.",
    'Gunning Fog': "Readability index estimating years of formal education needed. Higher score means harder to read.",
    'SMOG Index': "Readability index estimating years of education needed, often used for health literature. Higher score means harder to read.",
    'Dale-Chall Score': "Readability score based on average sentence length and percentage of words *not* on a list of common 'easy' words. Higher score means harder to read.",
    # Lexical Diversity
    'TTR': "The simplest diversity measure: `Unique Words / Total Words`. Highly sensitive to text length. Values closer to 1 indicate higher diversity within that specific text length.",
    'RTTR': "An attempt to correct TTR for length: `Unique Words / sqrt(Total Words)`. Less sensitive to length than TTR, but less robust than VOCD/MTLD.",
    "Herdan's C (LogTTR)": "Another length normalization attempt: `log(Unique Words) / log(Total Words)`. Assumes a specific logarithmic relationship. Higher values suggest greater diversity relative to length under this model.",
    f'MATTR (W=100)': "Calculates TTR over sliding windows (default 100 words) and averages the results. Measures *local* lexical diversity and is less sensitive to overall text length than TTR.",
    f'MSTTR (W=100)': "Calculates TTR over sequential, *non-overlapping* segments (default 100 words) and averages them. Provides a different view of segmental diversity compared to MATTR.",
    'VOCD': "A sophisticated and robust measure designed to be largely independent of text length. It models the theoretical relationship between TTR and text length using hypergeometric distribution probabilities (specifically, the HD-D variant). Higher scores indicate greater underlying vocabulary richness.",
    'MTLD': "Another highly robust, length-independent measure. It calculates the average number of sequential words required for the local TTR (calculated over the growing sequence) to fall below a specific threshold (typically 0.72). A higher MTLD score indicates greater diversity, meaning longer stretches of text were needed before vocabulary repetition became significant.",
    "Yule's K": "A measure focusing on vocabulary richness via word *repetition* patterns (derived from the frequency distribution). Unlike TTR-based measures, it's less sensitive to words occurring only once. Lower values indicate higher repetition of the most frequent words (lower diversity); higher values indicate more even word usage across the vocabulary (higher diversity). Calculated manually.",
    "Simpson's D": "Originally an ecological diversity index, applied here it measures the probability that two words selected randomly from the text will be of the same *type*. It reflects vocabulary *concentration* or the dominance of frequent words. Higher values (closer to 1) indicate *higher* concentration and therefore *lower* lexical diversity. Calculated manually.",
    # Syntactic/Phrase Diversity
    'Lexical Density': "The ratio of content words (nouns, verbs, adjectives, adverbs) to the total number of words. A higher density often suggests text that is more informationally packed or descriptive, as opposed to text with a higher proportion of function words (pronouns, prepositions, conjunctions).",
    'Distinct-2': "The proportion of unique bigrams (adjacent 2-word sequences) relative to the total number of bigrams in the text. A higher value indicates greater variety in word pairings and less reliance on repeated phrases.",
    'Repetition-2': "The proportion of bigram *tokens* that are repetitions of bigram *types* already seen earlier in the text. A higher value indicates more immediate phrase repetition, potentially impacting fluency.",
    'Distinct-3': "The proportion of unique trigrams (adjacent 3-word sequences) relative to the total number of trigrams. A higher value indicates greater variety in short three-word phrases.",
    'Repetition-3': "The proportion of trigram *tokens* that are repetitions of trigram *types* already seen earlier. Higher values indicate more repetition of three-word phrases.",
    # Sentiment/Emotion
    'Sentiment (VADER Comp)': "VADER compound sentiment score (-1=most negative, +1=most positive). Good for general/social media text.",
    'Sentiment (VADER Pos)': "Proportion of text classified as positive by VADER.",
    'Sentiment (VADER Neu)': "Proportion of text classified as neutral by VADER.",
    'Sentiment (VADER Neg)': "Proportion of text classified as negative by VADER.",
    'Sentiment (TextBlob Pol)': "TextBlob polarity score (-1=negative, +1=positive).",
    'Sentiment (TextBlob Subj)': "TextBlob subjectivity score (0=objective, 1=subjective).",
    'Emotion (NRCLex Dominant)': "The single emotion category with the highest score according to the NRC Emotion Lexicon.",
    'Emotion (NRCLex Scores JSON)': "A JSON string containing the raw counts or scores for each NRC emotion category (e.g., fear, anger, joy, sadness, trust, anticipation, surprise, disgust).",
    # Topic Modeling
    'Topic (NMF ID)': "The numerical ID (0 to N-1) of the topic assigned as most likely for this document by the NMF model.",
    'Topic (NMF Prob)': "The probability or weight associated with the document's assignment to its dominant NMF topic.",
    # Keyword Extraction
    'Keywords (TF-IDF Top 5)': "The top 5 keywords identified for this document based on TF-IDF scores (term frequency inverse document frequency), separated by semicolons.",
    'Keywords (YAKE! Top 5)': "The top 5 keywords identified for this document using the YAKE! algorithm, separated by semicolons.",
    # Collective Metrics
    'Self-BLEU': "The average pairwise BLEU score calculated between all pairs of documents in the set. Measures surface-level (N-gram) similarity *across* documents. Higher scores indicate the documents are textually very similar to each other (low diversity in the set).",
    'Avg Pairwise Cosine Similarity': "Calculates a vector embedding for each document and finds the average cosine similarity between all pairs of embeddings. Measures *semantic* similarity across documents. Higher scores (closer to 1) indicate the documents discuss very similar topics or convey similar meanings (low semantic diversity).",
    'Avg Distance to Centroid': "Calculates the average distance (using cosine distance) of each document's embedding from the mean embedding (centroid) of the entire set. Measures the semantic *spread* or dispersion of the documents. Higher values indicate the documents are more spread out semantically (high semantic diversity)."
}

# --- Updated PLOT_INFO with corrected keys and new plots ---
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
    'mtld_vs_vocd_scatter': { # Corrected key
        'title': 'MTLD vs. VOCD',
        'desc': "Plots each file as a point based on its MTLD score (Y-axis) and its VOCD score (X-axis). Points are colored by filename (see legend).",
        'read': "Look for patterns or trends. Since both are robust diversity measures, expect some positive correlation. Deviations might indicate differences in how the metrics capture diversity (e.g., sensitivity to rare words vs. overall repetition). Use the legend to identify specific files."
    },
    'unique_vs_sentlen_scatter': { # Corrected key
        'title': 'Unique Words vs. Average Sentence Length',
        'desc': "Plots each file based on its total Unique Words (Y-axis) and its Average Sentence Length (X-axis). Points are colored by filename (see legend).",
        'read': "Look for correlations. Do texts with longer sentences tend to use a larger vocabulary (positive correlation)? Or is there no clear relationship? Clusters might indicate different writing styles (e.g., simple sentences with limited vocabulary vs. complex sentences with rich vocabulary). Use the legend to identify specific files."
    },
    'grouped_profile_radar': {
        'title': 'Grouped Profile Comparison (Normalized)',
        'desc': "Creates average 'fingerprints' for groups of files. Files are grouped into quartiles (Low, Mid-Low, Mid-High, High) based on their MTLD scores. Each axis represents a different key metric (normalized and sometimes inverted so outward means 'more diverse/complex/readable'). Shows the *average* profile for files within each MTLD diversity tier.",
        'read': "Compare the shapes of the polygons for the different MTLD groups. Does the 'High MTLD' group consistently score higher on other diversity/complexity metrics? Are there trade-offs (e.g., does higher diversity correlate with lower readability in this dataset)? This shows average tendencies for different diversity levels. Filenames in each group are listed below."
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
        'read': "Files with similar overall profiles across the selected metrics will have lines that follow similar paths and cluster together visually. The color-coding helps see if files within the same MTLD group exhibit similar patterns across other metrics (e.g., do 'High MTLD' lines generally stay high on other diversity axes?). Outliers will have lines that deviate significantly. Observe group trends rather than trying to trace every individual line. Filenames in each group are listed below."
    },
    'readability_profiles': { # Corrected key
        'title': 'Readability Profiles (Normalized)',
        'desc': "Compares normalized readability scores (0-1 scale) across files. Higher values indicate *easier* reading for Flesch RE, and *harder* reading for Gunning Fog, SMOG, and Dale-Chall after normalization.",
        'read': "Each group of bars represents a file. Within each group, compare the heights of the bars for different indices. Files with consistently high bars across all indices (after normalization logic) are likely easier to read, while those with low bars are harder. Note: Flesch RE is inverted during normalization so higher always means 'easier' relative to the dataset range for this plot."
    },
    'grouped_emotion_bars': { # Corrected key
        'title': 'Average Emotion Scores by MTLD Group',
        'desc': "Shows the average score for each NRCLex emotion category, grouped by the file's MTLD diversity quartile.",
        'read': "This grouped bar chart compares the average emotional profile for files within different diversity tiers (Low to High MTLD). Look for emotions that are significantly higher or lower in certain diversity groups. For example, do high-diversity texts tend to express more 'anticipation' or 'trust' on average compared to low-diversity texts in this dataset? Filenames in each group are listed below."
    },
    'nrclex_scores_summary': { # Corrected key
        'title': 'Aggregated NRCLex Emotion Scores',
        'desc': "Shows the total score (summed across all files) for each NRC emotion category.",
        'read': "This horizontal bar chart indicates the overall emotional tone of the entire dataset according to the NRC lexicon. Longer bars represent emotions that appeared more frequently or with higher intensity across all documents combined. Useful for identifying the dominant emotional signals in the corpus as a whole."
    },
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
            start_line = separator_index + 1
            logging.info(f"Parsing collective metrics starting from line {start_line+1} (0-based index {start_line})")
            if start_line < len(lines):
                 if not lines[start_line].strip(): start_line += 1 # Skip potential blank line
                 for line_num, line in enumerate(lines[start_line:], start=start_line):
                      line = line.strip()
                      if not line: continue
                      match = re.match(r'^"?([^"]+)"?,\s*(.*)$', line)
                      if match:
                           key = match.group(1).strip()
                           value_str = match.group(2).strip()
                           logging.debug(f"Parsing collective metric: Key='{key}', Value='{value_str}'")
                           try:
                               if value_str.lower() == 'n/a (<=1 file)' or value_str == '' or value_str.lower() == 'fail': metrics[key] = None; logging.debug(f"  -> Parsed as None")
                               elif value_str.lower() == 'inf': metrics[key] = float('inf'); logging.debug(f"  -> Parsed as Inf")
                               elif value_str.lower() == 'nan': metrics[key] = float('nan'); logging.debug(f"  -> Parsed as NaN")
                               else: metrics[key] = float(value_str); logging.debug(f"  -> Parsed as float: {metrics[key]}")
                           except ValueError: metrics[key] = value_str; logging.debug(f"  -> Parsed as string: {metrics[key]}")
                      else: logging.warning(f"Unexpected format in collective metrics section (line {line_num+1}): {line}")
            else: logging.warning("No lines found after collective metrics separator.")
    except FileNotFoundError: logging.error(f"CSV file not found at {filename} for parsing collective metrics.")
    except Exception as e: logging.error(f"Error parsing collective metrics from CSV: {e}", exc_info=True)
    logging.info(f"Parsed collective metrics: {metrics}")
    return metrics


# --- Get Folder Path ---
if len(sys.argv) < 2:
    prompt = f"Enter path to folder containing outputs (or wait 5s for current: {os.getcwd()}): "
    user_path = get_input_with_timeout(prompt, 5)
    if user_path is None: target_folder = os.getcwd(); print(f"\nTimeout! Using current directory: {target_folder}")
    else:
        user_path_stripped = user_path.strip()
        if not user_path_stripped: target_folder = os.getcwd(); print(f"\nNo path entered. Using current directory: {target_folder}")
        else: target_folder = user_path_stripped; print(f"\nUsing provided path: {target_folder}")
else:
    target_folder = sys.argv[1]
    print(f"Using folder path from command line argument: {target_folder}")

# --- Validate Folder Path ---
if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
     print(f"CRITICAL Error: Invalid folder path '{target_folder}'"); sys.exit(1)

# --- Setup Paths based on target_folder ---
csv_path = os.path.join(target_folder, CSV_FILENAME)
html_path = os.path.join(target_folder, HTML_FILENAME)
if not os.path.isfile(csv_path): print(f"ERROR: CSV file '{CSV_FILENAME}' not found in '{target_folder}'"); sys.exit(1)
if not os.path.isfile(html_path): print(f"ERROR: HTML file '{HTML_FILENAME}' not found in '{target_folder}'. Run lexicon_analyzer.py first."); sys.exit(1)

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
        # Filter known collective keys if separator is missing
        known_collective_keys = list(collective_metrics_data.keys()) + ["--- Collective Metrics (E) ---"]
        if 'Filename' in df.columns:
            original_rows = len(df)
            df = df[~df['Filename'].isin(known_collective_keys)].copy()
            if len(df) < original_rows:
                logging.warning(f"Filtered out {original_rows - len(df)} rows suspected to be collective metrics.")
    else:
        rows_to_read = separator_idx
        logging.info(f"Separator line found at index {separator_idx}. Reading {rows_to_read} data rows.")
        df = pd.read_csv(csv_path, nrows=rows_to_read)
except Exception as e: logging.error(f"Could not read CSV file: {e}"); sys.exit(1)

# --- Data Cleaning & Filtering --- # <<< MOVED FILTERING HERE
logging.info(f"Loaded {df.shape[0]} rows initially from CSV (excluding header).")
df.dropna(how='all', inplace=True) # Drop rows where ALL values are NaN

# <<< FIX: Explicitly filter known collective keys AFTER loading nrows >>>
known_collective_keys = list(collective_metrics_data.keys()) + ["--- Collective Metrics (E) ---"]
if 'Filename' in df.columns:
    initial_rows = len(df)
    df = df[~df['Filename'].isin(known_collective_keys)].copy()
    if len(df) < initial_rows:
        logging.info(f"Filtered out {initial_rows - len(df)} collective metric rows from main data section.")
# <<< END FIX >>>

logging.info(f"DataFrame shape after dropping empty rows and filtering: {df.shape}")

# --- Store original filenames and indices AFTER filtering ---
original_filenames_list = []
if 'Filename' in df.columns:
    original_filenames_list = df['Filename'].tolist() # Use the filtered df
    # No need to filter again here
else:
     logging.warning("CSV has no 'Filename' column. Using index for filenames list.")
     original_filenames_list = [f"Index {i}" for i in df.index] # Use filtered df index
# Create a map from 0-based index to original filename
original_filenames_map = {i: fname for i, fname in enumerate(original_filenames_list)}
logging.info(f"Created filename map for {len(original_filenames_map)} files.")


# --- START: Parse Emotion JSON Early ---
emotion_json_col = 'Emotion (NRCLex Scores JSON)'
known_emotions = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy']
df_emotions = pd.DataFrame(index=df.index, columns=known_emotions) # Initialize with original index

if emotion_json_col in df.columns:
    logging.info(f"Parsing '{emotion_json_col}' column...")
    parsed_count = 0
    error_count = 0
    for index, json_str in df[emotion_json_col].items():
        scores = {emo: 0.0 for emo in known_emotions} # Default to 0
        try:
            if pd.notna(json_str) and isinstance(json_str, str) and json_str.strip().startswith('{'):
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    for emo in known_emotions:
                        scores[emo] = float(parsed.get(emo, 0.0)) # Ensure float, default 0
                    parsed_count += 1
                else:
                    logging.warning(f"Row index {index}: Parsed JSON is not a dictionary: {json_str[:100]}...")
                    error_count += 1
            elif pd.notna(json_str): # Handle non-string, non-null values if they somehow occur
                logging.warning(f"Row index {index}: Unexpected data type in JSON column: {type(json_str)}")
                error_count += 1
            # else: Keep default zeros for NaN/empty strings
        except json.JSONDecodeError:
            logging.warning(f"Row index {index}: Could not decode JSON string: {json_str[:100]}...")
            error_count += 1
        except Exception as e:
            logging.error(f"Row index {index}: Unexpected error parsing JSON '{json_str[:100]}...': {e}")
            error_count += 1
        df_emotions.loc[index] = scores # Assign scores to the correct row using original index
    logging.info(f"Finished parsing emotion JSON: {parsed_count} successful, {error_count} errors/skipped.")
else:
    logging.warning(f"Column '{emotion_json_col}' not found in CSV. Emotion plots will be skipped.")
# --- END: Parse Emotion JSON Early ---


# --- Convert to Numeric, Create df_numeric ---
numeric_cols = [col for col in df.columns if col != 'Filename' and col != emotion_json_col] # Exclude JSON col now
df_numeric = df.copy()
for col in numeric_cols:
    df_numeric[col] = df_numeric[col].replace(['FAIL', 'N/A (<=1 file)'], np.nan)
    df_numeric[col] = df_numeric[col].replace(['INF'], np.inf)
    df_numeric[col] = df_numeric[col].replace(['NaN'], np.nan)
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# --- Merge Parsed Emotions into df_numeric ---
# Ensure df_emotions contains numeric types
df_emotions = df_emotions.astype(float)
# Merge based on the index (should be the original 0..N-1 index for both at this stage)
df_numeric = df_numeric.merge(df_emotions, left_index=True, right_index=True, how='left')
logging.info(f"Merged parsed emotion scores into df_numeric. Shape after merge: {df_numeric.shape}")
# --- END Merge ---


# --- Set Index ---
filename_col_present = 'Filename' in df_numeric.columns
if filename_col_present:
    filenames_series = df_numeric['Filename']
    if df_numeric.index.name != 'Filename':
        try:
            # Check for duplicate filenames before setting index
            if df_numeric['Filename'].duplicated().any():
                logging.warning("Duplicate filenames found! Cannot set 'Filename' as index. Using default index.")
                filename_col_present = False
                df_numeric.index.name = 'File Index'
                filenames_series = pd.Series([f"Index {i}" for i in df_numeric.index], index=df_numeric.index)
            else:
                df_numeric.set_index('Filename', inplace=True)
                logging.info("Set 'Filename' as index for df_numeric.")
        except KeyError:
            logging.warning("Could not set 'Filename' as index (KeyError). Using default index.")
            filename_col_present = False
            df_numeric.index.name = 'File Index'
            filenames_series = pd.Series([f"Index {i}" for i in df_numeric.index], index=df_numeric.index)
else:
    logging.warning("'Filename' column not found after loading/cleaning. Plots will use default index.")
    df_numeric.index.name = 'File Index'
    filenames_series = pd.Series([f"Index {i}" for i in df_numeric.index], index=df_numeric.index)


# --- Function to generate plot, save, and encode ---
def generate_encode_plot(plot_func, filename_base, *args, **kwargs):
    """Generates plot using plot_func, saves it, encodes to base64."""
    plot_filepath = os.path.join(output_dir_path, f"{filename_base}.png")
    base64_encoded = None
    logging.info(f"Generating {filename_base}.png...")
    fig = None
    plot_return_value = None
    try:
        # Call the plot function, which might return figure/axes or just plot to current context
        plot_return_value = plot_func(*args, **kwargs)

        # Get the current figure
        fig = plt.gcf()

        # Check if the figure has any axes with data
        if fig.get_axes():
            has_content = any(ax.has_data() for ax in fig.get_axes() if hasattr(ax, 'has_data'))
            # Special case for clustermap which returns ClusterGrid, not axes with has_data()
            is_clustermap = plot_return_value is not None and 'ClusterGrid' in str(type(plot_return_value))

            if has_content or is_clustermap:
                fig.savefig(plot_filepath, bbox_inches='tight', dpi=150)
                with open(plot_filepath, "rb") as image_file:
                    base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
                logging.info(f"-> Saved and encoded {filename_base}.png.")
            else:
                logging.warning(f"-> Skipping save/encode for {filename_base}: No content plotted (axes empty or no data).")
        else:
            logging.warning(f"-> Skipping save/encode for {filename_base}: No axes found in figure.")

        # Return the encoded data AND the return value from the plot function (e.g., ClusterGrid)
        return base64_encoded, plot_return_value

    except Exception as e:
        logging.error(f"-> ERROR generating/encoding plot {filename_base}: {e}", exc_info=True)
        return None, None # Return None for both on error
    finally:
        if fig: plt.close(fig)
        else: plt.close() # Close the current figure context


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
    return ax # Return axes

def plot_distribution(series, metric_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(series, kde=True, bins=min(15, max(5, len(series)//2)), ax=ax)
    ax.set_title(f'Distribution of {metric_name}')
    ax.set_xlabel(metric_name)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    return ax # Return axes

# --- UPDATED Scatter Plot Function ---
def plot_scatter(df_plot, x_metric, y_metric, label_col='Filename'):
    """Plots scatter with labels using hue and returns palette/labels."""
    fig, ax = plt.subplots(figsize=(10, 7)) # Adjusted size back
    # Use the index (filename or File Index) for hue
    hue_data = df_plot.index if label_col not in df_plot.columns else df_plot[label_col]
    unique_hues = hue_data.unique()
    num_hues = len(unique_hues)
    show_legend_points = num_hues <= 30 # Only show legend in plot if 30 or fewer files

    # Generate a color palette
    palette = sns.color_palette("husl", num_hues) # Use husl for distinct colors

    sns.scatterplot(data=df_plot, x=x_metric, y=y_metric, hue=hue_data,
                    palette=palette, # Apply the palette
                    ax=ax, alpha=0.8, legend=show_legend_points, s=50)
    ax.set_title(f'{y_metric} vs. {x_metric}')
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Remove the automatic legend from the plot itself
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout() # Adjust layout

    # Return handles and labels for external legend box
    return palette, unique_hues
# --- End Updated Scatter Plot ---

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
    return ax # Return axes

def plot_heatmap_simple(df_norm_heat):
    fig, ax = plt.subplots(figsize=(15, max(10, len(df_norm_heat.columns) * 0.5)))
    # <<< FIX: Use reversed colormap 'viridis_r' >>>
    sns.heatmap(df_norm_heat.transpose(), annot=False, cmap="viridis_r", linewidths=.5, cbar=True, ax=ax)
    ax.set_title('Heatmap of Normalized Metrics Across Files')
    ax.set_xlabel('File' if filename_col_present else 'Index')
    ax.set_ylabel('Metric')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    return ax # Return axes

def plot_correlation_matrix(df_corr):
    plt.figure(figsize=(18, 14))
    ax = sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar=True, annot_kws={"size": 8})
    plt.title('Correlation Matrix of Numeric Metrics')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    return ax # Return axes

def plot_parallel_coordinates(df_plot, class_column='MTLD_Group', metrics=None, title='Parallel Coordinates Plot'):
    fig, ax = plt.subplots(figsize=(18, 8))
    if metrics is not None and len(metrics) > 0:
        cols_to_select = list(metrics) + ([class_column] if class_column in df_plot.columns else [])
        cols_to_select = [col for col in cols_to_select if col in df_plot.columns]
        df_plot_selected = df_plot[cols_to_select]
    else:
        df_plot_selected = df_plot

    if class_column not in df_plot_selected.columns:
         logging.warning(f"Class column '{class_column}' not found for parallel coordinates. Using default color.")
         df_plot_selected = df_plot_selected.copy()
         df_plot_selected['DummyClass'] = 'All Files'
         class_column = 'DummyClass'

    numeric_cols_in_selection = df_plot_selected.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols_in_selection:
        logging.error("No numeric columns found to plot in parallel coordinates.")
        ax.text(0.5, 0.5, 'No numeric data to plot', ha='center', va='center')
        return ax # Return axes

    colormap = plt.cm.get_cmap('viridis', max(1, len(df_plot_selected[class_column].unique())))
    try:
        parallel_coordinates(df_plot_selected, class_column=class_column, cols=numeric_cols_in_selection, colormap=colormap, alpha=0.7, ax=ax, axvlines_kwds={"color": "grey", "linestyle": ":", "alpha": 0.5})
    except Exception as e:
        logging.error(f"Error calling pandas parallel_coordinates: {e}", exc_info=True)
        ax.text(0.5, 0.5, f'Error plotting parallel coordinates:\n{e}', ha='center', va='center', color='red')
        return ax # Return axes

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
        if ax.get_legend(): ax.get_legend().remove()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    return ax # Return axes


# --- NEW Plotting Functions ---
def plot_nrclex_scores_summary(aggregated_scores):
    """Plots a horizontal bar chart of aggregated NRCLex scores."""
    if aggregated_scores.empty:
        logging.warning("No aggregated emotion scores to plot.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No valid emotion data to plot', ha='center', va='center')
        ax.set_title('Aggregated NRCLex Emotion Scores')
        return ax # Return axes

    aggregated_scores = aggregated_scores.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size for potentially many categories
    # <<< NOTE: Using 'viridis' here as requested (not reversed) >>>
    sns.barplot(x=aggregated_scores.values, y=aggregated_scores.index, palette='viridis', orient='h', ax=ax)
    ax.set_title('Aggregated NRCLex Emotion Scores Across All Files')
    ax.set_xlabel('Total Score (Sum across files)')
    ax.set_ylabel('Emotion Category')
    plt.tight_layout()
    return ax # Return axes

# --- UPDATED: Plot for Grouped Emotion Bars ---
def plot_grouped_emotion_bars(df_plot, group_col, emotion_cols):
    """
    Plots average emotion scores as a grouped bar chart, grouped by a specified column.
    """
    if group_col not in df_plot.columns:
        logging.warning(f"Grouping column '{group_col}' not found. Skipping grouped emotion plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, f"Grouping column '{group_col}' not found.", ha='center', va='center')
        ax.set_title('Average Emotion Scores by Group')
        return ax # Return axes

    if df_plot[group_col].isnull().all():
        logging.warning(f"Grouping column '{group_col}' contains only NaN values. Skipping grouped emotion plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, f"Grouping column '{group_col}' has no valid data.", ha='center', va='center')
        ax.set_title('Average Emotion Scores by Group')
        return ax # Return axes

    present_emotion_cols = [col for col in emotion_cols if col in df_plot.columns]
    if not present_emotion_cols:
        logging.warning("No valid emotion columns found in data. Skipping grouped emotion plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, "No valid emotion data columns found.", ha='center', va='center')
        ax.set_title('Average Emotion Scores by Group')
        return ax # Return axes

    # Calculate average scores for each emotion within each group
    # Ensure only numeric columns are selected for mean calculation
    numeric_emotion_cols = df_plot[present_emotion_cols].select_dtypes(include=np.number).columns.tolist()
    if not numeric_emotion_cols:
        logging.warning("No numeric emotion columns found after selection. Skipping grouped emotion plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, "No numeric emotion data columns found.", ha='center', va='center')
        ax.set_title('Average Emotion Scores by Group')
        return ax # Return axes

    grouped_means = df_plot.groupby(group_col, observed=False)[numeric_emotion_cols].mean()

    # Check if grouped_means is empty or all NaN
    if grouped_means.empty or grouped_means.isnull().all().all():
        logging.warning("Grouped means calculation resulted in empty or all-NaN data. Skipping grouped emotion plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, "No valid data after grouping.", ha='center', va='center')
        ax.set_title('Average Emotion Scores by Group')
        return ax # Return axes

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8)) # Create figure and axes
    grouped_means.plot(kind='bar', ax=ax, width=0.8) # Plot on the created axes
    ax.set_title(f'Average Emotion Scores by {group_col}')
    ax.set_ylabel('Average Score')
    ax.set_xlabel(group_col)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Emotion Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    return ax # Return axes
# --- END UPDATED ---

def plot_readability_profiles(df_plot, metrics):
    """Plots a grouped bar chart for normalized readability scores per file."""
    if df_plot.empty or not metrics:
        logging.warning("No data or metrics specified for readability profiles plot.")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.text(0.5, 0.5, 'No valid readability data to plot', ha='center', va='center')
        ax.set_title('Readability Profiles (Normalized)')
        return ax # Return axes

    # Use original filenames if possible
    if filename_col_present and df_plot.index.name != 'Filename':
        try: plot_labels = [original_filenames_map.get(i, f"Index {i}") for i in df_plot.index]
        except NameError: plot_labels = df_plot.index
    else: plot_labels = df_plot.index

    df_plot_copy = df_plot[metrics].copy()
    df_plot_copy.index = plot_labels # Set index for plotting

    # Normalize 0-1 (Important: Invert scores where lower is better)
    if MinMaxScaler and SKLEARN_AVAILABLE:
        scaler = MinMaxScaler()
        df_plot_copy[metrics] = scaler.fit_transform(df_plot_copy[metrics])
        for col in ['Gunning Fog', 'SMOG Index', 'Dale-Chall Score']:
            if col in df_plot_copy.columns: df_plot_copy[col] = 1.0 - df_plot_copy[col]
    else:
        logging.warning("MinMaxScaler not available. Manual normalization for readability plot.")
        for col in metrics:
            min_val = df_plot_copy[col].min(); max_val = df_plot_copy[col].max()
            if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val != 0):
                norm_vals = (df_plot_copy[col] - min_val) / (max_val - min_val)
                if col in ['Gunning Fog', 'SMOG Index', 'Dale-Chall Score']: df_plot_copy[col] = 1.0 - norm_vals
                else: df_plot_copy[col] = norm_vals
            else: df_plot_copy[col] = 0.5

    # Plotting
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot_copy)*0.6), 7)) # Create figure and axes
    df_plot_copy.plot(kind='bar', ax=ax, width=0.8) # Plot on the created axes
    ax.set_title('Readability Profiles (Normalized 0-1)')
    ax.set_ylabel('Normalized Score (Higher = Easier Reading)')
    ax.set_xlabel('File' if filename_col_present else 'Index')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.legend(title='Readability Index', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, 1.1) # Ensure y-axis goes from 0 to 1
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    return ax # Return axes


# --- UPDATED Clustered Heatmap Plot ---
def plot_clustered_heatmap(df_norm_cluster):
    """
    Plots a clustered heatmap using seaborn with file labels.
    Returns the ClusterGrid object which contains dendrogram info.
    """
    g = None # Initialize g
    if not SCIPY_AVAILABLE:
         logging.error("Cannot generate clustered heatmap: scipy library not found.")
         fig, ax = plt.subplots(figsize=(10, 8))
         ax.text(0.5, 0.5, 'Clustered Heatmap requires SciPy\nInstall: pip install scipy', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=12)
         ax.set_xticks([]); ax.set_yticks([])
         return None # Return None if SciPy is missing

    try:
        df_plot_final = df_norm_cluster.dropna(axis=1, how='all')
        if df_plot_final.empty or df_plot_final.shape[1] < 2:
             logging.error("Cannot generate clustered heatmap: Not enough valid numeric data remaining after cleaning.")
             fig, ax = plt.subplots(figsize=(10, 8))
             ax.text(0.5, 0.5, 'Not enough valid data for Clustered Heatmap', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
             ax.set_xticks([]); ax.set_yticks([])
             return None

        height = max(10, len(df_plot_final) * 0.4)
        width = max(12, len(df_plot_final.columns) * 0.6)

        # Use seaborn's clustermap function
        # <<< FIX: Use reversed colormap 'viridis_r' >>>
        g = sns.clustermap(df_plot_final, # Use the cleaned, normalized data
                           cmap="viridis_r",
                           figsize=(width, height),
                           linewidths=.5,
                           annot=False,
                           yticklabels=True, # Ensure row labels (filenames/indices) are shown
                           xticklabels=True,
                           cbar_pos=(0.02, 0.8, 0.03, 0.15), # Position color bar on left
                           cbar_kws={'label': 'Normalized Value (0-1)'}, # Add label to color bar
                           dendrogram_ratio=(.15, .2))

        g.ax_heatmap.set_title('Clustered Heatmap of All Metrics (Normalized)', y=1.05)
        g.ax_heatmap.set_xlabel('Metrics')
        g.ax_heatmap.set_ylabel('Files' if filename_col_present else 'Index')
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8) # Ensure y-labels are readable

        # Return the ClusterGrid object
        return g

    except Exception as e:
         logging.error(f"Error during clustermap generation: {e}", exc_info=True)
         fig, ax = plt.subplots(figsize=(10, 8))
         ax.text(0.5, 0.5, f'Error generating Clustered Heatmap:\n{e}', ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
         ax.set_xticks([]); ax.set_yticks([])
         return None # Return None on error


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
all_plot_metadata = {} # <<< Initialize dict to store metadata
scatter_legends = {} # <<< Store legend info for scatter plots

# --- Define Plot Order (Corrected) ---
plot_order = [
    'normalized_diversity_bar',
    'mtld_distribution',
    'mtld_vs_vocd_scatter',         # Corrected key
    'unique_vs_sentlen_scatter',    # Corrected key
    'grouped_profile_radar',
    'metrics_heatmap',
    'correlation_matrix',
    'parallel_coordinates',
    'readability_profiles',         # Corrected key
    'grouped_emotion_bars',         # Corrected key
    'nrclex_scores_summary',        # Corrected key
    'clustered_heatmap'
]

# --- Define paths for NPY files ---
linkage_matrix_path = os.path.join(target_folder, LINKAGE_MATRIX_FILENAME)
reordered_indices_path = os.path.join(target_folder, REORDERED_INDICES_FILENAME)
original_filenames_path = os.path.join(target_folder, ORIGINAL_FILENAMES_FILENAME)

# --- Loop through plot order to generate ---
for plot_key in plot_order:
    logging.info(f"--- Generating Plot: {plot_key} ---")
    if plot_key == 'clustered_heatmap': filename_base = "clustered_metrics_heatmap"
    else: filename_base = plot_key
    plot_info = PLOT_INFO.get(plot_key, {})
    plot_title = plot_info.get('title', plot_key.replace('_', ' ').title())
    plot_desc = plot_info.get('desc', 'No description available.')

    # <<< FIX: Add default metadata entry FIRST >>>
    all_plot_metadata[plot_key] = {
        "filename_base": None, # Default to None
        "title": plot_title,
        "description": plot_desc
    }

    encoded_plot = None # Initialize
    plot_return_value = None # To store ClusterGrid etc.

    try:
        # --- Plot Generation Logic ---
        if plot_key == 'normalized_diversity_bar':
            metrics_to_compare = ['TTR', 'MTLD', 'VOCD', 'Distinct-2']
            metrics_present = [m for m in metrics_to_compare if m in df_numeric.columns and df_numeric[m].notna().any()]
            if metrics_present and not df_numeric.empty:
                df_compare_norm = df_numeric[metrics_present].copy()
                if MinMaxScaler and SKLEARN_AVAILABLE: scaler = MinMaxScaler(); df_compare_norm[metrics_present] = scaler.fit_transform(df_compare_norm[metrics_present])
                else: # Manual fallback
                    for col in metrics_present:
                        min_val = df_compare_norm[col].min(); max_val = df_compare_norm[col].max()
                        if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val == 0): df_compare_norm[col] = 0.5
                        else: df_compare_norm[col] = (df_compare_norm[col] - min_val) / (max_val - min_val)
                encoded_plot, _ = generate_encode_plot(plot_normalized_bar, filename_base, df_compare_norm, metrics_present)
            else: logging.warning(f"Skipping {plot_key}: Not enough valid data or required columns missing.")

        elif plot_key == 'mtld_distribution':
            metric_for_dist = 'MTLD'
            if metric_for_dist in df_numeric.columns and df_numeric[metric_for_dist].notna().any():
                encoded_plot, _ = generate_encode_plot(plot_distribution, filename_base, df_numeric[metric_for_dist].dropna(), metric_for_dist)
            else: logging.warning(f"Skipping {plot_key}: Metric '{metric_for_dist}' not found or has no valid data.")

        elif plot_key == 'mtld_vs_vocd_scatter': # Changed
            x_metric_scatter = 'VOCD'; y_metric_scatter = 'MTLD'
            if x_metric_scatter in df_numeric.columns and y_metric_scatter in df_numeric.columns and df_numeric[x_metric_scatter].notna().any() and df_numeric[y_metric_scatter].notna().any():
                 df_scatter = df_numeric[[x_metric_scatter, y_metric_scatter]].dropna()
                 if not df_scatter.empty:
                     palette, labels = plot_scatter(df_scatter, x_metric_scatter, y_metric_scatter, label_col=filenames_series.name if filename_col_present else df_scatter.index.name)
                     encoded_plot, _ = generate_encode_plot(lambda: None, filename_base) # Save the plot generated by plot_scatter
                     scatter_legends[plot_key] = {'palette': palette, 'labels': labels} # Store legend info
                 else: logging.warning(f"Skipping {plot_key}: No overlapping valid data for '{x_metric_scatter}' and '{y_metric_scatter}'.")
            else: logging.warning(f"Skipping {plot_key}: One or both metrics ('{x_metric_scatter}', '{y_metric_scatter}') not found or have no valid data.")

        elif plot_key == 'unique_vs_sentlen_scatter': # Changed
            x_metric_scatter = 'Average Sentence Length'; y_metric_scatter = 'Unique Words'
            if x_metric_scatter in df_numeric.columns and y_metric_scatter in df_numeric.columns and df_numeric[x_metric_scatter].notna().any() and df_numeric[y_metric_scatter].notna().any():
                 df_scatter = df_numeric[[x_metric_scatter, y_metric_scatter]].dropna()
                 if not df_scatter.empty:
                     palette, labels = plot_scatter(df_scatter, x_metric_scatter, y_metric_scatter, label_col=filenames_series.name if filename_col_present else df_scatter.index.name)
                     encoded_plot, _ = generate_encode_plot(lambda: None, filename_base)
                     scatter_legends[plot_key] = {'palette': palette, 'labels': labels}
                 else: logging.warning(f"Skipping {plot_key}: No overlapping valid data for '{x_metric_scatter}' and '{y_metric_scatter}'.")
            else: logging.warning(f"Skipping {plot_key}: One or both metrics ('{x_metric_scatter}', '{y_metric_scatter}') not found or have no valid data.")

        elif plot_key == 'grouped_profile_radar':
            metrics_for_radar = ['MTLD', 'VOCD', 'Yule\'s K', 'Simpson\'s D', 'Distinct-2', 'Repetition-2', 'Average Sentence Length', 'Flesch Reading Ease']
            metrics_present_radar = [m for m in metrics_for_radar if m in df_numeric.columns and df_numeric[m].notna().any()]
            if grouping_possible and len(metrics_present_radar) >= 3:
                grouped_data = df_numeric.groupby(mtld_group_col, observed=False)[metrics_present_radar].mean()
                df_grouped_normalized = grouped_data.copy()
                if MinMaxScaler and SKLEARN_AVAILABLE:
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
                if not df_grouped_normalized.empty: encoded_plot, _ = generate_encode_plot(plot_grouped_radar, filename_base, df_grouped_normalized, df_grouped_normalized.columns.tolist())
                else: logging.warning(f"Skipping {plot_key}: No valid data after grouping/normalization.")
            else: logging.warning(f"Skipping {plot_key}: Grouping not possible or < 3 metrics.")

        elif plot_key == 'metrics_heatmap':
            if not df_numeric.empty and len(df_numeric.columns) > 1:
                df_norm_heat = df_numeric.select_dtypes(include=np.number).copy()
                imputed_columns = df_norm_heat.columns
                if df_norm_heat.isnull().any().any():
                    if SimpleImputer and SKLEARN_AVAILABLE:
                        imputer = SimpleImputer(strategy='mean'); df_imputed_values = imputer.fit_transform(df_norm_heat)
                        imputed_columns = imputer.get_feature_names_out(df_norm_heat.columns)
                        df_norm_heat = pd.DataFrame(df_imputed_values, index=df_norm_heat.index, columns=imputed_columns)
                    else: df_norm_heat.fillna(0.5, inplace=True); imputed_columns = df_norm_heat.columns
                if MinMaxScaler and SKLEARN_AVAILABLE:
                    scaler = MinMaxScaler(); scaled_vals = scaler.fit_transform(df_norm_heat)
                    df_norm_heat = pd.DataFrame(scaled_vals, index=df_norm_heat.index, columns=imputed_columns)
                else:
                    for col in imputed_columns:
                         min_val = df_norm_heat[col].min(); max_val = df_norm_heat[col].max()
                         if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val != 0): df_norm_heat[col] = (df_norm_heat[col] - min_val) / (max_val - min_val)
                         else: df_norm_heat[col] = 0.5
                df_norm_heat.dropna(axis=1, how='all', inplace=True)
                if not df_norm_heat.empty: encoded_plot, _ = generate_encode_plot(plot_heatmap_simple, filename_base, df_norm_heat)
                else: logging.warning(f"Skipping {plot_key}: No valid numeric data after normalization.")
            else: logging.warning(f"Skipping {plot_key}: No valid data.")

        elif plot_key == 'correlation_matrix':
            df_numeric_only = df_numeric.select_dtypes(include=np.number).dropna(axis=1, how='all')
            if not df_numeric_only.empty and len(df_numeric_only.columns) > 1:
                df_numeric_only = df_numeric_only.loc[:, df_numeric_only.apply(pd.Series.nunique) > 1]
                if len(df_numeric_only.columns) > 1:
                    df_corr = df_numeric_only.corr()
                    encoded_plot, _ = generate_encode_plot(plot_correlation_matrix, filename_base, df_corr)
                else: logging.warning(f"Skipping {plot_key}: Not enough columns with variance after dropping constant columns.")
            else: logging.warning(f"Skipping {plot_key}: Not enough numeric data or columns.")

        elif plot_key == 'parallel_coordinates':
            metrics_for_parallel = ['MTLD', 'VOCD', 'Yule\'s K', 'Simpson\'s D', 'Distinct-2', 'Repetition-2', 'Average Sentence Length', 'Flesch Reading Ease', 'Lexical Density']
            metrics_present_parallel = [m for m in metrics_for_parallel if m in df_numeric.columns and df_numeric[m].notna().any()]
            class_col_parallel = mtld_group_col if grouping_possible else ('Filename' if filename_col_present else 'File Index')
            if len(metrics_present_parallel) > 1 and not df_numeric.empty:
                cols_for_parallel = metrics_present_parallel + ([class_col_parallel] if class_col_parallel in df_numeric.columns else [])
                df_parallel_input = df_numeric[cols_for_parallel].copy()
                df_parallel_norm = df_parallel_input.copy()
                imputed_parallel_cols = metrics_present_parallel
                # Impute and Normalize
                if df_parallel_norm[metrics_present_parallel].isnull().any().any():
                     if SimpleImputer and SKLEARN_AVAILABLE:
                         imputer = SimpleImputer(strategy='mean'); imputed_vals = imputer.fit_transform(df_parallel_norm[metrics_present_parallel])
                         imputed_parallel_cols = imputer.get_feature_names_out(metrics_present_parallel)
                         df_parallel_norm_imputed = pd.DataFrame(imputed_vals, index=df_parallel_norm.index, columns=imputed_parallel_cols)
                     else:
                         df_parallel_norm[metrics_present_parallel] = df_parallel_norm[metrics_present_parallel].fillna(0.5)
                         df_parallel_norm_imputed = df_parallel_norm[metrics_present_parallel]
                         imputed_parallel_cols = metrics_present_parallel
                     df_parallel_to_scale = df_parallel_norm_imputed[imputed_parallel_cols]
                else: df_parallel_to_scale = df_parallel_norm[metrics_present_parallel]
                if MinMaxScaler and SKLEARN_AVAILABLE:
                    scaler = MinMaxScaler(); scaled_vals = scaler.fit_transform(df_parallel_to_scale)
                    df_parallel_scaled = pd.DataFrame(scaled_vals, index=df_parallel_to_scale.index, columns=imputed_parallel_cols)
                    for col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']:
                        if col in df_parallel_scaled.columns: df_parallel_scaled[col] = 1.0 - df_parallel_scaled[col]
                else: # Manual fallback
                    df_parallel_scaled = df_parallel_to_scale.copy()
                    for col in imputed_parallel_cols:
                        min_val = df_parallel_scaled[col].min(); max_val = df_parallel_scaled[col].max()
                        if pd.isna(min_val) or pd.isna(max_val) or (max_val - min_val == 0): df_parallel_scaled[col] = 0.5
                        else:
                            norm_val = (df_parallel_scaled[col] - min_val) / (max_val - min_val)
                            if col in ['Repetition-2', 'Repetition-3', 'Simpson\'s D']: df_parallel_scaled[col] = 1.0 - norm_val
                            else: df_parallel_scaled[col] = norm_val
                # Combine scaled data with class column
                if class_col_parallel in df_parallel_input.columns: df_parallel_final = pd.concat([df_parallel_scaled, df_parallel_input[[class_col_parallel]]], axis=1)
                else: # Handle case where class column wasn't in original df_numeric
                    df_parallel_final = df_parallel_scaled.copy()
                    if class_col_parallel == 'Filename' and filename_col_present: df_parallel_final['Filename'] = df_numeric.index # Use index if it's filename
                    elif class_col_parallel == 'File Index': df_parallel_final['File Index'] = df_numeric.index # Use index if it's numeric index
                df_parallel_final.dropna(subset=imputed_parallel_cols, inplace=True)
                if not df_parallel_final.empty and len(imputed_parallel_cols) > 1: encoded_plot, _ = generate_encode_plot(plot_parallel_coordinates, filename_base, df_parallel_final, class_column=class_col_parallel, metrics=imputed_parallel_cols, title=plot_title)
                else: logging.warning(f"Skipping {plot_key}: No valid data after normalization/NaN drop or too few metrics.")
            else: logging.warning(f"Skipping {plot_key}: Not enough valid metrics or data.")

        elif plot_key == 'readability_profiles': # Changed plot type
            readability_metrics = ['Flesch Reading Ease', 'Gunning Fog', 'SMOG Index', 'Dale-Chall Score']
            readability_metrics_present = [m for m in readability_metrics if m in df_numeric.columns and df_numeric[m].notna().any()]
            if len(readability_metrics_present) > 0:
                df_readability = df_numeric[readability_metrics_present].dropna()
                if not df_readability.empty: encoded_plot, _ = generate_encode_plot(plot_readability_profiles, filename_base, df_readability, readability_metrics_present)
                else: logging.warning(f"Skipping {plot_key}: No valid data after dropping NaNs.")
            else: logging.warning(f"Skipping {plot_key}: Required readability columns not found or contain only NaNs.")

        elif plot_key == 'grouped_emotion_bars': # New plot
            # Use the emotion columns already merged into df_numeric
            if grouping_possible and any(emo in df_numeric.columns for emo in known_emotions):
                emotion_cols_present = [emo for emo in known_emotions if emo in df_numeric.columns]
                # Select group column and present emotion columns
                df_plot_emotions = df_numeric[[mtld_group_col] + emotion_cols_present].copy()
                df_plot_emotions.dropna(subset=[mtld_group_col], inplace=True) # Drop rows where group is NaN

                # Check if there's still valid numeric data in emotion columns
                numeric_emotion_cols = df_plot_emotions[emotion_cols_present].select_dtypes(include=np.number).columns
                if not df_plot_emotions.empty and not df_plot_emotions[numeric_emotion_cols].isnull().all().all():
                    encoded_plot, _ = generate_encode_plot(plot_grouped_emotion_bars, filename_base, df_plot_emotions, mtld_group_col, numeric_emotion_cols)
                else:
                    logging.warning(f"Skipping {plot_key}: No valid numeric emotion data found after grouping or selecting columns.")
            elif not grouping_possible:
                logging.warning(f"Skipping {plot_key}: Grouping not possible.")
            else:
                logging.warning(f"Skipping {plot_key}: No known emotion columns found in df_numeric.")


        elif plot_key == 'nrclex_scores_summary': # New plot
            # Use the emotion columns already merged into df_numeric
            emotion_cols_present = [emo for emo in known_emotions if emo in df_numeric.columns]
            if emotion_cols_present:
                # Sum the scores across all files (rows)
                aggregated_scores_series = df_numeric[emotion_cols_present].sum(axis=0)
                # Filter out emotions with zero total score
                aggregated_scores_series = aggregated_scores_series[aggregated_scores_series > 0]
                if not aggregated_scores_series.empty:
                    encoded_plot, _ = generate_encode_plot(plot_nrclex_scores_summary, filename_base, aggregated_scores_series)
                else:
                    logging.warning(f"Skipping {plot_key}: All aggregated emotion scores are zero or less.")
            else:
                logging.warning(f"Skipping {plot_key}: No known emotion columns found in df_numeric.")


        elif plot_key == 'clustered_heatmap':
            if not df_numeric.empty and len(df_numeric.columns) > 1 and SCIPY_AVAILABLE:
                # Select only numeric columns for clustering, including emotions
                cols_for_cluster = df_numeric.select_dtypes(include=np.number).columns.tolist()
                # Optionally exclude specific columns if needed:
                # cols_to_exclude = ['Topic (NMF ID)'] # Example
                # cols_for_cluster = [c for c in cols_for_cluster if c not in cols_to_exclude]

                df_cluster_input = df_numeric[cols_for_cluster].copy()

                imputed_cluster_cols = df_cluster_input.columns
                if df_cluster_input.isnull().any().any():
                    if SimpleImputer and SKLEARN_AVAILABLE:
                        imputer = SimpleImputer(strategy='mean'); df_imputed_values = imputer.fit_transform(df_cluster_input)
                        imputed_cluster_cols = imputer.get_feature_names_out(df_cluster_input.columns)
                        df_cluster_input = pd.DataFrame(df_imputed_values, index=df_cluster_input.index, columns=imputed_cluster_cols)
                    else: df_cluster_input.fillna(0.5, inplace=True); imputed_cluster_cols = df_cluster_input.columns
                if MinMaxScaler and SKLEARN_AVAILABLE:
                    scaler = MinMaxScaler(); scaled_vals = scaler.fit_transform(df_cluster_input)
                    df_cluster_norm = pd.DataFrame(scaled_vals, index=df_cluster_input.index, columns=imputed_cluster_cols)
                else: # Manual fallback
                    df_cluster_norm = df_cluster_input.copy()
                    for col in imputed_cluster_cols:
                         min_val = df_cluster_norm[col].min(); max_val = df_cluster_norm[col].max()
                         if pd.notna(min_val) and pd.notna(max_val) and (max_val - min_val != 0): df_cluster_norm[col] = (df_cluster_norm[col] - min_val) / (max_val - min_val)
                         else: df_cluster_norm[col] = 0.5
                df_cluster_norm.dropna(axis=1, how='all', inplace=True)
                if not df_cluster_norm.empty and len(df_cluster_norm.columns) > 1:
                    # Call generate_encode_plot which calls plot_clustered_heatmap
                    encoded_plot, cluster_grid = generate_encode_plot(plot_clustered_heatmap, filename_base, df_cluster_norm)
                    if cluster_grid: # Check if clustermap was successful
                        try:
                            row_linkage = cluster_grid.dendrogram_row.linkage; reordered_row_indices = cluster_grid.dendrogram_row.reordered_ind
                            np.save(linkage_matrix_path, row_linkage); logging.info(f" -> Saved row linkage matrix to {linkage_matrix_path}")
                            np.save(reordered_indices_path, reordered_row_indices); logging.info(f" -> Saved reordered row indices to {reordered_indices_path}")
                            with open(original_filenames_path, 'w', encoding='utf-8') as f: json.dump(original_filenames_map, f, indent=4)
                            logging.info(f" -> Saved original filenames map to {original_filenames_path}")
                        except AttributeError as ae: logging.error(f"   -> ERROR accessing dendrogram data from ClusterGrid: {ae}. Cannot save .npy/.json files.")
                        except Exception as npy_e: logging.error(f"   -> ERROR saving dendrogram data to .npy/.json files: {npy_e}", exc_info=True)
                    else: logging.warning("   -> Clustermap generation failed, skipping image encoding and .npy/.json saving.")
                else: logging.warning(f"Skipping {plot_key}: No valid numeric data after normalization.")
            elif not SCIPY_AVAILABLE: logging.warning(f"Skipping {plot_key}: SciPy library not available.")
            else: logging.warning(f"Skipping {plot_key}: No valid data.")

        # --- Store results (Update metadata if successful) ---
        if encoded_plot:
            plot_data_base64[plot_key] = encoded_plot
            # <<< FIX: UPDATE the existing entry >>>
            all_plot_metadata[plot_key]["filename_base"] = filename_base
        # <<< ELSE: Keep the default entry with filename_base=None >>>

    except Exception as plot_err:
        logging.error(f"--- ERROR during generation of plot '{plot_key}': {plot_err}", exc_info=True)
        # Metadata entry already exists with filename_base=None
        plt.close('all') # Ensure plot context is cleared on error


# --- Append Descriptions, Plots, and JS to HTML ---
logging.info(f"Updating {html_path} with descriptions, plots, and sorting JS...")
soup = None # Initialize soup
parser_to_use = 'html.parser' # Default
if LXML_AVAILABLE: parser_to_use = 'lxml'

try:
    # --- Read original HTML ---
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    if not html_content.strip():
        logging.error(f"HTML file '{html_path}' is empty. Cannot update.")
        sys.exit(1)

    # --- Parse the original HTML ---
    soup = BeautifulSoup(html_content, parser_to_use)
    body = soup.body
    if not body:
        logging.error("HTML file is missing a <body> tag. Cannot append content.")
        sys.exit(1)

    # --- Remove existing appended sections to prevent duplication ---
    existing_toc = soup.find('div', id='toc')
    if existing_toc: existing_toc.decompose()
    existing_metrics_section_h2 = soup.find('h2', id='metric_descriptions')
    if existing_metrics_section_h2:
        logging.warning("Found existing metric descriptions section. Removing it.")
        dl_tag = existing_metrics_section_h2.find_next_sibling('dl')
        hr_tag = existing_metrics_section_h2.find_previous_sibling('hr')
        existing_metrics_section_h2.decompose()
        if dl_tag and dl_tag.name == 'dl': dl_tag.decompose()
        if hr_tag and hr_tag.name == 'hr': hr_tag.decompose()
    existing_plots_section_h2 = soup.find('h2', id='generated_plots')
    if existing_plots_section_h2:
        logging.warning("Found existing plots section. Removing it.")
        plots_container = existing_plots_section_h2.find_next_sibling('div')
        hr_tag = existing_plots_section_h2.find_previous_sibling('hr')
        existing_plots_section_h2.decompose()
        if plots_container and plots_container.name == 'div': plots_container.decompose()
        if hr_tag and hr_tag.name == 'hr': hr_tag.decompose()
    existing_script_tag = soup.find('script', string=lambda t: t and 'function sortTable' in t)
    if existing_script_tag:
        logging.warning("Found existing sorting script tag. Removing it.")
        existing_script_tag.decompose()


    # --- Find main H1 and results tables for insertion points ---
    main_h1 = soup.find('h1')
    results_table_main = soup.find('table', id='resultsTableMain')

    # --- Create Table of Contents ---
    toc_div = soup.new_tag('div', id='toc', style="margin-bottom: 20px; padding: 10px; border: 1px solid #eee; background-color: #f9f9f9;")
    toc_title = soup.new_tag('h3', style="margin-top: 0; margin-bottom: 10px;")
    toc_title.string = "Table of Contents"
    toc_div.append(toc_title)
    toc_list = soup.new_tag('ul', style="list-style: none; padding-left: 0;")

    # Define section IDs
    results_table_main_id = "resultsTableMain"
    results_table_secondary_id = "resultsTableSecondary"
    metric_desc_id = "metric_descriptions"
    plots_id = "generated_plots"
    collective_metrics_id = "collective_metrics_section"

    # Add links to tables, collective metrics, descriptions, plots
    if results_table_main: li_table = soup.new_tag('li'); a_table = soup.new_tag('a', href=f'#{results_table_main_id}'); a_table.string = "Individual File Metrics (Main)"; li_table.append(a_table); toc_list.append(li_table)
    results_table_secondary = soup.find('table', id=results_table_secondary_id)
    if results_table_secondary: li_table2 = soup.new_tag('li'); a_table2 = soup.new_tag('a', href=f'#{results_table_secondary_id}'); a_table2.string = "Individual File Metrics (Sentiment, Emotion & Keywords)"; li_table2.append(a_table2); toc_list.append(li_table2)
    li_coll = soup.new_tag('li'); a_coll = soup.new_tag('a', href=f'#{collective_metrics_id}'); a_coll.string = "Collective Metrics"; li_coll.append(a_coll); toc_list.append(li_coll)
    li_desc = soup.new_tag('li'); a_desc = soup.new_tag('a', href=f'#{metric_desc_id}'); a_desc.string = "Metric Descriptions"; li_desc.append(a_desc); toc_list.append(li_desc)
    li_plots = soup.new_tag('li'); a_plots = soup.new_tag('a', href=f'#{plots_id}'); a_plots.string = "Generated Plots (Visualizer)"; li_plots.append(a_plots); toc_list.append(li_plots)
    li_plots_sub = soup.new_tag('ul', style="margin-left: 20px; list-style: disc;")
    for key in plot_order: # Use the corrected plot_order list
         plot_info = all_plot_metadata.get(key)
         if plot_info:
              plot_title = plot_info.get('title', key.replace('_', ' ').title())
              plot_anchor = f"plot_{key}" # Create an anchor ID
              li_sub = soup.new_tag('li'); a_sub = soup.new_tag('a', href=f'#{plot_anchor}'); a_sub.string = plot_title; li_sub.append(a_sub); li_plots_sub.append(li_sub)
    li_plots.append(li_plots_sub)
    toc_div.append(toc_list)

    # --- Insert TOC AFTER the initial summary paragraphs ---
    insertion_point_for_toc = main_h1
    if main_h1:
        last_summary_p = None; current_element = main_h1.find_next_sibling()
        while current_element and current_element.name == 'p': last_summary_p = current_element; current_element = current_element.find_next_sibling()
        if last_summary_p: insertion_point_for_toc = last_summary_p
        else: logging.warning("Could not find summary paragraphs after H1. Inserting TOC directly after H1.")
        if insertion_point_for_toc and insertion_point_for_toc.parent: insertion_point_for_toc.insert_after(toc_div); logging.info("Added Table of Contents to HTML.")
        else: body.insert(0, toc_div); logging.warning("Could not insert TOC after H1/summary paragraphs. Prepended TOC to body.")
    else: body.insert(0, toc_div); logging.warning("Main H1 not found, prepended TOC to body.")


    # --- Construct Metric Descriptions HTML ---
    metrics_dl_tag = soup.new_tag('dl', style='margin-left: 20px;')
    try: from lexicon_analyzer import HEADER_ORDER as ANALYZER_HEADER_ORDER
    except ImportError:
        logging.warning("Could not import HEADER_ORDER from lexicon_analyzer.py. Using fallback list.")
        ANALYZER_HEADER_ORDER = [ # Fallback list - ensure it matches analyzer output
            'Filename', 'Total Words', 'Total Characters', 'Unique Words', 'Average Sentence Length',
            'Flesch Reading Ease', 'Gunning Fog', 'SMOG Index', 'Dale-Chall Score',
            'TTR', 'RTTR', "Herdan's C (LogTTR)", 'MATTR (W=100)', 'MSTTR (W=100)',
            'VOCD', 'MTLD', "Yule's K", "Simpson's D",
            'Lexical Density', 'Distinct-2', 'Repetition-2', 'Distinct-3', 'Repetition-3',
            'Sentiment (VADER Comp)', 'Sentiment (VADER Pos)', 'Sentiment (VADER Neu)', 'Sentiment (VADER Neg)',
            'Sentiment (TextBlob Pol)', 'Sentiment (TextBlob Subj)',
            'Emotion (NRCLex Dominant)', 'Emotion (NRCLex Scores JSON)',
            'Topic (NMF ID)', 'Topic (NMF Prob)',
            'Keywords (TF-IDF Top 5)', 'Keywords (YAKE! Top 5)'
        ]
    all_metric_keys_ordered = ANALYZER_HEADER_ORDER + list(collective_metrics_data.keys())
    processed_keys = set()
    for header in all_metric_keys_ordered:
        if header in processed_keys: continue
        processed_keys.add(header)
        lookup_key = header
        if header.startswith("MATTR"): lookup_key = f'MATTR (W=100)'
        if header.startswith("MSTTR"): lookup_key = f'MSTTR (W=100)'
        description = METRIC_DESCRIPTIONS.get(lookup_key, "No description available.")
        dt = soup.new_tag('dt', style='font-weight: bold; margin-top: 10px;'); dt.string = header
        dd = soup.new_tag('dd', style='margin-left: 20px; margin-bottom: 8px; font-size: 0.95em;')
        wrapped_lines = textwrap.wrap(description, width=120)
        for i, line in enumerate(wrapped_lines): dd.append(NavigableString(line));
        if i < len(wrapped_lines) - 1: dd.append(soup.new_tag('br'))
        metrics_dl_tag.append(dt); metrics_dl_tag.append(dd)

    # --- Construct Plots HTML ---
    plots_container_tag = soup.new_tag('div') # Create a container for plots
    for key in plot_order: # Use the corrected plot_order list
        plot_info = all_plot_metadata.get(key)
        plot_anchor = f"plot_{key}" # Anchor ID for TOC link
        plot_div = soup.new_tag('div', id=plot_anchor, style='margin-bottom: 30px; padding-top: 10px; border-bottom: 1px solid #eee;')
        if plot_info:
            title = plot_info.get('title', key.replace('_', ' ').title())
            desc = plot_info.get('description', 'No description.')
            read = PLOT_INFO.get(key, {}).get('read', 'No reading instructions.')
            h3 = soup.new_tag('h3'); h3.string = title; plot_div.append(h3)
            p_desc = soup.new_tag('p'); p_desc.append(soup.new_tag('b')); p_desc.b.string = "Description: "; p_desc.append(NavigableString(desc)); plot_div.append(p_desc)
            p_read = soup.new_tag('p'); p_read.append(soup.new_tag('b')); p_read.b.string = "How To Read: "; p_read.append(NavigableString(read)); plot_div.append(p_read)

            # --- Add Filename Lists Below Radar, Parallel Coords, and Grouped Emotion ---
            if key in ['grouped_profile_radar', 'parallel_coordinates', 'grouped_emotion_bars'] and grouping_possible:
                # --- PATCH: Corrected style for no indentation ---
                group_list_p = soup.new_tag('p', style="font-size: 0.85em; margin-left: 0px; margin-top: 5px; color: #333;")
                # --- END PATCH ---
                group_list_p.append(soup.new_tag('b'))
                group_list_p.b.string = "Files per MTLD Group:"
                group_list_p.append(soup.new_tag('br'))
                # Use df_numeric index which might be Filename or original index
                for group_name in df_numeric[mtld_group_col].cat.categories:
                    files_in_group_indices = df_numeric[df_numeric[mtld_group_col] == group_name].index
                    # Get original filenames using the map if index is numeric, otherwise use index directly
                    if pd.api.types.is_numeric_dtype(files_in_group_indices):
                        files_in_group_names = [original_filenames_map.get(i, f"Index {i}") for i in files_in_group_indices]
                    else: # Index is likely filename
                        files_in_group_names = files_in_group_indices.tolist()

                    group_text = f"{group_name}: {', '.join(files_in_group_names)}"
                    group_list_p.append(NavigableString(group_text))
                    group_list_p.append(soup.new_tag('br'))
                plot_div.append(group_list_p)
            # --- End Add Filename Lists ---

            # --- Add Scatter Plot Legend Box ---
            if key in scatter_legends:
                legend_info = scatter_legends[key]
                palette = legend_info['palette']
                labels = legend_info['labels']
                # --- PATCH: Removed outer div box, use simple paragraph ---
                legend_p = soup.new_tag('p', style="font-size: 0.8em; margin-left: 0px; margin-top: 5px; line-height: 1.4;")
                legend_title = soup.new_tag('b'); legend_title.string = "Filename Legend:"; legend_p.append(legend_title); legend_p.append(soup.new_tag('br'))
                for i, label in enumerate(labels):
                    color = palette[i]
                    hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    color_span = soup.new_tag('span', style=f"display: inline-block; width: 10px; height: 10px; background-color: {hex_color}; margin-right: 5px; border: 1px solid #ccc; vertical-align: middle;")
                    legend_p.append(color_span)
                    legend_p.append(NavigableString(f" {label}"))
                    legend_p.append(soup.new_tag('br'))
                plot_div.append(legend_p)
                # --- END PATCH ---
            # --- End Add Scatter Legend Box ---

        else: # Fallback if metadata missing
             title = key.replace('_', ' ').title()
             h3 = soup.new_tag('h3'); h3.string = title; plot_div.append(h3)
             p_info = soup.new_tag('p'); p_info.string = "Plot info missing."; plot_div.append(p_info)

        if key in plot_data_base64 and plot_data_base64[key]:
            # <<< START PATCH for clustered_heatmap styling >>>
            if key == 'clustered_heatmap':
                # Set width to 50%, remove max-width, keep other styles
                img_style = "width: 120%; height: auto; border: 1px solid #ccc; margin-top: 10px; display: block; margin-left: auto; margin-right: auto; background-color: white; padding: 1px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
            else:
                # Keep original style for other images
                img_style = "max-width: 90%; height: auto; border: 1px solid #ccc; margin-top: 10px; display: block; margin-left: auto; margin-right: auto; background-color: white; padding: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
            # <<< END PATCH for clustered_heatmap styling >>>
            img_tag = soup.new_tag('img', src=f"data:image/png;base64,{plot_data_base64[key]}", alt=title, style=img_style)
            plot_div.append(img_tag)
        else:
            p_err = soup.new_tag('p', style='color: red;'); p_err.string = f"Plot generation failed or skipped for '{title}'. Check logs/data."; plot_div.append(p_err)
        plots_container_tag.append(plot_div)

    # --- JavaScript for Table Sorting (Remains the same as previous version) ---
    js_code = """
function sortTable(tableId, columnIndex, isNumeric) {
    const table = document.getElementById(tableId);
    if (!table) { console.error("Table with ID '" + tableId + "' not found."); return; }
    const tbody = table.tBodies[0];
    if (!tbody) { console.error("Table body not found for table " + tableId); return; }
    const rows = Array.from(tbody.rows);
    const headerCell = table.tHead.rows[0].cells[columnIndex];
    const currentSort = headerCell.getAttribute('data-sort-direction') || 'none';
    let direction = 'asc';

    if (currentSort === 'asc') { direction = 'desc'; }
    else { direction = 'asc'; }

    // Reset other headers in the *same* table
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
            // Handle FAIL, NaN, INF as lowest value for sorting purposes
            const parseNumeric = (val) => {
                if (val === 'FAIL' || val === 'NaN' || val === 'N/A (<=1 file)') return -Infinity;
                if (val === 'INF') return Infinity;
                const num = parseFloat(val);
                return isNaN(num) ? -Infinity : num;
            };
            valA = parseNumeric(cellA);
            valB = parseNumeric(cellB);
        } else {
             // Case-insensitive string comparison
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

    // Re-append rows in sorted order
    rows.forEach(row => tbody.appendChild(row));
}

document.addEventListener('DOMContentLoaded', () => {
    const tableIds = ['resultsTableMain', 'resultsTableSecondary']; // IDs of the tables to make sortable

    // Define base names of numeric headers (excluding window sizes etc.)
    const numericHeadersBase = [
        "Total Words", "Total Characters", "Unique Words", "Average Sentence Length",
        "Flesch Reading Ease", "Gunning Fog", "SMOG Index", "Dale-Chall Score", // Readability
        "TTR", "RTTR", "Herdan's C (LogTTR)", "MATTR", "MSTTR", "VOCD", "MTLD", "Yule's K", "Simpson's D", // Diversity
        "Lexical Density", "Distinct-2", "Repetition-2", "Distinct-3", "Repetition-3", // Syntactic/Phrase
        "Sentiment (VADER Comp)", "Sentiment (VADER Pos)", "Sentiment (VADER Neu)", "Sentiment (VADER Neg)", // Sentiment
        "Sentiment (TextBlob Pol)", "Sentiment (TextBlob Subj)",
        "Topic (NMF ID)", "Topic (NMF Prob)", // Topics (ID treated as numeric for sorting)
        "Self-BLEU", "Avg Pairwise Cosine Similarity", "Avg Distance to Centroid" // Collective
    ];

    tableIds.forEach(tableId => {
        const table = document.getElementById(tableId);
        if (table && table.tHead && table.tHead.rows.length > 0) {
            const headers = table.tHead.rows[0].cells;
            for (let i = 0; i < headers.length; i++) {
                const header = headers[i];
                const baseHeaderText = header.textContent.split(' (W=')[0].trim();
                const isNumeric = numericHeadersBase.includes(baseHeaderText);
                header.classList.add('sortable-header');
                header.addEventListener('click', ((id, index, numeric) => {
                    return () => sortTable(id, index, numeric);
                })(tableId, i, isNumeric));
            }
            console.log("Table sorting enabled for: " + tableId);
        } else {
            console.error("Could not find table or table header for ID: " + tableId);
        }
    });
});
"""
    js_tag = soup.new_tag('script')
    js_tag.string = js_code

    # --- Modify HTML Structure ---
    last_inserted = soup.find('table', id='resultsTableSecondary')
    if not last_inserted: last_inserted = soup.find('table', id='resultsTableMain')

    # --- Move Collective Metrics ---
    collective_h2 = soup.find('h2', string=re.compile(r'Collective Metrics'))
    collective_ul = None; extracted_h2 = None; extracted_ul = None
    if collective_h2:
        collective_ul = collective_h2.find_next_sibling('ul')
        if collective_ul:
            logging.info("Found Collective Metrics section. Moving it.")
            extracted_h2 = collective_h2.extract(); extracted_ul = collective_ul.extract()
        else: logging.warning("Found Collective Metrics H2, but not the following UL. Cannot move."); extracted_h2 = collective_h2.extract()
    else: logging.warning("Collective Metrics section H2 not found in the original HTML.")

    # Insert Collective Metrics (if found) after the last table
    if last_inserted and last_inserted.parent:
        current_anchor = last_inserted
        if extracted_h2: current_anchor.insert_after(extracted_h2); current_anchor = extracted_h2
        if extracted_ul: extracted_ul['id'] = collective_metrics_id; current_anchor.insert_after(extracted_ul); current_anchor = extracted_ul
        last_inserted = current_anchor
        logging.info(f"Collective Metrics moved. Next insertion point: {last_inserted.name} id={last_inserted.get('id')}")
    else:
        logging.error("Cannot find results table(s) to insert Collective Metrics after.")
        if extracted_h2: body.append(extracted_h2)
        if extracted_ul: body.append(extracted_ul)
        last_inserted = body # Fallback

    # --- Append new sections (Descriptions, Plots, JS) ---
    if last_inserted and last_inserted.parent: # Ensure insertion point is valid
        h2_metrics = soup.new_tag('h2', id=metric_desc_id); h2_metrics.string = "Metric Descriptions"; hr_metrics = soup.new_tag('hr')
        last_inserted.insert_after(hr_metrics); hr_metrics.insert_after(h2_metrics); h2_metrics.insert_after(metrics_dl_tag)
        last_inserted = metrics_dl_tag

        h2_plots = soup.new_tag('h2', id=plots_id); h2_plots.string = "Generated Plots (Visualizer)"; hr_plots = soup.new_tag('hr')
        last_inserted.insert_after(hr_plots); hr_plots.insert_after(h2_plots); h2_plots.insert_after(plots_container_tag)
        last_inserted = plots_container_tag

        last_inserted.insert_after(js_tag)
        logging.info("Appended new descriptions, plots, and JS to HTML soup.")
    else:
        logging.warning("Could not find a valid insertion point after moving Collective Metrics. Appending remaining sections to body.")
        body.append(soup.new_tag('hr'))
        h2_metrics = soup.new_tag('h2', id=metric_desc_id); h2_metrics.string = "Metric Descriptions"; body.append(h2_metrics)
        body.append(metrics_dl_tag)
        body.append(soup.new_tag('hr'))
        h2_plots = soup.new_tag('h2', id=plots_id); h2_plots.string = "Generated Plots (Visualizer)"; body.append(h2_plots)
        body.append(plots_container_tag)
        body.append(js_tag)

    # --- WORKAROUND: Manually Insert Collective Metric Descriptions ---
    logging.info("Attempting Collective Metric description insertion workaround...")
    try:
        metrics_h2 = soup.find('h2', id='metric_descriptions')
        if metrics_h2:
            metrics_section_dl = metrics_h2.find_next_sibling('dl')
            if metrics_section_dl:
                all_dt_tags = metrics_section_dl.find_all('dt')
                if all_dt_tags:
                    last_dt = all_dt_tags[-1]
                    logging.debug(f"Found last <dt> tag: {last_dt.string}")
                    last_dd = last_dt.find_next_sibling('dd')
                    if last_dd:
                        logging.debug("Found last <dd> tag.")
                        next_dt = last_dd.find_next_sibling('dt')
                        if not next_dt or not re.match(r'^\s*Self-BLEU\s*$', next_dt.get_text(strip=True), re.IGNORECASE):
                            collective_keys_to_describe = [k for k in collective_metrics_data.keys() if k in METRIC_DESCRIPTIONS]
                            if not collective_keys_to_describe: logging.warning("No descriptions found for parsed collective metrics.")
                            else:
                                logging.info(f"Creating and inserting descriptions for collective metrics: {collective_keys_to_describe}")
                                current_insert_point = last_dd
                                for coll_key in collective_keys_to_describe:
                                    coll_desc_str = METRIC_DESCRIPTIONS.get(coll_key)
                                    if coll_desc_str:
                                        new_dt = soup.new_tag('dt', style='font-weight: bold; margin-top: 10px;'); new_dt.string = coll_key
                                        new_dd = soup.new_tag('dd', style='margin-left: 20px; margin-bottom: 8px; font-size: 0.95em;')
                                        wrapped_lines = textwrap.wrap(coll_desc_str, width=120)
                                        for i, line in enumerate(wrapped_lines): new_dd.append(NavigableString(line));
                                        if i < len(wrapped_lines) - 1: new_dd.append(soup.new_tag('br'))
                                        current_insert_point.insert_after(new_dd); current_insert_point.insert_after(new_dt)
                                        current_insert_point = new_dd # Move insert point forward
                                logging.info("Successfully inserted collective metric descriptions via workaround.")
                        else: logging.info("Collective metric descriptions already appear to be present. Skipping insertion.")
                    else: logging.warning("Could not find <dd> tag immediately following last <dt>. Cannot insert collective descriptions.")
                else: logging.warning("Could not find any <dt> tags within metrics DL. Cannot insert collective descriptions.")
            else: logging.warning("Could not find metric descriptions DL tag following H2. Cannot insert collective descriptions.")
        else: logging.warning("Could not find metric descriptions H2 tag. Cannot insert collective descriptions.")
    except Exception as workaround_e:
        logging.error(f"Error during collective metric description insertion workaround: {workaround_e}", exc_info=True)

    # --- Write the final modified HTML ---
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))
    logging.info("Successfully updated and saved final HTML file.")


except FileNotFoundError:
    logging.error(f"HTML file '{html_path}' not found. Cannot append content.")
except Exception as e:
    logging.error(f"Failed to update HTML file: {e}", exc_info=True)

# --- Save Plot Metadata JSON ---
plot_metadata_path = os.path.join(target_folder, PLOT_METADATA_FILENAME)
try:
    with open(plot_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_plot_metadata, f, indent=4)
    logging.info(f"Saved plot metadata to {plot_metadata_path}")
except Exception as e:
    logging.error(f"Failed to save plot metadata: {e}")


logging.info("Visualization script finished.")