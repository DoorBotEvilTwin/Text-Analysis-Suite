# lexicon_processor.py
# Patched to update HTML incrementally after each LLM label/summary call.
# WARNING: This will significantly increase runtime and I/O load.
# Patched update_html_report to insert AFTER transformer's section if present.
# REMOVED Markdown to HTML conversion for LLM outputs, using basic <br> instead.
# Added token multiplier for final synthesis call.
# PATCHED: Removed normalized metrics from file summary prompt to reduce hallucination.
# PATCHED: update_html_report now adds links to processor sections in TOC.
# PATCHED: Fixed imputation shape mismatch error in preprocess_data.
# PATCHED: Fixed file summary loop to iterate over processed data index.
# PATCHED: Updated KEY_METRICS lists to include new numeric metrics.
# PATCHED: Corrected data loading and loop logic to exclude collective metrics.
# PATCHED: Added checks to prevent errors on None during HTML update.
# PATCHED: Renamed file summary sub-section title based on mode and content summary status.
# PATCHED: Ensured content summary logic executes if enabled.
# PATCHED: Enabled CONTENT_SUMMARY_ENABLED flag by default.
# PATCHED: Integrated content summaries into the final synthesis prompt context.
# PATCHED: Increased bottom margin for <dd> elements in file summaries for better spacing. # <<< THIS FIX

import pandas as pd
import numpy as np
import os
import sys # Added for command-line args
import logging
import json
import time
import configparser
import re # Added for HTML searching
from datetime import datetime
from collections import defaultdict # Added for summarizing
import threading # Added for timed input

# --- Dependencies with Checks ---
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: scikit-learn is required. Install: pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: scipy is required for clustering. Install: pip install scipy")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup, NavigableString # Import NavigableString
    BS4_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: beautifulsoup4 is required for HTML parsing. Install: pip install beautifulsoup4")
    sys.exit(1)

# --- Requests library needed for local LLM calls ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not found (pip install requests). Local LLM mode will be unavailable.")

# --- REMOVED Markdown library import ---


# --- Configuration Files ---
INI_FILENAME = "lexicon_settings.ini"
CSV_FILENAME = "_LEXICON.csv"
HTML_FILENAME = "_LEXICON.html" # Target HTML file to modify
LOG_FILENAME = "_LEXICON_errors.log" # Appends to the same log
TXT_SUBDIR = "txt" # Subdirectory containing the original .txt files

# --- AI/Local LLM Configuration ---
MAX_RETRIES = 2
RETRY_DELAY = 3 # Shorter delay for local calls
DEFAULT_MAX_TOKENS = 256 # Default if not in INI
SYNTHESIS_MAX_TOKENS_MULTIPLIER = 8.0 # Multiplier for final synthesis token limit
# --- NEW: Config for Content Summarization ---
# <<< FIX: Enabled by default based on updated policy >>>
CONTENT_SUMMARY_ENABLED = True # Set to True to enable (Requires LLM access to raw text)
CONTENT_SUMMARY_MAX_TOKENS = 512 # Token limit for content summaries

# --- Clustering Configuration ---
CLUSTER_METRIC = 'euclidean'
CLUSTER_METHOD = 'ward'
N_CLUSTERS_TO_LABEL = 5
N_SPLITS_TO_ANALYZE = 3 # How many top splits to analyze for the final summary

# --- Metrics for Interpretation (Updated) ---
KEY_METRICS_FOR_CLUSTER_LABEL = [
    'MTLD', 'VOCD', 'Average Sentence Length', 'Flesch Reading Ease',
    'Distinct-2', 'Repetition-2', 'Lexical Density', 'Yule\'s K', 'Simpson\'s D',
    'Gunning Fog', 'Sentiment (VADER Comp)', 'Sentiment (TextBlob Pol)', 'Topic (NMF Prob)' # Added new
]
KEY_METRICS_FOR_FILE_SUMMARY = [
    'MTLD', 'VOCD', 'Average Sentence Length', 'Flesch Reading Ease',
    'Distinct-2', 'Repetition-2', 'Lexical Density', 'TTR', 'RTTR',
    'Gunning Fog', 'Sentiment (VADER Comp)', 'Sentiment (TextBlob Pol)', 'Topic (NMF Prob)' # Added new
]
KEY_METRICS_FOR_SPLIT_ANALYSIS = [
    'MTLD', 'VOCD', 'Average Sentence Length', 'Flesch Reading Ease',
    'Distinct-2', 'Repetition-2', 'Sentiment (VADER Comp)' # Added new
]


# --- Basic Logging Setup ---
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
log_level = logging.INFO # Change to logging.DEBUG if needed
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'), # Append mode 'a'
        logging.StreamHandler()
    ]
)
logging.info(f"\n{'='*10} Starting Offline Interpretation Run {'='*10}")

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


# --- Helper Functions (Existing + New) ---

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

# --- PATCHED: load_data respects separator AND filters collective keys ---
def load_data(csv_path):
    """Loads the main metrics data from the CSV file, stopping before collective metrics."""
    separator_idx = find_separator_index(csv_path)
    df = None
    try:
        if separator_idx is None:
            logging.warning(f"Could not find collective metrics separator in {csv_path}. Reading entire file.")
            df = pd.read_csv(csv_path)
        else:
            rows_to_read = separator_idx
            logging.info(f"Separator line found at index {separator_idx}. Reading {rows_to_read} data rows.")
            df = pd.read_csv(csv_path, nrows=rows_to_read)

        df.dropna(how='all', inplace=True) # Drop rows where ALL values are NaN

        # <<< FIX: Explicitly filter known collective keys AFTER loading nrows >>>
        # Parse collective keys first to know what to filter
        temp_collective_metrics = {}
        if separator_idx is not None:
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    start_line = separator_idx + 1
                    if start_line < len(lines):
                        if not lines[start_line].strip(): start_line += 1
                        for line in lines[start_line:]:
                            line = line.strip()
                            if not line: continue
                            match = re.match(r'^"?([^"]+)"?,\s*(.*)$', line)
                            if match: temp_collective_metrics[match.group(1).strip()] = True # Just need the keys
            except Exception: pass # Ignore errors here, just trying to get keys

        known_collective_keys = list(temp_collective_metrics.keys()) + ["--- Collective Metrics (E) ---"]
        if 'Filename' in df.columns:
            initial_rows = len(df)
            df = df[~df['Filename'].isin(known_collective_keys)].copy()
            if len(df) < initial_rows:
                logging.info(f"Filtered out {initial_rows - len(df)} collective metric rows from main data section.")
        # <<< END FIX >>>

        logging.info(f"Loaded {df.shape[0]} data rows from CSV.")
        return df
    except FileNotFoundError:
        logging.error(f"CSV file '{csv_path}' not found.")
        return None
    except Exception as e:
        logging.error(f"Could not read CSV file: {e}", exc_info=True)
        return None

def preprocess_data(df):
    """Prepares data for clustering and interpretation."""
    if df is None: return None, None, None, None, None # Added None for df_numeric_orig

    # Convert to numeric, coercing errors and handling placeholders
    df_numeric = df.copy()
    numeric_cols = [col for col in df.columns if col != 'Filename']
    for col in numeric_cols:
        df_numeric[col] = df_numeric[col].replace(['FAIL', 'N/A (<=1 file)'], np.nan)
        df_numeric[col] = df_numeric[col].replace(['INF'], np.inf)
        df_numeric[col] = df_numeric[col].replace(['NaN'], np.nan)
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Separate filename if present
    original_filenames = None
    df_numeric_orig_for_stats = df_numeric.copy() # Keep copy before dropping filename for stats

    if 'Filename' in df_numeric.columns:
        original_filenames = df_numeric['Filename'].tolist() # Keep original names
        try:
            # Keep Filename in df_numeric_orig_for_stats but drop from df_numeric for processing
            df_numeric = df_numeric.drop(columns=['Filename'])
        except KeyError: pass
    else:
        logging.warning("No 'Filename' column found. Using index.")
        original_filenames = [f"File Index {i}" for i in df_numeric.index]
        # Add index as Filename to df_numeric_orig_for_stats for consistency
        df_numeric_orig_for_stats['Filename'] = original_filenames

    # Create anonymized IDs (still useful for prompts even if not sent externally)
    anonymized_ids = [f"File {i+1}" for i in range(len(original_filenames))]

    # Select only numeric columns for processing
    df_numeric_only = df_numeric.select_dtypes(include=np.number)
    if df_numeric_only.empty:
        logging.error("No numeric columns found after attempting conversion.")
        return None, None, None, None, None

    # Impute missing values (using mean for simplicity)
    imputer = SimpleImputer(strategy='mean')
    logging.debug(f"Columns before imputation: {df_numeric_only.columns.tolist()}")
    df_imputed_values = imputer.fit_transform(df_numeric_only)
    imputed_columns = imputer.get_feature_names_out(df_numeric_only.columns)
    logging.debug(f"Columns after imputation: {imputed_columns.tolist()}")

    if np.isnan(df_imputed_values).any():
         logging.warning("NaNs detected *after* imputation with mean. This might indicate columns with all NaNs. Filling remaining NaNs with 0.")
         df_imputed_values = np.nan_to_num(df_imputed_values, nan=0.0)

    df_imputed = pd.DataFrame(df_imputed_values,
                              columns=imputed_columns,
                              index=df_numeric_only.index) # Use original index

    # Normalize data (0-1 scaling) for consistent interpretation and clustering
    scaler = MinMaxScaler()
    if df_imputed.isnull().values.any():
         logging.error("NaNs still present before scaling, despite imputation attempts. Aborting.")
         return None, None, None, None, None

    df_normalized_values = scaler.fit_transform(df_imputed)
    df_normalized = pd.DataFrame(df_normalized_values,
                                 columns=imputed_columns,
                                 index=df_imputed.index) # Use original index

    logging.info(f"Data preprocessed: Imputed NaNs, Normalized {df_normalized.shape[1]} numeric columns.")
    # Ensure df_numeric_orig_for_stats has the Filename column for later lookup
    return df_numeric_orig_for_stats, df_imputed, df_normalized, original_filenames, anonymized_ids

def perform_clustering(df_processed):
    """Performs hierarchical clustering."""
    if not SCIPY_AVAILABLE or df_processed is None or df_processed.empty:
        logging.error("Clustering cannot be performed (SciPy unavailable or no data).")
        return None
    if len(df_processed) < 2:
        logging.warning("Clustering requires at least 2 data points. Skipping clustering.")
        return None
    try:
        if not np.all(np.isfinite(df_processed.values)):
             logging.error("Non-finite values (NaN/inf) detected in data passed to clustering. Aborting clustering.")
             return None

        logging.debug(f"Performing clustering on data with shape: {df_processed.shape}")
        row_dist = pdist(df_processed.values, metric=CLUSTER_METRIC)
        row_linkage = linkage(row_dist, method=CLUSTER_METHOD, metric=CLUSTER_METRIC)
        logging.info(f"Hierarchical clustering performed using metric='{CLUSTER_METRIC}', method='{CLUSTER_METHOD}'. Linkage matrix shape: {row_linkage.shape}")
        return row_linkage
    except ValueError as ve:
         logging.error(f"ValueError during clustering (check data): {ve}", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"Error during clustering: {e}", exc_info=True)
        return None

def get_flat_clusters(linkage_matrix, n_clusters, num_original_points):
    """Gets flat cluster assignments from the linkage matrix."""
    if linkage_matrix is None:
        logging.warning("Linkage matrix is None, cannot derive flat clusters.")
        return None
    if n_clusters <= 0:
        logging.warning(f"Invalid number of clusters requested ({n_clusters}). Cannot derive flat clusters.")
        return None
    if n_clusters >= num_original_points:
        logging.warning(f"Requested clusters ({n_clusters}) >= number of points ({num_original_points}). Assigning each point to its own cluster.")
        return np.arange(1, num_original_points + 1)

    try:
        logging.debug(f"Deriving {n_clusters} flat clusters from linkage matrix (shape {linkage_matrix.shape}) for {num_original_points} points.")
        assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        if len(assignments) != num_original_points:
             logging.error(f"Mismatch in cluster assignment length ({len(assignments)}) and original points ({num_original_points}).")
             return None
        logging.info(f"Derived {n_clusters} flat clusters. Assignment counts: {np.unique(assignments, return_counts=True)}")
        return assignments
    except Exception as e:
        logging.error(f"Error deriving flat clusters: {e}", exc_info=True)
        return None

def get_cluster_avg_metrics(df_processed, cluster_assignments, key_metrics):
    """Calculates the average metric profile for each cluster."""
    if df_processed is None or cluster_assignments is None:
        logging.warning("Cannot calculate cluster averages: missing data or assignments.")
        return {}
    avg_metrics = {}
    unique_clusters = sorted(np.unique(cluster_assignments))
    if len(df_processed) != len(cluster_assignments):
        logging.error(f"Mismatch between data length ({len(df_processed)}) and cluster assignments length ({len(cluster_assignments)}). Cannot calculate averages.")
        return {}

    # Ensure df_processed has a simple range index for iloc below
    df_with_clusters = df_processed.reset_index(drop=True).copy()
    df_with_clusters['ClusterID'] = cluster_assignments

    logging.debug(f"Calculating average metrics for clusters: {unique_clusters}")
    for cluster_id in unique_clusters:
        cluster_mask = df_with_clusters['ClusterID'] == cluster_id
        cluster_data = df_with_clusters.loc[cluster_mask]

        if cluster_data.empty:
            logging.warning(f"No data found for Cluster {cluster_id}. Skipping average calculation.")
            continue

        relevant_cols = [col for col in key_metrics if col in cluster_data.columns]
        if relevant_cols:
            cluster_means = cluster_data[relevant_cols].mean(skipna=True).to_dict()
            avg_metrics[cluster_id] = cluster_means
            logging.debug(f"Cluster {cluster_id} average metrics (first 3): {list(cluster_means.items())[:3]}")
        else:
            avg_metrics[cluster_id] = {}
            logging.warning(f"No key metrics found in data columns for cluster {cluster_id}.")

    logging.info(f"Calculated average profiles for {len(avg_metrics)} clusters.")
    return avg_metrics

def get_dataset_stats(df_numeric_orig_with_filename, key_metrics):
    """Calculates overall mean and std dev for key metrics from original numeric data."""
    stats = {}
    if df_numeric_orig_with_filename is None or df_numeric_orig_with_filename.empty:
        logging.warning("Cannot calculate dataset stats: original numeric data is missing or empty.")
        return {}

    # Select only numeric columns, excluding potential non-numeric 'Filename'
    numeric_cols_for_stats = df_numeric_orig_with_filename.select_dtypes(include=np.number).columns
    df_numeric_for_stats = df_numeric_orig_with_filename[numeric_cols_for_stats]

    relevant_cols = [col for col in key_metrics if col in df_numeric_for_stats.columns]
    if relevant_cols and not df_numeric_for_stats.empty:
        stats['mean'] = df_numeric_for_stats[relevant_cols].mean(skipna=True).to_dict()
        stats['std'] = df_numeric_for_stats[relevant_cols].std(skipna=True).to_dict()
        stats['min'] = df_numeric_for_stats[relevant_cols].min(skipna=True).to_dict()
        stats['max'] = df_numeric_for_stats[relevant_cols].max(skipna=True).to_dict()
        logging.info(f"Calculated dataset statistics (mean, std, min, max) for {len(relevant_cols)} key metrics.")
        logging.debug(f"Dataset Mean MTLD (example): {stats.get('mean', {}).get('MTLD', 'N/A')}")
    else:
        logging.warning("Could not calculate dataset stats - no relevant key metrics found or numeric data empty.")
    return stats


def generate_cluster_label_rules(cluster_id, avg_metrics_norm):
    """Generates a descriptive label for a cluster using simple rules."""
    if not avg_metrics_norm: return "N/A (No metrics)"

    # Define thresholds (on normalized 0-1 scale) - Adjust these based on observation!
    high_thr = 0.65
    low_thr = 0.35
    mid_range = (low_thr, high_thr)

    # Get key metric values (handle missing keys gracefully)
    mtld = avg_metrics_norm.get('MTLD', 0.5)
    avg_len = avg_metrics_norm.get('Average Sentence Length', 0.5)
    flesch = avg_metrics_norm.get('Flesch Reading Ease', 0.5) # Higher = easier
    rep2 = avg_metrics_norm.get('Repetition-2', 0.5) # Higher = more repetitive
    distinct2 = avg_metrics_norm.get('Distinct-2', 0.5) # Higher = more distinct pairs
    vader_comp = avg_metrics_norm.get('Sentiment (VADER Comp)', 0.0) # -1 to 1, 0 is neutral

    # Simple Logic (can be expanded significantly)
    label = "Mixed/Average Profile" # Default

    # Combine diversity and complexity
    if mtld > high_thr and distinct2 > high_thr:
        if avg_len > high_thr and flesch < low_thr: label = "Diverse & Complex Syntax"
        elif avg_len < low_thr and flesch > high_thr: label = "Diverse & Simple Syntax"
        else: label = "High Lexical & Phrase Diversity"
    elif mtld < low_thr and rep2 > high_thr:
        if avg_len < low_thr and flesch > high_thr: label = "Repetitive & Simple Syntax"
        else: label = "Repetitive, Low Diversity"
    # Focus on readability/complexity
    elif flesch < low_thr and avg_len > high_thr: label = "Low Readability, Long Sentences"
    elif flesch > high_thr and avg_len < low_thr: label = "High Readability, Short Sentences"
    # Focus on diversity/repetition alone
    elif mtld > high_thr: label = "High Lexical Diversity"
    elif mtld < low_thr: label = "Low Lexical Diversity"
    elif rep2 > high_thr: label = "High Repetition (Bigrams)"
    elif distinct2 < low_thr: label = "Low Phrase Variety (Bigrams)"

    # Add sentiment modifier if strong
    if vader_comp > 0.5: # Significantly positive (adjust threshold as needed)
        label += ", Positive Sentiment"
    elif vader_comp < -0.3: # Significantly negative (adjust threshold as needed)
        label += ", Negative Sentiment"

    logging.debug(f"Rule-based label for Cluster {cluster_id}: '{label}'")
    return label

def generate_file_summary_rules(anonymized_id, file_metrics_raw, dataset_stats):
    """Generates a comparative summary for a single file using simple rules."""
    if not file_metrics_raw or not dataset_stats or 'mean' not in dataset_stats or 'std' not in dataset_stats:
        return "Dataset statistics unavailable for comparison."

    summary_parts = []
    mean = dataset_stats['mean']
    std = dataset_stats['std']

    # Compare key metrics (including new ones if available in stats)
    metrics_to_compare = ['MTLD', 'Average Sentence Length', 'Flesch Reading Ease', 'Repetition-2', 'Sentiment (VADER Comp)']
    for metric in metrics_to_compare:
        if metric not in file_metrics_raw or metric not in mean or metric not in std:
            continue # Skip if metric or stats missing

        val = file_metrics_raw.get(metric) # Use .get for safety
        m = mean.get(metric)
        s = std.get(metric)

        # Check if any value is None or NaN before calculation
        if val is None or m is None or s is None or pd.isna(val) or pd.isna(m) or pd.isna(s) or s == 0:
            continue # Cannot compare

        try:
            z_score = (val - m) / s
        except TypeError:
             logging.warning(f"TypeError calculating z-score for {metric} in {anonymized_id}. Values: val={val}, mean={m}, std={s}")
             continue # Skip if calculation fails

        desc = ""
        if z_score > 1.5: desc = "significantly above average"
        elif z_score > 0.5: desc = "above average"
        elif z_score < -1.5: desc = "significantly below average"
        elif z_score < -0.5: desc = "below average"

        if desc:
            if metric == 'Flesch Reading Ease': desc += f" (implying {'easier' if z_score > 0 else 'harder'} reading)"
            elif metric == 'Repetition-2': desc += f" (implying {'more' if z_score > 0 else 'less'} bigram repetition)"
            elif metric == 'Sentiment (VADER Comp)': desc += f" (implying {'more positive' if z_score > 0 else 'more negative'} sentiment)"
            summary_parts.append(f"{metric} ({val:.2f}) is {desc}.")

    # Add TTR vs MTLD check
    ttr = file_metrics_raw.get('TTR')
    mtld = file_metrics_raw.get('MTLD')
    if pd.notna(ttr) and pd.notna(mtld) and 'TTR' in mean and 'MTLD' in mean:
        ttr_mean = mean.get('TTR')
        mtld_mean = mean.get('MTLD')
        ttr_std = std.get('TTR', 1) # Default std to 1 if missing
        mtld_std = std.get('MTLD', 1)
        if pd.notna(ttr_mean) and pd.notna(mtld_mean) and ttr_std != 0 and mtld_std != 0:
            ttr_z = (ttr - ttr_mean) / ttr_std
            mtld_z = (mtld - mtld_mean) / mtld_std
            if ttr_z > 1.0 and mtld_z < 0.0:
                summary_parts.append("High TTR but average/low MTLD might suggest initial novelty fading quickly or short text length effects.")

    if not summary_parts:
        return "Metrics are generally around the dataset average."
    else:
        # Limit summary length
        return " ".join(summary_parts[:3]) # Join first 2-3 notable points


def call_local_llm_api(prompt, llm_config, max_tokens_override=None):
    """
    Calls a local LLM API (e.g., KoboldCpp) via HTTP POST.
    Allows overriding max_tokens.
    Uses 'processor_model' from config if available, else 'model'.
    """
    if not REQUESTS_AVAILABLE or requests is None:
        return "Error: 'requests' library not installed (pip install requests)."

    api_base = llm_config.get('api_base')
    if not api_base:
        return "Error: 'api_base' URL not configured in [LocalLLM] section of INI file."

    current_max_tokens = max_tokens_override if max_tokens_override is not None else int(llm_config.get('max_tokens', DEFAULT_MAX_TOKENS))
    model_name = llm_config.get('processor_model', llm_config.get('model', 'local-model')) # Prioritize processor_model

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(llm_config.get('temperature', 0.0)),
        "max_tokens": current_max_tokens,
    }
    if 'top_p' in llm_config: payload['top_p'] = float(llm_config['top_p'])
    if 'top_k' in llm_config: payload['top_k'] = int(llm_config['top_k'])

    payload = {k: v for k, v in payload.items() if v is not None}
    endpoint = api_base

    logging.debug(f"Attempting local LLM call to: {endpoint}")
    logging.debug(f"Payload: {json.dumps(payload, indent=2)}")

    last_error = None
    for attempt in range(MAX_RETRIES):
        response = None
        try:
            logging.info(f"Sending request to LLM (Attempt {attempt+1}/{MAX_RETRIES}) with NO client-side timeout...")
            response = requests.post(endpoint, json=payload)
            logging.debug(f"Local LLM request sent. Status Code: {response.status_code if response else 'N/A'}")
            if response is None: raise requests.exceptions.RequestException("Request returned None unexpectedly.")
            response.raise_for_status()

            raw_response_text = response.text
            logging.debug(f"Raw response text received: {raw_response_text[:500]}...")
            response_json = response.json()
            logging.debug(f"Local LLM Response JSON: {json.dumps(response_json, indent=2)}")

            if 'choices' in response_json and len(response_json['choices']) > 0:
                 message = response_json['choices'][0].get('message', {})
                 content = message.get('content')
                 if content:
                      logging.debug("Local LLM call successful.")
                      return content.strip()
                 else: last_error = "Error: 'content' field missing in response message."; logging.warning(last_error)
            else: last_error = "Error: Expected 'choices' array missing or empty in response."; logging.warning(last_error)

        except requests.exceptions.ConnectionError as e: last_error = f"Error: Connection failed to {endpoint} ({e}). Is the backend running?"; logging.error(last_error); break
        except requests.exceptions.RequestException as e: response_text_on_error = response.text[:500] if response is not None else "N/A"; status_code = response.status_code if response is not None else "N/A"; last_error = f"Error: Request failed (Status: {status_code}, Error: {e}). Response: {response_text_on_error}"; logging.error(f"Local LLM request failed (Attempt {attempt + 1}/{MAX_RETRIES}): {last_error}")
        except json.JSONDecodeError as e: raw_text_on_decode_error = response.text if response is not None else "N/A"; last_error = f"Error: Failed to decode JSON response from {endpoint}."; logging.error(f"{last_error} Raw Response: '{raw_text_on_decode_error[:500]}...'", exc_info=True); break
        except Exception as e: last_error = f"Error: Unexpected error during local LLM call ({type(e).__name__}: {e})"; logging.error(last_error, exc_info=True); break

        if attempt < MAX_RETRIES - 1: logging.debug(f"Retrying in {RETRY_DELAY}s..."); time.sleep(RETRY_DELAY)

    return last_error if last_error else "Error: LLM call failed after multiple retries."


def generate_cluster_label_prompt(cluster_id, avg_metrics):
    """Generates the prompt for cluster labeling."""
    if not avg_metrics: return None
    valid_metrics = {k: v for k, v in avg_metrics.items() if pd.notna(v) and k in KEY_METRICS_FOR_CLUSTER_LABEL}
    if not valid_metrics:
        logging.warning(f"No valid (non-NaN) average metrics found for Cluster {cluster_id} from the key list.")
        return None

    metrics_str = "\n".join([f"- {k}: {v:.3f}" for k, v in valid_metrics.items()])
    sorted_metrics = sorted(valid_metrics.items(), key=lambda item: item[1])
    lowest_3 = sorted_metrics[:3]
    highest_3 = sorted_metrics[-3:][::-1]

    prompt = f"""
Analyze the average *normalized* (0-1 scale) linguistic metrics for Cluster {cluster_id}:
{metrics_str}

Key metrics indicating potential characteristics:
- Highest normalized values: {', '.join([f'{k} ({v:.3f})' for k, v in highest_3])}
- Lowest normalized values: {', '.join([f'{k} ({v:.3f})' for k, v in lowest_3])}

Based *only* on these average normalized metrics, provide a concise, descriptive label (max 10 words) summarizing the dominant linguistic profile of files in this cluster. Focus on aspects like lexical diversity, sentence complexity, readability, sentiment, and repetition. Examples: 'High Diversity, Complex Syntax, Neutral Sentiment', 'Simple Vocabulary, Repetitive, Positive', 'Balanced Profile', 'Low Readability, Varied Phrasing, Negative'.
"""
    return prompt

def generate_file_summary_prompt(anonymized_id, file_metrics_raw, file_metrics_norm, dataset_stats): # Keep norm param for now, though unused
    """Generates the prompt for file summary (using anonymized ID)."""
    if file_metrics_raw is None:
         logging.warning(f"Missing raw metrics for {anonymized_id}. Cannot generate prompt.")
         return None

    # Format raw metrics, handling potential non-numeric values gracefully
    raw_metrics_str_parts = []
    for k in KEY_METRICS_FOR_FILE_SUMMARY:
        if k in file_metrics_raw:
            val = file_metrics_raw[k]
            if isinstance(val, (int, float, np.number)) and pd.notna(val): raw_metrics_str_parts.append(f"- {k}: {val:.3f}")
            else: raw_metrics_str_parts.append(f"- {k}: {val}") # Keep as string if not numeric or NaN
    raw_metrics_str = "\n".join(raw_metrics_str_parts)

    context_str = ""
    if dataset_stats and isinstance(dataset_stats.get('mean'), dict):
        context_str += "\nDataset Averages (Raw):\n" + "\n".join([f"- {k}: {dataset_stats['mean'].get(k, 'N/A'):.3f}" if pd.notna(dataset_stats['mean'].get(k)) else f"- {k}: N/A" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in dataset_stats['mean']])
    if dataset_stats and isinstance(dataset_stats.get('std'), dict):
        context_str += "\nDataset Std Dev (Raw):\n" + "\n".join([f"- {k}: {dataset_stats['std'].get(k, 'N/A'):.3f}" if pd.notna(dataset_stats['std'].get(k)) else f"- {k}: N/A" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in dataset_stats['std']])

    prompt = f"""
Analyze the linguistic metrics for the document identified as: "{anonymized_id}"

Raw Metrics:
{raw_metrics_str}

Context (Dataset Statistics):
{context_str}

Provide a brief summary (2-4 sentences) interpreting these metrics for this specific document.
- Compare its key metrics (especially {', '.join(KEY_METRICS_FOR_FILE_SUMMARY[:4])}) to the dataset averages/context provided, using the **Raw Metrics** values shown above.
- Highlight 1-2 notable characteristics (e.g., significantly higher/lower diversity, complexity, readability, repetition, sentiment than average).
- Briefly explain any interesting combinations (e.g., if TTR is high but RTTR/MTLD are average/low, what might that imply?).
- Focus on providing insight based *only* on the provided numbers (Raw Metrics and Context). Do not mention the anonymized ID in your response.
""" # Removed Markdown instruction
    return prompt

# --- NEW: Prompt for Content Summarization ---
def generate_content_summary_prompt(anonymized_id, text_content):
    """Generates a prompt for summarizing the actual text content."""
    # Limit content length to avoid excessive prompt size
    max_content_chars = 4000 # Adjust as needed
    truncated_content = text_content[:max_content_chars]
    if len(text_content) > max_content_chars:
        truncated_content += "..."

    prompt = f"""
Please provide a very brief (1-2 sentence) abstract summarizing the main topic or purpose of the following text content, identified as "{anonymized_id}". Focus on *what* the text is about. Do not mention the anonymized ID in your response.

Text Content (potentially truncated):
--- START CONTENT ---
{truncated_content}
--- END CONTENT ---

Generate the 1-2 sentence abstract:
"""
    return prompt


# --- NEW: Helper Functions for Final Synthesis ---

def get_cluster_leaves(linkage_matrix, cluster_id, num_leaves):
    """Recursively find the original leaf indices belonging to a cluster node."""
    if cluster_id < num_leaves:
        return [cluster_id]
    else:
        node_idx = cluster_id - num_leaves
        if node_idx >= len(linkage_matrix):
             logging.error(f"[get_cluster_leaves] Node index {node_idx} out of bounds for linkage matrix (size {len(linkage_matrix)}). Cluster ID: {cluster_id}")
             return []
        left_child = int(linkage_matrix[node_idx, 0])
        right_child = int(linkage_matrix[node_idx, 1])
        return get_cluster_leaves(linkage_matrix, left_child, num_leaves) + \
               get_cluster_leaves(linkage_matrix, right_child, num_leaves)

def analyze_top_splits(linkage_matrix, df_normalized, n_splits=3, key_metrics=None):
    """Analyzes the top N splits in the linkage matrix based on distance."""
    if linkage_matrix is None or len(linkage_matrix) == 0 or df_normalized is None:
        return "Clustering information unavailable for split analysis."
    if key_metrics is None:
        key_metrics = KEY_METRICS_FOR_SPLIT_ANALYSIS

    num_leaves = len(df_normalized)
    if num_leaves <= 1:
         return "Not enough data points for split analysis."

    df_norm_reset = df_normalized.reset_index(drop=True)
    valid_key_metrics = [m for m in key_metrics if m in df_norm_reset.columns]
    if not valid_key_metrics:
        return "No key metrics for split analysis found in the processed data."

    analysis_parts = []
    sorted_linkage_indices = np.argsort(linkage_matrix[:, 2])[::-1]
    num_splits_to_analyze = min(n_splits, len(linkage_matrix))

    analysis_parts.append(f"Hierarchical clustering identified {len(linkage_matrix)} merge points (splits). Analyzing the top {num_splits_to_analyze} splits based on distance:")

    for i in range(num_splits_to_analyze):
        linkage_idx = sorted_linkage_indices[i]
        node_id = num_leaves + linkage_idx
        distance = linkage_matrix[linkage_idx, 2]

        left_child_id = int(linkage_matrix[linkage_idx, 0])
        right_child_id = int(linkage_matrix[linkage_idx, 1])

        left_leaves_orig_indices = get_cluster_leaves(linkage_matrix, left_child_id, num_leaves)
        right_leaves_orig_indices = get_cluster_leaves(linkage_matrix, right_child_id, num_leaves)

        if not left_leaves_orig_indices or not right_leaves_orig_indices:
            analysis_parts.append(f"- Split {i+1} (Node {node_id}, Dist {distance:.3f}): Involves an empty child cluster.")
            continue

        left_profile = df_norm_reset.iloc[left_leaves_orig_indices][valid_key_metrics].mean()
        right_profile = df_norm_reset.iloc[right_leaves_orig_indices][valid_key_metrics].mean()

        diff = (left_profile - right_profile).abs().sort_values(ascending=False)
        if not diff.empty and diff.iloc[0] > 0.01: # Only report meaningful differences
            top_diff_metric = diff.index[0]
            l_val = left_profile.get(top_diff_metric, np.nan)
            r_val = right_profile.get(top_diff_metric, np.nan)
            analysis_parts.append(f"- Split {i+1} (Node {node_id}, Dist {distance:.3f}): Separated groups primarily by '{top_diff_metric}' (avg norm. values ~{l_val:.2f} vs ~{r_val:.2f}). Left side had {len(left_leaves_orig_indices)} files, right side had {len(right_leaves_orig_indices)} files.")
        else:
            analysis_parts.append(f"- Split {i+1} (Node {node_id}, Dist {distance:.3f}): No single metric showed a strong difference between the {len(left_leaves_orig_indices)} files on the left and {len(right_leaves_orig_indices)} files on the right.")

    return "\n".join(analysis_parts)


def summarize_interpretations_by_cluster(file_summaries, cluster_assignments, original_filenames):
    """Condenses file summaries into key points per cluster."""
    if not file_summaries or cluster_assignments is None or not original_filenames:
        return {}
    if not isinstance(file_summaries, list):
         logging.warning(f"Cannot summarize interpretations: file_summaries is not a list (type: {type(file_summaries)}).")
         return {}
    if len(file_summaries) != len(cluster_assignments) or len(file_summaries) != len(original_filenames):
         logging.warning(f"Mismatch in lengths for summarizing interpretations. Files: {len(original_filenames)}, Summaries: {len(file_summaries)}, Assignments: {len(cluster_assignments) if cluster_assignments is not None else 'None'}. Skipping.")
         return {}

    summary_by_cluster = defaultdict(list)
    for i, summary in enumerate(file_summaries):
        if i >= len(cluster_assignments): # Safety check
             logging.warning(f"Index {i} out of bounds for cluster_assignments (len {len(cluster_assignments)}). Skipping summary.")
             continue
        cluster_id = cluster_assignments[i]
        if summary and not str(summary).startswith("Error:") and "pending" not in str(summary):
            summary_by_cluster[cluster_id].append(str(summary))

    condensed_summaries = {}
    for cluster_id, summaries in summary_by_cluster.items():
        if not summaries: continue
        num_summaries = len(summaries)
        representative_summary = summaries[0] # Take the first one
        key_points = []
        if "above average" in representative_summary: key_points.append("some metrics above average")
        if "below average" in representative_summary: key_points.append("some metrics below average")
        if "diversity" in representative_summary: key_points.append("mentions diversity")
        if "readability" in representative_summary: key_points.append("mentions readability")
        if "repetition" in representative_summary: key_points.append("mentions repetition")
        if "sentiment" in representative_summary: key_points.append("mentions sentiment") # Added

        if key_points:
            condensed_summaries[cluster_id] = f"Cluster {cluster_id} ({num_summaries} files): Representative summary suggests {', '.join(key_points)}."
        else:
            condensed_summaries[cluster_id] = f"Cluster {cluster_id} ({num_summaries} files): Representative summary: '{representative_summary[:100]}...'"

    return condensed_summaries


# --- PATCHED: Added content_summaries, cluster_assignments, original_filenames ---
def generate_final_summary_prompt(split_analysis, llm_cluster_labels, rule_cluster_labels, llm_summary_trends, rule_summary_trends, dataset_stats, llm_content_summaries=None, cluster_assignments=None, original_filenames=None):
    """Generates the prompt for the final overall synthesis."""

    prompt = "Synthesize the following analysis results into a cohesive summary of the text dataset's linguistic characteristics. Focus on integrating the cluster structure, LLM interpretations (prioritize these), rule-based findings, and content insights. Avoid simple repetition; aim for an informative overview.\n\n"

    prompt += "--- Clustering Structure Analysis ---\n"
    prompt += split_analysis + "\n\n"

    prompt += "--- Cluster Interpretations ---\n"
    if llm_cluster_labels:
        prompt += "LLM-Generated Labels:\n"
        for cid, label in llm_cluster_labels.items(): prompt += f"- Cluster {cid}: {label}\n"
    if rule_cluster_labels:
        prompt += "Rule-Based Labels:\n"
        for cid, label in rule_cluster_labels.items():
            if not llm_cluster_labels or cid not in llm_cluster_labels or llm_cluster_labels.get(cid, "").startswith("Error:") or llm_cluster_labels.get(cid, "") != label:
                 prompt += f"- Cluster {cid}: {label} (Rule)\n"
    prompt += "\n"

    prompt += "--- Individual File Metric Summary Trends (by Cluster) ---\n"
    if llm_summary_trends:
        prompt += "LLM Metric Summary Trends:\n"
        for cid, trend in llm_summary_trends.items(): prompt += f"- {trend}\n" # Trend already includes cluster ID
    if rule_summary_trends:
        prompt += "Rule-Based Metric Summary Trends:\n"
        for cid, trend in rule_summary_trends.items():
             if not llm_summary_trends or cid not in llm_summary_trends: prompt += f"- {trend} (Rule)\n"
    prompt += "\n"

    # --- START: Add Content Summary Insights (if available) ---
    if CONTENT_SUMMARY_ENABLED and llm_content_summaries and cluster_assignments is not None and original_filenames is not None:
        prompt += "\n--- Content Summary Insights (LLM - Sampled by Cluster) ---\n"
        clusters_mentioned = set(llm_summary_trends.keys()) # Get clusters already discussed
        added_content_summary = False
        # Ensure unique_clusters is defined if needed, or use clusters_mentioned directly
        unique_clusters = sorted(list(clusters_mentioned)) if clusters_mentioned else []

        for cluster_id in unique_clusters:
            # Find filenames for this cluster
            cluster_indices = [idx for idx, c_id in enumerate(cluster_assignments) if c_id == cluster_id]
            cluster_filenames = [original_filenames[idx] for idx in cluster_indices if idx < len(original_filenames)]

            # Get summaries for these files (limit to first 2 for brevity)
            summaries_for_cluster = [llm_content_summaries.get(fname) for fname in cluster_filenames[:2] if fname in llm_content_summaries and llm_content_summaries.get(fname) and not str(llm_content_summaries.get(fname)).startswith("Error:")]

            if summaries_for_cluster:
                prompt += f"Cluster {cluster_id} Content Snippets:\n"
                for snippet in summaries_for_cluster:
                    # Ensure snippet is treated as string, truncate safely
                    snippet_str = str(snippet)
                    prompt += f"- \"{snippet_str[:150]}{'...' if len(snippet_str) > 150 else ''}\"\n"
                added_content_summary = True

        if not added_content_summary:
             prompt += "No valid content summaries available for the discussed clusters.\n"
        prompt += "\n"
    # --- END: Add Content Summary Insights ---


    prompt += "--- Overall Dataset Context ---\n"
    if dataset_stats and 'mean' in dataset_stats:
        prompt += "Key Dataset Averages (Raw):\n"
        means = {k: v for k, v in dataset_stats['mean'].items() if k in KEY_METRICS_FOR_FILE_SUMMARY[:5]} # Show first 5 key metrics
        for k, v in means.items(): prompt += f"- {k}: {v:.2f}\n"
    prompt += "\n"

    prompt += "--- Synthesis Task ---\n"
    prompt += "Based on all the above (clustering, metric trends, content snippets), provide a detailed synthesis (a few paragraphs). Describe the main groups found, their characteristics according to the LLM and rules, how they differ based on the split analysis, and any overall trends observed in the dataset compared to the average file. Integrate the content insights to explain *what* the different clusters might be about. Prioritize the LLM's interpretations but incorporate rule-based insights where relevant." # Removed Markdown instruction

    return prompt


# --- HTML Update Function (Patched for Incremental Updates, Correct Order, No Markdown, TOC Update) ---
def update_html_report(html_path, rules_results, llm_results, final_synthesis_summary, original_filenames, cluster_assignments, llm_config):
    """
    Adds/Updates the generated interpretations and final synthesis to the HTML report.
    Designed to be called multiple times, replacing existing sections.
    Ensures processor output follows transformer output if present.
    Updates the Table of Contents.
    Uses basic <br> for line breaks instead of Markdown.
    """
    if not BS4_AVAILABLE:
        logging.error("Cannot update HTML report: BeautifulSoup4 not available.")
        return False

    try:
        # --- Read current HTML ---
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        # --- Find a stable insertion point ---
        # Try to find the last section added by the transformer script
        stable_anchor_element = soup.find('div', id='augmented_analysis_section') # Try augmented first
        if not stable_anchor_element: stable_anchor_element = soup.find('div', id='numerical_analysis_section') # Then numerical
        if not stable_anchor_element: stable_anchor_element = soup.find('div', id='dendrogram_explanation_section') # Then main visual

        if stable_anchor_element: logging.info("Found transformer's analysis section. Inserting processor output after it.")
        else:
            # Fallback: Find the end of the plots section or the last known element
            logging.debug("Transformer section not found. Looking for plots section or other fallback anchor...")
            stable_anchor_element = soup.find('div', id='generated_plots') # Plots container added by visualizer
            if not stable_anchor_element: stable_anchor_element = soup.find('ul', id='collective_metrics_section') # Collective metrics list
            if not stable_anchor_element: stable_anchor_element = soup.find('table', id='resultsTableSecondary') # Secondary results table
            if not stable_anchor_element: stable_anchor_element = soup.find('table', id='resultsTableMain') # Main results table

            if stable_anchor_element: logging.info(f"Found fallback anchor point: <{stable_anchor_element.name} id='{stable_anchor_element.get('id', 'N/A')}'>.")
            else: logging.warning("Could not find transformer section or fallback anchors. Will append new sections to the end of the body."); stable_anchor_element = soup.body

        if not stable_anchor_element: logging.error("Could not find a suitable insertion point (even body) in HTML."); return False

        # --- Remove existing processor sections if present ---
        existing_interp_section = soup.find('div', id='interpretations_section')
        if existing_interp_section: logging.debug("Removing previous processor interpretation section."); existing_interp_section.decompose()
        existing_synth_section = soup.find('div', id='synthesis_section')
        if existing_synth_section: logging.debug("Removing previous processor synthesis section."); existing_synth_section.decompose()


        # --- Create Main Interpretation Container (only if results exist) ---
        main_interp_div = None
        if rules_results or llm_results:
            main_interp_div = soup.new_tag('div', id='interpretations_section', style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #666;")
            main_title = soup.new_tag('h2', style="margin-bottom: 15px;"); main_title.string = "Generated Interpretations (Processor)"; main_interp_div.append(main_title)

            # --- Function to add results for a specific mode ---
            def add_mode_results_to_html(mode_name, results_dict, config_dict=None):
                if not results_dict or (not results_dict.get('cluster_labels') and not results_dict.get('file_summaries')): return None

                mode_div = soup.new_tag('div', style="margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #eee;")
                mode_h3 = soup.new_tag('h3'); mode_h3.string = f"Interpretation Mode: {mode_name}"; mode_div.append(mode_h3)
                is_llm_mode = (mode_name == 'Local LLM')

                if is_llm_mode and config_dict:
                     model_info = soup.new_tag('p', style="font-size: 0.9em; color: #555;")
                     model_info.string = f"Model Used (from INI): {config_dict.get('processor_model', config_dict.get('model', 'N/A'))}"
                     mode_div.append(model_info)

                # Add Cluster Labels
                cluster_labels = results_dict.get('cluster_labels')
                if cluster_labels and isinstance(cluster_labels, dict):
                    cluster_h4 = soup.new_tag('h4', style="margin-top: 15px;"); cluster_h4.string = f"Cluster Labels (Top {len(cluster_labels)} Clusters)"; mode_div.append(cluster_h4)
                    cluster_p = soup.new_tag('p', style="font-size: 0.9em; color: #555;"); cluster_p.string = "Estimated linguistic profiles based on average metrics within each cluster."; mode_div.append(cluster_p)
                    cluster_dl = soup.new_tag('dl', style="margin-left: 20px;")
                    for cluster_id, label in sorted(cluster_labels.items()):
                        dt = soup.new_tag('dt', style="font-weight: bold; margin-top: 8px;"); dt.string = f"Cluster {cluster_id}:"
                        dd = soup.new_tag('dd', style="margin-left: 20px; margin-bottom: 5px;")
                        if label and not str(label).startswith("Error:"):
                            # REMOVED Markdown conversion
                            try:
                                # Insert raw text with <br> for newlines
                                parsed_content = BeautifulSoup(str(label).replace('\n', '<br/>'), 'html.parser')
                                # Append the parsed content (which now includes <br/> tags)
                                for content_node in parsed_content.contents:
                                    dd.append(content_node.extract())
                            except Exception as parse_err:
                                logging.error(f"Error parsing cluster label {cluster_id}: {parse_err}. Inserting raw text.")
                                dd.append(NavigableString(str(label))) # Fallback
                        else: error_tag = soup.new_tag('i', style='color:red;'); error_tag.string = str(label) if label else "Label generation failed."; dd.append(error_tag)
                        cluster_dl.append(dt); cluster_dl.append(dd)
                    mode_div.append(cluster_dl)
                elif cluster_labels: logging.warning(f"Cluster labels for mode {mode_name} is not a dictionary: {cluster_labels}"); p_err = soup.new_tag('p', style="color:red;"); p_err.string = "Error displaying cluster labels (invalid format)."; mode_div.append(p_err)

                # Add File Summaries
                file_summaries = results_dict.get('file_summaries')
                content_summaries = results_dict.get('content_summaries', {})
                summaries_exist = isinstance(file_summaries, (list, dict)) and len(file_summaries) > 0

                if summaries_exist and original_filenames:
                    if isinstance(file_summaries, dict):
                         temp_summaries = [file_summaries.get(fname, "Summary pending or missing...") for fname in original_filenames]
                         file_summaries = temp_summaries
                    elif not isinstance(file_summaries, list):
                         logging.warning(f"File summaries for {mode_name} has unexpected type: {type(file_summaries)}. Skipping."); file_summaries = []

                    if len(file_summaries) > 0: # Check again after potential conversion/clearing
                        # --- PATCH: Update section title based on mode and content summary status ---
                        summary_h4 = soup.new_tag('h4', style="margin-top: 25px;")
                        if is_llm_mode and CONTENT_SUMMARY_ENABLED:
                            summary_h4.string = "Individual File Summaries: Metrics & Content (LLM)"
                        elif is_llm_mode:
                            summary_h4.string = "Individual File Summaries: Metrics (LLM)"
                        else: # Rule-based
                            summary_h4.string = "Individual File Summaries: Metrics (Rule-Based)"
                        mode_div.append(summary_h4)
                        # --- END PATCH ---
                        summary_p_text = "Brief interpretation comparing each file's metrics to the dataset."
                        if is_llm_mode and CONTENT_SUMMARY_ENABLED:
                            summary_p_text += " Followed by an LLM-generated content summary (if enabled and successful)."
                        summary_p = soup.new_tag('p', style="font-size: 0.9em; color: #555;"); summary_p.string = summary_p_text; mode_div.append(summary_p)
                        summary_dl = soup.new_tag('dl', style="margin-left: 20px;")

                        for i, orig_fname in enumerate(original_filenames):
                            cluster_id_str = 'N/A'
                            if cluster_assignments is not None and i < len(cluster_assignments): cluster_id_str = str(cluster_assignments[i])
                            summary = file_summaries[i] if i < len(file_summaries) else "Summary pending..."
                            content_summary = content_summaries.get(orig_fname, None) if isinstance(content_summaries, dict) else None

                            dt = soup.new_tag('dt', style="font-weight: bold; margin-top: 8px;"); dt.string = f"{orig_fname} (Cluster {cluster_id_str}):"
                            # <<< MODIFIED LINE: Increased margin-bottom for spacing >>>
                            dd = soup.new_tag('dd', style="margin-left: 20px; margin-bottom: 15px;") # Changed 5px to 15px

                            # --- Add Metric Summary ---
                            if summary and not str(summary).startswith("Error:") and "pending" not in str(summary):
                                # REMOVED Markdown conversion
                                try:
                                    parsed_content = BeautifulSoup(str(summary).replace('\n', '<br/>'), 'html.parser')
                                    for content_node in parsed_content.contents:
                                        dd.append(content_node.extract())
                                except Exception as parse_err:
                                    logging.error(f"Error parsing file summary for {orig_fname}: {parse_err}. Inserting raw text.")
                                    dd.append(NavigableString(str(summary))) # Fallback
                            else: # Handle errors or pending state
                                error_tag = soup.new_tag('i', style='color:red;' if str(summary).startswith("Error:") else 'color:gray;')
                                error_tag.string = str(summary) if summary else "Summary generation failed."; dd.append(error_tag)

                            # --- Add Content Summary ---
                            if content_summary and not str(content_summary).startswith("Error:"):
                                dd.append(soup.new_tag('br')) # Add first separator line break
                                dd.append(soup.new_tag('br')) # Add second separator line break for more space
                                content_label = soup.new_tag('b'); content_label.string = "LLM Content Summary: " # Updated label text
                                dd.append(content_label)
                                # dd.append(soup.new_tag('br')) # Optional: Add break after label if needed
                                # REMOVED Markdown conversion
                                try:
                                    parsed_content = BeautifulSoup(str(content_summary).replace('\n', '<br/>'), 'html.parser')
                                    for content_node in parsed_content.contents:
                                        # Append directly without checking for <p> tag
                                        dd.append(content_node.extract())
                                except Exception as parse_err:
                                    logging.error(f"Error parsing content summary for {orig_fname}: {parse_err}. Inserting raw text.")
                                    dd.append(NavigableString(str(content_summary))) # Fallback
                            elif content_summary: # Handle error case for content summary
                                dd.append(soup.new_tag('br'))
                                dd.append(soup.new_tag('br'))
                                content_label = soup.new_tag('b'); content_label.string = "LLM Content Summary: "
                                dd.append(content_label)
                                error_tag = soup.new_tag('i', style='color:red;')
                                error_tag.string = str(content_summary)
                                dd.append(error_tag)
                            # --- END Content Summary ---

                            summary_dl.append(dt); summary_dl.append(dd)
                        mode_div.append(summary_dl)
                return mode_div # Return the generated div for this mode

            # --- Add Rule-Based Results ---
            if rules_results:
                rules_div = add_mode_results_to_html("Rule-Based", rules_results)
                if rules_div: main_interp_div.append(rules_div)

            # --- Add Local LLM Results ---
            if llm_results:
                llm_div = add_mode_results_to_html("Local LLM", llm_results, local_llm_config) # Pass llm_config
                if llm_div: main_interp_div.append(llm_div)

        # --- Create Synthesis Section (only if summary exists) ---
        synthesis_div = None
        if final_synthesis_summary:
            synthesis_div = soup.new_tag('div', id='synthesis_section', style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #333;")
            synth_title = soup.new_tag('h2', style="margin-bottom: 15px;"); synth_title.string = "Overall Synthesis (Processor)"; synthesis_div.append(synth_title)
            synth_content_div = soup.new_tag('div') # Container for content
            if not final_synthesis_summary.startswith("Error:") and not final_synthesis_summary.startswith("Skipped"):
                 # REMOVED Markdown conversion
                 try:
                     parsed_content = BeautifulSoup(final_synthesis_summary.replace('\n', '<br/>'), 'html.parser')
                     for content_node in parsed_content.contents:
                         synth_content_div.append(content_node.extract())
                 except Exception as parse_err:
                     logging.error(f"Error parsing final synthesis: {parse_err}. Inserting raw text.")
                     synth_content_div.append(NavigableString(final_synthesis_summary)) # Fallback
            else: # Handle error or skipped case
                 error_tag = soup.new_tag('i', style='color:red;' if final_synthesis_summary.startswith("Error:") else 'color:gray;')
                 error_tag.string = final_synthesis_summary; synth_content_div.append(error_tag)
            synthesis_div.append(synth_content_div)


        # --- Insert the Sections into the HTML ---
        last_inserted = stable_anchor_element # Start after the stable anchor

        if main_interp_div: # If there are interpretations to insert
             if last_inserted and last_inserted.parent: last_inserted.insert_after(main_interp_div); last_inserted = main_interp_div; logging.debug("Inserted processor interpretation section.")
             else: logging.warning("Could not insert processor interpretation section after anchor, appending to body.");
             if soup.body: soup.body.append(main_interp_div); last_inserted = main_interp_div
             else: logging.error("Cannot append processor interpretation to body (no body tag).")

        if synthesis_div: # If there is a synthesis to insert
             if last_inserted and last_inserted.parent: last_inserted.insert_after(synthesis_div); logging.debug("Inserted processor synthesis section.")
             else: logging.warning("Could not insert processor synthesis section after previous element, appending to body.");
             if soup.body: soup.body.append(synthesis_div)
             else: logging.error("Cannot append processor synthesis to body (no body tag).")


        # --- START: Update Table of Contents ---
        toc_div = soup.find('div', id='toc')
        if toc_div:
            toc_list = toc_div.find('ul')
            if toc_list:
                logging.debug("Found TOC list. Checking for processor section links...")
                interp_section_id = "interpretations_section"; synth_section_id = "synthesis_section"
                interp_link_exists = toc_list.find('a', href=f"#{interp_section_id}")
                if not interp_link_exists and main_interp_div:
                    li_interp = soup.new_tag('li'); a_interp = soup.new_tag('a', href=f"#{interp_section_id}"); a_interp.string = "Generated Interpretations (Processor)"; li_interp.append(a_interp); toc_list.append(li_interp)
                    logging.debug("Added 'Generated Interpretations' link to TOC.")
                synth_link_exists = toc_list.find('a', href=f"#{synth_section_id}")
                if not synth_link_exists and synthesis_div:
                    li_synth = soup.new_tag('li'); a_synth = soup.new_tag('a', href=f"#{synth_section_id}"); a_synth.string = "Overall Synthesis (Processor)"; li_synth.append(a_synth); toc_list.append(li_synth)
                    logging.debug("Added 'Overall Synthesis' link to TOC.")
            else: logging.warning("Could not find UL tag within TOC div to add links.")
        else: logging.warning("Could not find TOC div (id='toc') to update.")
        # --- END: Update Table of Contents ---


        # --- Save the modified HTML back to the ORIGINAL file ---
        output_html_path = html_path # Write back to the input path
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(str(soup.prettify()))

        return True

    except FileNotFoundError: logging.error(f"HTML file '{html_path}' not found for updating."); return False
    except Exception as e:
        if isinstance(e, ValueError) and "Element has no parent" in str(e): logging.error(f"Failed to update HTML file due to insertion error: {e}. This likely means the insertion anchor point was removed or invalid.", exc_info=True)
        elif "bs4.FeatureNotFound" in str(type(e)): logging.critical(f"CRITICAL: Failed to parse HTML. Ensure 'html.parser' or another parser like 'lxml' is installed and the HTML is valid. Error: {e}", exc_info=True)
        else: logging.error(f"Failed to update HTML file due to an unexpected error: {e}", exc_info=True)
        if soup is None:
             logging.error("HTML parsing failed, cannot proceed with update.")
        return False


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load INI Configuration
    config = configparser.ConfigParser()
    ini_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), INI_FILENAME)
    if not os.path.exists(ini_path): logging.critical(f"CRITICAL: Configuration file '{INI_FILENAME}' not found."); sys.exit(1)
    try: config.read(ini_path); logging.info(f"Loaded configuration from '{INI_FILENAME}'.")
    except configparser.Error as e: logging.critical(f"CRITICAL: Error parsing configuration file '{INI_FILENAME}': {e}"); sys.exit(1)

    # Check if Local LLM mode is enabled and configured
    local_llm_enabled = False; local_llm_config = {}
    if 'LocalLLM' in config:
        local_llm_enabled = config.getboolean('LocalLLM', 'enabled', fallback=False)
        if local_llm_enabled:
            local_llm_config = dict(config.items('LocalLLM'))
            if 'max_tokens' not in local_llm_config: local_llm_config['max_tokens'] = str(DEFAULT_MAX_TOKENS)
            logging.info(f"Local LLM mode is enabled. Config: {local_llm_config}")
            if not local_llm_config.get('api_base'): logging.warning("Local LLM mode enabled, but 'api_base' is not set in [LocalLLM] section. LLM calls will likely fail.")
            if not REQUESTS_AVAILABLE: logging.error("Local LLM mode enabled, but 'requests' library is not installed. Cannot use Local LLM."); local_llm_enabled = False
        else: logging.info("Local LLM mode is disabled in configuration.")
    else: logging.info("[LocalLLM] section not found in INI file. Local LLM mode disabled.")

    # REMOVED Markdown check


    # 2. Get Target Folder
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

    if not os.path.isdir(target_folder): log_func = logging.critical if 'logging' in sys.modules else print; log_func(f"CRITICAL Error: Target folder not found or is not a directory: {target_folder}"); sys.exit(1)

    csv_path = os.path.join(target_folder, CSV_FILENAME)
    html_path = os.path.join(target_folder, HTML_FILENAME)
    txt_dir_path = os.path.join(target_folder, TXT_SUBDIR) # Path to .txt files
    if not os.path.isfile(csv_path): logging.critical(f"CRITICAL: CSV file '{CSV_FILENAME}' not found."); sys.exit(1)
    if not os.path.isfile(html_path): logging.critical(f"CRITICAL: HTML file '{HTML_FILENAME}' not found."); sys.exit(1)
    if CONTENT_SUMMARY_ENABLED and not os.path.isdir(txt_dir_path):
        logging.warning(f"Content summary enabled, but text directory '{txt_dir_path}' not found. Disabling content summaries.")
        CONTENT_SUMMARY_ENABLED = False


    # 3. Load and Preprocess Data
    logging.info("--- Loading and Preprocessing Data ---")
    df_raw = load_data(csv_path) # Now loads only data rows and filters collective keys
    if df_raw is None: sys.exit(1)
    df_numeric_orig, df_imputed, df_normalized, original_filenames, anonymized_ids = preprocess_data(df_raw)
    if df_normalized is None or original_filenames is None or anonymized_ids is None or df_numeric_orig is None:
        logging.critical("Data preprocessing failed. Exiting.")
        sys.exit(1)
    logging.info("--- Data Loading and Preprocessing Complete ---")


    # 4. Perform Clustering
    logging.info("--- Performing Clustering ---")
    linkage_matrix = perform_clustering(df_normalized)
    if linkage_matrix is None: logging.warning("Clustering failed or skipped.")
    logging.info("--- Clustering Step Complete ---")


    # 5. Get Flat Cluster Assignments
    logging.info("--- Deriving Flat Clusters ---")
    num_files = len(df_imputed) # Use length of imputed/normalized df
    actual_n_clusters = min(N_CLUSTERS_TO_LABEL, num_files) if num_files > 0 else 0
    cluster_assignments = None
    if linkage_matrix is not None and actual_n_clusters >= 2:
        cluster_assignments = get_flat_clusters(linkage_matrix, actual_n_clusters, num_files)
        if cluster_assignments is None: logging.warning("Failed to derive flat clusters."); actual_n_clusters = 0
        else: logging.info(f"Successfully derived {actual_n_clusters} cluster assignments.")
    else:
        logging.warning(f"Skipping flat cluster derivation (num_files={num_files}, actual_n_clusters={actual_n_clusters}).")
        actual_n_clusters = 0
    logging.info("--- Flat Cluster Derivation Complete ---")


    # 6. Calculate Cluster Averages and Dataset Stats
    logging.info("--- Calculating Averages and Stats ---")
    cluster_avg_metrics_norm = {}
    if actual_n_clusters >= 2 and cluster_assignments is not None:
        cluster_avg_metrics_norm = get_cluster_avg_metrics(df_normalized, cluster_assignments, KEY_METRICS_FOR_CLUSTER_LABEL)

    dataset_stats = get_dataset_stats(df_numeric_orig, KEY_METRICS_FOR_FILE_SUMMARY)
    logging.info("--- Averages and Stats Calculation Complete ---")


    # --- 7. Generate Interpretations ---
    rules_results = {}; llm_results = {}; llm_generation_failed = False

    # --- Always run Rule-Based Interpretation ---
    logging.info(f"--- Starting Interpretation Generation (Mode: Rule-Based) ---")
    rules_results['mode'] = 'Rules'
    rule_cluster_labels = {}
    if actual_n_clusters >= 2 and cluster_avg_metrics_norm:
        logging.info("--- Generating Rule-Based Cluster Labels ---")
        for cluster_id, avg_metrics in cluster_avg_metrics_norm.items():
            label = generate_cluster_label_rules(cluster_id, avg_metrics)
            rule_cluster_labels[cluster_id] = label
        rules_results['cluster_labels'] = rule_cluster_labels
    else: rules_results['cluster_labels'] = {}

    rule_file_summaries = []
    logging.info("--- Generating Rule-Based File Summaries ---")
    for i in range(num_files): # Iterate up to the actual number of processed files
        fname = original_filenames[i] if i < len(original_filenames) else f"File Index {i}"
        anonymized_id = anonymized_ids[i] if i < len(anonymized_ids) else f"File {i+1}"
        # Find the corresponding row in df_numeric_orig using the original filename
        if 'Filename' in df_numeric_orig.columns:
             file_metrics_raw_series = df_numeric_orig[df_numeric_orig['Filename'] == fname].iloc[0]
        else: # Fallback to index if Filename column wasn't present initially
             file_metrics_raw_series = df_numeric_orig.iloc[i]

        file_metrics_raw = file_metrics_raw_series.to_dict() if not file_metrics_raw_series.empty else None
        summary = generate_file_summary_rules(anonymized_id, file_metrics_raw, dataset_stats)
        rule_file_summaries.append(summary)
    rules_results['file_summaries'] = rule_file_summaries
    logging.info(f"--- Rule-Based Interpretation Generation Complete ---")


    # --- Optionally run Local LLM Interpretation ---
    if local_llm_enabled:
        logging.info(f"--- Starting Interpretation Generation (Mode: Local LLM) ---")
        llm_results['mode'] = 'Local LLM'
        llm_results['model_used'] = local_llm_config.get('processor_model', local_llm_config.get('model', 'N/A'))
        llm_cluster_labels = {}
        llm_file_summaries = ["Summary pending..."] * num_files # Use correct num_files
        llm_content_summaries = {} # NEW: Dict for content summaries {filename: summary}
        llm_results['cluster_labels'] = llm_cluster_labels
        llm_results['file_summaries'] = llm_file_summaries
        llm_results['content_summaries'] = llm_content_summaries # Add to results

        # Generate Cluster Labels
        if actual_n_clusters >= 2 and cluster_avg_metrics_norm:
            logging.info(f"--- Generating Local LLM Cluster Labels ({llm_results['model_used']}) ---")
            for cluster_id, avg_metrics in cluster_avg_metrics_norm.items():
                if llm_generation_failed: break
                prompt = generate_cluster_label_prompt(cluster_id, avg_metrics)
                if prompt:
                    label = call_local_llm_api(prompt, local_llm_config)
                    llm_cluster_labels[cluster_id] = label
                    llm_results['cluster_labels'] = llm_cluster_labels
                    update_success = update_html_report(html_path, rules_results, llm_results, None, original_filenames, cluster_assignments, local_llm_config)
                    if update_success: logging.info(f"HTML updated incrementally after LLM label for Cluster {cluster_id}.")
                    else: logging.error("Failed incremental HTML update after cluster label."); llm_generation_failed = True; break
                    if label is not None and str(label).startswith("Error:"):
                         logging.error(f"Critical error generating cluster label {cluster_id} with LLM. Stopping LLM generation.")
                         llm_generation_failed = True; break
            if llm_generation_failed: llm_results['cluster_labels'] = {"Error": "LLM generation failed during cluster labeling."}

        # Generate File Summaries ONLY if cluster labels didn't fail critically
        if not llm_generation_failed:
            logging.info(f"--- Generating Local LLM File Summaries ({llm_results['model_used']}) ---")
            for i in range(num_files): # Iterate up to the actual number of processed files
                if llm_generation_failed: break
                fname = original_filenames[i] if i < len(original_filenames) else f"File Index {i}"
                logging.info(f"Generating LLM summary for file {i+1}/{num_files} ({fname})...")
                anonymized_id = anonymized_ids[i] if i < len(anonymized_ids) else f"File {i+1}"

                # Find the corresponding rows in df_numeric_orig and df_normalized
                if 'Filename' in df_numeric_orig.columns:
                    file_metrics_raw_series = df_numeric_orig[df_numeric_orig['Filename'] == fname].iloc[0]
                else: # Fallback to index
                    file_metrics_raw_series = df_numeric_orig.iloc[i]

                file_metrics_norm_series = df_normalized.iloc[i] # Index should align

                file_metrics_raw = file_metrics_raw_series.to_dict() if not file_metrics_raw_series.empty else None
                file_metrics_norm = file_metrics_norm_series.to_dict() if not file_metrics_norm_series.empty else None

                # --- Generate Metric Summary ---
                prompt = generate_file_summary_prompt(anonymized_id, file_metrics_raw, file_metrics_norm, dataset_stats)
                metric_summary = "Error: Could not generate prompt." # Default
                if prompt:
                    metric_summary = call_local_llm_api(prompt, local_llm_config)
                    llm_file_summaries[i] = metric_summary
                    if metric_summary is not None and str(metric_summary).startswith("Error:"):
                         logging.error(f"Critical error generating metric summary {i+1} ({fname}) with LLM. Stopping LLM generation.")
                         llm_generation_failed = True
                else:
                     logging.warning(f"Could not generate metric summary prompt for file index {i} ({anonymized_id}). Setting error message.")
                     llm_file_summaries[i] = metric_summary # Store error

                # --- Generate Content Summary (if enabled and metric summary didn't fail critically) ---
                if CONTENT_SUMMARY_ENABLED and not llm_generation_failed:
                    logging.info(f"Attempting content summary for file {i+1} ({fname})...")
                    content_summary = f"Error: Content summary disabled or file not found."
                    txt_file_path = os.path.join(txt_dir_path, fname)
                    if os.path.exists(txt_file_path):
                        try:
                            with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f_txt:
                                text_content = f_txt.read()
                            content_prompt = generate_content_summary_prompt(anonymized_id, text_content)
                            content_summary = call_local_llm_api(content_prompt, local_llm_config, max_tokens_override=CONTENT_SUMMARY_MAX_TOKENS)
                            llm_content_summaries[fname] = content_summary # Store content summary
                            if content_summary is not None and str(content_summary).startswith("Error:"):
                                logging.error(f"Critical error generating content summary {i+1} ({fname}) with LLM. Stopping LLM generation.")
                                llm_generation_failed = True
                            else:
                                logging.info(f"Content summary generated for file {i+1} ({fname}).")
                        except Exception as e_content:
                            logging.error(f"Error reading file {fname} for content summary: {e}")
                            llm_content_summaries[fname] = f"Error reading file content."
                    else:
                        logging.warning(f"Text file not found for content summary: {txt_file_path}")
                        llm_content_summaries[fname] = f"Error: Source .txt file not found."
                # --- END Content Summary ---

                # --- Incremental Update (after both summaries attempted) ---
                llm_results['file_summaries'] = llm_file_summaries
                llm_results['content_summaries'] = llm_content_summaries # Update results dict
                update_success = update_html_report(html_path, rules_results, llm_results, None, original_filenames, cluster_assignments, local_llm_config)
                if update_success: logging.info(f"HTML updated incrementally after LLM summaries for file {i+1} ({fname}).")
                else: logging.error("Failed incremental HTML update after file summary."); llm_generation_failed = True; break

                # Break outer loop if critical error occurred in either summary
                if llm_generation_failed: break

            if llm_generation_failed:
                 for j in range(i + 1, num_files): # Start from next index if loop broke
                      if llm_file_summaries[j] == "Summary pending...": llm_file_summaries[j] = "Error: LLM generation failed before processing this file."
                 update_success = update_html_report(html_path, rules_results, llm_results, None, original_filenames, cluster_assignments, local_llm_config)
                 if update_success: logging.info("HTML updated finally after LLM failure to mark remaining files.")
                 else: logging.error("Failed final HTML update after LLM failure.")


        if llm_generation_failed: logging.error("Local LLM interpretation generation stopped due to errors.")
        else: logging.info(f"--- Local LLM Interpretation Generation Complete ---")
    else:
        logging.info("Skipping Local LLM interpretation as it's disabled in the INI file.")


    # --- 8. Generate Final Synthesis (Optional) ---
    final_synthesis_summary = None
    if local_llm_enabled and not llm_generation_failed:
        logging.info("--- Generating Final Synthesis Summary (LLM) ---")
        try:
            split_analysis_text = analyze_top_splits(linkage_matrix, df_normalized, N_SPLITS_TO_ANALYZE, KEY_METRICS_FOR_SPLIT_ANALYSIS)
            logging.debug(f"Split Analysis for Synthesis:\n{split_analysis_text}")

            llm_summary_trends = summarize_interpretations_by_cluster(llm_results.get('file_summaries', []), cluster_assignments, original_filenames)
            rule_summary_trends = summarize_interpretations_by_cluster(rules_results.get('file_summaries', []), cluster_assignments, original_filenames)
            logging.debug(f"LLM Summary Trends for Synthesis:\n{llm_summary_trends}")
            logging.debug(f"Rule Summary Trends for Synthesis:\n{rule_summary_trends}")

            # --- PATCHED: Pass content summaries, assignments, and filenames ---
            final_prompt = generate_final_summary_prompt(
                split_analysis_text, llm_results.get('cluster_labels', {}), rules_results.get('cluster_labels', {}),
                llm_summary_trends, rule_summary_trends, dataset_stats,
                llm_results.get('content_summaries'), # Pass content summaries
                cluster_assignments, original_filenames # Pass assignments and filenames
            )
            # --- END PATCH ---
            logging.debug(f"--- Final Synthesis Prompt ---\n{final_prompt}\n--- End Final Synthesis Prompt ---")

            if final_prompt:
                synthesis_llm_config = local_llm_config.copy()
                base_max_tokens = int(synthesis_llm_config.get('max_tokens', DEFAULT_MAX_TOKENS))
                synthesis_tokens = int(base_max_tokens * SYNTHESIS_MAX_TOKENS_MULTIPLIER)
                logging.info(f"Requesting max_tokens={synthesis_tokens} for final synthesis.")
                final_synthesis_summary = call_local_llm_api(final_prompt, synthesis_llm_config, max_tokens_override=synthesis_tokens)

                if final_synthesis_summary and not str(final_synthesis_summary).startswith("Error:"): logging.info("Successfully generated final synthesis summary.")
                else: logging.error(f"Failed to generate final synthesis summary: {final_synthesis_summary}")
            else:
                 logging.warning("Could not generate final synthesis prompt.")
                 final_synthesis_summary = "Error: Could not generate prompt for final synthesis."

        except Exception as e:
            logging.error(f"Error during final synthesis generation: {e}", exc_info=True)
            final_synthesis_summary = f"Error: Exception during final synthesis generation - {e}"
    elif local_llm_enabled and llm_generation_failed:
        logging.warning("Skipping final synthesis summary because LLM generation failed earlier.")
        final_synthesis_summary = "Skipped due to earlier LLM errors."
    else:
        logging.info("Skipping final synthesis summary as Local LLM is disabled.")


    # --- 9. Final HTML Update (includes synthesis) ---
    logging.info("--- Performing Final HTML Update (with Synthesis if available) ---")
    success = update_html_report(
        html_path, rules_results, llm_results, final_synthesis_summary,
        original_filenames, cluster_assignments, local_llm_config
    )
    if success: logging.info("Interpretation script finished successfully.")
    else: logging.error("Interpretation script finished with errors during FINAL HTML update."); sys.exit(1)

    logging.info("--- Script Finished ---")