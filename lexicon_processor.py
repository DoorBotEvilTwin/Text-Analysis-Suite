# lexicon_processor.py

import pandas as pd
import numpy as np
import os
import sys
import logging
import json
import time
import configparser
from datetime import datetime

# --- Dependencies with Checks ---
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: scikit-learn is required. Install: pip install scikit-learn")
    sys.exit(1)

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("CRITICAL: scipy is required for clustering. Install: pip install scipy")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
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


# --- Configuration Files ---
INI_FILENAME = "lexicon_processor.ini"
CSV_FILENAME = "_LEXICON.csv"
HTML_FILENAME = "_LEXICON.html" # Target HTML file to modify
LOG_FILENAME = "_LEXICON_errors.log" # Appends to the same log
# OUTPUT_HTML_SUFFIX = "_interpreted" # No longer creating a new file

# --- AI/Local LLM Configuration ---
MAX_RETRIES = 2
RETRY_DELAY = 3 # Shorter delay for local calls
# REQUEST_TIMEOUT = 60 # Timeout for local LLM calls - REMOVED

# --- Clustering Configuration ---
CLUSTER_METRIC = 'euclidean'
CLUSTER_METHOD = 'ward'
N_CLUSTERS_TO_LABEL = 5

# --- Metrics for Interpretation ---
KEY_METRICS_FOR_CLUSTER_LABEL = [
    'MTLD', 'VOCD', 'Average Sentence Length', 'Flesch Reading Ease',
    'Distinct-2', 'Repetition-2', 'Lexical Density', 'Yule\'s K', 'Simpson\'s D'
]
KEY_METRICS_FOR_FILE_SUMMARY = [
    'MTLD', 'VOCD', 'Average Sentence Length', 'Flesch Reading Ease',
    'Distinct-2', 'Repetition-2', 'Lexical Density', 'TTR', 'RTTR'
]

# --- Basic Logging Setup ---
log_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), LOG_FILENAME)
log_level = logging.INFO # Set to DEBUG for more verbose local LLM call info
logging.basicConfig(
    level=log_level,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'), # Append mode 'a'
        logging.StreamHandler()
    ]
)
logging.info(f"\n{'='*10} Starting Offline Interpretation Run {'='*10}")

# --- Helper Functions (find_separator_index, load_data, preprocess_data, perform_clustering, get_flat_clusters, get_cluster_avg_metrics, get_dataset_stats) ---
# These remain largely the same as the previous version. Ensure they are included here.
# ... (Paste the helper functions from the previous version here) ...
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

def load_data(csv_path):
    """Loads the main metrics data from the CSV file."""
    separator_idx = find_separator_index(csv_path)
    try:
        if separator_idx is None:
            logging.warning(f"Could not find collective metrics separator in {csv_path}. Reading entire file.")
            df = pd.read_csv(csv_path)
        else:
            rows_to_read = separator_idx
            logging.info(f"Separator line found at index {separator_idx}. Reading {rows_to_read} data rows.")
            df = pd.read_csv(csv_path, nrows=rows_to_read)
        df.dropna(how='all', inplace=True) # Drop rows where ALL values are NaN
        logging.info(f"Loaded {df.shape[0]} rows from CSV.")
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

    # Convert to numeric, coercing errors
    df_numeric = df.copy()
    numeric_cols = [col for col in df.columns if col != 'Filename']
    for col in numeric_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # Separate filename if present
    original_filenames = None
    df_numeric_orig_for_stats = df_numeric.copy() # Keep copy before dropping filename for stats

    if 'Filename' in df_numeric.columns:
        original_filenames = df_numeric['Filename'].tolist() # Keep original names
        try:
            df_numeric = df_numeric.drop(columns=['Filename'])
        except KeyError: pass
    else:
        logging.warning("No 'Filename' column found. Using index.")
        original_filenames = [f"File Index {i}" for i in df_numeric.index]

    # Create anonymized IDs (still useful for prompts even if not sent externally)
    anonymized_ids = [f"File {i+1}" for i in range(len(original_filenames))]

    # Select only numeric columns for processing
    df_numeric_only = df_numeric.select_dtypes(include=np.number)
    if df_numeric_only.empty:
        logging.error("No numeric columns found after attempting conversion.")
        return None, None, None, None, None

    # Impute missing values (using mean for simplicity)
    imputer = SimpleImputer(strategy='mean')
    df_imputed_values = imputer.fit_transform(df_numeric_only)
    if np.isnan(df_imputed_values).any():
         logging.warning("NaNs detected *after* imputation with mean. This might indicate columns with all NaNs. Filling remaining NaNs with 0.")
         df_imputed_values = np.nan_to_num(df_imputed_values, nan=0.0)

    df_imputed = pd.DataFrame(df_imputed_values,
                              columns=df_numeric_only.columns,
                              index=df_numeric_only.index)

    # Normalize data (0-1 scaling) for consistent interpretation and clustering
    scaler = MinMaxScaler()
    if df_imputed.isnull().values.any():
         logging.error("NaNs still present before scaling, despite imputation attempts. Aborting.")
         return None, None, None, None, None

    df_normalized_values = scaler.fit_transform(df_imputed)
    df_normalized = pd.DataFrame(df_normalized_values,
                                 columns=df_imputed.columns,
                                 index=df_imputed.index)

    logging.info(f"Data preprocessed: Imputed NaNs, Normalized {df_normalized.shape[1]} numeric columns.")
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


# --- Rule-Based Interpretation Functions (generate_cluster_label_rules, generate_file_summary_rules) ---
# These remain the same as the previous version. Ensure they are included here.
# ... (Paste the rule-based functions from the previous version here) ...
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

    # Simple Logic (can be expanded significantly)
    label = "Mixed/Average Profile" # Default

    if mtld > high_thr and distinct2 > high_thr:
        if avg_len > high_thr and flesch < low_thr:
            label = "Diverse & Complex Syntax"
        elif avg_len < low_thr and flesch > high_thr:
            label = "Diverse & Simple Syntax"
        else:
            label = "High Lexical & Phrase Diversity"
    elif mtld < low_thr and rep2 > high_thr:
        if avg_len < low_thr and flesch > high_thr:
            label = "Repetitive & Simple Syntax"
        else:
            label = "Repetitive, Low Diversity"
    elif flesch < low_thr and avg_len > high_thr:
        label = "Low Readability, Long Sentences"
    elif flesch > high_thr and avg_len < low_thr:
        label = "High Readability, Short Sentences"
    elif mtld > high_thr:
        label = "High Lexical Diversity"
    elif mtld < low_thr:
        label = "Low Lexical Diversity"
    elif rep2 > high_thr:
        label = "High Repetition (Bigrams)"
    elif distinct2 < low_thr:
         label = "Low Phrase Variety (Bigrams)"

    logging.debug(f"Rule-based label for Cluster {cluster_id}: '{label}'")
    return label

def generate_file_summary_rules(anonymized_id, file_metrics_raw, dataset_stats):
    """Generates a comparative summary for a single file using simple rules."""
    if not file_metrics_raw or not dataset_stats or 'mean' not in dataset_stats or 'std' not in dataset_stats:
        return "Dataset statistics unavailable for comparison."

    summary_parts = []
    mean = dataset_stats['mean']
    std = dataset_stats['std']

    # Compare key metrics
    for metric in ['MTLD', 'Average Sentence Length', 'Flesch Reading Ease', 'Repetition-2']:
        if metric not in file_metrics_raw or metric not in mean or metric not in std:
            continue # Skip if metric or stats missing

        val = file_metrics_raw[metric]
        m = mean[metric]
        s = std[metric]

        if pd.isna(val) or pd.isna(m) or pd.isna(s) or s == 0: continue # Cannot compare

        z_score = (val - m) / s

        desc = ""
        if z_score > 1.5: desc = "significantly above average"
        elif z_score > 0.5: desc = "above average"
        elif z_score < -1.5: desc = "significantly below average"
        elif z_score < -0.5: desc = "below average"
        # else: desc = "around average" # Optional to include average cases

        if desc:
            # Add context for readability/repetition
            if metric == 'Flesch Reading Ease':
                desc += f" (implying {'easier' if z_score > 0 else 'harder'} reading)"
            elif metric == 'Repetition-2':
                desc += f" (implying {'more' if z_score > 0 else 'less'} bigram repetition)"
            summary_parts.append(f"{metric} ({val:.2f}) is {desc}.")

    # Add TTR vs MTLD check
    ttr = file_metrics_raw.get('TTR')
    mtld = file_metrics_raw.get('MTLD')
    if pd.notna(ttr) and pd.notna(mtld) and 'TTR' in mean and 'MTLD' in mean:
        ttr_z = (ttr - mean['TTR']) / std.get('TTR', 1) if std.get('TTR', 0) != 0 else 0
        mtld_z = (mtld - mean['MTLD']) / std.get('MTLD', 1) if std.get('MTLD', 0) != 0 else 0
        if ttr_z > 1.0 and mtld_z < 0.0:
            summary_parts.append("High TTR but average/low MTLD might suggest initial novelty fading quickly or short text length effects.")

    if not summary_parts:
        return "Metrics are generally around the dataset average."
    else:
        # Limit summary length
        return " ".join(summary_parts[:3]) # Join first 2-3 notable points


# --- Local LLM API Call Function ---

def call_local_llm_api(prompt, llm_config):
    """Calls a local LLM API (e.g., KoboldCpp) via HTTP POST."""
    if not REQUESTS_AVAILABLE or requests is None:
        return "Error: 'requests' library not installed (pip install requests)."

    api_base = llm_config.get('api_base')
    if not api_base:
        return "Error: 'api_base' URL not configured in [LocalLLM] section of INI file."

    # Construct payload
    payload = {
        "model": llm_config.get('model', 'local-model'),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(llm_config.get('temperature', 1.0)),
        "max_tokens": int(llm_config.get('max_tokens', 256)),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    endpoint = api_base

    logging.debug(f"Attempting local LLM call to: {endpoint}")
    logging.debug(f"Payload: {json.dumps(payload, indent=2)}")

    last_error = None
    for attempt in range(MAX_RETRIES):
        response = None # Initialize response to None
        try:
            response = requests.post(endpoint, json=payload) # No client-side timeout
            logging.debug(f"Local LLM request sent. Status Code: {response.status_code}")
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

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
                 else:
                      last_error = "Error: 'content' field missing in response message."
                      logging.warning(last_error)
            else:
                 last_error = "Error: Expected 'choices' array missing or empty in response."
                 logging.warning(last_error)

        except requests.exceptions.ConnectionError as e:
            last_error = f"Error: Connection failed to {endpoint} ({e}). Is the backend running?"
            logging.error(last_error)
            break
        except requests.exceptions.RequestException as e:
            last_error = f"Error: Request failed ({e})"
            response_text_on_error = response.text[:500] if response is not None else "N/A"
            logging.error(f"Local LLM request failed (Attempt {attempt + 1}/{MAX_RETRIES}): {e}. Response: {response_text_on_error}")
        except json.JSONDecodeError as e:
             raw_text_on_decode_error = response.text if response is not None else "N/A"
             last_error = f"Error: Failed to decode JSON response from {endpoint}."
             # *** Log the raw text that caused the error ***
             logging.error(f"{last_error} Raw Response: '{raw_text_on_decode_error[:500]}...'", exc_info=True)
             break
        except Exception as e:
             last_error = f"Error: Unexpected error during local LLM call ({type(e).__name__}: {e})"
             logging.error(last_error, exc_info=True)
             break

        if attempt < MAX_RETRIES - 1:
            logging.debug(f"Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    return last_error


# --- Prompt Generation Functions (generate_cluster_label_prompt, generate_file_summary_prompt) ---
# These remain the same as the previous version. Ensure they are included here.
# ... (Paste the prompt generation functions from the previous version here) ...
def generate_cluster_label_prompt(cluster_id, avg_metrics):
    """Generates the prompt for cluster labeling."""
    if not avg_metrics: return None
    valid_metrics = {k: v for k, v in avg_metrics.items() if pd.notna(v)}
    if not valid_metrics:
        logging.warning(f"No valid (non-NaN) average metrics found for Cluster {cluster_id}.")
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

Based *only* on these average normalized metrics, provide a concise, descriptive label (max 10 words) summarizing the dominant linguistic profile of files in this cluster. Focus on aspects like lexical diversity, sentence complexity, readability, and repetition. Examples: 'High Diversity, Complex Syntax', 'Simple Vocabulary, Repetitive', 'Balanced Profile', 'Low Readability, Varied Phrasing'.
"""
    return prompt

def generate_file_summary_prompt(anonymized_id, file_metrics_raw, file_metrics_norm, dataset_stats):
    """Generates the prompt for file summary (using anonymized ID)."""
    if file_metrics_raw is None or file_metrics_norm is None:
         logging.warning(f"Missing raw or normalized metrics for {anonymized_id}. Cannot generate prompt.")
         return None

    raw_metrics_str = "\n".join([f"- {k}: {file_metrics_raw.get(k, 'N/A'):.3f}" if isinstance(file_metrics_raw.get(k), (int, float)) else f"- {k}: {file_metrics_raw.get(k, 'N/A')}" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in file_metrics_raw])
    norm_metrics_str = "\n".join([f"- {k}: {file_metrics_norm.get(k, 'N/A'):.3f}" if pd.notna(file_metrics_norm.get(k)) else f"- {k}: N/A" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in file_metrics_norm]) # Check for NaN in norm

    context_str = ""
    if dataset_stats and isinstance(dataset_stats.get('mean'), dict):
        context_str += "\nDataset Averages (Raw):\n" + "\n".join([f"- {k}: {dataset_stats['mean'].get(k, 'N/A'):.3f}" if pd.notna(dataset_stats['mean'].get(k)) else f"- {k}: N/A" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in dataset_stats['mean']])
    if dataset_stats and isinstance(dataset_stats.get('std'), dict):
        context_str += "\nDataset Std Dev (Raw):\n" + "\n".join([f"- {k}: {dataset_stats['std'].get(k, 'N/A'):.3f}" if pd.notna(dataset_stats['std'].get(k)) else f"- {k}: N/A" for k in KEY_METRICS_FOR_FILE_SUMMARY if k in dataset_stats['std']])

    prompt = f"""
Analyze the linguistic metrics for the document identified as: "{anonymized_id}"

Raw Metrics:
{raw_metrics_str}

Normalized Metrics (0-1 scale):
{norm_metrics_str}

Context (Dataset Statistics):
{context_str}

Provide a brief summary (2-4 sentences) interpreting these metrics for this specific document.
- Compare its key metrics (especially {', '.join(KEY_METRICS_FOR_FILE_SUMMARY[:4])}) to the dataset averages/context provided.
- Highlight 1-2 notable characteristics (e.g., significantly higher/lower diversity, complexity, readability, repetition than average).
- Briefly explain any interesting combinations (e.g., if TTR is high but RTTR/MTLD are average/low, what might that imply?).
- Focus on providing insight based *only* on the provided numbers. Do not mention the anonymized ID in your response.
"""
    return prompt


# --- HTML Update Function (update_html_report) ---
# This remains the same as the previous version. Ensure it is included here.
# ... (Paste the HTML update function from the previous version here) ...
def update_html_report(html_path, rules_results, llm_results, original_filenames, cluster_assignments, llm_config):
    """Adds the generated interpretations (Rules and optionally LLM) to the HTML report."""
    if not BS4_AVAILABLE:
        logging.error("Cannot update HTML report: BeautifulSoup4 not available.")
        return False
    if not rules_results and not llm_results:
        logging.warning("No interpretations generated to add to HTML.")
        return False

    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')

        heatmap_img = soup.find('img', alt=lambda x: x and 'Clustered Heatmap' in x)
        insertion_point_element = None
        if heatmap_img:
            parent_div = heatmap_img.find_parent('div')
            insertion_point_element = parent_div if parent_div else heatmap_img
            logging.info("Found clustered heatmap image/div for insertion.")
        else:
            logging.warning("Clustered heatmap image not found in HTML. Appending interpretations to the end of the body.")
            insertion_point_element = soup.body

        if not insertion_point_element:
             logging.error("Could not find a suitable insertion point (heatmap or body tag) in HTML.")
             return False

        # --- Remove existing section if present ---
        existing_section = soup.find('div', id='interpretations_section')
        if existing_section:
            logging.warning("Previous interpretation section found in HTML. Removing it before adding new content.")
            existing_section.decompose()

        # --- Create Main Interpretation Container ---
        main_interp_div = soup.new_tag('div', id='interpretations_section', style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #666;")
        main_title = soup.new_tag('h2', style="margin-bottom: 15px;")
        main_title.string = "Generated Interpretations"
        main_interp_div.append(main_title)

        # --- Function to add results for a specific mode ---
        def add_mode_results_to_html(mode_name, results_dict, config_dict=None):
            mode_div = soup.new_tag('div', style="margin-bottom: 30px; padding-bottom: 15px; border-bottom: 1px solid #eee;")
            mode_h3 = soup.new_tag('h3')
            mode_h3.string = f"Interpretation Mode: {mode_name}"
            mode_div.append(mode_h3)

            if mode_name == 'Local LLM' and config_dict:
                 model_info = soup.new_tag('p', style="font-size: 0.9em; color: #555;")
                 model_info.string = f"Model Used (from INI): {config_dict.get('model', 'N/A')}"
                 mode_div.append(model_info)

            # Add Cluster Labels
            cluster_labels = results_dict.get('cluster_labels')
            if cluster_labels:
                cluster_h4 = soup.new_tag('h4', style="margin-top: 15px;")
                cluster_h4.string = f"Cluster Labels (Top {len(cluster_labels)} Clusters)"
                mode_div.append(cluster_h4)
                cluster_p = soup.new_tag('p', style="font-size: 0.9em; color: #555;")
                cluster_p.string = "Estimated linguistic profiles based on average metrics within each cluster."
                mode_div.append(cluster_p)
                cluster_dl = soup.new_tag('dl', style="margin-left: 20px;")
                for cluster_id, label in sorted(cluster_labels.items()):
                    dt = soup.new_tag('dt', style="font-weight: bold; margin-top: 8px;")
                    dt.string = f"Cluster {cluster_id}:"
                    dd = soup.new_tag('dd', style="margin-left: 20px; margin-bottom: 5px;")
                    label_text = label if label and not label.startswith("Error:") else f"<i style='color:red;'>{label}</i>" if label else "<i style='color:red;'>Label generation failed.</i>"
                    dd.append(BeautifulSoup(label_text, 'html.parser'))
                    cluster_dl.append(dt)
                    cluster_dl.append(dd)
                mode_div.append(cluster_dl)
            else:
                 p_no_labels = soup.new_tag('p', style="font-style: italic;")
                 p_no_labels.string = "No cluster labels were generated (or clustering was skipped)."
                 mode_div.append(p_no_labels)

            # Add File Summaries
            file_summaries = results_dict.get('file_summaries')
            if file_summaries and original_filenames and len(file_summaries) == len(original_filenames):
                summary_h4 = soup.new_tag('h4', style="margin-top: 25px;")
                summary_h4.string = "Individual File Summaries"
                mode_div.append(summary_h4)
                summary_p = soup.new_tag('p', style="font-size: 0.9em; color: #555;")
                summary_p.string = "Brief interpretation comparing each file's metrics to the dataset."
                mode_div.append(summary_p)
                summary_dl = soup.new_tag('dl', style="margin-left: 20px;")
                summary_map = dict(zip(original_filenames, file_summaries))

                for i, orig_fname in enumerate(original_filenames):
                    cluster_id_str = 'N/A'
                    if cluster_assignments is not None and i < len(cluster_assignments):
                         cluster_id_str = str(cluster_assignments[i])
                    summary = summary_map.get(orig_fname, "Error: Summary not found for this file.")

                    dt = soup.new_tag('dt', style="font-weight: bold; margin-top: 8px;")
                    dt.string = f"{orig_fname} (Cluster {cluster_id_str}):"
                    dd = soup.new_tag('dd', style="margin-left: 20px; margin-bottom: 5px;")
                    summary_lines = summary.split('\n') if summary and not summary.startswith("Error:") else [f"<i style='color:red;'>{summary}</i>"] if summary else ["<i style='color:red;'>Summary generation failed.</i>"]
                    for line_num, line in enumerate(summary_lines):
                         dd.append(BeautifulSoup(line, 'html.parser'))
                         if line_num < len(summary_lines) - 1:
                              dd.append(soup.new_tag('br'))
                    summary_dl.append(dt)
                    summary_dl.append(dd)
                mode_div.append(summary_dl)
            else:
                 p_no_summaries = soup.new_tag('p', style="font-style: italic;")
                 if not original_filenames: p_no_summaries.string = "Cannot display file summaries: Original filenames missing."
                 elif not file_summaries: p_no_summaries.string = "No file summaries were generated."
                 else: p_no_summaries.string = f"Cannot display file summaries: Mismatch between summary count ({len(file_summaries)}) and file count ({len(original_filenames)})."
                 mode_div.append(p_no_summaries)

            return mode_div # Return the generated div for this mode

        # --- Add Rule-Based Results ---
        if rules_results:
            rules_div = add_mode_results_to_html("Rule-Based", rules_results)
            if rules_div: main_interp_div.append(rules_div)

        # --- Add Local LLM Results ---
        if llm_results:
            llm_div = add_mode_results_to_html("Local LLM", llm_results, llm_config)
            if llm_div: main_interp_div.append(llm_div)


        # --- Insert the Main Interpretation Section into the HTML ---
        if insertion_point_element == soup.body:
             insertion_point_element.append(main_interp_div)
             logging.info("Appended interpretations to the end of the HTML body.")
        else:
             insertion_point_element.insert_after(main_interp_div)
             logging.info("Inserted interpretations after the heatmap element.")

        # --- Save the modified HTML back to the ORIGINAL file ---
        output_html_path = html_path # Write back to the input path
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(str(soup.prettify()))
        logging.info(f"Successfully updated HTML file: {output_html_path}")
        return True

    except FileNotFoundError:
        logging.error(f"HTML file '{html_path}' not found for updating.")
        return False
    except Exception as e:
        logging.error(f"Failed to update HTML file: {e}", exc_info=True)
        return False


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load INI Configuration
    config = configparser.ConfigParser()
    ini_path = os.path.join(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd(), INI_FILENAME)
    if not os.path.exists(ini_path):
        logging.critical(f"CRITICAL: Configuration file '{INI_FILENAME}' not found.")
        sys.exit(1)
    try:
        config.read(ini_path)
        logging.info(f"Loaded configuration from '{INI_FILENAME}'.")
    except configparser.Error as e:
        logging.critical(f"CRITICAL: Error parsing configuration file '{INI_FILENAME}': {e}")
        sys.exit(1)

    # Check if Local LLM mode is enabled and configured
    local_llm_enabled = False
    local_llm_config = {}
    if 'LocalLLM' in config:
        local_llm_enabled = config.getboolean('LocalLLM', 'enabled', fallback=False)
        if local_llm_enabled:
            local_llm_config = dict(config.items('LocalLLM'))
            logging.info(f"Local LLM mode is enabled. Config: {local_llm_config}")
            if not local_llm_config.get('api_base'):
                 logging.warning("Local LLM mode enabled, but 'api_base' is not set in [LocalLLM] section. LLM calls will likely fail.")
            if not REQUESTS_AVAILABLE:
                 logging.error("Local LLM mode enabled, but 'requests' library is not installed. Cannot use Local LLM.")
                 local_llm_enabled = False # Disable if library missing
        else:
            logging.info("Local LLM mode is disabled in configuration.")
    else:
        logging.info("[LocalLLM] section not found in INI file. Local LLM mode disabled.")


    # 2. Get Target Folder
    target_folder = input(f"Enter the path to the folder containing '{CSV_FILENAME}' and '{HTML_FILENAME}': ")
    if not os.path.isdir(target_folder):
        logging.critical(f"CRITICAL: Invalid folder path '{target_folder}'")
        sys.exit(1)

    csv_path = os.path.join(target_folder, CSV_FILENAME)
    html_path = os.path.join(target_folder, HTML_FILENAME) # Path to the file to be modified
    if not os.path.isfile(csv_path): logging.critical(f"CRITICAL: CSV file '{CSV_FILENAME}' not found."); sys.exit(1)
    if not os.path.isfile(html_path): logging.critical(f"CRITICAL: HTML file '{HTML_FILENAME}' not found."); sys.exit(1)

    # 3. Load and Preprocess Data
    logging.info("--- Loading and Preprocessing Data ---")
    df_raw = load_data(csv_path)
    if df_raw is None: sys.exit(1)
    df_numeric_orig, df_imputed, df_normalized, original_filenames, anonymized_ids = preprocess_data(df_raw)
    if df_normalized is None or original_filenames is None or anonymized_ids is None:
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
    num_files = len(df_normalized)
    actual_n_clusters = min(N_CLUSTERS_TO_LABEL, num_files) if num_files > 0 else 0
    cluster_assignments = None
    if linkage_matrix is not None and actual_n_clusters >= 2:
        cluster_assignments = get_flat_clusters(linkage_matrix, actual_n_clusters, num_files)
        if cluster_assignments is None:
             logging.warning("Failed to derive flat clusters.")
             actual_n_clusters = 0
        else:
             logging.info(f"Successfully derived {actual_n_clusters} cluster assignments.")
    else:
        logging.warning(f"Skipping flat cluster derivation.")
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
    rules_results = {}
    llm_results = {}
    llm_generation_failed = False # Flag to track critical LLM failures

    # --- Always run Rule-Based Interpretation ---
    logging.info(f"--- Starting Interpretation Generation (Mode: Rule-Based) ---")
    rules_results['mode'] = 'Rules'
    # Generate Cluster Labels
    rule_cluster_labels = {}
    if actual_n_clusters >= 2 and cluster_avg_metrics_norm:
        logging.info("--- Generating Rule-Based Cluster Labels ---")
        for cluster_id, avg_metrics in cluster_avg_metrics_norm.items():
            label = generate_cluster_label_rules(cluster_id, avg_metrics)
            rule_cluster_labels[cluster_id] = label
        rules_results['cluster_labels'] = rule_cluster_labels
    else:
         rules_results['cluster_labels'] = {}

    # Generate File Summaries
    rule_file_summaries = []
    logging.info("--- Generating Rule-Based File Summaries ---")
    for i in range(num_files):
        anonymized_id = anonymized_ids[i]
        file_metrics_raw = df_numeric_orig.iloc[i].to_dict() if i < len(df_numeric_orig) else None
        summary = generate_file_summary_rules(anonymized_id, file_metrics_raw, dataset_stats)
        rule_file_summaries.append(summary)
    rules_results['file_summaries'] = rule_file_summaries
    logging.info(f"--- Rule-Based Interpretation Generation Complete ---")


    # --- Optionally run Local LLM Interpretation ---
    if local_llm_enabled:
        logging.info(f"--- Starting Interpretation Generation (Mode: Local LLM) ---")
        llm_results['mode'] = 'Local LLM'
        llm_results['model_used'] = local_llm_config.get('model', 'N/A')
        # Generate Cluster Labels
        llm_cluster_labels = {}
        if actual_n_clusters >= 2 and cluster_avg_metrics_norm:
            logging.info(f"--- Generating Local LLM Cluster Labels ({local_llm_config.get('model')}) ---")
            for cluster_id, avg_metrics in cluster_avg_metrics_norm.items():
                prompt = generate_cluster_label_prompt(cluster_id, avg_metrics)
                if prompt:
                    label = call_local_llm_api(prompt, local_llm_config)
                    llm_cluster_labels[cluster_id] = label
                    if label is not None and label.startswith("Error:"):
                         logging.error(f"Critical error generating cluster label {cluster_id} with LLM. Stopping LLM generation.")
                         llm_generation_failed = True
                         break # Stop trying cluster labels
                    # time.sleep(0.1) # Optional small delay
            if not llm_generation_failed:
                 llm_results['cluster_labels'] = llm_cluster_labels
            else:
                 llm_results['cluster_labels'] = {"Error": "LLM generation failed during cluster labeling."} # Indicate failure
        else:
             llm_results['cluster_labels'] = {}

        # Generate File Summaries ONLY if cluster labels didn't fail critically
        if not llm_generation_failed:
            llm_file_summaries = []
            logging.info(f"--- Generating Local LLM File Summaries ({local_llm_config.get('model')}) ---")
            for i in range(num_files):
                # *** ADDED PROGRESS LOGGING ***
                logging.info(f"Generating LLM summary for file {i+1}/{num_files}...")
                anonymized_id = anonymized_ids[i]
                file_metrics_raw = df_numeric_orig.iloc[i].to_dict() if i < len(df_numeric_orig) else None
                file_metrics_norm = df_normalized.iloc[i].to_dict() if i < len(df_normalized) else None
                prompt = generate_file_summary_prompt(anonymized_id, file_metrics_raw, file_metrics_norm, dataset_stats)
                if prompt:
                    summary = call_local_llm_api(prompt, local_llm_config)
                    llm_file_summaries.append(summary)
                    if summary is not None and summary.startswith("Error:"):
                         logging.error(f"Critical error generating file summary {i+1} with LLM. Stopping LLM generation.")
                         llm_generation_failed = True
                         break # Stop trying file summaries
                    # time.sleep(0.1) # Optional small delay
                else:
                     logging.warning(f"Could not generate prompt for file index {i} ({anonymized_id}). Appending error message.")
                     llm_file_summaries.append("Error: Could not generate prompt for this file.")
            # Store summaries only if loop completed without critical failure
            if not llm_generation_failed:
                 llm_results['file_summaries'] = llm_file_summaries
            else:
                 # Add partial summaries if any were generated before failure? Or just mark as failed?
                 # Let's just mark as failed for simplicity
                 llm_results['file_summaries'] = ["Error: LLM generation failed during file summaries."] * num_files


        if llm_generation_failed:
             logging.error("Local LLM interpretation generation stopped due to errors.")
             # Clear potentially partial results to avoid confusion? Or keep errors? Keep errors for now.
             # llm_results = {} # Option to clear results on failure
        else:
             logging.info(f"--- Local LLM Interpretation Generation Complete ---")
    else:
        logging.info("Skipping Local LLM interpretation as it's disabled in the INI file.")


    # 8. Update HTML Report with all collected interpretations
    logging.info("--- Updating HTML Report ---")
    # Pass both results dictionaries to the update function
    # llm_results might be empty if disabled or if generation failed critically
    success = update_html_report(html_path, rules_results, llm_results, original_filenames, cluster_assignments, local_llm_config)
    if success:
        logging.info("Interpretation script finished successfully.")
    else:
        logging.error("Interpretation script finished with errors during HTML update.")
        sys.exit(1)

    logging.info("--- Script Finished ---")