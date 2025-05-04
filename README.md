# Text Analysis Suite

Version 1.5

## Creator

**https://github.com/DoorBotEvilTwin/**

## Overview

This Python project provides a modular suite of scripts designed for offline, comprehensive analysis of text file collections (`.txt`). It calculates a wide array of linguistic metrics, generates insightful visualizations, and optionally leverages local Large Language Models (LLMs) for automated interpretation and synthesis. The suite is structured for a sequential workflow, ideally managed by the `run_suite.py` wrapper script, ensuring data flows correctly between modules.

**Version 1.5 represents a major overhaul**, introducing several new analysis dimensions (Sentiment, Emotion, Topic Modeling, Keywords, expanded Readability), new visualizations, LLM-driven plot interpretation (Transformer module), and enhanced LLM metric interpretation capabilities (Processor module), along with workflow improvements like incremental HTML updates and a unified settings file.

The intended workflow is:
**Analyzer -> Visualizer -> Transformer -> Processor -> Optimizer**

## Modules

1.  **`lexicon_analyzer.py` (Metrics Calculation)**
    *   **Input:** Reads `.txt` files from a `txt` subdirectory within the target folder.
    *   **Function:** Calculates a broad spectrum of metrics for each file, including basic counts, readability scores (Flesch, Gunning Fog, SMOG, Dale-Chall via `textstat`), extensive lexical diversity measures (TTR, RTTR, LogTTR, MATTR, MSTTR, VOCD, MTLD, Yule's K, Simpson's D), syntactic/phrase diversity (Lexical Density, N-gram distinctiveness/repetition), sentiment scores (VADER, TextBlob), emotion detection (NRCLex), topic modeling (NMF), and keyword extraction (TF-IDF, YAKE!). It also computes collective metrics (Self-BLEU, Embedding Similarity/Distance) across the entire dataset.
    *   **Output:** Generates `_LEXICON.csv` (raw data), `_LEXICON.txt` (formatted text), and `_LEXICON.html` (basic report structure with sortable tables). Also creates `_LEXICON_errors.log`.

2.  **`lexicon_visualizer.py` (Visualization & Report Enhancement)**
    *   **Input:** Reads `_LEXICON.csv`.
    *   **Function:** Processes the metric data to generate a variety of plots visualizing different aspects of the analysis (distributions, comparisons, correlations, profiles, heatmaps). It embeds these plots directly into `_LEXICON.html`, adds detailed descriptions for each metric and plot, and inserts a Table of Contents for navigation.
    *   **Output:** Updates `_LEXICON.html` into a rich visual report. Saves individual plot images (`.png`) into a `plots` subdirectory. Critically, it also saves clustering data (`_linkage_matrix.npy`, `_reordered_indices.npy`), filename mapping (`_original_filenames.json`), and plot metadata (`_plot_metadata.json`) required by the Transformer and Processor modules.

3.  **`lexicon_transformer.py` (LLM Plot Interpretation - Experimental)**
    *   **Input:** Reads `_LEXICON.html`, `plots/clustered_metrics_heatmap.png`, `_linkage_matrix.npy`, `_reordered_indices.npy`, `_plot_metadata.json`, and uses `lexicon_settings.ini` for LLM configuration.
    *   **Function:** Uses a **multimodal** local LLM to first interpret the row dendrogram structure within the `clustered_metrics_heatmap.png`. It prompts the LLM to identify major branches and provide a numbered list explaining their inferred meaning. Based on the LLM's response, it modifies the heatmap image by adding corresponding numbered labels (`#1`, `#2`, etc.) using Pillow. It then iterates through the *other* plots (identified via `_plot_metadata.json`), encodes them, and prompts the multimodal LLM to generate a summary for each.
    *   **Output:** Updates `_LEXICON.html` *incrementally* after each LLM call. It replaces the original heatmap with the numbered version, inserts the LLM's dendrogram explanation below it, and inserts the LLM's summary for each of the other plots directly below their respective images. Saves the numbered heatmap as `plots/clustered_metrics_heatmap_numbered.png`. Appends to `_LEXICON_errors.log`. *(Note: This module can be slow due to multiple multimodal LLM calls).*

4.  **`lexicon_processor.py` (Metric Interpretation & Synthesis - Optional)**
    *   **Input:** Reads `_LEXICON.csv`, `_LEXICON.html`, clustering/filename data (`.npy`, `.json`), and uses `lexicon_settings.ini`.
    *   **Function:** Performs further analysis and interpretation based on the calculated metrics. It runs offline and offers two interpretation modes:
        *   **Rule-Based Mode (Always Run):** Uses predefined Python logic and dataset statistics to generate simple descriptive labels for file clusters (derived from the heatmap clustering) and brief comparative summaries for individual files based on their metrics.
        *   **Local LLM Mode (Optional):** If enabled in the `.ini` file, connects to a locally running **text-based** LLM. It sends prompts (with anonymized file IDs and relevant metrics) to generate potentially more nuanced cluster labels and file summaries. It can optionally generate brief content summaries if `CONTENT_SUMMARY_ENABLED` is set to `True` (requires access to the `txt` subfolder). It concludes by prompting the LLM for a final synthesis report integrating all available analysis findings (numerical, rule-based, LLM interpretations).
    *   **Output:** Updates `_LEXICON.html` *incrementally*, appending interpretation sections (Rule-Based and optionally LLM-Based) *after* the content added by the Transformer module. Appends to `_LEXICON_errors.log`.

5.  **`lexicon_optimizer.py` (LLM Meta-Analysis - Optional)**
    *   **Input:** Reads the final `_LEXICON.html` and uses `lexicon_settings.ini`.
    *   **Function:** Performs a final meta-analysis using a **text-based** local LLM. It reads the *entire* textual content and plot references from the HTML report and prompts the LLM to provide a high-level overview and synthesis of the complete analysis. It also calculates and adds an estimated LLM context size requirement near the top of the report.
    *   **Output:** Updates `_LEXICON.html` one last time by appending the LLM meta-analysis section (within `<pre>` tags for raw formatting) and adding a link to it in the Table of Contents. Appends to `_LEXICON_errors.log`.

The suite is particularly useful for analyzing and comparing outputs from Large Language Models (LLMs), but can be applied to any collection of text documents seeking detailed linguistic profiling and interpretation. A top-level shortcut script `FileAnalysisSuite_v1.5.py` is also provided to launch `run_suite.py` from a parent directory, assuming the suite scripts are in a subfolder named `FAS_1.5`.

*Note on Data Privacy:* While the Processor script *can* be configured to send raw text snippets to a *local* LLM for content summarization (if `CONTENT_SUMMARY_ENABLED=True` in the script), the default configuration and core design emphasize analyzing metrics derived from the text, not the raw text itself, especially concerning potential future API integration. Filenames are never sent to the LLM; anonymized IDs are used in prompts.

## Features: Calculated Metrics (v1.5)

The main script (`lexicon_analyzer.py`) calculates the following metrics for each input text file. These are grouped logically in the output files and HTML report.

### Basic Info

1.   **Filename:** The name of the input text file.
2.   **Total Words:** Total number of tokens (words) identified in the file using NLTK's tokenizer.
3.   **Total Characters:** Total number of characters (including spaces and punctuation) in the raw file content.
4.   **Unique Words:** Number of distinct word types (case-insensitive) found in the file.
5.   **Average Sentence Length:** The average number of words per sentence (Total Words / Number of Sentences). Provides a basic measure of syntactic complexity.

### Readability Indices

6.   **Flesch Reading Ease:** A standard score indicating text difficulty (0-100 scale). Higher scores indicate easier readability, typically based on average sentence length and syllables per word.
7.   **Gunning Fog:** Readability index estimating years of formal education needed. Higher score means harder to read.
8.   **SMOG Index:** Readability index estimating years of education needed, often used for health literature. Higher score means harder to read.
9.   **Dale-Chall Score:** Readability score based on average sentence length and percentage of words *not* on a list of common 'easy' words. Higher score means harder to read.

### Lexical Diversity (Word Level)

10.   **TTR:** The simplest diversity measure: `Unique Words / Total Words`. Highly sensitive to text length. Values closer to 1 indicate higher diversity within that specific text length.
11.   **RTTR:** An attempt to correct TTR for length: `Unique Words / sqrt(Total Words)`. Less sensitive to length than TTR, but less robust than VOCD/MTLD.
12.   **Herdan's C (LogTTR):** Another length normalization attempt: `log(Unique Words) / log(Total Words)`. Assumes a specific logarithmic relationship. Higher values suggest greater diversity relative to length under this model.
13.   **MATTR (W=100):** Calculates TTR over sliding windows (default 100 words) and averages the results. Measures *local* lexical diversity and is less sensitive to overall text length than TTR.
14.   **MSTTR (W=100):** Calculates TTR over sequential, *non-overlapping* segments (default 100 words) and averages them. Provides a different view of segmental diversity compared to MATTR.
15.   **VOCD:** A sophisticated and robust measure designed to be largely independent of text length. It models the theoretical relationship between TTR and text length using hypergeometric distribution probabilities (specifically, the HD-D variant). Higher scores indicate greater underlying vocabulary richness.
16.   **MTLD:** Another highly robust, length-independent measure. It calculates the average number of sequential words required for the local TTR (calculated over the growing sequence) to fall below a specific threshold (typically 0.72). A higher MTLD score indicates greater diversity, meaning longer stretches of text were needed before vocabulary repetition became significant.
17.   **Yule's K:** A measure focusing on vocabulary richness via word *repetition* patterns (derived from the frequency distribution). Unlike TTR-based measures, it's less sensitive to words occurring only once. Lower values indicate higher repetition of the most frequent words (lower diversity); higher values indicate more even word usage across the vocabulary (higher diversity). Calculated manually.
18.   **Simpson's D:** Originally an ecological diversity index, applied here it measures the probability that two words selected randomly from the text will be of the same *type*. It reflects vocabulary *concentration* or the dominance of frequent words. Higher values (closer to 1) indicate *higher* concentration and therefore *lower* lexical diversity. Calculated manually.

### Syntactic/Phrase Diversity

19.   **Lexical Density:** The ratio of content words (nouns, verbs, adjectives, adverbs) to the total number of words. A higher density often suggests text that is more informationally packed or descriptive, as opposed to text with a higher proportion of function words (pronouns, prepositions, conjunctions).
20.   **Distinct-2:** The proportion of unique bigrams (adjacent 2-word sequences) relative to the total number of bigrams in the text. A higher value indicates greater variety in word pairings and less reliance on repeated phrases.
21.   **Repetition-2:** The proportion of bigram *tokens* that are repetitions of bigram *types* already seen earlier in the text. A higher value indicates more immediate phrase repetition, potentially impacting fluency.
22.   **Distinct-3:** The proportion of unique trigrams (adjacent 3-word sequences) relative to the total number of trigrams. A higher value indicates greater variety in short three-word phrases.
23.   **Repetition-3:** The proportion of trigram *tokens* that are repetitions of trigram *types* already seen earlier. Higher values indicate more repetition of three-word phrases.

### Sentiment & Emotion

24.   **Sentiment (VADER Comp):** VADER compound sentiment score (-1=most negative, +1=most positive). Good for general/social media text.
25.   **Sentiment (VADER Pos):** Proportion of text classified as positive by VADER.
26.   **Sentiment (VADER Neu):** Proportion of text classified as neutral by VADER.
27.   **Sentiment (VADER Neg):** Proportion of text classified as negative by VADER.
28.   **Sentiment (TextBlob Pol):** TextBlob polarity score (-1=negative, +1=positive).
29.   **Sentiment (TextBlob Subj):** TextBlob subjectivity score (0=objective, 1=subjective).
30.   **Emotion (NRCLex Dominant):** The single emotion category with the highest score according to the NRC Emotion Lexicon.
31.   **Emotion (NRCLex Scores JSON):** A JSON string containing the raw counts or scores for each NRC emotion category (e.g., fear, anger, joy, sadness, trust, anticipation, surprise, disgust).

### Topic Modeling

32.   **Topic (NMF ID):** The numerical ID (0 to N-1) of the topic assigned as most likely for this document by the NMF model.
33.   **Topic (NMF Prob):** The probability or weight associated with the document's assignment to its dominant NMF topic.

### Keyword Extraction

34.   **Keywords (TF-IDF Top 5):** The top 5 keywords identified for this document based on TF-IDF scores (term frequency inverse document frequency), separated by semicolons.
35.   **Keywords (YAKE! Top 5):** The top 5 keywords identified for this document using the YAKE! algorithm, separated by semicolons.

### Collective Metrics (Calculated Across the Entire Set - Reported Separately)

36.   **Self-BLEU:** The average pairwise BLEU score calculated between all pairs of documents in the set. Measures surface-level (N-gram) similarity *across* documents. Higher scores indicate the documents are textually very similar to each other (low diversity in the set).
37.   **Avg Pairwise Cosine Similarity:** Calculates a vector embedding for each document and finds the average cosine similarity between all pairs of embeddings. Measures *semantic* similarity across documents. Higher scores (closer to 1) indicate the documents discuss very similar topics or convey similar meanings (low semantic diversity).
38.   **Avg Distance to Centroid:** Calculates the average distance (using cosine distance) of each document's embedding from the mean embedding (centroid) of the entire set. Measures the semantic *spread* or dispersion of the documents. Higher values indicate the documents are more spread out semantically (high semantic diversity).

*(Note: Collective metrics are calculated only if 2 or more files are successfully processed.)*

## Features: Visualizations & Interpretations

The suite generates a comprehensive HTML report (`_LEXICON.html`) that includes the calculated metrics, collective metrics, and the following visualizations and interpretations added sequentially by the different modules:

### Visualizer (`lexicon_visualizer.py`)

*   **Table of Contents:** Adds a navigable TOC at the top of the HTML report.
*   **Embedded Plots:** Generates and embeds the following plots with descriptions:
    *   **Normalized Diversity Metrics Comparison (Bar Chart):**
        *   *Description:* Compares key diversity metrics (TTR, MTLD, VOCD, Distinct-2) across all files.
        *   *How To Read:* Values are normalized (0-1 scale) to allow visual comparison despite different original scales. Higher bars generally indicate higher diversity for that specific metric relative to the other files in the set. Helps quickly identify files that are consistently high or low on these common measures.
    *   **MTLD Score Distribution (Histogram/KDE):**
        *   *Description:* Shows the distribution (histogram and density curve) of MTLD scores across all analyzed files.
        *   *How To Read:* The histogram bars show how many files fall into specific MTLD score ranges. The curve estimates the underlying probability distribution. This helps understand the overall range, central tendency (peak), and spread of MTLD scores in your dataset. Is the diversity generally high or low? Is it consistent or highly variable?
    *   **MTLD vs. VOCD (Scatter Plot):**
        *   *Description:* Plots each file as a point based on its MTLD score (Y-axis) and its VOCD score (X-axis). Points are colored by filename (see legend).
        *   *How To Read:* Look for patterns or trends. Since both are robust diversity measures, expect some positive correlation. Deviations might indicate differences in how the metrics capture diversity (e.g., sensitivity to rare words vs. overall repetition). Use the legend to identify specific files.
    *   **Unique Words vs. Average Sentence Length (Scatter Plot):**
        *   *Description:* Plots each file based on its total Unique Words (Y-axis) and its Average Sentence Length (X-axis). Points are colored by filename (see legend).
        *   *How To Read:* Look for correlations. Do texts with longer sentences tend to use a larger vocabulary (positive correlation)? Or is there no clear relationship? Clusters might indicate different writing styles (e.g., simple sentences with limited vocabulary vs. complex sentences with rich vocabulary). Use the legend to identify specific files.
    *   **Grouped Profile Comparison (Radar Chart):**
        *   *Description:* Creates average "fingerprints" for groups of files. Files are grouped into quartiles (Low, Mid-Low, Mid-High, High) based on their MTLD scores. Each axis represents a different key metric (normalized and sometimes inverted so outward means "more diverse/complex/readable"). Shows the *average* profile for files within each MTLD diversity tier.
        *   *How To Read:* Compare the shapes of the polygons for the different MTLD groups. Does the "High MTLD" group consistently score higher on other diversity/complexity metrics? Are there trade-offs (e.g., does higher diversity correlate with lower readability in this dataset)? This shows average tendencies for different diversity levels. Filenames in each group are listed below the plot.
    *   **Metrics Heatmap (Normalized):**
        *   *Description:* Provides a grid view where rows are files and columns are metrics. The color intensity of each cell represents the normalized value (0-1) of that metric for that file (typically, brighter colors like yellow mean higher normalized values, darker colors like purple mean lower).
        *   *How To Read:* Look for rows (files) or columns (metrics) with consistently bright or dark colors. Identify blocks of similar colors, which might indicate groups of files with similar metric profiles. Useful for spotting overall patterns and relationships visually at a glance.
    *   **Metrics Correlation Matrix (Heatmap):**
        *   *Description:* Shows the pairwise Pearson correlation coefficient (-1 to +1) between all numeric metrics calculated across all files.
        *   *How To Read:* Colors indicate the strength and direction of the correlation (e.g., warm colors like red for strong positive correlation, cool colors like blue for strong negative). The number in each cell is the correlation coefficient. Look for strongly correlated metrics (measuring similar underlying properties) or strongly anti-correlated metrics (measuring opposite properties). Helps understand redundancy and relationships between measures in *your specific dataset*.
    *   **Parallel Coordinates Profile (Normalized & Grouped):**
        *   *Description:* A "super-chart" visualizing multiple key metrics simultaneously. Each file is represented by a line that connects points on parallel vertical axes. Each axis represents a different metric, normalized to a 0-1 scale (with Repetition/Simpson's D inverted for consistent interpretation where higher=better/more diverse). Lines are color-coded based on the file's MTLD quartile group.
        *   *How To Read:* Files with similar overall profiles across the selected metrics will have lines that follow similar paths and cluster together visually. The color-coding helps see if files within the same MTLD group exhibit similar patterns across other metrics (e.g., do 'High MTLD' lines generally stay high on other diversity axes?). Outliers will have lines that deviate significantly. Observe group trends rather than trying to trace every individual line. Filenames in each group are listed below the plot.
    *   **Readability Profiles (Grouped Bar Chart):**
        *   *Description:* Compares normalized readability scores (0-1 scale) across files. Higher values indicate *easier* reading for Flesch RE, and *harder* reading for Gunning Fog, SMOG, and Dale-Chall after normalization.
        *   *How To Read:* Each group of bars represents a file. Within each group, compare the heights of the bars for different indices. Files with consistently high bars across all indices (after normalization logic) are likely easier to read, while those with low bars are harder. Note: Flesch RE is inverted during normalization so higher always means 'easier' relative to the dataset range for this plot.
    *   **Average Emotion Scores by MTLD Group (Grouped Bar Chart):**
        *   *Description:* Shows the average score for each NRCLex emotion category, grouped by the file's MTLD diversity quartile.
        *   *How To Read:* This grouped bar chart compares the average emotional profile for files within different diversity tiers (Low to High MTLD). Look for emotions that are significantly higher or lower in certain diversity groups. For example, do high-diversity texts tend to express more 'anticipation' or 'trust' on average compared to low-diversity texts in this dataset? Filenames in each group are listed below the plot.
    *   **Aggregated NRCLex Emotion Scores (Horizontal Bar Chart):**
        *   *Description:* Shows the total score (summed across all files) for each NRC emotion category.
        *   *How To Read:* This horizontal bar chart indicates the overall emotional tone of the entire dataset according to the NRC lexicon. Longer bars represent emotions that appeared more frequently or with higher intensity across all documents combined. Useful for identifying the dominant emotional signals in the corpus as a whole.
    *   **Clustered Heatmap of All Metrics (Normalized, with Dendrograms):**
        *   *Description:* An enhanced heatmap where both the rows (files) and columns (metrics) have been reordered using hierarchical clustering based on their similarity. Similar files are placed near each other, and metrics that behave similarly across the files are placed near each other. Dendrograms (tree diagrams) alongside show the clustering hierarchy. **This image is the input for `lexicon_transformer.py`.**
        *   *How To Read:* This chart is excellent for identifying distinct groups (clusters) of files that share similar linguistic profiles across *all* measured dimensions. Look for blocks of color indicating these groups. Observe which metrics cluster together – this reinforces findings from the correlation matrix about related measures. **Interpret the characteristics of a file cluster by examining the typical color patterns (high/low normalized values) across the metric columns for that block.** The dendrograms show how clusters are related. *(The HTML report will show a **numbered version** of this image with an LLM-generated explanation of the dendrogram below it, added by `lexicon_transformer.py`. The numbers on the image correspond vertically to the numbered points in the explanation. Further rule-based and/or LLM interpretations for clusters and files are added below that by `lexicon_processor.py` if run).*
*   **Data Export:** Saves clustering data (`.npy`) and plot metadata (`.json`) for downstream use.

### Transformer (`lexicon_transformer.py`)

*   **Numbered Heatmap:** Replaces the original clustered heatmap in the HTML with a version that has numbered labels (`#1`, `#2`, etc.) added to the row dendrogram area.
*   **LLM Dendrogram Explanation:** Inserts a section below the numbered heatmap containing a textual explanation generated by a multimodal LLM, interpreting the major branches of the dendrogram. The numbered points in the text correspond to the numbered labels added to the image.
*   **LLM Plot Summaries:** Inserts a short textual summary generated by the multimodal LLM directly below *each* of the other plots in the HTML report, providing an AI interpretation of the visual patterns.

### Processor (`lexicon_processor.py`)

*   **Rule-Based Interpretations:** Appends a section to the HTML containing:
    *   *Cluster Labels:* Simple descriptive labels for the main file clusters identified in the heatmap (e.g., "High Diversity, Complex Syntax").
    *   *File Summaries:* Brief comparisons of each file's key metrics against the dataset average.
*   **Local LLM Interpretations (Optional):** If enabled, appends another section containing:
    *   *LLM Cluster Labels:* Potentially more nuanced labels generated by a text LLM.
    *   *LLM File Summaries:* More detailed interpretations of individual file metrics generated by a text LLM.
    *   *LLM Content Summaries (Optional):* If `CONTENT_SUMMARY_ENABLED=True`, includes brief abstracts of the raw text content for each file.
    *   *LLM Final Synthesis:* A concluding paragraph generated by the text LLM, attempting to synthesize all the preceding analysis results (numerical, rule-based, LLM-based).

### Optimizer (`lexicon_optimizer.py`)

*   **LLM Context Size Estimator:** Adds a paragraph near the top of the HTML report estimating the approximate token count needed for an LLM to process the entire report's content.
*   **LLM Meta-Analysis:** Appends a final section to the HTML report containing a high-level meta-analysis of the *entire* report, generated by a text LLM. This is presented within `<pre>` tags to preserve raw formatting.

## Requirements

*   **Python:** Version 3.8 or higher recommended.
*   **Libraries:** See installation steps below. Key libraries include `pandas`, `numpy`, `nltk`, `spacy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `requests`, `Pillow`, `beautifulsoup4`, `lexical-diversity`, `textstat`, `vaderSentiment`, `textblob`, `NRCLex`, `yake`, `sentence-transformers`.
*   **Internet Connection:** Required only for downloading libraries and potentially NLTK/spaCy/HuggingFace data on first use. The core analysis, visualization, and interpretation scripts run offline once dependencies are met.
*   **Local LLM Backend (Optional, for Transformer, Processor, Optimizer):** To use the Local LLM features, you need a separate LLM server application (like KoboldCpp, LM Studio, Oobabooga with API enabled) running on your machine or local network.
    *   **Crucially, for `lexicon_transformer.py`, the LLM model loaded in the backend *must support image interpretation* (multimodal models like LLaVA, CogVLM, Moondream, etc.).** Check your backend and model documentation. Based on internal testing (see `lexicon_settings.ini` notes), a 12B parameter multimodal model is recommended as a minimum for reliable heatmap interpretation.
    *   `lexicon_processor.py` and `lexicon_optimizer.py` only require text generation capabilities.
    *   Ensure the LLM backend has sufficient context size (e.g., **at least 2048 tokens**, more recommended for the Optimizer's meta-analysis). Check the estimate added by the Optimizer script to the HTML report for guidance.

## Installation

1.  **Ensure Python and Pip are installed:** Make sure you have a working Python installation and that the `pip` package installer is available (use `python -m pip` or `py -m pip` if `pip` isn't recognized directly).

2.  **Install Core Analysis & Utility Dependencies:**
    ```bash
    py -m pip install pandas numpy nltk scipy spacy scikit-learn requests Pillow beautifulsoup4 lxml configparser tqdm lexical-diversity textstat vaderSentiment textblob NRCLex yake sentence-transformers torch
    ```
    *(Note: This is a large installation, primarily due to `torch` and `sentence-transformers`. Estimated size: **1 GB - 3 GB+** depending on torch version. `lxml` is recommended for HTML parsing.)*

3.  **Download spaCy Model:**
    ```bash
    py -m spacy download en_core_web_sm
    ```

4.  **Install Visualization Dependencies:**
    ```bash
    py -m pip install matplotlib seaborn
    ```
    *(Estimated size: **~50-100 MB**)*

5.  **NLTK Data (Automatic Download):** `lexicon_analyzer.py` will attempt to automatically download required NLTK data packages (`punkt`, `averaged_perceptron_tagger`, `stopwords`) on its first run if they are not found. TextBlob may also trigger downloads (`brown`, `punkt`).

## Usage

The recommended way to run the full suite is using the `run_suite.py` wrapper script, typically launched via the top-level `FileAnalysisSuite_v1.5.py` shortcut.

### Using the Shortcut (`FileAnalysisSuite_v1.5.py`)

1.  **Place Files:**
    *   Place `FileAnalysisSuite_v1.5.py` in your desired project directory.
    *   Ensure the actual suite scripts (`lexicon_*.py`, `run_suite.py`, `lexicon_settings.ini`) are located in a subfolder named `FAS_1.5` within that same project directory.
    *   Place your `.txt` files to be analyzed inside a folder named `txt` within the `FAS_1.5` subfolder (i.e., `YourProject/FAS_1.5/txt/your_file.txt`).
2.  **Configure LLM (Optional but needed for Transformer/Processor/Optimizer):**
    *   Edit `FAS_1.5/lexicon_settings.ini`.
    *   Set `[LocalLLM]` `enabled = true`.
    *   Set `api_base` to your local LLM server's **full API endpoint URL**.
    *   Specify the `transformer_model` (must be multimodal) and `processor_model` (text is fine).
    *   Adjust `temperature`, `max_tokens` as needed.
3.  **Start Local LLM Backend:** Launch your LLM server application (e.g., KoboldCpp) with the appropriate models loaded (multimodal for transformer, text for processor/optimizer). Ensure sufficient context size (e.g., >2k tokens).
4.  **Run the Shortcut:** Double-click `FileAnalysisSuite_v1.5.py` or run `py FileAnalysisSuite_v1.5.py` from the command line in your project directory.
5.  **Use the Menu:** The `run_suite.py` menu will appear.
    *   Select `1` to run the full sequence (Analyze > Visualize > Transform > Process > Optimize).
    *   Select `2-7` to run individual scripts.
    *   Select `8` to edit `lexicon_settings.ini`.
    *   Select `9` to view the `_LEXICON.html` report.
    *   Select `0` to quit.
6.  **Wait & View:** Monitor the console output. After completion, check the `FAS_1.5` folder for outputs, especially the updated `_LEXICON.html`.

### Running Scripts Individually

You can run scripts one by one from within the `FAS_1.5` folder, but ensure they are run in the correct order: **Analyzer -> Visualizer -> Transformer -> Processor -> Optimizer**. Pass the target folder path (which will be the `FAS_1.5` folder itself) as a command-line argument.

1.  `cd FAS_1.5`
2.  `py lexicon_analyzer.py .` (Note the `.` for current directory)
3.  `py lexicon_visualizer.py .`
4.  *(Start multimodal LLM server)* `py lexicon_transformer.py .`
5.  *(Start text LLM server if using LLM mode)* `py lexicon_processor.py .`
6.  `py lexicon_optimizer.py .`

## Output Files

Generated/Updated inside the **target folder** (e.g., `FAS_1.5`):

1.  **`_LEXICON.csv`:** Comma-separated values file with individual metrics per row (ordered as described above). Collective metrics appended after a separator. Used by subsequent scripts.
2.  **`_LEXICON.txt`:** Formatted text file with the same data for console viewing.
3.  **`_LEXICON.html`:** The main report file. Initially created by the Analyzer, then significantly enhanced by the Visualizer (plots, descriptions, TOC), potentially modified by the Transformer (numbered heatmap, plot summaries), potentially modified again by the Processor (interpretations, synthesis), and finally potentially modified by the Optimizer (meta-analysis).
4.  **`_LEXICON_errors.log`:** Log file with execution details, warnings, and errors from all scripts. Check this first if issues occur.
5.  **`_linkage_matrix.npy`:** NumPy file containing the row linkage matrix from the clustered heatmap (saved by Visualizer). Used by Transformer/Processor.
6.  **`_reordered_indices.npy`:** NumPy file containing the order of rows (files) in the clustered heatmap (saved by Visualizer). Used by Transformer/Processor.
7.  **`_original_filenames.json`:** JSON file mapping the 0-based row index (after filtering) to the original filename (saved by Visualizer). Used by Transformer/Processor.
8.  **`_plot_metadata.json`:** JSON file containing titles and descriptions for generated plots (saved by Visualizer). Used by Transformer.
9.  **`plots/` (subdirectory):** Contains all generated plot images (`.png` files), including:
    *   `normalized_diversity_bar.png`
    *   `mtld_distribution.png`
    *   `mtld_vs_vocd_scatter.png`
    *   `unique_vs_sentlen_scatter.png`
    *   `grouped_profile_radar.png`
    *   `metrics_heatmap.png`
    *   `correlation_matrix.png`
    *   `parallel_coordinates.png`
    *   `readability_profiles.png`
    *   `grouped_emotion_bars.png`
    *   `nrclex_scores_summary.png`
    *   `clustered_metrics_heatmap.png` (Original, saved by Visualizer)
    *   `clustered_metrics_heatmap_numbered.png` (Modified with labels, saved by Transformer)

## Configuration Notes (`lexicon_settings.ini`)

*   This file controls the behavior of the LLM-dependent scripts (Transformer, Processor, Optimizer). Create/edit it in the same directory as the scripts (e.g., `FAS_1.5`).
*   **`[LocalLLM]` Section:**
    *   `enabled`: Set to `true` to enable LLM features, `false` to disable (Transformer/Optimizer will fail, Processor will only use rules).
    *   `api_base`: **Crucial.** Set to the **full URL** of your running local LLM's OpenAI-compatible API endpoint (e.g., `http://127.0.0.1:5001/v1/chat/completions`).
    *   `transformer_model`: Specify the model name loaded in your backend for the Transformer script (e.g., `llava-v1.6-vicuna-13b.Q5_K_M.gguf`). **Must be multimodal.**
    *   `processor_model`: Specify the model name for the Processor and Optimizer scripts (e.g., `Mistral-7B-Instruct-v0.2.Q5_K_M.gguf`). Text-only is sufficient.
    *   `temperature`: Controls randomness (0.0 = deterministic, >1.0 = more creative/random).
    *   `max_tokens`: Default maximum tokens for LLM responses (e.g., `256`). This acts as a base value.
    *   `top_p`, `top_k`: Optional parameters for nucleus/top-k sampling.
*   **Token Length Control:** The scripts use multipliers on the base `max_tokens` for different tasks:
    *   **SHORT:** Base `max_tokens` (e.g., 256) or lower - Used for cluster labels.
    *   **MEDIUM:** Base `max_tokens` x 2.0 (e.g., 512) - Used for individual plot summaries (Transformer) and file metric summaries (Processor).
    *   **LONG:** Base `max_tokens` x 4.0 (e.g., 1024) or higher - Used for detailed heatmap analysis (Transformer) and final synthesis/meta-analysis (Processor/Optimizer).
    *   Adjust `max_tokens` in the `.ini` file to scale all these response lengths.

## Potential Issues & Troubleshooting

*   **Long Runtime:** Processing many large files or using the LLM modules (especially Transformer with many plots) can be time-consuming.
*   **Memory Usage:** Loading embeddings (`lexicon_analyzer.py`) or large local LLMs can consume significant RAM (>16GB often needed for larger models).
*   **Dependencies:** Ensure all libraries are installed in the *same* Python environment. Check `_LEXICON_errors.log` for `ImportError`.
*   **Local LLM Backend:**
    *   Requires a separate application (KoboldCpp, LM Studio, etc.) running *before* executing the scripts.
    *   **Transformer requires a multimodal model.** Check compatibility. A 12B parameter model is recommended minimum.
    *   Verify the `api_base` URL in `lexicon_settings.ini` is correct and accessible. Check for 404s or connection errors in the log.
    *   Ensure the LLM context window is sufficient (>2k tokens recommended).
    *   The backend server might crash or have internal errors. Check the backend's console.
    *   LLM interpretation quality varies greatly. See model comparison notes in `lexicon_settings.ini`.
*   **HTML Structure Dependency:** Scripts modify the HTML sequentially. Running them out of order or significant changes to one script's output structure could break later scripts.
*   **Image Numbering Precision (`lexicon_transformer.py`):** The numbered labels on the heatmap correspond *vertically by order* to the LLM's explanation points but are placed programmatically and may not align perfectly with the geometric branch location.
*   **Logging:** If a script crashes hard, the last few log messages might not be written to `_LEXICON_errors.log`. Check the console output as well.
*   **`_original_filenames.json`:** This file maps the internal row index (0, 1, 2...) used *after* loading and filtering the CSV to the original filename. If you add/remove/reorder files in the `txt` folder and re-run only later scripts (e.g., Transformer), this mapping might become outdated, leading to incorrect filenames being associated with analyses. It's best to re-run the full suite (or at least Analyzer and Visualizer) if the input files change.
