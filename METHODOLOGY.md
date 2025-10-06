# Methodology - Hybrid NLP Riz Score Pipeline

## Overview

This pipeline uses a hybrid machine learning approach combining rule-based keyword matching with pre-trained transformer models to detect Muslim misrepresentation in Bollywood films across five dimensions.

## Architecture

### Three-Signal Hybrid System

The pipeline integrates three complementary signals to produce a weighted Riz score:

#### 1. Keyword Filtering (30% weight)
- **Method**: TF-IDF vectorization with predefined keywords
- **Purpose**: Captures explicit mentions of problematic terms
- **Output**: Normalized keyword density scores (0-1)
- **Advantages**: Fast, interpretable, captures explicit language
- **Limitations**: No context awareness, misses implicit bias

#### 2. Zero-Shot Classification (50% weight)
- **Model**: `facebook/bart-large-mnli`
- **Method**: Classifies script chunks against dimension labels without training
- **Labels**:
  - "terrorism, violence, extremism, religious radicalization"
  - "anger, aggression, violent behavior, rage"
  - "superstition, religious backwardness, primitive beliefs"
  - "threat to society, anti-democratic values, fundamentalism"
  - "misogyny, gender oppression, discrimination against women"
- **Output**: Probability scores (0-1) for each dimension
- **Advantages**: Context-aware, detects implicit bias, no training needed
- **Limitations**: Computationally intensive, may miss cultural nuances

#### 3. Contextual Sentiment Analysis (20% weight)
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Method**:
  - Extracts text windows around Muslim character mentions
  - Analyzes sentiment (positive/negative) in those contexts
  - Calculates negative sentiment ratio
- **Output**: Ratio of negative contexts (0-1)
- **Advantages**: Character-specific, reduces false positives
- **Limitations**: Requires character identification, English-centric

### Scoring System

**Combined Score (per dimension):**
```
combined_score = 0.3 × keyword_norm + 0.5 × zeroshot_score + 0.2 × sentiment_ratio
```

**Weight Rationale:**
- **Zero-shot (50%)**: Highest weight due to contextual understanding and thematic detection
- **Keywords (30%)**: Moderate weight for explicit term detection
- **Sentiment (20%)**: Lower weight as supplementary signal for context validation

**Two scoring variants:**
1. **Binary Riz Score** (0-5): Sum of flagged dimensions (threshold > 0.3)
2. **Weighted Riz Score** (0-5 continuous): Sum of combined scores

---

## Pipeline Steps

### Step 1: Data Collection

**Input:** SRT subtitle files for 60 Bollywood films (2004-2023)

**Process:**
1. Parse SRT files to extract dialogue (`parse_subtitles.py`)
2. Fetch character data from IMDb API (`get_movie_cast.py`)
3. Identify Muslim characters via name matching (`movie_muslim_check.ipynb`)

**Output:**
- `final_structured_dialogues.csv` - All movie dialogues
- `movie_characters.csv` - Character names and actors
- Muslim character list per movie

### Step 2: Preprocessing

**Text Cleaning:**
```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic
    doc = nlp(text)  # spaCy NLP
    tokens = [token.lemma_ for token in doc if not token.is_stop]  # Lemmatize, remove stopwords
    return " ".join(tokens)
```

**Outputs:**
- Cleaned scripts (lowercase, lemmatized, stopwords removed)
- Full scripts (for sentiment/zero-shot analysis)

### Step 3: Keyword-Based Scoring

**Method:** TF-IDF vectorization

**Keywords per dimension:**
- **Terrorism** (24 terms): jihad, bomb, martyr, extremist, militant, etc.
- **Anger** (15 terms): rage, furious, violent, yelled, exploded, etc.
- **Superstition** (14 terms): backward, primitive, orthodox, superstition, etc.
- **Threat to Western** (13 terms): fundamentalism, sharia law, anti-democracy, etc.
- **Misogyny** (20 terms): honor killing, forced marriage, rape, submission, etc.

**Normalization:**
```python
keyword_norm = keyword_score / max(keyword_scores)  # Scale to 0-1
```

### Step 4: Muslim Character Detection

**Process:**
1. Load Muslim names database (2000+ names)
2. Match character names against database
3. Create character list per movie

**Handling missing cast data:**
- 17 movies lack cast data → assign empty list `[]`
- Prevents NaN errors in downstream processing

### Step 5: Contextual Sentiment Analysis

**Context Extraction:**
```python
def extract_muslim_context(script, muslim_chars, window=150):
    # Find character name mentions in script
    # Extract ±150 characters around each mention
    # Return up to 20 contexts per movie
```

**Sentiment Analysis:**
```python
sentiments = sentiment_analyzer(contexts)
negative_ratio = count(NEGATIVE) / total_contexts
```

**Output:** Negative sentiment ratio (0-1)

### Step 6: Zero-Shot Classification

**Script Chunking:**
- Split long scripts into 5 chunks (1000 chars each)
- Sample from beginning, middle, end for diversity

**Classification:**
```python
result = zero_shot_classifier(
    chunk[:1024],
    candidate_labels=all_labels,
    multi_label=True  # Allow multiple dimensions
)
```

**Averaging:** Mean probability across all chunks per dimension

### Step 7: Weighted Score Calculation

**Per-Dimension Combined Score:**
```python
combined_score = (
    0.3 * keyword_norm +
    0.5 * zeroshot_score +
    0.2 * sentiment_ratio
)
```

**Binary Flag:** `combined_score > 0.3` → flagged (1) or not (0)

**Final Riz Scores:**
- **Weighted:** Sum of 5 dimension combined scores (0-5 continuous)
- **Binary:** Count of flagged dimensions (0-5 integer)

---

## Improvements Over Keyword-Only Methods

| Aspect | Keyword-Only | Hybrid Method |
|--------|--------------|---------------|
| **Context** | None | Character-specific + thematic |
| **Scoring** | Binary (0/1) | Continuous (0-1 probabilities) |
| **Implicit bias** | Missed | Detected via zero-shot |
| **False positives** | High ("bomb" in any context) | Reduced (sentiment + context) |
| **Nuance** | No gradation | 0-1 scale per dimension |
| **Interpretability** | High | Moderate (ML black-box) |

---

## Validation & Reliability

### Internal Consistency
- **Dimension correlations:** Check if dimensions correlate as expected
  - E.g., terrorism + anger should correlate moderately

### Temporal Validity
- **Pre/post 2014 comparison:** Statistical significance testing (t-test)
- **Trend correlation:** Pearson r with year

### External Validity
- **Manual spot-checking:** Review high-scoring films for accuracy
- **Riz Test alignment:** Compare with manual Riz Test assessments

---

## Limitations

### 1. English-Centric Models
- Models trained primarily on English data
- May miss Hindi/Urdu cultural nuances
- Subtitle translation artifacts

**Mitigation:** Future work - fine-tune on Bollywood corpus

### 2. Character Identification Gaps
- 17/60 movies missing cast data
- Name matching may miss non-standard names

**Mitigation:** Manual supplementation for critical films

### 3. Computational Cost
- 2-5 hours on CPU for 60 movies
- Requires 4-8 GB RAM

**Mitigation:** GPU acceleration (3-5x speedup)

### 4. Western Model Bias
- Sentiment/zero-shot models reflect Western perspectives
- May not fully capture Bollywood tropes

**Mitigation:** Cross-validate with cultural experts

### 5. Binary Dimension Boundaries
- Real-world misrepresentation is multifaceted
- 5 dimensions may oversimplify complexity

**Mitigation:** Treat as analytical framework, not absolute truth

---

## Future Enhancements

### 1. Fine-Tuning
- Train on labeled Bollywood dataset
- Incorporate Hindi/Urdu language models

### 2. Multimodal Analysis
- Audio features (tone, music, sound effects)
- Visual features (cinematography, costume, setting)
- Character network analysis

### 3. Temporal Granularity
- Track representation changes within films
- Analyze character arcs over time

### 4. Expanded Dataset
- Include regional cinema (Tamil, Telugu, etc.)
- Extend to TV shows, web series

### 5. Interactive Dashboard
- Real-time Riz score calculator
- Dimension-level breakdowns
- Comparative film analysis

---

## Reproducibility

### Required Files
- `final_structured_dialogues.csv`
- `movie_characters.csv`
- `muslim_names.csv`

### Software Dependencies
```
pandas, numpy, scikit-learn, transformers, torch,
spacy, matplotlib, scipy
```

### Random Seeds
- No randomness in pipeline (deterministic)
- Model inference uses greedy decoding

### Expected Runtime
- **CPU:** 2-5 hours (standard), 45-90 min (quick mode)
- **GPU:** 30-60 min (standard), 15-30 min (quick mode)

---

## Citation

If using this methodology, please cite:

**Riz Test Framework:**
- https://www.riztest.com/

**Models:**
- BART-MNLI: Lewis et al., 2020 (https://arxiv.org/abs/1910.13461)
- DistilBERT: Sanh et al., 2019 (https://arxiv.org/abs/1910.01108)
- spaCy: Honnibal et al., 2020 (https://spacy.io)

**This Implementation:**
- [Your paper/repository citation]

---

## Ethical Considerations

### Purpose
This tool is designed for:
- Academic research on media representation
- Awareness of cultural bias in cinema
- Data-driven advocacy for fair representation

### Misuse Prevention
This tool should **NOT** be used for:
- Individual character or actor judgment
- Censorship or content banning
- Ethnic/religious profiling

### Transparency
- All code and methodology are open-source
- Scores are probabilistic, not absolute truth
- Human review should supplement automated scoring

---

## Contact & Contributions

For questions, issues, or contributions:
- [GitHub repository]
- [Email contact]

This methodology is open to peer review and improvement.
