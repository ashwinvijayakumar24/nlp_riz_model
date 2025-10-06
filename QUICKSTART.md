# Quick Start Guide - Riz Score NLP Pipeline

## Prerequisites

Ensure you have Python 3.8+ installed.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Verify installation:**
   ```bash
   python -c "import transformers; import torch; print('✓ All packages installed')"
   ```

---

## Usage

### Option 1: Run via Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook nlp_riz_score_hybrid.ipynb
```

Then execute cells sequentially. This allows you to:
- Inspect intermediate results
- Modify parameters
- Visualize step-by-step

**Expected runtime:** 2-5 hours (CPU)

### Option 2: Run via Python Script (Recommended for production)

**Standard mode (most accurate):**
```bash
python run_hybrid_pipeline.py
```

**Quick mode (3x faster, slightly less accurate):**
```bash
python run_hybrid_pipeline.py --quick
```

**With GPU acceleration (if available):**
```bash
python run_hybrid_pipeline.py --gpu
```

**Expected runtime:**
- CPU: 2-5 hours (standard), 45-90 minutes (quick)
- GPU: 30-60 minutes (standard), 15-30 minutes (quick)

---

## Analyzing Results

After running the pipeline, analyze the results:

```bash
python analyze_results.py
```

This generates:
1. **`riz_analysis_visualization.png`** - 6-panel comprehensive analysis:
   - Temporal trend with trendline
   - Distribution histogram
   - Pre/post 2014 box plots
   - Dimension flags by year (stacked bars)
   - Sentiment analysis over time
   - Dimension intensity heatmap

2. **`riz_scoring_methods.png`** - Binary vs weighted scoring comparison

3. **`riz_analysis_by_year.csv`** - Yearly statistics

4. **`dimension_correlations.csv`** - Dimension correlation matrix

5. **Console output** with comprehensive statistics:
   - Overall statistics
   - Temporal trend analysis
   - Pre/post 2014 comparison
   - Statistical significance tests
   - Dimension breakdown
   - Top/bottom scoring movies

---

## Understanding Results

### Weighted Riz Score Scale (0-5):
- **0.0-1.0**: Minimal misrepresentation
- **1.0-2.0**: Low misrepresentation
- **2.0-3.0**: Moderate misrepresentation
- **3.0-4.0**: High misrepresentation
- **4.0-5.0**: Severe misrepresentation

### Dimension Flags:
Each movie gets binary flags (0/1) for:
- **Terrorism** (threshold: 0.3)
- **Anger** (threshold: 0.3)
- **Superstition** (threshold: 0.3)
- **Threat to Western** (threshold: 0.3)
- **Misogyny** (threshold: 0.3)

### Output Files:

**`final_riz_test_results_hybrid.csv`** contains:
- Movie Title, Year
- Has_Muslim_Character (boolean)
- sentiment_negative_ratio (0-1)
- [dimension]_combined_score (0-1) × 5 dimensions
- [dimension]_flag (0/1) × 5 dimensions
- riz_score_weighted (0-5 continuous)
- riz_score_binary (0-5 integer)

---

## Troubleshooting

### Issue: "Out of memory" error

**Solution 1:** Use quick mode
```bash
python run_hybrid_pipeline.py --quick
```

**Solution 2:** Process fewer chunks at a time

Edit `run_hybrid_pipeline.py`, line ~180:
```python
num_chunks=3 if quick_mode else 5  # Change 5 to 3
```

### Issue: Models take forever to download

**Solution:** Download models manually first:
```python
from transformers import pipeline

# This downloads models once
pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

### Issue: "CUDA out of memory" with GPU

**Solution:** Force CPU usage:
```bash
python run_hybrid_pipeline.py  # Omit --gpu flag
```

### Issue: Missing data files

**Solution:** Ensure these files exist in the project directory:
- `final_structured_dialogues.csv`
- `movie_characters.csv`
- `muslim_names.csv`

If missing, run the data collection scripts first:
```bash
python parse_subtitles.py
python get_movie_cast.py
# Then run movie_muslim_check.ipynb
```

### Issue: TypeError about 'float' object not iterable

**Solution:** This is fixed in the latest version. Ensure you have:
- Updated `nlp_riz_score_hybrid.ipynb` (cell 10)
- Updated `run_hybrid_pipeline.py` (lines 110-113)

The fix converts NaN → empty list `[]` for movies without cast data.

---

## Customization

### Adjust Signal Weights

Edit `run_hybrid_pipeline.py` or notebook cell, lines ~220-222:

```python
KEYWORD_WEIGHT = 0.3    # Increase for more emphasis on explicit keywords
ZEROSHOT_WEIGHT = 0.5   # Increase for more emphasis on thematic patterns
SENTIMENT_WEIGHT = 0.2  # Increase for more emphasis on tone around Muslim characters
```

**Note:** Weights should sum to 1.0

### Adjust Flagging Threshold

More sensitive (flags more movies):
```python
df_grouped[f'{category}_flag'] = (combined > 0.2).astype(int)  # Was 0.3
```

Less sensitive (flags fewer movies):
```python
df_grouped[f'{category}_flag'] = (combined > 0.4).astype(int)  # Was 0.3
```

### Use Different Models

Replace in model loading section:

**Alternative sentiment analyzer:**
```python
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=device
)
```

**Alternative zero-shot classifier:**
```python
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    device=device
)
```

---

## Performance Tips

1. **Use GPU if available:** 3-5x speedup
2. **Use quick mode for testing:** Verify pipeline works before full run
3. **Close other applications:** Free up RAM
4. **Use smaller batches:** If memory-constrained

---

## Validation Checklist

After running, verify:

- [ ] Output file exists: `final_riz_test_results_hybrid.csv`
- [ ] Contains 60 movies (or your dataset size)
- [ ] No NaN values in score columns
- [ ] Scores in reasonable range (0-5)
- [ ] Positive correlation with year (r > 0)
- [ ] Pre-2014 mean < Post-2014 mean

---

## Workflow Example

**Complete analysis from scratch:**

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run pipeline (choose one)
python run_hybrid_pipeline.py              # Standard mode
python run_hybrid_pipeline.py --quick      # Quick mode
python run_hybrid_pipeline.py --gpu        # With GPU

# 3. Analyze results
python analyze_results.py

# 4. Review outputs
open riz_analysis_visualization.png
open final_riz_test_results_hybrid.csv
```

**Expected outputs:**
- Console: Detailed statistics and analysis
- `final_riz_test_results_hybrid.csv` - Full results
- `riz_analysis_visualization.png` - 6-panel analysis
- `riz_scoring_methods.png` - Scoring comparison
- `riz_analysis_by_year.csv` - Yearly stats
- `dimension_correlations.csv` - Correlation matrix

---

## Next Steps

1. **Interpret Results:**
   - Identify high-scoring films for detailed review
   - Examine dimension patterns
   - Validate findings against cultural knowledge

2. **Statistical Analysis:**
   - Use `riz_analysis_by_year.csv` for time-series analysis
   - Import into R/SPSS for advanced stats
   - Perform regression analysis

3. **Publication:**
   - Include methodology from `METHODOLOGY.md`
   - Cite models and Riz Test framework
   - Share visualizations and findings

---

## Support

For issues or questions:
1. Check `METHODOLOGY.md` for detailed methodology
2. Review code comments in notebook/scripts
3. Open GitHub issue (if applicable)

---

## Citation

If using this pipeline in research, please cite:
- **Riz Test:** https://www.riztest.com/
- **HuggingFace Transformers:** https://huggingface.co/transformers/
- **This implementation:** [Your paper/repository]
