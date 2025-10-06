# NLP Analysis to Detect Muslim Misrepresentation in Bollywood Films

A hybrid machine learning model utilizing natural language processing (NLP) to detect Muslim misrepresentation in Bollywood movies over the past 20 years. This project uses statistical analysis to highlight the increase in right-wing nationalism and Muslim discrimination through the analysis of Bollywood films.

## Methodology

The pipeline combines three complementary ML signals:
- **Keyword filtering** (TF-IDF) - 30% weight
- **Zero-shot classification** (BART-MNLI) - 50% weight
- **Contextual sentiment analysis** (DistilBERT) - 20% weight

This hybrid approach provides continuous probability scores (0-5 scale) with improved context awareness and reduced false positives compared to pure keyword-based methods.

## Key Findings

Analysis reveals a **significant increase** in negative Muslim misrepresentation in films released after the Modi era (2014+), uncovering clear trends of rising right-wing nationalism and cultural bias in Indian cinema.

This model uses the concept of the [Riz Test](https://www.riztest.com/) to determine Muslim misrepresentation, similar to the Bechdel test for gender representation.

## Project Structure

### Data Collection
- `parse_subtitles.py` - Extracts dialogues from SRT subtitle files
- `get_movie_cast.py` - Fetches cast/character data from IMDb
- `movie_muslim_check.ipynb` - Identifies Muslim characters using name matching
- `clean_muslim_names.ipynb` - Maintains Muslim names database

### Analysis Pipeline
- `nlp_riz_score_hybrid.ipynb` - **Main analysis notebook** (interactive)
- `run_hybrid_pipeline.py` - **CLI runner** (production)
- `analyze_results.py` - Visualization and statistical analysis

### Data Files
- `final_structured_dialogues.csv` - Movie scripts (60 films, 2004-2023)
- `movie_characters.csv` - Character names from IMDb
- `muslim_names.csv` - Muslim name database (2000+ names)

### Documentation
- `METHODOLOGY.md` - Detailed methodology and approach
- `QUICKSTART.md` - Step-by-step usage guide
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Option 1: Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook nlp_riz_score_hybrid.ipynb
```

Execute cells sequentially to analyze scripts and generate Riz scores.

### Option 2: Python Script (Recommended for production)

**Standard mode:**
```bash
python run_hybrid_pipeline.py
```

**Quick mode (3x faster):**
```bash
python run_hybrid_pipeline.py --quick
```

**With GPU:**
```bash
python run_hybrid_pipeline.py --gpu
```

### Analyze Results

After running the pipeline:

```bash
python analyze_results.py
```

Generates:
- Comprehensive 6-panel visualization
- Temporal trend analysis
- Pre/post 2014 comparison
- Dimension-wise breakdown
- Statistical significance tests

## Results

The pipeline outputs:
- **`final_riz_test_results_hybrid.csv`** - Riz scores for each movie
- **`riz_analysis_visualization.png`** - Comprehensive 6-panel analysis
- **`riz_scoring_methods.png`** - Binary vs weighted scoring comparison
- **`riz_analysis_by_year.csv`** - Yearly statistics
- **`dimension_correlations.csv`** - Dimension correlation matrix

### Riz Score Scale (0-5):
- **0.0-1.0**: Minimal misrepresentation
- **1.0-2.0**: Low misrepresentation
- **2.0-3.0**: Moderate misrepresentation
- **3.0-4.0**: High misrepresentation
- **4.0-5.0**: Severe misrepresentation

### Five Dimensions Analyzed:
1. **Terrorism** - Association with violence/extremism
2. **Anger** - Portrayal as aggressive/temperamental
3. **Superstition** - Depiction as backward/primitive
4. **Threat to Western** - Anti-democratic/fundamentalist framing
5. **Misogyny** - Gender oppression/discrimination

## Technical Details

**Dataset:** 60 Bollywood films (2004-2023)
**Languages:** Hindi/Urdu with English subtitles
**Models Used:**
- `facebook/bart-large-mnli` (zero-shot classification)
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment)
- `en_core_web_sm` (NLP preprocessing)

**Performance:**
- Runtime: 2-5 hours (CPU), 30-60 min (GPU)
- Memory: 4-8 GB RAM recommended

## Citation

If using this methodology in research, please cite:
- Riz Test: https://www.riztest.com/
- HuggingFace Transformers: https://huggingface.co/transformers/

Full analysis on methodology will be available after publication.

## License

[Specify license]

## Contact

[Your contact information]
