"""
Run Hybrid Riz Score Pipeline

This script provides a command-line interface to run the hybrid pipeline
with progress tracking and automatic GPU detection.

Usage:
    python run_hybrid_pipeline.py [--gpu] [--quick]

Options:
    --gpu    : Force GPU usage (will fail if not available)
    --quick  : Quick mode - process fewer chunks per movie (faster but less accurate)
"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import torch

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("ℹ No GPU detected, using CPU")
        return False

def load_models(use_gpu=False):
    """Load pre-trained models"""
    from transformers import pipeline

    device = 0 if use_gpu else -1

    print("\n📦 Loading models...")
    print("  - Sentiment analysis model (DistilBERT)...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    print("  - Zero-shot classification model (BART-MNLI)...")
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

    print("✓ Models loaded successfully\n")
    return sentiment_analyzer, zero_shot_classifier

def preprocess_data():
    """Load and preprocess data"""
    import spacy
    import en_core_web_sm

    print("📂 Loading data files...")
    df = pd.read_csv('final_structured_dialogues.csv')
    movie_characters_df = pd.read_csv('movie_characters.csv')
    muslim_names_df = pd.read_csv('muslim_names.csv')

    print(f"  - {len(df)} dialogue lines")
    print(f"  - {len(movie_characters_df)} character entries")
    print(f"  - {len(muslim_names_df)} Muslim names")

    # Preprocessing
    print("\n🔧 Preprocessing...")
    df = df.fillna('-')
    df = df.groupby('Movie Title', group_keys=False).apply(
        lambda x: x.drop_duplicates(subset='Dialogue', keep='first')
    )

    df_grouped = df.groupby(["Movie Title", "Year"])["Dialogue"].apply(" ".join).reset_index()
    df_grouped.rename(columns={"Dialogue": "Full Script"}, inplace=True)

    # Text preprocessing
    nlp = en_core_web_sm.load()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        return " ".join(tokens)

    print("  - Cleaning and lemmatizing scripts...")
    df_grouped["Processed Script"] = df_grouped["Full Script"].progress_apply(preprocess_text)

    # Extract Muslim characters
    muslim_name_set = set(muslim_names_df['Name'].str.lower())
    movie_muslim_characters = {}

    for movie, group in movie_characters_df.groupby('Movie Title'):
        muslim_chars = []
        for character in group['Character'].dropna():
            name_parts = [part.strip(" '\"") for part in str(character).split()]
            if any(part.lower() in muslim_name_set for part in name_parts):
                muslim_chars.append(character)
        movie_muslim_characters[movie] = muslim_chars

    df_grouped['Muslim_Characters'] = df_grouped['Movie Title'].map(movie_muslim_characters)
    # Replace NaN with empty lists for movies without cast data
    df_grouped['Muslim_Characters'] = df_grouped['Muslim_Characters'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    print(f"✓ Preprocessed {len(df_grouped)} movies\n")
    return df_grouped, muslim_name_set

def run_pipeline(df_grouped, sentiment_analyzer, zero_shot_classifier, quick_mode=False):
    """Run the hybrid pipeline"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Define categories
    misrepresentation_keywords = {
        "terrorism": ["jihad", "jihadist", "suicide bomber", "bomb", "martyr", "radical", "extremist", "militant"],
        "anger": ["rage", "furious", "violent", "yelled", "screamed"],
        "superstition": ["superstition", "backward", "primitive", "orthodox"],
        "threat_to_western": ["fundamentalism", "sharia law", "anti-democracy"],
        "misogyny": ["honor killing", "forced marriage", "rape", "harassment", "submission"]
    }

    zero_shot_labels = {
        "terrorism": "terrorism, violence, extremism",
        "anger": "anger, aggression, rage",
        "superstition": "superstition, religious backwardness",
        "threat_to_western": "threat to society, fundamentalism",
        "misogyny": "misogyny, gender oppression"
    }

    # Step 1: Keyword scores
    print("📊 Step 1/3: Calculating keyword-based scores...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_grouped["Processed Script"])
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    for category, keywords in misrepresentation_keywords.items():
        existing_keywords = [word for word in keywords if word in tfidf_df.columns]
        if existing_keywords:
            df_grouped[f'{category}_keyword_score'] = tfidf_df[existing_keywords].sum(axis=1).values
        else:
            df_grouped[f'{category}_keyword_score'] = 0.0

    # Step 2: Sentiment analysis
    print("\n💭 Step 2/3: Analyzing contextual sentiment...")
    def extract_muslim_context(script, muslim_chars, window=150):
        # muslim_chars is guaranteed to be a list (empty or populated)
        if not muslim_chars:
            return []

        contexts = []
        script_lower = script.lower()

        for char in muslim_chars:
            first_name = str(char).split()[0].lower()
            start = 0
            while True:
                pos = script_lower.find(first_name, start)
                if pos == -1:
                    break
                context_start = max(0, pos - window)
                context_end = min(len(script), pos + len(first_name) + window)
                context = script[context_start:context_end].strip()
                if len(context) > 20:
                    contexts.append(context)
                start = pos + 1

        return contexts[:10 if quick_mode else 20]

    def analyze_sentiment(contexts):
        if not contexts:
            return 0.0
        try:
            truncated = [ctx[:512] for ctx in contexts]
            sentiments = sentiment_analyzer(truncated)
            return sum(1 for s in sentiments if s['label'] == 'NEGATIVE') / len(sentiments)
        except:
            return 0.0

    tqdm.pandas(desc="Extracting contexts")
    df_grouped['muslim_contexts'] = df_grouped.progress_apply(
        lambda row: extract_muslim_context(row['Full Script'], row['Muslim_Characters']),
        axis=1
    )

    tqdm.pandas(desc="Analyzing sentiment")
    df_grouped['sentiment_negative_ratio'] = df_grouped['muslim_contexts'].progress_apply(
        analyze_sentiment
    )

    # Step 3: Zero-shot classification
    print("\n🎯 Step 3/3: Running zero-shot classification...")
    def chunk_script(script, chunk_size=1000, num_chunks=3 if quick_mode else 5):
        script_len = len(script)
        if script_len < chunk_size:
            return [script]

        chunks = []
        step = script_len // num_chunks
        for i in range(num_chunks):
            start = i * step
            end = min(start + chunk_size, script_len)
            chunks.append(script[start:end])
        return chunks

    def zero_shot_classify(script):
        chunks = chunk_script(script)
        category_scores = {cat: [] for cat in zero_shot_labels.keys()}

        for chunk in chunks:
            if len(chunk) < 50:
                continue
            try:
                all_labels = list(zero_shot_labels.values())
                result = zero_shot_classifier(chunk[:1024], candidate_labels=all_labels, multi_label=True)

                label_to_category = {v: k for k, v in zero_shot_labels.items()}
                for label, score in zip(result['labels'], result['scores']):
                    category = label_to_category.get(label)
                    if category:
                        category_scores[category].append(score)
            except:
                continue

        return {cat: np.mean(scores) if scores else 0.0
                for cat, scores in category_scores.items()}

    zero_shot_results = []
    for idx, row in tqdm(df_grouped.iterrows(), total=len(df_grouped), desc="Classifying scripts"):
        scores = zero_shot_classify(row['Full Script'])
        zero_shot_results.append(scores)

    for category in zero_shot_labels.keys():
        df_grouped[f'{category}_zeroshot_score'] = [r[category] for r in zero_shot_results]

    # Calculate weighted scores
    print("\n⚖️  Calculating weighted Riz scores...")
    KEYWORD_WEIGHT = 0.3
    ZEROSHOT_WEIGHT = 0.5
    SENTIMENT_WEIGHT = 0.2

    for category in misrepresentation_keywords.keys():
        col = f'{category}_keyword_score'
        max_val = df_grouped[col].max()
        if max_val > 0:
            df_grouped[f'{category}_keyword_norm'] = df_grouped[col] / max_val
        else:
            df_grouped[f'{category}_keyword_norm'] = 0.0

    for category in misrepresentation_keywords.keys():
        combined = (
            KEYWORD_WEIGHT * df_grouped[f'{category}_keyword_norm'] +
            ZEROSHOT_WEIGHT * df_grouped[f'{category}_zeroshot_score'] +
            SENTIMENT_WEIGHT * df_grouped['sentiment_negative_ratio']
        )
        df_grouped[f'{category}_combined_score'] = combined
        df_grouped[f'{category}_flag'] = (combined > 0.3).astype(int)

    df_grouped['riz_score_weighted'] = sum(
        df_grouped[f'{cat}_combined_score'] for cat in misrepresentation_keywords.keys()
    )
    df_grouped['riz_score_binary'] = sum(
        df_grouped[f'{cat}_flag'] for cat in misrepresentation_keywords.keys()
    )

    return df_grouped

def save_results(df_grouped):
    """Save results to CSV"""
    output_df = df_grouped[[
        'Movie Title', 'Year',
        'sentiment_negative_ratio',
        'terrorism_combined_score', 'terrorism_flag',
        'anger_combined_score', 'anger_flag',
        'superstition_combined_score', 'superstition_flag',
        'threat_to_western_combined_score', 'threat_to_western_flag',
        'misogyny_combined_score', 'misogyny_flag',
        'riz_score_weighted', 'riz_score_binary'
    ]].copy()

    output_df = output_df.sort_values('Year')
    output_df.to_csv('final_riz_test_results_hybrid.csv', index=False)

    print("\n✓ Results saved to 'final_riz_test_results_hybrid.csv'")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nMean Weighted Riz Score: {output_df['riz_score_weighted'].mean():.3f}")
    print(f"Correlation with Year:   r = {output_df['Year'].corr(output_df['riz_score_weighted']):.3f}")

    print("\nDimension Flags:")
    for cat in ['terrorism', 'anger', 'superstition', 'threat_to_western', 'misogyny']:
        count = output_df[f'{cat}_flag'].sum()
        print(f"  {cat.capitalize():20s}: {count} films")

    # Pre/post 2014
    pre = output_df[output_df['Year'] < 2014]
    post = output_df[output_df['Year'] >= 2014]
    change = ((post['riz_score_weighted'].mean() - pre['riz_score_weighted'].mean())
              / pre['riz_score_weighted'].mean() * 100)

    print(f"\nPre-2014 Mean:  {pre['riz_score_weighted'].mean():.3f}")
    print(f"Post-2014 Mean: {post['riz_score_weighted'].mean():.3f}")
    print(f"Change:         {change:+.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Run Hybrid Riz Score Pipeline')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--quick', action='store_true', help='Quick mode (faster, less accurate)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("HYBRID RIZ SCORE PIPELINE")
    print("="*60)

    # Check GPU
    use_gpu = check_gpu() if not args.gpu else True
    if args.gpu and not torch.cuda.is_available():
        print("❌ Error: GPU requested but not available")
        sys.exit(1)

    if args.quick:
        print("⚡ Quick mode enabled (fewer chunks per movie)")

    # Enable progress bars
    tqdm.pandas()

    # Load models
    sentiment_analyzer, zero_shot_classifier = load_models(use_gpu)

    # Preprocess data
    df_grouped, _ = preprocess_data()

    # Run pipeline
    print("\n" + "="*60)
    print("RUNNING PIPELINE")
    print("="*60)
    df_results = run_pipeline(df_grouped, sentiment_analyzer, zero_shot_classifier, args.quick)

    # Save results
    save_results(df_results)

    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review: final_riz_test_results_hybrid.csv")
    print("  2. Compare: python compare_methods.py")
    print("  3. Visualize: Open nlp_riz_score_hybrid.ipynb")

if __name__ == "__main__":
    main()
