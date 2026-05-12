# CIS 5190 Final Project: News Source Classification

## Project Overview

Binary text classification to distinguish Fox News vs NBC News headlines. Dataset: 3,815 URLs (2,010 Fox, 1,805 NBC). Models: DistilBERT transformer and hybrid SVM with embeddings + stylistic features.

**Key Files for Quick Access:**
- Main dataset: `data/processed/combined_base_data.csv`
- Feature data: `data/processed/style_features.csv`
- Embeddings: `data/processed/minilm_embeddings.npy`

## Setup & Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Navigation Guide

### Data Collection (Scraping)
**Scripts:**
- `scripts/scrape_fox.py` - Multi-threaded Fox News scraper with BeautifulSoup
- `scripts/scrape_nbc.py` - NBC News scraper using JSON-LD schema
- `scripts/backfill_nbc.py` - NBC live blog and missing article scraper
- `scripts/combine_data.py` - Merge all scraped data into combined CSV

**Notebooks:**
- `notebooks/01_rk_scraping.ipynb` - Scraping implementation and validation
- `notebooks/01_scraping_demo.ipynb` - Scraping techniques demonstration

**Data Output:** Raw scraped files in `data/raw/`, combined in `data/raw/combined_scraped_all.csv`

### Data Processing & Cleaning
**Scripts:**
- `scripts/combine_scraped_fox.py` - Consolidate Fox scraping batches
- `scripts/split_url_data.py` - Partition URL lists for distributed scraping

**Notebooks:**
- `notebooks/04_combined_exploration.ipynb` - Combined dataset analysis and cleaning

**Data Output:** Cleaned dataset in `data/processed/combined_base_data.csv`

### Exploratory Data Analysis (EDA)
**Notebooks:**
- `notebooks/02_rk_eda.ipynb` - Initial headline analysis (distributions, lengths, word frequencies)
- `notebooks/03_rk_style_eda.ipynb` - Stylistic feature exploration and validation
- `notebooks/03_nbc_news.ipynb` - NBC-specific data exploration
- `notebooks/04_combined_exploration.ipynb` - Combined Fox+NBC analysis
- `notebooks/05_nbc_live_news.ipynb` - Live news blog analysis
- `notebooks/06_nbc_news.ipynb` - Additional NBC exploration

**Visualizations:** Generated plots in `figures/` folder

### Feature Engineering
**Scripts:**
- `scripts/rk_feature_engineering.py` - Extract stylistic features (word count, punctuation, sentiment, NER)

**Notebooks:**
- `notebooks/03_rk_style_eda.ipynb` - Feature selection justification and analysis

**Features:** 5-dimensional style vectors saved in `data/processed/style_features.csv`

### Modeling & Training
**DistilBERT Model:**
- `scripts/distilbert/model.py` - NewsClassifier class using DistilBertForSequenceClassification
- `scripts/distilbert/preprocess.py` - Tokenization and preprocessing
- `scripts/distilbert/model.pt` - Trained model weights

**Notebooks:**
- `notebooks/05_rk_distilbert.ipynb` - DistilBERT fine-tuning experiments
- `notebooks/06_rk_full_distilbert_training.ipynb` - Complete training pipeline with metrics

**Hybrid SVM Model:**
- `scripts/embedding_style_svm/model.py` - SVM classifier with embeddings + features
- `scripts/embedding_style_svm/preprocess.py` - Feature vector preparation
- `scripts/embedding_style_svm/train_save.py` - Training and model saving
- `scripts/embedding_style_svm/model.pt` - Trained SVM model

**Notebooks:**
- `notebooks/04_rk_modeling.ipynb` - Model development, training, and comparison

### Evaluation & Results
**Scripts:**
- `scripts/distilbert/eval_project_b.py` - DistilBERT evaluation
- `scripts/embedding_style_svm/eval_project_b.py` - SVM evaluation

**Notebooks:**
- `notebooks/04_rk_modeling.ipynb` - Model performance comparison (accuracy, precision, recall, F1)
- `notebooks/05_rk_distilbert.ipynb` - DistilBERT metrics and analysis
- `notebooks/06_rk_full_distilbert_training.ipynb` - Full evaluation results

**Metrics Tracked:** Accuracy, precision, recall, F1-score; performance by topics/authors

## Project Topics (from plan.md)
1. Sentiment Analysis (headlines/subtitles, by author/topic)
2. Topic Stratification (model performance by news topics)
3. LDA Topic Modeling on headlines
4. Author Bias analysis

## File Structure Summary
- `data/external/` - URL lists
- `data/raw/` - Scraped data
- `data/processed/` - Final datasets
- `scripts/` - All Python scripts
- `notebooks/` - Jupyter notebooks (main analysis)
- `_notebooks/` - Archived notebooks
- `figures/` - Generated visualizations