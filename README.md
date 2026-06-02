# 🌍 Tourist Sentiment & Trend Analytics

NLP-powered analysis of 20,000+ TripAdvisor hotel reviews using sentiment analysis, topic modeling, and ML-based rating prediction.

## 📊 What This Project Does

- Sentiment classification (Positive/Negative/Neutral) using TextBlob
- Topic modeling using LDA — identifies 5 key review themes
- Rating prediction using Random Forest (TF-IDF features)
- Visual analytics — heatmaps, word clouds, polarity distributions

## 🔍 Key Findings

- 86.2% of reviews are Positive
- Polarity score increases consistently with rating (1★ = -0.04 → 5★ = 0.36)
- Top topics: Room & Service, Location, Resort & Amenities, Cleanliness, Beach Activities

## 🛠️ Tech Stack

Python | Pandas | TextBlob | Scikit-learn | LDA | Matplotlib | Seaborn | WordCloud

## 📁 Dataset

[TripAdvisor Hotel Reviews — Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
20,491 reviews | 2 columns | Usability score: 10/10

## 🚀 How to Run

```bash
pip install pandas textblob scikit-learn matplotlib seaborn wordcloud
jupyter notebook tourist-sentiment-trend-analytics.ipynb
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Total Reviews | 20,491 |
| Model Accuracy | ~65% |
| Positive Sentiment | 86.2% |
| Average Rating | 3.9/5 |
