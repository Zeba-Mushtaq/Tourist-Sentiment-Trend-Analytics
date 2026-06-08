# 🌍 Tourist Sentiment & Trend Analytics
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://tourist-sentiment-zeba.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Zeba-Mushtaq/tourist-sentiment-analytics)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

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

Python | Pandas | TextBlob | Scikit-learn | LDA | Matplotlib | Seaborn | WordCloud |Machine Learning

## 📁 Dataset

[TripAdvisor Hotel Reviews — Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
20,491 reviews | 2 columns | Usability score: 10/10

## 🔮 Future Improvements

- Explore transformer-based models (BERT/RoBERTa) to improve sentiment classification and rating prediction accuracy.

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
