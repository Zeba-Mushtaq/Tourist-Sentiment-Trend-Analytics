import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Tourist Sentiment Analytics", page_icon="🌍", layout="wide")

st.title("🌍 Tourist Sentiment & Trend Analytics Dashboard")
st.markdown("**NLP-powered analysis of 20,000+ TripAdvisor hotel reviews**")

@st.cache_data
def load_and_process():
    df = pd.read_csv('tripadvisor_hotel_reviews.csv')
    df['polarity'] = df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['subjectivity'] = df['Review'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    def get_sentiment(p):
        if p > 0.1: return 'Positive'
        elif p < -0.1: return 'Negative'
        else: return 'Neutral'
    df['Sentiment'] = df['polarity'].apply(get_sentiment)
    df['review_length'] = df['Review'].apply(lambda x: len(str(x).split()))
    return df

df = load_and_process()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Average Rating", f"{df['Rating'].mean():.2f}/5")
col3.metric("Positive Reviews", f"{(df['Sentiment']=='Positive').sum():,}")
col4.metric("Avg Review Length", f"{df['review_length'].mean():.0f} words")

st.markdown("---")

# Charts row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Rating Distribution")
    fig, ax = plt.subplots()
    df['Rating'].value_counts().sort_index().plot(
        kind='bar', ax=ax,
        color=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#27ae60']
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("🎯 Sentiment Distribution")
    fig, ax = plt.subplots()
    df['Sentiment'].value_counts().plot(
        kind='pie', ax=ax, autopct='%1.1f%%',
        colors=['#2ecc71','#95a5a6','#e74c3c']
    )
    st.pyplot(fig)

st.markdown("---")

# Charts row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Rating vs Sentiment Heatmap")
    fig, ax = plt.subplots(figsize=(8,5))
    sentiment_rating = pd.crosstab(df['Rating'], df['Sentiment'])
    sns.heatmap(sentiment_rating, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("📏 Avg Polarity by Rating")
    fig, ax = plt.subplots()
    df.groupby('Rating')['polarity'].mean().plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel("Rating")
    ax.set_ylabel("Polarity")
    st.pyplot(fig)

st.markdown("---")
st.subheader("🔍 Interactive Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Polarity vs Rating (Scatter)**")
    fig = px.scatter(
        df.sample(2000), 
        x='polarity', 
        y='Rating',
        color='Sentiment',
        hover_data=['Review'],
        color_discrete_map={
            'Positive':'#2ecc71',
            'Negative':'#e74c3c',
            'Neutral':'#95a5a6'
        },
        title='Polarity vs Rating'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Rating Distribution (Interactive)**")
    fig2 = px.histogram(
        df, 
        x='Rating',
        color='Sentiment',
        barmode='group',
        color_discrete_map={
            'Positive':'#2ecc71',
            'Negative':'#e74c3c',
            'Neutral':'#95a5a6'
        },
        title='Ratings by Sentiment'
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.subheader("📝 Review Explorer")

sentiment_filter = st.selectbox(
    "Filter by Sentiment:",
    ['All', 'Positive', 'Negative', 'Neutral']
)

rating_filter = st.slider("Filter by Rating:", 1, 5, (1, 5))

if sentiment_filter == 'All':
    filtered_df = df[
        (df['Rating'] >= rating_filter[0]) & 
        (df['Rating'] <= rating_filter[1])
    ]
else:
    filtered_df = df[
        (df['Sentiment'] == sentiment_filter) & 
        (df['Rating'] >= rating_filter[0]) & 
        (df['Rating'] <= rating_filter[1])
    ]

st.write(f"Showing {len(filtered_df):,} reviews")
st.dataframe(
    filtered_df[['Review', 'Rating', 'Sentiment', 'polarity']].head(50),
    use_container_width=True
)

st.markdown("---")

# Live sentiment analyzer
st.subheader("🔍 Try It Yourself — Live Review Analyzer")
user_input = st.text_area("Enter a hotel review:")
if st.button("Analyze"):
    if user_input:
        blob = TextBlob(user_input)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiment = "✅ Positive"
            color = "green"
        elif polarity < -0.1:
            sentiment = "❌ Negative"
            color = "red"
        else:
            sentiment = "➖ Neutral"
            color = "gray"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", sentiment)
        col2.metric("Polarity", f"{polarity:.3f}")
        col3.metric("Subjectivity", f"{blob.sentiment.subjectivity:.3f}")
