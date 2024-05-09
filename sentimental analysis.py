import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


# Download NLTK resources (you only need to do this once)
nltk.download('vader_lexicon')

# Load your Airbnb reviews data
reviews_df = pd.read_csv('reviews.csv')

# Drop rows with NaN values in the 'comments' column
reviews_df = reviews_df.dropna(subset=['comments'])

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def get_sentiment_scores(comment):
    if isinstance(comment, str):  # Check if the comment is a string
        scores = sia.polarity_scores(comment)
        return scores['compound']
    else:
        return 0.0  # Return a neutral score for non-string values

# Apply the sentiment analysis function to the 'comments' column
reviews_df['sentiment_score'] = reviews_df['comments'].apply(get_sentiment_scores)

# Categorize sentiment into positive, neutral, or negative
reviews_df['sentiment'] = reviews_df['sentiment_score'].apply(lambda score: 'positive' if score > 0 else ('neutral' if score == 0 else 'negative'))

# Display the results
print(reviews_df[['comments', 'sentiment_score', 'sentiment']])
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=reviews_df, palette='viridis')
plt.title('Sentiment Distribution in Airbnb Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

