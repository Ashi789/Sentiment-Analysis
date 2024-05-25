import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('reddit_opinion_PSE_ISR.csv')

# Display the first few rows of the dataframe
df.head()

# Initialize the SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Define a function to analyze sentiment
def analyze_sentiment(text):
    if pd.isna(text):
        return 0  # Return neutral sentiment for NaN values
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Apply sentiment analysis to the 'self_text' column
df['Sentiment_Score'] = df['self_text'].apply(analyze_sentiment)

# Display the first few rows of the DataFrame with sentiment scores
df[['comment_id', 'self_text', 'Sentiment_Score']].head()

# Visualize the sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Sentiment_Score'], kde=True, bins=30)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Categorize sentiment into Positive, Neutral, Negative
def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment_Category'] = df['Sentiment_Score'].apply(categorize_sentiment)

# Count plot for sentiment categories with data labels
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Sentiment_Category', data=df, palette='viridis')

# Adding data labels on each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

# Removing the y-axis values
ax.yaxis.set_visible(False)

# Title and labels
plt.title('Sentiment Category Counts')
plt.xlabel('Sentiment Category')
plt.ylabel('')

# Adding total number of reviews outside the plot
total_reviews = len(df)
plt.figtext(0.9, 0.02, f'Total Reviews: {total_reviews}', horizontalalignment='right', fontsize=12)

plt.show()
