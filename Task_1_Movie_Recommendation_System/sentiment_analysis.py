import pandas as pd
from textblob import TextBlob

# Sample movie reviews
reviews = ["This movie is amazing and fantastic!", "I loved the acting and the story.", "It was an average movie.", "The movie was boring and too long.", "Worst movie I have ever watched."]

# Create DataFrame
df = pd.DataFrame({"Review":reviews})

# Calculate sentiment polarity
df["Sentiment Score"] = df["Review"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Classify sentiment
def classify(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"
df["Sentiment"] = df["Sentiment Score"].apply(classify)
print(df)