import nltk
import json
import re
import plotly.graph_objects as go
from nltk.corpus import stopwords
from collections import defaultdict
from statistics import mean
from plotly.subplots import make_subplots
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))
words.add('dogecoin')
words.add('shibcoin')
words.add('DOGE')
words.add('doge')
words.add('SHIB')
words.add('shib')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def clean_data(message):
    # filter whitespace
    message = message.strip()
    # filter numbers
    message = re.sub(r'\d+', '', message)
    # lowercase
    message = message.lower()

    return message


def filter_english_words(message):
    result = " ".join(lemmatizer.lemmatize(stemmer.stem(w)) for w in nltk.wordpunct_tokenize(message) if
                      (w.lower() in words and w not in stop_words) or not w.isalpha())
    return result


def read_telegram_file(path):
    message_dated_dict = defaultdict(list)
    with open(path, "r", encoding='utf-8') as read_file:
        file_data = json.load(read_file)
        for document in tqdm(file_data['messages'], desc='Reading Telegram Message data ...', colour='Green'):
            date = document['date'][0:10]
            text_message = document['text']

            # Check if message has web links or additional nested messages
            if type(text_message) is list:
                flattened_message = ''
                for i in range(len(text_message)):
                    if type(text_message[i]) is str:
                        flattened_message += (text_message[i])
                    else:
                        flattened_message += (text_message[i]['text'])
                message_dated_dict[date].append(clean_data(flattened_message))
            else:
                message_dated_dict[date].append(clean_data(text_message))
        return message_dated_dict


if __name__ == "__main__":
    sentiments = SentimentIntensityAnalyzer()
    preprocessed_messages = defaultdict(list)
    date_sentiment_dict = defaultdict(float)
    date_message_count_dict = defaultdict(int)
    crypto_coins = ['doge', 'shib']

    # Read Telegram Messages file
    data = read_telegram_file("result.json")

    # Preprocess Messages
    for date, messages in tqdm(data.items(), desc='Pre-Processing Messages ...', colour='Green'):
        for sentence in messages:
            if any(coins in sentence.lower() for coins in crypto_coins):
                preprocessed_messages[date].append(
                    sentiments.polarity_scores(filter_english_words(sentence))["compound"])

    # Calculate Mean Sentiment for each day
    for date, sentiments in tqdm(preprocessed_messages.items(), desc='Calculating Mean Sentiment per day ...',
                                 colour='Green'):
        date_sentiment_dict[date] = mean(sentiments)
        date_message_count_dict[date] = len(sentiments)

    # Create figure with secondary y-axis
    sentimentPlot = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    sentimentPlot.add_trace(
        go.Scatter(x=list(date_message_count_dict.keys()), y=list(date_message_count_dict.values()),
                   name="Y-Axis-1 data"),
        secondary_y=False,
    )

    sentimentPlot.add_trace(
        go.Scatter(x=list(date_sentiment_dict.keys()), y=list(date_sentiment_dict.values()), name="Y-Axis-2 data"),
        secondary_y=True,
    )

    # Add figure title
    sentimentPlot.update_layout(
        title_text="Average Sentiment per day plot for SHIB and DOGE Coins."
    )

    # Set x-axis title
    sentimentPlot.update_xaxes(title_text="Date")

    # Set y-axes titles
    sentimentPlot.update_yaxes(title_text="Number-Of-Messages", secondary_y=False)
    sentimentPlot.update_yaxes(title_text="Sentiment", secondary_y=True)

    sentimentPlot.show()
