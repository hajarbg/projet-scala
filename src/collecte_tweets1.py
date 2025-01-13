import requests
import os
import json
import socket
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lower
from pyspark.sql.types import StringType
from textblob import TextBlob
import threading
import time
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import tweepy

# Twitter API credentials
API_KEY = 'C7lKJza57kBmoimEVY7ZzPPbr'
API_SECRET = '5qn0SzFNLQCX4zk8WdYjKTz7ny2yPCFdsfYT0kWONQqIpW32e0'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAALZ0yAEAAAAAFNBww%2B5TvNDW0La%2B8XuMlmLWmLQ%3Dip2IuWQQG6o2fU6Aav3U9wo76Lq6Npix3mjHZFBCEd9pv0fo6t'
query = "messi"
# Initialize Tweepy client for API v2
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Output directories
STATS_DIR = "output/sentiment_stats"
PROCESSED_TWEETS_DIR = "output/processed_tweets"

os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(PROCESSED_TWEETS_DIR, exist_ok=True)

def check_arrow_configuration(spark):
    try:
        arrow_enabled = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")
        print("Apache Arrow in PySpark is enabled:", arrow_enabled)
    except Exception as e:  # Catching a general exception to handle any kind of error when fetching the config
        print("Apache Arrow in PySpark is not enabled. Defaulting to false.")
        arrow_enabled = "false"
    return arrow_enabled
#fonction pour obtenir le stream et creer un proxy pour passer chaque tweet recu a spark
def fetch_tweets_avec_tweepy(query, max_results, host, port):
    try:
        
        response = client.search_recent_tweets(query=query, tweet_fields=["created_at", "text", "author_id"], max_results=max_results)

        if not response.data:
            print("Pas de tweets.")
            return

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"initiation socket {host}:{port}")

        conn, addr = s.accept()
        print(f"Connection etabli avec {addr}")

        for tweet in response.data:
            nettoyee_text = nettoye_text(tweet.text)
            print(f"Cleaned Tweet: {nettoyee_text}")
            conn.send((nettoye_text + "\n").encode("utf-8"))
            time.sleep(1)  # Simulate streaming delay

        conn.close()
        s.close()

    except Exception as e:
        print(f"Error: {e}")

# Fonction le nettoyage du text
def nettoye_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

#fonction pour determiner la polarite du chaque text
def analyser_sentiment(text):
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

# Fonction pour le process du stream arrivee de la part de proxy
def process_stream_avec_spark(host, port):
    spark = SparkSession.builder \
        .appName("TwitterProcessing") \
        .config("spark.ui.showConsoleProgress", "true")\
        .config("spark.sql.execution.pythonUDF.arrow.enabled", "true")\
        .getOrCreate()
    check_arrow_configuration(spark)

    sentiment_udf = udf(analyser_sentiment, StringType())
    spark.udf.register("sentiment_udf", sentiment_udf)

    # Stream tweet du socket
    lines = spark.readStream \
        .format("socket") \
        .option("host", host) \
        .option("port", port) \
        .load()

    processed_tweets = lines \
        .withColumn("tweet_nettoyee", lower(col("value"))) \
        .withColumn("sentiment", sentiment_udf(col("tweet_nettoyee")))

    stats = processed_tweets.groupBy("sentiment").count()

    try:
        processed_tweets.writeStream \
        .outputMode("append") \
        .format("console") \
        .start() \
        .awaitTermination()
    except Exception as e:
        print("Error in stream processing:", e)

    try:
        stats.writeStream \
        .outputMode("complete") \
        .format("console") \
        .start() \
        .awaitTermination()
    except Exception as e:
        print("Error in stream stats:", e)
    
    #fonction pour charger les donnees enregistree par spark
def load_sentiment_data(stats_dir):
    all_files = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir) if f.endswith(".csv")]
    if not all_files:
        return pd.DataFrame({"sentiment": [], "count": [], "timestamp": []})

    dfs = [pd.read_csv(f) for f in all_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["timestamp"] = pd.to_datetime(time.ctime(os.path.getmtime(all_files[0])))
    return combined_df
    #lancement de dashboard
def start_dashboard():
    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Dashboard"),
        dcc.Graph(id="sentiment-trend"),
        dcc.Interval(
            id="interval-update",
            interval=5000,  
            n_intervals=0
        )
    ])

    @app.callback(
        Output("sentiment-trend", "figure"),
        [Input("interval-update", "n_intervals")]
    )
    #fonction pour la mise a jour du dashboard
    def update_graph(n_intervals):
        df = load_sentiment_data(STATS_DIR)  
        if not df.empty:
            figure = {
                "data": [
                    {"x": df["timestamp"], "y": df[df["sentiment"] == "Positive"]["count"], "type": "line", "name": "Positive"},
                    {"x": df["timestamp"], "y": df[df["sentiment"] == "Negative"]["count"], "type": "line", "name": "Negative"},
                    {"x": df["timestamp"], "y": df[df["sentiment"] == "Neutral"]["count"], "type": "line", "name": "Neutral"},
                ],
                "layout": {
                    "title": "Sentiment Evolution Over Time",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Count"}
                }
            }
        else:
            figure = {"data": [], "layout": {"title": "No Data Available"}}
        return figure

    print('Pour accéder au Dashboard visiter le lien : http://127.0.0.1:8050')
    app.run_server(debug=True)


    
if __name__ == "__main__":
    # initialisation des threads:
    
    twitter_thread = threading.Thread(target=fetch_tweets_avec_tweepy, args=(query, 10, "localhost", 9999))
    twitter_thread.start()

   
    spark_thread = threading.Thread(target=process_stream_avec_spark, args=("localhost", 9999))
    spark_thread.start()

    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.start()