# Positive News Aggregator

## by Niko and Will

We are using Keras & TensorFlow to filter positive from non-positive news, and present the results to a user.

Based on https://towardsdatascience.com/sentiment-analysis-on-news-headlines-classic-supervised-learning-vs-deep-learning-approach-831ac698e276


## Development:

Run this in your terminal to set up:

    conda create -n positive
    conda activate positive
    pip install -r requirements.txt -U --user

Then run:

    python src/models/tf_classifier.py

If you add new requirements, save them with:

    pip list --format=freeze > requirements.txt