# Positive News Aggregator

## by Niko and Will

Adapted from [this guide](https://towardsdatascience.com/sentiment-analysis-on-news-headlines-classic-supervised-learning-vs-deep-learning-approach-831ac698e276), we are using Keras & TensorFlow to filter positive from non-positive news, and present the results to a user.

This is hosted on Github [here](https://github.com/willschuerman/positive-news-aggregator)


## Development:

Run this in your terminal to set up:

    conda create -n positive
    conda activate positive
    pip install -r requirements.txt -U --user

----
Train the model with:

    python src/models/tf_classifier.py

or run open the file in VSCode or Jupyter

----

If you add new requirements, save them with:

    pip list --format=freeze > requirements.txt