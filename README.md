# [Memotion Analysis](https://competitions.codalab.org/competitions/20629), SemEval-2020 International Workshop, COLING

**Authors**: Vishal Keswani, Sakshi Singh

In this repository, we present our approaches for the Sentiment Analysis of Internet Memes. The challenge was organized under the SemEval-2020 International Workshop (Task 8: Memotion Analysis). We stood **first** in the Sentiment Analysis subtask. <br>

## Abstract [[Paper]](https://www.aclweb.org/anthology/2020.semeval-1.150/)
Social media is abundant in visual and textual information presented together or in isolation.
Memes are the most popular form, belonging to the former class. In this paper, we present our
approaches for the Memotion Analysis problem as posed in SemEval-2020 Task 8. The goal
of this task is to classify memes based on their emotional content and sentiment. We leverage
techniques from Natural Language Processing (NLP) and Computer Vision (CV) towards the
sentiment classification of internet memes (Subtask A). We consider Bimodal (text and image) as
well as Unimodal (text-only) techniques in our study ranging from the Na¨ıve Bayes classifier to
Transformer-based approaches. Our results show that a text-only approach, a simple Feed Forward
Neural Network (FFNN) with Word2vec embeddings as input, performs superior to all the others.
We stand first in the Sentiment analysis task with a relative improvement of 63% over the baseline
macro-F1 score. Our work is relevant to any task concerned with the combination of different
modalities.

## Approaches
Memes are essentially bimodal. They have both Text and Image components. We used both bimodal (text and image) and unimodal (text-only) approaches as listed below: <br>

**Bimodal**
* ffnn_cnn_svm: Feed Forward Neural Network with Word2vec for text, CNN for images, combined via an SVM classifier 
* mmbt: Multimodal Bitransformer 

**Unimodal**
* naive_bayes: Simple Naive Bayes classifier 
* ffnn_w2v: Feed Forward Neural Network with Word2ve
* bert: Bidirectional Encoder Representations via Transformers

The details of the architectures and performance are discussed in the [paper](https://www.aclweb.org/anthology/2020.semeval-1.150/).

## Citation
```
@inproceedings{keswani-etal-2020-iitk-semeval,
    title = "{IITK} at {S}em{E}val-2020 Task 8: Unimodal and Bimodal Sentiment Analysis of {I}nternet Memes",
    author = "Keswani, Vishal  and
      Singh, Sakshi  and
      Agarwal, Suryansh  and
      Modi, Ashutosh",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://aclanthology.org/2020.semeval-1.150",
    pages = "1135--1140",
    abstract = "Social media is abundant in visual and textual information presented together or in isolation. Memes are the most popular form, belonging to the former class. In this paper, we present our approaches for the Memotion Analysis problem as posed in SemEval-2020 Task 8. The goal of this task is to classify memes based on their emotional content and sentiment. We leverage techniques from Natural Language Processing (NLP) and Computer Vision (CV) towards the sentiment classification of internet memes (Subtask A). We consider Bimodal (text and image) as well as Unimodal (text-only) techniques in our study ranging from the Na ̈{\i}ve Bayes classifier to Transformer-based approaches. Our results show that a text-only approach, a simple Feed Forward Neural Network (FFNN) with Word2vec embeddings as input, performs superior to all the others. We stand first in the Sentiment analysis task with a relative improvement of 63{\%} over the baseline macro-F1 score. Our work is relevant to any task concerned with the combination of different modalities.",
}

```

## Resources
* https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7
* https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
* https://textblob.readthedocs.io/en/dev/_modules/textblob/classifiers.html
