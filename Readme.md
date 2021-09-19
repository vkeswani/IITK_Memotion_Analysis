# Memotion Analysis, SemEval-2020 International Workshop

**Authors**: Vishal Keswani, Sakshi Singh

In this repository, we present our approaches for the Sentiment Analysis of Internet Memes. The challenge was organized under the SemEval-2020 International Workshop (Task 8: Memotion Analysis). We stood **first** in the Sentiment Analysis subtask. <br>

Memes are essentially bimodal. They have both Text and Image components. We used both bimodal (text and image) and unimodal (text-only) approaches as listed below: <br>

**Bimodal**
* FFNN+CNN+SVM
* MMBT 

**Unimodal**
* Naive Bayes
* FFNN with Word2vec
* BERT

The details of the architectures and performance are discussed in the corresponding paper (https://www.aclweb.org/anthology/2020.semeval-1.150/).

### Resources
* https://blog.insightdatascience.com/using-bert-for-state-of-the-art-pre-training-for-natural-language-processing-1d87142c29e7
* https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
* https://textblob.readthedocs.io/en/dev/_modules/textblob/classifiers.html
