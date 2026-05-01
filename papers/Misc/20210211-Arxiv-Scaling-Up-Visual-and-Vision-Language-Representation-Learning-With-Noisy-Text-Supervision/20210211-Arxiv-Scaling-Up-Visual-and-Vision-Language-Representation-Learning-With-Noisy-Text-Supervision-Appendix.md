<a id="A1"></a>

## Appendix A Remove Near-Duplicate Test Images from Training Data

To detect near-duplicate images, we first train a separate high-quality image embedding model following (Wang et al., [2014](#ref-67)) with a large-scale labeled dataset as in (Juan et al., [2020](#ref-26)), and then generate 4K clusters via k-means based on all training images of the embedding model. For each query image (from the ALIGN dataset) and index image (from test sets of downstream tasks), we find their top-10 nearest clusters based on the embedding distance. Each image is then assigned to $10\choose 3$ buckets (all possible combinations of 3 clusters out of 10). For any query-index image pair that falls into the same bucket, we mark it as near-duplicated if their embedding cosine similarity is larger than 0.975. This threshold is trained on a large-scale dataset built with human rated data and synthesized data with random augmentation.
<a id="A2"></a>

## Appendix B Evaluation on SimLex-999

The image-text co-training could also help the natural language understanding as shown in Kiros et al. ([2018](#ref-29)).
For instance, with language only, it is very hard to learn antonyms.
In order to test this capability of ALIGN model, we also evaluate the word representation from ALIGN model^5 on SimLex-999  (Hill et al., [2015](#ref-22)), which is a task to compare word similarity for 999 word pairs.
We follow Kiros et al. ([2018](#ref-29)) to report the results on 9 sub-tasks each contains a subset of word pairs: *all, adjectives, nouns, verbs, concreteness quartiles (1-4)*, and *hard*.

> Footnote 5: As ALIGN uses the wordpiece tokens, one word can be split into multiple pieces. We feed the wordpieces of a word into ALIGN model and use the [CLS] token representation before the project layers as the word embeddings.

<a id="figure-5"></a>

> Table 12: SimLex-999 results (Spearman’s $\rho$).

The results are listed in the Table [12](#table-12) compared to Picturebook (Kiros et al., [2018](#ref-29)) and GloVe (Pennington et al., [2014](#ref-52)) embeddings. Overall the learned ALIGN perform better than Picturebook but slightly worse than GloVe embeddings. What is interesting is that the ALIGN word embeddings has a similar trend of Picturebook embeddings, with better performance on *nouns* and *most concrete* categories but worse on *adjs* and *less concrete* categories compared to GloVe embeddings.
ALIGN word embedding achieves the highest performance on the *hard* category, which similarity is difficult to distinguish from relatedness. This observation confirmed the hypothesis from Kiros et al. ([2018](#ref-29)) that image-based word embeddings are less likely to confuse similarity with relatedness than text learned distributional-based methods.