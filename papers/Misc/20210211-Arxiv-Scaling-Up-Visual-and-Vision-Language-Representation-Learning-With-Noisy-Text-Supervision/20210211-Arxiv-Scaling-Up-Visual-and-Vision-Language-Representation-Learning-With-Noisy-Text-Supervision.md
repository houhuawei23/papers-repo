# Title: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
- ArXiv: 2102.05918
- Authors:
  - Chao Jia — Google (United States), Google Research,
  - Yinfei Yang — Google (United States), Google Research,
  - Ye Xia — Google (United States), Google,,,,,
  - Yi-Ting Chen — Google (United States), Google,,,,,
  - Zarana Parekh — Google (United States), Google,,,,,
  - Hieu Pham — Google (United States), Google,,,,,
  - Quoc V. Le — Google (United States), [Google Brain]
  - Yunhsuan Sung — Google (United States), Google Research,
  - Zhen Li — Google (United States), Google,,,,,
  - Tom Duerig — Google (United States), Google,,,,,
- Sections: 24
- Estimated tokens: 17.6k

## Abstract

Pre-trained representations are becoming crucial for many NLP and perception tasks. While representation learning in NLP has transitioned to training on raw text without human annotations, visual and vision-language representations still rely heavily on curated training datasets that are
expensive or require expert knowledge. For vision applications, representations are mostly learned using datasets with explicit class labels such as ImageNet or OpenImages. For vision-language, popular datasets like Conceptual Captions, MSCOCO, or CLIP all involve a non-trivial data collection (and cleaning) process. This costly curation process limits the size of datasets and hence hinders the scaling of trained models. In this paper, we leverage a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset. A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss. We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme. Our visual representation achieves strong performance when transferred to classification tasks such as ImageNet and VTAB. The aligned visual and language representations enables zero-shot image classification and also set new state-of-the-art results on Flickr30K and MSCOCO image-text retrieval benchmarks, even when compared with more sophisticated cross-attention models.
The representations also enable cross-modality search with complex text and text + image queries.

<a id="S1"></a>

## 1 Introduction

In the existing literature, visual and vision-language representation learning are mostly studied separately with different training data sources. In the vision domain, pre-training
on large-scale supervised data such as ImageNet (Deng et al., [2009](#ref-8)), OpenImages (Kuznetsova et al., [2020](#ref-34)), and JFT-300M (Sun et al., [2017](#ref-63); Kolesnikov et al., [2020](#ref-31)) has proven to be critical for improving performance on downstream tasks via transfer learning. Curation of such pre-training datasets
requires heavy work on data gathering, sampling, and human annotation, and hence is difficult to scale.

Pre-training has also become the de-facto approach in vision-language modeling (Lu et al., [2019](#ref-41); Chen et al., [2020c](#ref-7); Li et al., [2020](#ref-38)). However, vision-language pre-training datasets such as Conceptual Captions (Sharma et al., [2018](#ref-61)), Visual Genome Dense Captions (Krishna et al., [2016](#ref-33)),
and ImageBERT
(Qi et al., [2020](#ref-55)) require even heavier work on human annotation, semantic parsing, cleaning and balancing. As a result, the scales of these datasets are only in the realm of $\sim$10M examples. This is at least an order of magnitude smaller than their counterparts in the vision domain, and much smaller than large corpora of text from the internet for
NLP
pre-training (e.g., Devlin et al. ([2019](#ref-10)); Radford et al. ([2019](#ref-56)); Yang et al. ([2019](#ref-70)); Liu et al. ([2019b](#ref-40)); Raffel et al. ([2020](#ref-58))).

In this work, we leverage a dataset of over one billion noisy image alt-text pairs to scale visual and vision-language representation learning.
We follow the procedures described in the Conceptual Captions dataset (Sharma et al., [2018](#ref-61)) to have a large noisy dataset. But instead of applying the complex filtering and post-processing steps as proposed by (Sharma et al., [2018](#ref-61)) to clean the dataset, we only apply simple frequency-based filtering.
The resulting dataset is noisy, but is two orders of magnitude larger than the Conceptual Captions dataset. We show that visual and vision-language representations pre-trained on our exascale dataset achieve very strong performance on a wide range of tasks.

To train our model, we use an objective that aligns the visual and language representations in a shared latent embedding space using a simple dual-encoder architecture. Similar objectives has been applied to learning visual-semantic embeddings (VSE) (Frome et al., [2013](#ref-16); Faghri et al., [2018](#ref-14)). We name our model  **ALIGN** :  **A**  **L** arge-scale  **I** ma **G** e and  **N** oisy-text embedding. Image and text encoders are learned via a contrastive loss (formulated as normalized softmax) that pushes the embeddings of matched image-text pair together while pushing those of non-matched image-text pair apart. This is one of the most effective loss functions for both self-supervised (Chen et al., [2020b](#ref-5)) and supervised (Zhai & Wu, [2019](#ref-74); Musgrave et al., [2020](#ref-47)) representation learning. Considering paired texts as fine-grained labels of images, our image-to-text contrastive loss is analogous to the conventional label-based classification objective; and the key difference is that the text encoder generates the “label” weights. The top-left of Figure [1](#figure-1) summarizes the method we use in ALIGN.

The aligned image and text representations are naturally suited for cross-modality matching/retrieval tasks and achieve state-of-the-art (SOTA) results in corresponding benchmarks. For instance, ALIGN outperforms the previous SOTA method by over 7% in most zero-shot and fine-tuned R@1 metrics in Flickr30K and MSCOCO. Moreover, such cross-modality matching naturally enables zero-shot image classification when feeding the classnames into the text encoder, achieving 76.4% top-1 accuracy in ImageNet without using any of its training samples. The image representation itself also achieves superior performance in various downstream visual tasks. For example, ALIGN achieves 88.64% top-1 accuracy in ImageNet. Figure [1](#figure-1)-bottom shows the cross-modal retrieval examples that come from a real retrieval system built by ALIGN.
<a id="S2"></a>

## 2 Related Work

High-quality visual representations for classification or retrieval are usually pre-trained on large-scale labeled datasets (Mahajan et al., [2018](#ref-42); Kolesnikov et al., [2020](#ref-31); Dosovitskiy et al., [2021](#ref-11); Juan et al., [2020](#ref-26)). Recently, self-supervised (Chen et al., [2020b](#ref-5); Tian et al., [2020](#ref-65); He et al., [2020](#ref-19); Misra & Maaten, [2020](#ref-46); Li et al., [2021](#ref-36); Grill et al., [2020](#ref-18); Caron et al., [2020](#ref-3)) and semi-supervised learning (Yalniz et al., [2019](#ref-69); Xie et al., [2020](#ref-68); Pham et al., [2020](#ref-53)) have been studied as alternative paradigms. However, models trained by these methods so far show limited transferability to downstream tasks (Zoph et al., [2020](#ref-78)).

Leveraging images and natural language captions is another direction of learning visual representations. Joulin et al. ([2015](#ref-25)); Li et al. ([2017](#ref-35)); Desai & Johnson ([2020](#ref-9)); Sariyildiz et al. ([2020](#ref-60)); Zhang et al. ([2020](#ref-76)) show that a good visual representation can be learned by predicting the captions from images, which inspires our work. These works are however limited to small datasets such as Flickr (Joulin et al., [2015](#ref-25); Li et al., [2017](#ref-35)) and COCO Captions (Desai & Johnson, [2020](#ref-9); Sariyildiz et al., [2020](#ref-60)), and the resulting models don’t produce a vision-language representation that is needed for tasks like cross-modal retrieval.

In the vision-language representation learning domain, visual-semantic embeddings (VSE) (Frome et al., [2013](#ref-16); Faghri et al., [2018](#ref-14)) and improved versions (e.g., leveraging object detectors, dense feature maps, or multi-attention layers) (Socher et al., [2014](#ref-62); Karpathy et al., [2014](#ref-28); [Kiros et al., ](#ref-30); Nam et al., [2017](#ref-48); Li et al., [2019](#ref-37); Messina et al., [2020](#ref-43); Chen et al., [2020a](#ref-4)) have been proposed.
Recently more advanced models emerge with cross-modal attention layers (Liu et al., [2019a](#ref-39); Lu et al., [2019](#ref-41); Chen et al., [2020c](#ref-7); Huang et al., [2020b](#ref-24)) and show superior performance in image-text matching tasks. However, they are orders of magnitudes slower and hence impractical for image-text retrieval systems in the real world. In contrast, our model inherits the simplest VSE form, but still outperforms all previous cross-attention models in image-text matching benchmarks.

Closely related to our work is CLIP (Radford et al., [2021](#ref-57)), which proposes visual representation learning via natural language supervision in a similar contrastive learning setting. Besides using different vision and language encoder architectures, the key difference is on training data: ALIGN follows the natural distribution of image-text pairs from the raw alt-text data, while CLIP collects the dataset by first constructing an allowlist of high-frequency visual concepts from English Wikipedia. We demonstrate that strong visual and vision-language representations can be learned with a dataset that doesn’t require expert knowledge to curate.
<a id="S3"></a>

## 3 A Large-Scale Noisy Image-Text Dataset

The focus of our work is to scale up visual and vision-language representation learning. For this purpose, we resort to a much larger dataset than existing ones. Specifically, we follow the methodology of constructing Conceptual Captions dataset (Sharma et al., [2018](#ref-61)) to get a version of raw English alt-text data (image and alt-text pairs). The Conceptual Captions dataset was cleaned by heavy filtering and post-processing. Here, for the purpose of scaling, we trade quality for scale by relaxing most of the cleaning steps in the original work. Instead, we only apply minimal frequency-based filtering as detailed below. The result is a much larger (1.8B image-text pairs) but noisier dataset.
Figure [2](#figure-2) shows some sample image-text pairs from the dataset.

<a id="figure-2"></a>

![Refer to caption](images/ablation_capacity.png)

> Figure 2: Example image-text pairs randomly sampled from the training dataset of ALIGN. One clearly noisy text annotation is marked in *italics*.

<a id="S3.SS0.SSS0.Px1"></a>

#### Image-based filtering.

Following Sharma et al. ([2018](#ref-61)), we remove pornographic images and keep only images whose shorter dimension is larger than 200 pixels and aspect ratio is smaller than 3. Images with more than 1000 associated alt-texts are discarded. To ensure that we don’t train on test images, we also remove duplicates or near-duplicates of test images in all downstream evaluation datasets (e.g., ILSVRC-2012, Flickr30K, and MSCOCO). See Appendix [A](#A1) for more details.

<a id="S3.SS0.SSS0.Px2"></a>

#### Text-based filtering.

We exclude alt-texts that are shared by more than 10 images. These alt-texts are often irrelevant to the content of the images (e.g., “1920x1080”, “alt_img”, and “cristina”). We also discard alt-texts that contain any rare token (outside of 100 million most frequent unigrams and bigrams from the raw dataset), and those that are either too short ($<$3 unigrams) or too long ($>$20 unigrams). This removes noisy texts like “image_tid 25&id mggqpuweqdpd&cache 0&lan_code 0”, or texts that are too generic to be useful.
<a id="S4"></a>

## 4 Pre-training and Task Transfer

<a id="S4.SS1"></a>

### 4.1 Pre-training on Noisy Image-Text Pairs

We pre-train ALIGN using a dual-encoder architecture. The model consists of a pair of image and text encoders with a cosine-similarity combination function at the top. We use EfficientNet with global pooling (without training the 1x1 conv layer in the classification head) as the image encoder and BERT with [CLS] token embedding as the text embedding encoder (we generate 100k wordpiece vocabulary from our training dataset). A fully-connected layer with linear activation is added on top of BERT encoder to match the dimension from the image tower. Both image and text encoders are trained from scratch.

The image and text encoders are optimized via normalized softmax loss (Zhai & Wu, [2019](#ref-74)). In training, we treat matched image-text pairs as positive and all other random image-text pairs that can be formed in a training batch as negative.

We minimize the sum of two losses: one for image-to-text classification

<a id="eq-1"></a>

$$
\small L_{i2t}=-\frac{1}{N}\sum_{i}^{N}\log{\frac{\exp(x_{i}^{\top}y_{i}/\sigma)}{\sum_{j=1}^{N}\exp(x_{i}^{\top}y_{j}/\sigma)}} \tag{(1)}
$$

and the other for text-to-image classification

<a id="eq-2"></a>

$$
\small L_{t2i}=-\frac{1}{N}\sum_{i}^{N}\log{\frac{\exp(y_{i}^{\top}x_{i}/\sigma)}{\sum_{j=1}^{N}\exp(y_{i}^{\top}x_{j}/\sigma)}} \tag{(2)}
$$

Here, $x_{i}$ and $y_{j}$ are the normalized embedding of image in the $i$-th pair and that of text in the $j$-th pair, respectively. $N$ is the batch size, and $\sigma$ is the temperature to scale the logits. For in-batch negatives to be more effective, we concatenate embeddings from all computing cores to form a much larger batch. The temperature variable is crucial as both image and text embeddings are L2-normalized. Instead of manually sweeping for the optimal temperature value, we find that it can be effectively learned together with all the other parameters.

<a id="S4.SS2"></a>

### 4.2 Transferring to Image-Text Matching & Retrieval

We evaluate ALIGN models on image-to-text and text-to-image retrieval tasks, with and without finetuning. Two benchmark datasets are considered: Flickr30K (Plummer et al., [2015](#ref-54)) and MSCOCO (Chen et al., [2015](#ref-6)). We also evaluate ALIGN on Crisscrossed Captions (CxC) (Parekh et al., [2021](#ref-50)), which is an extension of MSCOCO with additional human semantic similarity judgments for caption-caption, image-image, and image-caption pairs.
With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-to-image, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).
As the training set is identical to the original MSCOCO, we can directly evaluate the MSCOCO fine-tuned ALIGN model on CxC annotations.

<a id="S4.SS3"></a>

### 4.3 Transferring to Visual Classification

We first apply zero-shot transfer of ALIGN to visual classification tasks on ImageNet ILSVRC-2012 benchmark (Deng et al., [2009](#ref-8)) and its variants including ImageNet-R(endition) (Hendrycks et al., [2020](#ref-20)) (non-natural images such as art, cartoons, sketches), ImageNet-A(dversarial) (Hendrycks et al., [2021](#ref-21)) (more challenging images for ML models), and ImageNet-V2 (Recht et al., [2019](#ref-59)). All of these variants follow the same set (or a subset) of ImageNet classes, while the images in ImageNet-R and ImageNet-A are sampled from drastically different distributions from ImageNet.

We also transfer the image encoder to downstream visual classification tasks. For this purpose, we use the ImageNet as well as a handful of smaller fine-grained classification datasets such as Oxford Flowers-102 (Nilsback & Zisserman, [2008](#ref-49)), Oxford-IIIT Pets (Parkhi et al., [2012](#ref-51)), Stanford Cars (Krause et al., [2013](#ref-32)), and Food101 (Bossard et al., [2014](#ref-2)). For ImageNet, results from two settings are reported: training the top classification layer only (with frozen ALIGN image encoder) and fully fine-tuned. Only the latter setting is reported for fine-grained classification benchmarks. Following Kolesnikov et al. ([2020](#ref-31)), we also evaluate the robustness of our model on Visual Task Adaptation Benchmark (VTAB) (Zhai et al., [2019](#ref-75)) which consists of 19 diverse (covering subgroups of natural, specialized and structured image classification tasks) visual classification tasks with 1000 training samples each.
<a id="S5"></a>

## 5 Experiments and Results

We train our ALIGN models from scratch, using the open-sourced implementation of EfficientNet as the image encoder and BERT as the text encoder. Unless in the ablation study,
we use the results of ALIGN where the image encoder is EfficientNet-L2 and the text encoder is BERT-Large.
The image encoder is trained at resolution of 289 $\times$ 289 pixels no matter what EfficientNet variant is used. We first resize input images to 346 $\times$ 346 resolution and then perform random crop (with additional random horizontal flip) in training and central crop in evaluation. For BERT we use wordpiece sequence of maximum 64 tokens since the input texts are no longer than 20 unigrams. The softmax temperature variable is initialized as 1.0 (this temperature variable is shared between image-to-text loss and text-to-image loss) and we use 0.1 as label smoothing parameter in the softmax losses. We use LAMB optimizer (You et al., [2020](#ref-71))^1 with weight decay ratio 1e-5. The learning rate is warmed up linearly to 1e-3 from zero in 10k steps, and then linearly decay to zero in 1.2M steps ($\sim$12 epochs). We train the model on 1024 Cloud TPUv3 cores with 16 positive pairs on each core. Therefore the total effective batch size is 16384.

> Footnote 1: We tried SGD with momentum and ADAM which are known to work well for CNNs and BERT respectively. LAMB appears to be a better choice for training both image and text encoders.

<a id="table-1"></a>

|  |  | Flickr30K (1K test set) | MSCOCO (5K test set) |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | image $\to$ text | text $\to$ image | image $\to$ text | text $\to$ image |  |  |  |  |  |  |  |  |
|  |  | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |
| Zero-shot | ImageBERT | 70.7 | 90.2 | 94.0 | 54.3 | 79.6 | 87.5 | 44.0 | 71.2 | 80.4 | 32.3 | 59.0 | 70.2 |
| UNITER | 83.6 | 95.7 | 97.7 | 68.7 | 89.2 | 93.9 | - | - | - | - | - | - |  |
| CLIP | 88.0 | 98.7 | 99.4 | 68.7 | 90.6 | 95.2 | 58.4 | 81.5 | 88.1 | 37.8 | 62.4 | 72.2 |  |
|  **ALIGN**  |  **88.6**  |  **98.7**  |  **99.7**  |  **75.7**  |  **93.8**  |  **96.8**  |  **58.6**  |  **83.0**  |  **89.7**  |  **45.6**  |  **69.8**  |  **78.6**  |  |
| Fine-tuned | GPO | 88.7 | 98.9 | 99.8 | 76.1 | 94.5 | 97.1 | 68.1 | 90.2 | - | 52.7 | 80.2 | - |
| UNITER | 87.3 | 98.0 | 99.2 | 75.6 | 94.1 | 96.8 | 65.7 | 88.6 | 93.8 | 52.9 | 79.9 | 88.0 |  |
| ERNIE-ViL | 88.1 | 98.0 | 99.2 | 76.7 | 93.6 | 96.4 | - | - | - | - | - | - |  |
| VILLA | 87.9 | 97.5 | 98.8 | 76.3 | 94.2 | 96.8 | - | - | - | - | - | - |  |
| Oscar | - | - | - | - | - | - | 73.5 | 92.2 | 96.0 | 57.5 | 82.8 |  **89.8**  |  |
|  **ALIGN**  |  **95.3**  |  **99.8**  |  **100.0**  |  **84.9**  |  **97.4**  |  **98.6**  |  **77.0**  |  **93.5**  |  **96.9**  |  **59.9**  |  **83.3**  |  **89.8**  |  |

<a id="table-2"></a>

|  | image $\to$ text | text $\to$ image | text $\to$ text | image $\to$ image |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |
| VSE++ | 43.1 | 74.3 | 84.2 | 32.5 | 62.7 | 75.4 | 38.7 | 62.3 | 72.2 | 36.4 | 70.4 | 81.3 |
| VSRN | 52.4 | 81.9 | 90.0 | 40.1 | 71.1 | 81.5 | 41.0 | 64.8 | 74.5 | 44.2 | 76.7 | 86.2 |
| DE_I2T | 53.9 | 82.7 | 91.2 | 39.8 | 70.2 | 80.9 | 26.0 | 47.1 | 57.5 | 38.3 | 74.1 | 85.0 |
| DE_T2T+I2T | 55.9 | 84.2 | 91.8 | 41.7 | 72.3 | 83.0 | 42.4 | 64.9 | 74.0 | 38.5 | 73.6 | 84.9 |
|  **ALIGN**  |  **78.1**  |  **94.3**  |  **97.4**  |  **61.8**  |  **84.9**  |  **91.1**  |  **45.4**  |  **66.8**  |  **75.2**  |  **49.4**  |  **81.4**  |  **89.1**  |

<a id="table-3"></a>

|  **Model**  |  **STS**  |  **SIS**  |  **SITS**  |  **Mean Avg**  |
| --- | --- | --- | --- | --- |
| VSE++ |  **74.4$\pm$0.4**  | 73.3$\pm$0.9 | 55.2$\pm$1.5 | 67.6 |
| VSRN | 73.0$\pm$0.4 | 70.1$\pm$1.0 | 60.4$\pm$1.3 | 67.8 |
| DE_I2T | 50.9$\pm$0.6 |  **81.3$\pm$0.7**  | 61.6$\pm$1.4 | 64.6 |
| DE_T2T+I2T | 74.2$\pm$0.4 | 74.5$\pm$0.9 | 61.9$\pm$1.3 | 70.2 |
|  **ALIGN**  | 72.9$\pm$0.4 | 77.2$\pm$0.8 |  **67.6$\pm$1.2**  |  **72.6**  |

<a id="S5.SS1"></a>

### 5.1 Image-Text Matching & Retrieval

We evaluate ALIGN on Flickr30K and MSCOCO cross-modal retrieval benchmarks, in both zero-shot and fully fine-tuned settings. We follow (Karpathy & Fei-Fei, [2015](#ref-27)) and most existing works to obtain the train/test splits. Specifically, for Flickr30K, we evaluate on the standard 1K test set, and finetune on the 30k training set. For MSCOCO, we evaluate on the 5K test set, and finetune on 82K training plus 30K additional validation images that are not in the 5K validation or 5K test sets.

During fine-tuning, the same loss function is used. But there can be false negatives when the batch size is comparable to the total number of training samples. So we reduce the global batch size from 16384 to 2048. We also reduce the initial learning rate to 1e-5 and train for 3K and 6K steps (with linear decay) respectively on Flickr30K and MSCOCO. All the other hyper-parameters are kept the same as pre-training.

Table [1](#table-1) shows that, compared to previous works, ALIGN achieves SOTA results in all metrics of Flickr30K and MSCOCO benchmarks. In the zero-shot setting, ALIGN gets more than 7% improvement in image retrieval task compared to the previous SOTA, CLIP (Radford et al., [2021](#ref-57)). With fine-tuning, ALIGN outperforms all existing methods by a large margin, including those that employ more complex cross-modal attention layers such as ImageBERT (Qi et al., [2020](#ref-55)), UNITER (Chen et al., [2020c](#ref-7)), ERNIE-ViL (Yu et al., [2020](#ref-73)), VILLA (Gan et al., [2020](#ref-17)) and Oscar (Li et al., [2020](#ref-38)).

Table [2](#table-2) reports the performance of ALIGN on Crisscrossed Captions (CxC) retrieval tasks. Again, ALIGN achieves SOTA results in all metrics, especially by a large margin on image-to-text (+22.2% R@1) and text-to-image (20.1% R@1) tasks.
Table [3](#table-3) shows that ALIGN also outperforms the previous SOTA on SITS task with an improvement of 5.7%.
One interesting observation is that, despite being much better on inter-modal tasks, ALIGN is not as impressive on intra-modal tasks.
For instance, the improvements on text-to-text and image-to-image retrieval tasks (in particular the former) are less significant compared to those on image-to-text and text-to-image tasks.
The performance on STS and SIS tasks is also slightly worse than VSE++ and DE_I2T.
We suspect it is because the training objective of ALIGN focuses on cross-modal (image-text) matching instead of intra-modal matching.
Parekh et al. ([2021](#ref-50)) suggest multitask learning could produce more balanced representations.
We leave it to the future work.

<a id="S5.SS2"></a>

### 5.2 Zero-shot Visual Classification

If we directly feed the texts of classnames into the text encoder, ALIGN is able to classify images into candidate classes via image-text retrieval. Table [4](#table-4) compares ALIGN with CLIP on Imagenet and its variants. Similar to CLIP, ALIGN shows great robustness on classification tasks with different image distributions. In order to make a fair comparison, we use the same prompt ensembling method as CLIP. Each classname is expanded with a set of prompt templates defined by CLIP such as “A photo of a {classname}”. The class embedding is computed by averaging the embeddings of all templates followed by an L2-normalization. We find that such ensembling gives 2.9% improvement on ImageNet top-1 accuracy.

<a id="table-4"></a>

| Model | ImageNet | ImageNet-R | ImageNet-A | ImageNet-V2 |
| --- | --- | --- | --- | --- |
| CLIP | 76.2 | 88.9 |  **77.2**  |  **70.1**  |
|  **ALIGN**  |  **76.4**  |  **92.2**  | 75.8 |  **70.1**  |

<a id="table-5"></a>

| Model (backbone) | Acc@1 w/ frozen features | Acc@1 | Acc@5 |
| --- | --- | --- | --- |
| WSL (ResNeXt-101 32x48d) | 83.6 | 85.4 | 97.6 |
| CLIP (ViT-L/14) | 85.4 | - | - |
| BiT (ResNet152 x 4) | - | 87.54 | 98.46 |
| NoisyStudent (EfficientNet-L2) | - | 88.4 | 98.7 |
| ViT (ViT-H/14) | - | 88.55 | - |
| Meta-Pseudo-Labels (EfficientNet-L2) | - |  **90.2**  |  **98.8**  |
|  **ALIGN**  (EfficientNet-L2) |  **85.5**  | 88.64 | 98.67 |

<a id="S5.SS3"></a>

### 5.3 Visual Classification w/ Image Encoder Only

On the ImageNet benchmark, we first freeze the learned visual features and only train the classification head. Afterwards we fine-tune all layers. We use basic data augmentations including random cropping (same as in Szegedy et al. ([2015](#ref-64))) and horizontal flip. In evaluation we apply a single central crop with ratio of 0.875. Following Touvron et al. ([2019](#ref-66)), we use 0.8 scale ratio between training and evaluation to mitigate the resolution discrepancy introduced by random crop. Specifically, train/eval resolution is 289/360 with frozen visual features, and is 475/600 when fine-tuning all variables.

In both stages of training, we use a global batch size of 1024, SGD optimizer with momentum 0.9, and learning rate decayed every 30 epochs with ratio 0.2 (100 epochs in total). Weight decay is set to zero. With frozen visual features, we use the initial learning rate of 0.1. When fine-tuning all layers with use the initial learning rate of 0.01, and use 10x smaller learning rate on the backbone network compared to the classification head.

Table [5](#table-5) compares ALIGN with previous methods on the ImageNet benchmark. With frozen features, ALIGN slightly outperforms CLIP and achieves SOTA result of 85.5% top-1 accuracy. After fine-tuning ALIGN achieves higher accuracy than BiT and ViT models, and is only worse than Meta Pseudo Labels which requires deeper interaction between ImageNet training and large-scale unlabeled data. Compared to NoisyStudent and Meta-Pseudeo-Labels which also use EfficientNet-L2, ALIGN saves 44% FLOPS by using smaller test resolution (600 instead of 800).

In VTAB eval, we follow a hyper-parameter sweep as shown in the Appendix I in (Zhai et al., [2019](#ref-75)) with 50 trials for each task. Each task is trained on 800 images and the hyperparameters are selected using the validation set of 200 images. After the sweep, the selected hyperparameters are used to train on the combined training and validation splits of 1000 images for each task. Table [6](#table-6) reports the mean accuracy (including the breakdown results on each subgroup) with standard deviation from three fine-tuning runs and shows that ALIGN outperforms BiT-L (Kolesnikov et al., [2020](#ref-31)) with similar hyper-parameter selection method applied.

<a id="table-6"></a>

| Model | All tasks | Natural | Specialized | Structured |
| --- | --- | --- | --- | --- |
| Bit-L | 78.72 | - | - | - |
|  **ALIGN**  |  **79.99$\pm$0.15**  | 83.38 | 87.56 | 73.25 |

To evaluate on smaller fine-grained classification benchmarks, we adopt a simple fine-tuning strategy for all tasks. We use the same data augmentation and optimizer as in ImageNet fine-tuning. Similarly, we first train the classification head and then fine-tune all layers, except with batch norm statistics frozen. The train/eval resolution is fixed at 289/360. We use batch size 256 and weight decay 1e-5. The initial learning rate is set to 1e-2 and 1e-3 respectively, with cosine learning rate decay in 20k steps. Table [7](#table-7) compares ALIGN with BiT-L (Kolesnikov et al., [2020](#ref-31)) and SAM (Foret et al., [2021](#ref-15)) which both apply same fine-tuning hyper-parameters for all tasks.^2 For small tasks like these, details in fine-tuning matter. So we list the baseline results in (Foret et al., [2021](#ref-15)) without using SAM optimization for a fairer comparison. Our result (average of three runs) is comparable to the SOTA results without tweaking on optimization algorithms.

> Footnote 2: ViT (Dosovitskiy et al., [2021](#ref-11)) uses different hyper-parameters for different tasks and hence is not included in comparison.

<a id="table-7"></a>

| Model | Oxford | Oxford | Stanford | Food101 |
| --- | --- | --- | --- | --- |
| BiT-L | 99.63 | 96.62 | - | - |
| SAM-baseline | 99.60 | 96.92 | 95.07 | 96.03 |
| SAM-final |  **99.65**  |  **97.10**  | 95.96 |  **96.18**  |
|  **ALIGN**  |  **99.65**  | 96.19 |  **96.13**  | 95.88 |
<a id="S6"></a>

## 6 Ablation Study

In the ablation study, we compare model performance mostly on MSCOCO zero-shot retrieval and ImageNet K-Nearest-neighbor (KNN) tasks.^3 We find these two metrics are representative and correlate well with other metrics reported in the section above. If not mentioned, hyper-parameters other than the ablated factor are kept the same as in the baseline model.

> Footnote 3: For each image in the validation set of ImageNet, we retrieve its nearest neighbors from the training set w/ pre-trained image encoder. Recall@K metric is calculated based on if the groundtruth label of the query image appears in the top-K retrieved images.

<a id="S6.SS1"></a>

### 6.1 Model Architectures

We first study the performance of ALIGN models using different image and text backbones. We train EfficientNet from B1 to L2 for the image encoder and BERT-Mini to BERT-Large for the text encoder. We add an additional fully-connected layer with linear activation on top of B1, B3, B5 and L2 globally-pooled features to match the output dimension of B7 (640). A similar linear layer is added to all text encoders. We reduce the training steps to 1M in ablation to save some runtime.

Figures [3](#figure-3) shows MSCOCO zero-shot retrieval and ImageNet KNN results with different combinations of image and text backbones. Model quality improves nicely with larger backbones except that the ImageNet KNN metric starts to saturate from BERT-Base to BERT-Large with EfficientNet-B7 and EfficientNet-L2. As expected, scaling up image encoder capacity is more important for vision tasks (e.g., even with BERT-Mini text tower, L2 performs better than B7 with BERT-Large). In image-text retrieval tasks the image and text encoder capacities are equally important. Based on the nice scaling property shown in Figure [3](#figure-3), we only fine-tune the model with EfficientNet-L2 + BERT-Large as reported in Section [5](#section-5).

<a id="figure-3"></a>

![Refer to caption](images/NNIQ_t2i_new.png)

> Figure 3: Zero-shot image-text retrieval and ImageNet KNN accuracy@1 with different image and text encoder sizes.

We then study key architecture hyperparameters including embedding dimensions,
number of random negatives in the batch, and the softmax temperature. Table [8](#table-8) compares a number of model variants to a baseline model (first row) trained with the following settings: EfficientNet-B5 image encoder, BERT-Base text encoder, embedding dimension 640, all negatives in the batch, and a learnable softmax temperature.

Rows 2-4 of Table [8](#table-8) show that model performance improves with higher embedding dimensions. Hence, we let the dimension scale with larger EfficientNet backbone (L2 uses 1376).
Rows 5 and 6 show that using fewer in-batch negatives (50% and 25%) in the softmax loss will degrade the performance.
Rows 7-9 study the effect of the temperature parameter in the softmax loss. Compared to the baseline model that learns the temperature parameter (converged to about 1/64), some hand-selected, fixed temperatures could be slightly better. However, we choose to use the learnable temperature as it performs competitively and makes learning easier. We also notice that the temperature usually quickly decrease to only around 1.2x of the converged values in the first 100k steps, and then slowly converges until the end of training.

<a id="table-8"></a>

|  **Model**  |  **MSCOCO**  |  **ImangeNet KNN**  |  |
| --- | --- | --- | --- |
|
B5 + BERT-base | 51.7 |  **37.5**  | 64.6 |
|       w/ embedding dim=320 | 50.3 | 34.1 | 64.0 |
|       w/ embedding dim=160 | 47.0 | 34.4 | 63.7 |
|       w/ embedding dim=80 | 42.0 | 29.3 | 61.9 |
|       w/ 50% in-batch negs | 50.2 | 37.0 | 63.8 |
|       w/ 25% in-batch negs | 48.7 | 35.8 | 63.3 |
|       w/ softmax temp=1/128 |  **52.2**  | 36.5 |  **64.8**  |
|       w/ softmax temp=1/64 |  **52.2**  | 37.3 |  **64.8**  |
|       w/ softmax temp=1/32 | 39.6 | 26.9 | 61.2 |

<a id="S6.SS2"></a>

### 6.2 Pre-training Datasets

It’s also important to understand how the model performs when trained on different datasets with varying size. For this purpose, we train two models: EfficientNet-B7 + BERT-base and EfficientNet-B3 + BERT-mini on three different datasets: full ALIGN training data, 10% randomly sampled ALIGN training data, and Conceptual Captions (CC-3M, around 3M images). CC-3M is much smaller so we train the model with 1/10 of the default number of steps. All models are trained from scratch. As shown in Table [9](#table-9), a large scale training set is essential to allow scaling up of our models and to achieve better performance. For instance, models trained on ALIGN data clearly outperform those trained on CC-3M data. On CC-3M, B7+BERT-base starts to overfit and performs even worse than B3+BERT-mini. Conversely, a larger model is required to fully utilize the larger dataset – the smaller B3+BERT-mini almost saturate at 10% of ALIGN data, while with the larger B7+BERT-base, there is a clear improvement with full ALIGN data.

<a id="table-9"></a>

|  **Model + Data**  |  **MSCOCO**  |  **ImangeNet KNN**  |  |
| --- | --- | --- | --- |
|        + ALIGN full data | 55.4 | 41.7 | 69.3 |
|        + ALIGN 10% data | 52.0 | 39.2 | 68.8 |
|        + CC-3M data | 18.9 | 15.5 | 48.7 |
| B3 + BERT-mini |  |  |  |
|        + ALIGN full data | 37.4 | 24.5 | 56.5 |
|        + ALIGN 10% data | 36.7 | 24.4 | 55.8 |
|        + CC-3M data | 22.1 | 17.3 | 48.9 |

To understand better how data size scaling wins over the increased noise, we further randomly sample 3M, 6M, and 12M ALIGN training data and compare them with the cleaned CC-3M data on B7+BERT-base model. Table [10](#table-10) shows that while the ALIGN data performs much worse than CC data with the same size (3M), the model quality trained on 6M and 12M ALIGN data rapidly catches up. Despite being noisy, ALIGN data outperforms Conceptual Captions with only 4x size.

<a id="table-10"></a>

|  **Model + Data**  |  **MSCOCO**  |  **ImangeNet KNN**  |  |
| --- | --- | --- | --- |
|        + ALIGN 12M data | 23.8 | 17.5 | 51.4 |
|        + ALIGN 6M data | 15.8 | 11.9 | 47.9 |
|        + ALIGN 3M data | 8.1 | 6.3 | 41.3 |
|        + CC-3M data | 18.9 | 15.5 | 48.7 |
<a id="S7"></a>

## 7 Analysis of Learned Embeddings

We build a simple image retrieval system to study the behaviors of embeddings trained by ALIGN. For demonstration purposes, we use an index consisting of 160M CC-BY licensed images that are separate from our training set. Figure [4](#figure-4) shows the top 1 text-to-image retrieval results for a handful of text queries not existing in the training data. ALIGN can retrieve precise images given detailed descriptions of a scene, or fine-grained or instance-level concepts like landmarks and artworks. These examples demonstrate that our ALIGN model can align images and texts with similar semantics, and that ALIGN can generalize to novel complex concepts.

<a id="figure-4"></a>

![Refer to caption](images/NNIQ_i+t.png)

> Figure 4: Image retrieval with fine-grained text queries using ALIGN’s embeddings.

<a id="figure-5"></a>

![Refer to caption](/html/2102.05918/assets/x5.png)

> Figure 5: Image retrieval with image$\pm$text queries. We add (or subtract) text query embedding
to (or from) the image query embedding, and then use the resulting embedding to retrieve relevant images using cosine similarity.

Previously word2vec (Mikolov et al., [2013a](#ref-44), [b](#ref-45)) shows that linear relationships between word vectors emerge as a result of training them to predict adjacent words in sentences and paragraphs. We show that linear relationships between image and text embeddings also emerge in ALIGN. We perform image retrieval using a combined image+text query. Specifically, given a query image and a text string, we add their ALIGN embeddings together and use it to retrieve relevant images.^4 Figure [5](#figure-5) shows results for a variety of image+text queries. These examples not only demonstrate great compositionality of ALIGN embeddings across vision and language domains, but also show the feasibility of a new paradigm of “search with multi-modal query” that would otherwise be hard using only text query or image query. For instance, one could now look for the “Australia” or “Madagascar” equivalence of pandas, or turn a pair of black shoes into identically-looking shoes with the color of “beige”. Finally, as shown in the last three rows of Figure [5](#figure-5), removing objects/attributes from a scene is possible by performing subtraction in the embedding space.

> Footnote 4: We normalize the text and image embeddings before adding them. We also tried various scale factor and found that a scale of 2 for the text embedding and 1 for the image embedding give best results as shown in the figure, although 1:1 also works well.
<a id="S8"></a>

## 8 Multilingual ALIGN Model

One advantage of ALIGN is that the model is trained on noisy web image text data with very simple filters, and none of the filters are language specific.
Given that, we further lift the language constraint of the conceptual caption data processing pipeline to extend the dataset to multilingual (covering 100+ languages) and match its size to the English dataset (1.8B image-text pairs).
A multilingual model ALIGN_mling is trained using this data.
We created a new mutlilingual wordpiece vocabulary with size 250k to cover all languages. Model training follows the exact English configuration.

We test the multilingual model on Multi30k, a multilingual image text retrieval dataset extends Flickr30K (Plummer et al., [2015](#ref-54)) to German (de) (Elliott et al., [2016](#ref-12)), French (fr) (Elliott et al., [2017](#ref-13)) and Czech (cs) (Barrault et al., [2018](#ref-1)). The dataset consists of 31,783 images with 5 captions per image in English and German and 1 caption per image in French and Czech. The train/dev/test splits are defined in Young et al. ([2014](#ref-72)).
We evaluate the zero-shot model performance of ALIGN and compare it with M^3P (Huang et al., [2020a](#ref-23)) and UC2 (Zhou et al., [2021](#ref-77)).
The evaluation metric is mean Recall (mR), which computes the average score of Recall@1, Recall@5 and Recall@10 on image-to-text retrieval and text-to-image retrieval tasks.

Table [11](#table-11) shows that the zero-shot performance of ALIGN_mling outperforms M^3P on all languages by a large margin, with the largest +57.8 absolution mR improvement on fr. The zero-shot performance of ALIGN_mling is even comparable to the fine-tuned (w/ training splits) M^3P and UC2 except on cs. On en, ALIGN_mling performs slightly worse on its counterpart ALIGN_EN (trained on EN-only data.)

<a id="table-11"></a>

|  **Model**  |  **en**  |  **de**  |  **fr**  |  **cs**  |
| --- | --- | --- | --- | --- |
| *zero-shot* |  |  |  |  |
|       M^3P | 57.9 | 36.8 | 27.1 | 20.4 |
|  **ALIGN_EN**  |  **92.2**  | - | - | - |
|  **ALIGN_mling**  | 90.2 | 84.1 |  **84.9**  | 63.2 |
| * w/ fine-tuning* |  |  |  |  |
|       M^3P | 87.7 | 82.7 | 73.9 | 72.2 |
|       UC2 | 88.2 |  **84.5**  | 83.9 |  **81.2**  |
<a id="S9"></a>

## 9 Conclusion

We present a simple method of leveraging large-scale noisy image-text data to scale up visual and vision-language representation learning. Our method avoids heavy work on data curation and annotation, and only requires minimal frequency-based cleaning. On this dataset, we train a simple dual-encoder model using a contrastive loss. The resulting model, named ALIGN, is capable of cross-modal retrieval and significantly outperforms SOTA VSE and cross-attention vision-language models. In visual-only downstream tasks, ALIGN is also comparable to or outperforms SOTA models trained with large-scale labeled data.
<a id="S10"></a>

## 10 Social Impacts and Future Work

While this work shows promising results from a methodology perspective with a simple data collection method, additional analysis of the data and the resulting model is necessary before the use of the model in practice. For instance, considerations should be made towards the potential for the use of harmful text data in alt-texts to reinforce such harms. On the fairness front, data balancing efforts may be required to prevent reinforcing stereotypes from the web data. Additional testing and training around sensitive religious or cultural items should be taken to understand and mitigate the impact from possibly mislabeled data.

Further analysis should also be taken to ensure that the demographic distribution of humans and related cultural items like clothing, food, and art do not cause model performance to be skewed. Analysis and balancing would be required if such models will be used in production.

Finally, unintended misuse of such models for surveillance or other nefarious purposes should be prohibited.
<a id="Sx1"></a>

## Acknowledgements

This work was done with invaluable help from colleagues from Google. We would like to thank Jan Dlabal and Zhe Li for continuous support in training infrastructure, Simon Kornblith for building the zero-shot & robustness model evaluation on ImageNet variants, Xiaohua Zhai for help on conducting VTAB evaluation, Mingxing Tan and Max Moroz for suggestions on EfficientNet training, Aleksei Timofeev for the early idea of multimodal query retrieval, Aaron Michelony and Kaushal Patel for their early work on data generation, and Sergey Ioffe, Jason Baldridge and Krishna Srinivasan for the insightful feedback and discussion.