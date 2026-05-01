## 附录 A：从训练数据中移除近似重复的测试图像（Appendix A Remove Near-Duplicate Test Images from Training Data）

为了检测近似重复的图像，我们首先按照 (Wang et al., [2014](#ref-67)) 的方法，使用 (Juan et al., [2020](#ref-26)) 中的大规模标注数据集训练一个独立的高质量 **图像嵌入模型（image embedding model）** ，然后基于该嵌入模型的所有训练图像，通过  **k-means**  聚类生成 4K 个簇。对于每个查询图像（来自 ALIGN 数据集）和索引图像（来自下游任务的测试集），我们根据嵌入距离找到它们各自的前 10 个最近邻簇。然后，每张图像被分配到 $10\choose 3$ 个桶中（即从 10 个簇中任选 3 个簇的所有可能组合）。对于任何落入同一桶的查询-索引图像对，如果它们的嵌入余弦相似度大于 0.975，则将其标记为近似重复。该阈值是在一个大规模数据集上训练的，该数据集由人工评分数据以及通过随机增广生成的合成数据构建而成。

## 附录 B：在 SimLex-999 上的评估（Appendix B Evaluation on SimLex-999）

如 Kiros 等人 ([2018](#ref-29)) 所示， **图文联合训练（image-text co-training）**  也有助于自然语言理解。例如，仅依靠语言很难学习反义词。为了测试 ALIGN 模型的这一能力，我们还在  **SimLex-999**  (Hill et al., [2015](#ref-22)) 上评估了来自 ALIGN 模型^5的词表示，该任务旨在比较 999 个词对的相似度。我们遵循 Kiros 等人 ([2018](#ref-29)) 的方法，在 9 个子任务上报告结果，每个子任务包含一个词对子集：*全部（all）、形容词（adjectives）、名词（nouns）、动词（verbs）、具体性四分位数（concreteness quartiles, 1-4）* 以及 *困难（hard）*。

> 脚注 5：由于 ALIGN 使用 **词片元（wordpiece tokens）** ，一个词可能被拆分为多个片段。我们将一个词的各个词片元输入 ALIGN 模型，并使用投影层之前的 [CLS] 词元表示作为词嵌入。

> 表 12：SimLex-999 结果（斯皮尔曼相关系数 $\rho$）。

结果列于表 [12](#table-12) 中，并与  **Picturebook**  (Kiros et al., [2018](#ref-29)) 和  **GloVe**  (Pennington et al., [2014](#ref-52)) 嵌入进行了比较。总体而言，学习到的 ALIGN 嵌入性能优于 Picturebook，但略逊于 GloVe 嵌入。有趣的是，ALIGN 词嵌入与 Picturebook 嵌入具有相似的趋势：与 GloVe 嵌入相比，在 *名词（nouns）* 和 *最具体（most concrete）* 类别上表现更好，但在 *形容词（adjs）* 和 *较不具体（less concrete）* 类别上表现更差。
ALIGN 词嵌入在 *困难（hard）* 类别上取得了最高性能，该类别的相似度难以与相关性区分开来。这一观察结果证实了 Kiros 等人 ([2018](#ref-29)) 的假设：与基于文本学习的分布方法相比，基于图像的词嵌入更不容易混淆相似性与相关性。