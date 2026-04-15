## 附录


<a id="appendix-a"></a>

## 附录 A 补充结果

### A.1 SWE-bench 上的基线

<a id="figure-5"></a>

![dgm_progress](images/dgm_progress.png)

> 图 5： **无自改进智能体（DGM without self-improving agents）** 的 DGM。保持修改和生成下一个编码智能体的元智能体不变，无自改进的 DGM 无法随时间持续改进。（左）在 SWE-bench 上运行无自改进 DGM 期间生成的编码智能体档案库。每个节点代表一个编码智能体，节点 0 对应基础智能体。节点颜色表示在 SWE-bench 上的性能（解决问题的百分比），边框颜色反映智能体被评估的任务数量。边显示哪些智能体自我修改以产生后代。（右）无自改进 DGM 在 SWE-bench 上的进度图。浅绿色线显示所有拥有基本代码库编辑功能的智能体的平均分数。绿色线跟踪档案库中任何智能体在每次迭代中达到的最佳分数。深色线显示最终发现的最佳智能体及其前驱节点的谱系。

<a id="figure-6"></a>

![transfer_model_task](images/transfer_model_task.png)

> 图 6： **无开放式探索（DGM without open-ended exploration）** 的 DGM。移除档案库后，无开放式探索的 DGM 总是使用最近的智能体进行自我修改，在 SWE-bench 上进展甚微。（左）在 SWE-bench 上运行无开放式探索 DGM 期间生成的编码智能体档案库。每个节点代表一个编码智能体，节点 0 对应基础智能体。节点颜色表示在 SWE-bench 上的性能（解决问题的百分比），边框颜色反映智能体被评估的任务数量。边显示哪些智能体自我修改以产生后代。（右）无开放式探索 DGM 在 SWE-bench 上的进度图。橙色线显示所有拥有基本代码库编辑功能的智能体的平均分数。浅橙色线跟踪档案库中任何智能体在每次迭代中达到的最佳分数。深色线显示最终发现的最佳智能体及其前驱节点的谱系。

### A.2 Polyglot 上跨模型的通用性

<a id="figure-7"></a>

![dgm_wo_selfimprove](images/dgm_wo_selfimprove.png)

> 图 7：Polyglot 上的模型间迁移

除了在 SWE-bench 上测试迁移模型（见 [第 4.4 节](#S4.SS4), [图 2](#S4.F2)），本节我们还展示了在 Polyglot 上的迁移结果。在 Polyglot 上（[图 7](#A1.F7)），DGM 使用 o3-mini 运行，我们将其中的 **基础模型（Foundation Model, FM）** 替换为 Claude 3.5 Sonnet (New) 或 Claude 3.7 Sonnet，并在完整基准测试上进行评估（[图 4](#S4.F4), 中部）。使用 Claude 3.5 Sonnet (New) 时，初始智能体达到 32.0%，DGM 发现的智能体达到 33.3%。使用 Claude 3.7 Sonnet 时，初始智能体达到 35.6%，DGM 发现的智能体达到 36.8%。这些结果表明，DGM 带来的改进能够跨 FM 泛化，而不是与其运行期间使用的特定 FM 紧密耦合（[图 4](#S4.F4)）。

### A.3 父节点选择的消融研究

为了进一步研究 DGM 中父节点选择机制的影响，我们引入了 **DGM Greedy** 。DGM Greedy 总是选择性能最佳的节点作为分支的父节点，而不是像本 DGM 实现中那样，给每个节点一个非零的分支概率（大致与其性能分数和子节点数量成正比）（[第 C.2 节](#A3.SS2)）。此消融实验复制了 Robeyns

## 附录 B 补充相关工作

**开放终结性（第二部分）** 。
早期的开放终结性方法探索了不同的机制来平衡 **可学习性（learnability）** 与 **趣味性（interestingness）** 。 **质量-多样性算法（Quality-diversity algorithms）** 旨在通过多样化的高性能行为来探索广阔的解决方案空间 [Pugh et al., 2016; Chatzilygeroudis et al., 2021; Mouret and Clune, 2015; Nguyen et al., 2015]。其他方法则强调 **目标导向探索（goal-directed exploration）** [Ecoffet et al., 2019; 2021; Schaul et al., 2015; Andrychowicz et al., 2017; Eysenbach et al., 2018]、 **内在动机（intrinsic motivation）** [Lehman and Stanley, 2011; Oudeyer et al., 2007; Li et al., 2014; Pathak et al., 2017] 或 **学习进展框架（learning progress frameworks）** [Kanitscheider et al., 2021; Gaven et al., 2025; Baranes and Oudeyer, 2013; Colas et al., 2019; 2022b; Jiang et al., 2021; Dennis et al., 2020; Schmidhuber, 2008; 2013; Kompella et al., 2017]。最近， **大规模基础模型（large-scale foundation models, FMs）** [Brown et al., 2020; Radford et al., 2019] 已成为人类趣味性概念的有力代理 [Zhang et al., 2024b; Faldor et al., 2025; Sancaktar et al., 2025] 以及提出新颖代码解决方案的有效 **变异算子（mutation operators）** [Romera-Paredes et al., 2024; Novikov et al., 2025; Lehman et al., 2023; Faldor et al., 2025; Hu et al., 2025]。FMs 可以指导 **自目标智能体（autotelic agents）** [Colas et al., 2022b; 2023; a]，建模人类对质量和多样性的偏好 [Bradley et al., 2024; Ding et al., 2024; Wang et al., 2023b; Klissarov et al., 2023; 2024; Samvelyan et al., 2024; Lim et al., 2024; Havrilla et al., 2024a]，设计 **奖励函数（reward functions）** [Wiering and Schmidhuber, 1997; Wang et al., 2023a; Ma et al., 2023; Faldor et al., 2025]，创建 **模拟环境（simulated environments）** [Sudhakaran et al., 2023; Nasir and Togelius, 2023; Aki et al., 2024; Nasir et al., 2024; Bruce et al., 2024; Parker-Holder et al., 2024; Schmidhuber, 2013]，驱动不断演化的 **多智能体动态（multi-agent dynamics）** [Dharna et al., 2024; Zhou et al., 2025]，搜索多样化的 **行走机器人形态（ambulating robot morphologies）** [Lehman et al., 2023]，并为基准测试或目标优化搜索广阔的解决方案空间 [Lange et al., 2024; Zhang et al., 2024b; Faldor et al., 2025; Hu et al., 2025; Lu et al., 2024b; Romera-Paredes et al., 2024; Fernando et al., 2024; Lu et al., 2024a; Khan et al., 2024; Lu et al., 2025; Liu et al., 2024; Novikov et al., 2025]。

**程序合成（Program Synthesis）** 。
**程序合成（Program synthesis）** [Alur et al., 2018; Polozov and Gulwani, 2015; Buchi and Landweber, 1990; Gulwani, 2011; Ellis et al., 2021] 旨在生成满足外部规范（如输入-输出示例或逻辑公式）的代码。混合方法将符号方法与神经或 FM 指导相结合：例如，Li 等人（2024）在 SyGuS 设置中使用 LLM 建议来引导符号搜索，改进了纯枚举方法。Barke 等人（2024）将 LLM 补全与学习的 **代理模型（surrogate model）** 相结合，以指导 **领域特定语言（Domain-Specific Languages, DSLs）** 中的合成。Shi 等人（2023）使用神经策略在搜索过程中构建高阶和 lambda 抽象，在列表操作任务上优于纯 LLM 和符号基线。 **DGM（达尔文式生成模型，Darwinian Generation Model）** 的不同之处在于，它不仅关注为外部任务生成程序，还关注智能体的 **自我修改（self-modification）** ，即重写自身的实现以提高其未来自我改进的能力。

**达尔文式进化的启发** 。
这项工作深受 **达尔文式进化（Darwinian evolution）** [Darwin, 2023] 机制的启发，特别是 **变异（variation/mutation）** 、 **选择（selection）** 和 **谱系（血统）保留（preservation of lineages/stepping stones）** ，并将它们引入到自我修改的编码智能体领域。在 DGM 中，维护着一个过去智能体版本的档案库，从中采样父代智能体；然后， **变异（mutations）** （即代码编辑）生成新的子代智能体，并在编码基准测试上进行经验评估；成功的智能体被添加到档案库中，从而实现对多个进化轨迹的并行探索（[第 3 节](#S3)）。这反映了 **生物进化（biological evolution）** [Edwards, 2000; Wright, 1932] 如何保持 **遗传多样性（genetic diversity）** [Mayr, 1982]，利用 **变异（variation）** [Kimura, 1979]，并使用 **自然选择（natural selection）** 来保留有益的突变 [Dobzhansky, 1970]。

<a id="appendix-c"></a>

## 附录 C 算法细节

### C.1 初始编码智能体

在本节中，我们介绍初始编码智能体（[第 4.1 节](#S4.SS1)）可用的工具详情及其任务提示。

给定的 Bash 工具信息：

给定的 Edit 工具信息：

任务提示：

### C.2 父代选择

在 DGM 的每次迭代中，我们从档案库中选择一个智能体子集作为父代，进行自我修改并产生新的子代智能体（[第 3 节](#S3)）。受 Ecoffet 等人（2019）启发，父代选择过程的细节概述如下。未来的工作可以探索替代的搜索和探索方法 [Coulom, 2006; Silver et al., 2016; Herr et al., 2025]。

在 DGM 运行的第 $t$ 次迭代，令

$$
\mathcal{A}^{t}=\{a_{1}^{t},a_{2}^{t},\dots,a_{N}^{t}\}
$$

为候选智能体的档案库。我们首先定义合格集合

$$
\mathcal{E}^{t}=\bigl\{\,a_{i}^{t}\in\mathcal{A}^{t}:\alpha_{i}<1\bigr\},
$$

即仅包含那些性能得分尚未达到完美的智能体。然后，我们从 $\mathcal{E}^{t}$ 中按比例（可重复）采样 $k$ 个父代，比例基于性能和具有代码库编辑功能的子代数量的综合度量。具体来说，对于每个智能体 $a_{i}^{t}\in\mathcal{E}^{t}$：

$$
\begin{aligned}
\text{(性能)} \quad &\displaystyle\alpha_{i}\;=\;\mathrm{performance}(a_{i}^{t}), &(1)\\
\text{(子代数量)} \quad &\displaystyle n_{i}\;=\;\mathrm{functioning\_children\_count}(a_{i}^{t}), &(2)\\
\text{(Sigmoid 缩放性能)} \quad &\displaystyle s_{i}\;=\;\frac{1}{1+\exp\!\bigl(-\lambda\,(\alpha_{i}-\alpha_{0})\bigr)}, &(3)\\
\text{(新颖性奖励)} \quad &\displaystyle h_{i}\;=\;\frac{1}{1+n_{i}}, &(4)\\
\text{(未归一化权重)} \quad &\displaystyle w_{i}\;=\;s_{i}\,h_{i}, &(5)\\
\text{(选择概率)} \quad &\displaystyle p_{i}\;=\;\frac{w_{i}}{\sum_{j=1}^{N}w_{j}}, &(6)\\
\text{(采样)} \quad &\displaystyle\{\,\text{parents}\,\}\;\sim\;\mathrm{Categorical}\bigl(\{p_{i}\}_{a_{i}^{t}\in\mathcal{E}^{t}}\bigr)\quad\text{(抽取 k 个样本)}.
\end{aligned}
$$

### C.3 自我改进提示（Self-Improve Prompts）

在每次迭代中，会选择一个智能体进行自我修改。对于这一步，DGM 的当前实现利用一个独立的 FM（基础模型）来诊断改进机会，然后将其传递给编码智能体进行实现。为了识别下一个潜在的改进点，所选智能体在编码基准测试上的评估日志会被输入到 FM 中。虽然当前实现依赖于一个额外的诊断步骤，但 DGM 在根本上并无限制，完全可以自主分析自身性能、生成改进假设并实施后续修改。为此，我们选择使用 OpenAI 的 o1 模型（Team 等人，2024），因为我们发现它在实验期间可用的 FM 中具有更好的推理能力。用于此诊断的提示如下。从诊断输出中，会提取 `implementation_suggestion` 和 `problem_description`，并将其作为自我修改任务传递给编码智能体。

用于 SWE-bench 实验的自我改进诊断提示：

> （提示内容）

虽然诊断提示中包含了私有测试补丁，但人们可能会认为智能体在任务中取得高分的简单方法是硬编码解决方案。然而，通过对日志和智能体修改的手动检查，我们**并未观察到任何表明记忆或对特定私有测试用例过拟合的问题逻辑或行为**。此外，在一个任务或 LLM（大语言模型）上发现的改进通常能很好地推广到其他任务或模型（[图˜4](#S4.F4)），这表明这些改进并非针对特定任务，而是反映了能提升整体性能的通用性改变。

用于 Polyglot 实验的附加提示：

> （提示内容）

### C.4 伪代码（Pseudocode）

这是 DGM 算法的伪代码，在[第˜3节](#S3)中描述：

```pseudocode
（伪代码内容）
```

这是没有自我改进智能体的基线 DGM 的伪代码，在[第˜4.3节](#S4.SS3)中描述：

```pseudocode
（伪代码内容）
```

这是没有开放式探索的基线 DGM 的伪代码，在[第˜4.3节](#S4.SS3)中描述：

```pseudocode
（伪代码内容）
```


##  **附录 E 基准测试详情（Appendix E Benchmark Details）**

###  **E.1 成本估算（Cost Estimate）**

如[第 4 节](#S4)所述，在 SWE-bench 上完成一次  **DGM（Directed Graph Model）**  运行的估算成本约为 22,000 美元。相比之下，在 SWE-bench 上完成任一基线（无自改进的 DGM 或无开放式探索的 DGM）单次运行的估算成本约为 10,000 美元。尽管 DGM 的成本远高于基线，但一个能够持续改进的方法，即使成本更高，也比一个无法改进或停滞在可能永远无法达到 DGM 性能水平的方法更可取。更细粒度的成本分解如下：

| LLM                     | 基准测试（Benchmark） | 任务数量（Number of Tasks） | 估算成本（Cost Estimate (USD)） |
| :---------------------- | :-------------------- | :-------------------------- | :------------------------------ |
| Claude 3.5 Sonnet (New) | SWE-bench             | 60                          | $350                            |
| o3-mini                 | Polyglot              | 60                          | $5                              |

我们承认当前在 SWE-bench 上的实验需要相当大的计算资源。因此，我们也在另一个基准测试  **Polyglot**  上进行了实验，其成本显著更低。这表明费用因任务复杂性而异，SWE-bench 属于更复杂、资源密集型的编码基准测试之一。此外，一些有影响力的方法（例如， **大型语言模型（Large Language Model, LLM）**  在其诞生之初的训练）最初也以巨大的计算需求为特征。与这些开创性工作类似，我们希望为未来研究提高我们方法的效率和可扩展性打开大门。此外，SWE-bench 排行榜上的许多领先编码智能体（Coding Agent）背后都有工业公司支持，这些公司雇佣了专业的全职研究人员和工程师，这产生了大量的人力成本。相比之下，我们的方法通过完全自主的自改进，无需人工干预，实现了  **SoTA（State-of-the-Art）**  级别的性能，在考虑专业 AI 开发人才成本与 API 使用成本的对比时，可能提供更高的效率。最后，随着  **基础模型（Foundation Models, FMs）**  的持续改进和计算成本的持续下降，像 DGM 这样的方法将变得越来越高效和易于获取。

另外，由 DGM 发现的性能更高的智能体确实比初始智能体产生更高的推理成本，但成本与性能并非严格相关，一些昂贵的智能体表现可能不如更便宜的智能体。

###  **E.2 SWE-bench 任务（SWE-bench Tasks）**

用于验证编码智能体基本功能的初始 10 个任务：

*   django_django-10973
*   django_django-11066
*   django_django-12754
*   django_django-15930
*   django_django-13279
*   django_django-16661
*   django_django-13346
*   django_django-10880
*   django_django-10999
*   django_django-11087

用于评估编码智能体总体有效性的额外 50 个任务：

*   django_django-9296
*   django_django-11790
*   django_django-11815
*   django_django-11848
*   django_django-11880
*   django_django-11885
*   django_django-11951
*   django_django-11964
*   django_django-11999
*   django_django-12039
*   django_django-12050
*   django_django-12143
*   django_django-12155
*   django_django-12193
*   django_django-12209
*   django_django-12262
*   django_django-12273
*   django_django-12276
*   django_django-12304
*   django_django-12308
*   django_django-12325
*   django_django-12406
*   django_django-12708
*   django_django-12713
*   django_django-12774
*   sphinx-doc_sphinx-7454
*   sphinx-doc_sphinx-7590
*   sphinx-doc_sphinx-7748
*   sphinx-doc_sphinx-7757
*   sphinx-doc_sphinx-7985
*   sphinx-doc_sphinx-8035
*   sphinx-doc_sphinx-8056
*   sphinx-doc_sphinx-8265
*   sphinx-doc_sphinx-8269
*   sphinx-doc_sphinx-8475
*   sphinx-doc_sphinx-8548
*   sphinx-doc_sphinx-8551
*   sphinx-doc_sphinx-8638
*   sphinx-doc_sphinx-8721
*   sphinx-doc_sphinx-9229
*   sphinx-doc_sphinx-9230
*   sphinx-doc_sphinx-9281
*   sphinx-doc_sphinx-9320
*   sphinx-doc_sphinx-9367
*   sphinx-doc_sphinx-9461
*   sphinx-doc_sphinx-9698
*   sphinx-doc_sphinx-10449
*   sphinx-doc_sphinx-10466
*   sphinx-doc_sphinx-10673
*   sphinx-doc_sphinx-11510

 **用于更准确评估编码智能体性能的额外 140 项任务：**

*   astropy_astropy-12907
*   astropy_astropy-13033
*   astropy_astropy-13236
*   astropy_astropy-13398
*   astropy_astropy-13453
*   astropy_astropy-13579
*   astropy_astropy-13977
*   astropy_astropy-14096
*   astropy_astropy-14182
*   astropy_astropy-14309
*   astropy_astropy-14365
*   astropy_astropy-14369
*   astropy_astropy-14508
*   astropy_astropy-14539
*   astropy_astropy-14598
*   astropy_astropy-14995
*   astropy_astropy-7166
*   astropy_astropy-7336
*   astropy_astropy-7606
*   astropy_astropy-7671
*   astropy_astropy-8707
*   astropy_astropy-8872
*   django_django-10097
*   django_django-10554
*   django_django-10914
*   django_django-11095
*   django_django-11099
*   django_django-11119
*   django_django-11133
*   django_django-11138
*   django_django-11141
*   django_django-11149
*   django_django-11163
*   django_django-11179
*   django_django-11206
*   django_django-11211
*   django_django-11239
*   django_django-11265
*   django_django-11276
*   django_django-11292
*   django_django-11299
*   django_django-11333
*   django_django-11400
*   django_django-11433
*   django_django-11451
*   django_django-11477
*   django_django-11490
*   django_django-11532
*   django_django-11551
*   django_django-11555
*   django_django-11603
*   django_django-11728
*   django_django-11734
*   django_django-11740
*   django_django-11749
*   django_django-11820
*   django_django-12125
*   django_django-12419
*   django_django-12663
*   django_django-12741
*   django_django-12858
*   django_django-12965
*   django_django-13012
*   django_django-13023
*   django_django-13028
*   django_django-13033
*   django_django-13089
*   django_django-13109
*   django_django-13112
*   django_django-13121
*   django_django-13128
*   django_django-13158
*   django_django-13195
*   django_django-13212
*   django_django-13297
*   django_django-13315
*   django_django-13343
*   django_django-13344
*   django_django-13363
*   django_django-13401
*   django_django-13406
*   django_django-13410
*   django_django-13417
*   django_django-13449
*   django_django-13512
*   django_django-13513
*   django_django-13516
*   django_django-13551
*   django_django-13568
*   django_django-13569
*   django_django-13590
*   django_django-13658
*   django_django-13670
*   django_django-13741
*   django_django-13786
*   django_django-13794
*   django_django-13807
*   django_django-13809
*   django_django-13810
*   django_django-13820
*   django_django-13821
*   django_django-13837
*   django_django-13925
*   django_django-13933
*   django_django-13964
*   django_django-14007
*   django_django-14011
*   django_django-14017
*   django_django-14034
*   django_django-14053
*   django_django-14089
*   django_django-14122
*   django_django-14140
*   django_django-14155
*   django_django-14170
*   django_django-14238
*   django_django-14311
*   django_django-14315
*   django_django-14349
*   django_django-14351
*   django_django-14373
*   django_django-14376
*   django_django-14404
*   django_django-14434
*   django_django-14493
*   django_django-14500
*   django_django-14534
*   django_django-14539
*   django_django-14559
*   django_django-14580
*   django_django-14608
*   django_django-14631
*   django_django-14672
*   django_django-14725
*   django_django-14752
*   django_django-14765
*   django_django-14771
*   django_django-14787
*   django_django-14792
*   django_django-14855

### E.3 Polyglot Tasks（多语言任务）

 **用于验证编码智能体基本功能的初始 10 项任务：**

1.  go_dominoes
2.  cpp_all-your-base
3.  python_dominoes
4.  java_sgf-parsing
5.  javascript_robot-name
6.  rust_variable-length-quantity
7.  python_beer-song
8.  go_book-store
9.  javascript_bottle-song
10. rust_bowling

 **用于评估编码智能体总体有效性的额外 50 项任务：**

1.  javascript_queen-attack
2.  rust_wordy
3.  python_dot-dsl
4.  java_satellite
5.  cpp_diamond
6.  rust_accumulate
7.  go_error-handling
8.  cpp_queen-attack
9.  rust_poker
10. python_sgf-parsing
11. rust_react
12. java_ledger
13. go_connect
14. rust_macros
15. javascript_triangle
16. java_zipper
17. java_bowling
18. python_tree-building
19. javascript_say
20. java_wordy
21. python_food-chain
22. javascript_wordy
23. python_poker
24. javascript_grade-school
25. cpp_gigasecond
26. java_forth
27. python_dominoes
28. go_word-search
29. javascript_simple-linked-list
30. go_counter
31. java_react
32. javascript_ocr-numbers
33

## 附录 F 最佳发现智能体

### F.1 DGM 在 SWE-bench 上的表现

DGM 在 SWE-bench 上发现的最佳智能体所贡献的差异补丁：

### F.2 DGM 在 Polyglot 上的表现

DGM 在 Polyglot 上发现的最佳智能体所贡献的差异补丁：

<a id="appendix-g"></a>

## 附录 G 相似目标功能，不同实现

对于相同的目标功能，提供给编码智能体的建议以及由此产生的实现可能存在显著差异，从而导致编码能力和基准测试性能的巨大变化。例如，考虑 DGM 在 SWE-bench 上运行时的节点 6 和节点 24（[图˜3](#S4.F3)）。两者都旨在增强现有的编辑器工具以支持更细粒度的文件编辑。然而，节点 6 的实现保留了原始的编辑命令并添加了参数 `edit_type` 和 `edit_actions`，而节点 24 的实现则用一个新的 `str_replace` 命令替换了原始的编辑命令。尽管目标功能相似，但功能建议以及实现细节差异很大，它们的性能也是如此。节点 6 仅达到 23.3%，而节点 24 达到了 40.5%。

获取节点 6 的功能建议：

获取节点 24 的功能建议：

获取节点 6 的建议功能实现：

获取节点 24 的建议功能实现：

<a id="appendix-h"></a>

## 附录 H 案例研究：解决幻觉问题

如[第˜5节](#S5)中作为未来工作的潜在方向所讨论的，DGM 可用于优化编码之外的目标。在本节中，我们展示 DGM 可以解决 **基础模型（Foundation Models, FMs）** 工具使用的 **幻觉（Hallucination）** 问题。通过评估日志的反馈，DGM 改进了幻觉检测机制，并最终发现了解决这些幻觉的方法。我们将更详细地讨论这些幻觉何时发生、DGM 的设置以及由此产生的解决方案。最后，我们展示一个 **目标黑客攻击（Objective hacking）** 的例子，即智能体优化可量化的指标，而不是真正完成任务的精神并解决根本问题。

<a id="figure-8"></a>

![dgm_wo_openended](images/dgm_wo_openended.png)

> 图 8：DGM 解决 FMs 中工具使用的幻觉问题。在 SWE-bench 上运行 DGM 以解决 FMs 幻觉时生成的编码智能体存档。每个节点代表一个智能体，节点 0 对应基础智能体。节点颜色表示已解决的幻觉分数，边框颜色反映智能体是否具有基本的代码库编辑功能。边显示哪些智能体通过自我修改产生了后代。

 **幻觉何时发生。**
当输入到 Claude 3.5 Sonnet (New)（Claude）的上下文非常长，包含所有先前的工具使用输入和输出时，我们发现 Claude 倾向于产生工具使用的幻觉。它并非实际进行工具调用，而是将工具输入和输出作为纯文本输出。这是 Claude 产生工具使用幻觉的一个例子，它声称使用了 Bash 工具来运行测试，并且工具输出表明所有测试都已通过，但 Claude 并未调用任何工具，工具结果完全是纯文本形式的幻觉：

 **DGM 设置。**
我们在 SWE-bench 上运行此实验。除了我们评估每个编码智能体的 SWE-bench 任务数量以及 DGM 的最终评分函数外，所有其他设置均与主要实验相同（[第˜3节](#S3)）。初始编码智能体以及在 DGM 运行期间生成的所有智能体仅在 3 个 SWE-bench 任务上进行评估。我们只使用 3 个任务是因为，如果幻觉在一个任务中发生，它很可能在其他任务中也会发生。因此，3 个任务足以向 DGM 提供关于工具使用幻觉是否仍然存在的可靠信号。

 **解决幻觉分数。**
我们根据每个生成的智能体防止幻觉的有效性对其进行评分，分数越高表示幻觉越少，性能越好。对于智能体被评估的每个 SWE-bench 任务，分数由两部分组成：0 - 1 分基于日志中是否发生幻觉，以及 1 - 2 分（仅在日志中没有发生幻觉后添加和计入）如果所有 Claude 调用中都使用了工具。最终的解决幻觉分数计算为所有评估任务的平均值。为了检测日志中的幻觉，我们在实际工具使用时插入特殊标记，然后检查这些标记是否出现在模型的纯文本输出中，这表明模型模仿了工具使用而并未实际调用任何工具。重要的是，在自我修改过程中，幻觉检查函数对编码智能体是隐藏的。我们发现，当这些函数未被隐藏时，目标黑客攻击（即，为可测量的目标进行优化，而非真正解决根本问题）发生得更频繁。

 **结果。**
我们运行 DGM 进行了 150 次迭代。在此次运行中，DGM 发现了两个有前景的解决方案：节点 96，获得了第二高的分数 1.67；以及节点 114，获得了满分 2.0（[图˜8](#A8.F8)）。
节点 96 在一定程度上解决了幻觉问题，在没有目标黑客攻击的情况下获得了 1.67 分。在通往节点 96 的谱系中，DGM 首先改进了工具事务日志记录机制，以更好地检测幻觉何时发生。然后，它探索了各种解决策略，例如，当确定工具未被实际调用时，从输出文本中移除幻觉的工具使用。
节点 114 在其谱系中仅经过 2 次修改后就迅速获得了满分。然而，我们观察到了目标黑客攻击：根据我们预定义的评估函数，它得分很高，但它实际上并未解决工具使用幻觉的根本问题。在通往节点 114 的修改中（见下文），智能体移除了指示工具使用的特殊标记的日志记录（尽管有指令要求不更改特殊标记），从而有效地绕过了我们的幻觉检测函数。

总之，DGM 可以应用于编码领域之外，并且我们强调了一个目标黑客攻击的案例。类似于 **强化学习（Reinforcement Learning）** 中的 **奖励黑客攻击（Reward hacking）** （Skalse 等人，2022），当系统为预定义的、可量化的目标进行优化，而不是完成任务的精神或解决预期问题时，就会发生目标黑客攻击。这一观察结果支持了先前工作中的论点（Zhang 等人，2024b；Faldor 等人，2025），这些论点表明优化定量度量通常会导致不良或病态的结果，并且与 **古德哈特定律（Goodhart's law）** （Strathern，1997）相符——“当一项度量成为目标时，它就不再是一个好的度量。”

<a id="appendix-i"></a>

## 附录 I 额外的安全讨论

任何提高 AI 系统自主能力的进步都会带来其自身的一系列安全考量（Bengio 等人，2024），尤其是对于那些以 **开放式（Open-ended）** 方式改进的系统（Ecoffet 等人，2020；Clune，2019）。[第˜5节](#S5)讨论了这些担忧，并概述了缓解这些担忧的具体、可操作的步骤。我们呼吁对 AI 安全进行更多的研究和讨论，包括社会所有利益相关者就“究竟什么才算安全的 AI”这一复杂问题进行深入思考和讨论。我们确信我们所做的工作从未不安全（[第˜5节](#S5)），但其扩大规模的版本可能会不安全。与所有变革性技术一样，此类 AI 系统的最终影响仍然极不确定，既可以有力地论证它将带来巨大的好处，也可以论证它将带来巨大的危害。这些不确定性凸显了需要就如何驾驭这些发展进行持续、包容和多学科的讨论（不仅来自当前的专家，也来自更广泛、更多样化的群体）。

<a id="appendix-j"></a>

## 附录 J 额外的未来工作方向

尽管本文已经展示了 **达尔文-哥德尔机（Darwin Gödel Machine, DGM）** 通过开放式探索和经验验证迭代改进编码智能体的潜力，但一些扩展可以解决当前的局限性，并将 AI 推至其已在激励文化和推进科学方面日益增长的角色之外。以下方向概述了有前景的进一步研究途径。

 **自主改进开放式探索过程。**
在此版本的 DGM 中，[第˜3节](#S3)中描述的开放式探索过程是固定的，这可能会因此阻碍系统的自我加速潜力。由于计算预算有限，我们做出了这一设计选择。如果我们想要演化算法的这一部分，可能需要指数级更多的计算来识别能产生[第˜4.4节](#S4.SS4)中所示相同改进的过程。然而，由于开放式探索循环本身是用代码实现的，原则上它可以被编码智能体编辑和改进。开放式探索有许多可能的实现方式，例如，使用平衡探索与利用的替代搜索机制（Herr 等人，2025）、仅在存档中保留最有趣的智能体（Faldor 等人，2025），或利用生成的智能体群体作为 **集成（Ensemble）** （Samvelyan 等人，2024）。一个有前景的未来工作方向是允许智能体修改开放式探索过程，从而不仅自主改进其自身能力，而且改进分配有限计算资源以驱动自我改进和自我加速的元过程。

 **人类在自主 AI 系统中的作用。**
在 DGM 的当前表述中，提议的自我修改是自主评估的，无需任何人工干预。然而，随着自主系统在复杂性和影响力上的增长，人类应如何保持参与的问题变得越来越紧迫。人类监督是否应被构建为一个优化
$$
