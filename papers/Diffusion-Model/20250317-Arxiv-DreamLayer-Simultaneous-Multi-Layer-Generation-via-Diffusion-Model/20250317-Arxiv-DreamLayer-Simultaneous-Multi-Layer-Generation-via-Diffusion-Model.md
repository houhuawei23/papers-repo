# Title: DreamLayer: Simultaneous Multi-Layer Generation via Diffusion Model

- ArXiv: 2503.12838
- Authors: Junjia Huang, Pengxiang Yan, Jinhang Cai, Jiyang Liu, Zhao Wang, Yitong Wang, Xinglong Wu, Guanbin Li
- Sections: 25
- Estimated tokens: 25.2k

## Contents

- 1 Introduction
- 2 Related Work
- 3 Methodology
  - 3.1 Context-Aware Cross-Attention
  - 3.2 Layer-Shared Self-Attention
  - 3.3 Information Retained Harmonization
  - 3.4 Dataset Preparation
- 4 Experiments
  - 4.1 Implementation Details
  - 4.2 Comparisons of Multi-Layer Image Generation
  - 4.3 Ablation Study
  - 4.4 Further Application
- 5 Conclusion

## Abstract

###### Abstract

Text-driven image generation using diffusion models has recently gained significant attention. To enable more flexible image manipulation and editing, recent research has expanded from single image generation to transparent layer generation and multi-layer compositions. However, existing approaches often fail to provide a thorough exploration of multi-layer structures, leading to inconsistent inter-layer interactions, such as occlusion relationships, spatial layout, and shadowing. In this paper, we introduce DreamLayer, a novel framework that enables coherent text-driven generation of multiple image layers, by explicitly modeling the relationship between transparent foreground and background layers. DreamLayer incorporates three key components, i.e., Context-Aware Cross-Attention (CACA) for global-local information exchange, Layer-Shared Self-Attention (LSSA) for establishing robust inter-layer connections, and Information Retained Harmonization (IRH) for refining fusion details at the latent level. By leveraging a coherent full-image context, DreamLayer builds inter-layer connections through attention mechanisms and applies a harmonization step to achieve seamless layer fusion. To facilitate research in multi-layer generation, we construct a high-quality, diverse multi-layer dataset including $400k$ samples. Extensive experiments and user studies demonstrate that DreamLayer generates more coherent and well-aligned layers, with broad applicability, including latent-space image editing and image-to-layer decomposition.

<a id="section-1"></a>

## 1 Introduction

In recent years, text-to-image generation based on diffusion models [23, 26, 1, 21, 4, 3, 39] has demonstrated impressive capabilities to create high-quality, detail-rich images from text prompts. However, most methods focus on generating a single, complete image, significantly limiting their potential in applications like content editing and graphic design, which rely heavily on layered compositions. Layered structures are particularly advantageous for images containing multiple objects, as they allow for more flexible and versatile editing and creative modifications. This paper investigates the application of diffusion models to generate coherent, multi-layered images through a simple text-driven process.

<a id="figure-1"></a>

![Intro](images/Intro.png)

> Figure 1: DreamLayer can handle multiple tasks: (a) Text-to-layer: Given a text input, we use GPT-4 to decompose foreground and background elements, feeding them into DreamLayer to generate a multi-layered image. (b) Image-to-layer: By using inversion to initialize starting latent, DreamLayer can decompose an image based on text prompts. (c) Latent-space editing: During denoising, DreamLayer can respond to editing instructions, producing more harmonious and consistent edited images.

<a id="figure-2"></a>

![Visual_2](images/Visual_2.png)

> Figure 2: Multi-layer Dataset: Each image consists of a multi-layer structure, including a background and multiple foreground objects, with each foreground object represented as a transparent layer.

Recent methods have started to explore the simultaneous generation of layered image structures to better support AI-driven image editing workflows. Most existing methods [44, 41] are limited to the generation of two-layer structures, _i.e_., foreground and background layers. While certain approaches [41, 10] attempt to model the multi-layer generation task, they lack consideration for the relationship between different foreground layers and the background. For instance, LayerDiffusion [41] disregards the spatial relationships between layers when adding new ones, leading to unintended overlaps between layers. LayerDiff [10] attempts to generate multi-layer composite images simultaneously, but it can only generate isolated, non-overlapping layers. These methods typically rely on simple stacking for layer composition, neglecting essential effects like shadows and occlusion, which are important for cohesive multi-layer generation and editing. Furthermore, a significant challenge in multi-layer generation is the lack of large-scale, high-quality open-source datasets. Current approaches and their datasets often rely on randomly stacked segmentation data [36], suffer from limited data volume [31], or lack strictly defined multi-layered images [10].

To address these challenges, we propose a comprehensive multi-layer data generation pipeline that decomposes images generated by advanced text-to-image models, createing a dataset of $400k$ multi-layer samples, as shown in [Fig. 2](#figure-2). In existing text-to-image generation models, when given a text prompt containing a background and multiple foregrounds, the models often demonstrate the ability to automatically arrange objects in a reasonable layout and generate harmonious compositions. Building upon this, we introduce DreamLayer, a framework that utilizes global layer information to guide inter-layer attention and integrates a harmonization mechanism. To address layout issues in foreground layers, we begin by generating a cohesive global image from the full-text prompt. Subsequently, we employ Context-Aware Cross-Attention to extract contextual information from the global image, guiding the generation of foreground layers. To establish connections between layers, we adapt Layer-Shared Self-Attention, which further facilitates the sharing of global information across independent layers. Finally, we apply Information Retained Harmonization to fuse the composite image in latent space, ensuring a harmonious final result and improving flexibility and consistency for subsequent editing tasks. DreamLayer enables the generation of multi-layer images with cohesive layouts and seamless integration across layers. It also supports a variety of tasks. As illustrated in [Fig. 1](#figure-1), (a) DreamLayer can perform text-to-layer to generate multi-layer images by adaptively decomposing user text prompts using GPT-4; (b) DreamLayer supports image-to-layer decomposition by initializing the denoising latent via inversion and directing it based on text prompts in a training-free manner; (c) DreamLayer supports user-driven edits during the denoising process, ensuring stable and harmonious edited images. In summary, our key contributions are threefold:

- We introduce DreamLayer, a simultaneous multi-layer generation framework that enhances harmony and consistency across layers via inter-layer interaction.
- We propose a layer-level harmonization approach to achieve smoother inter-layer blending, making it more adaptable for subsequent editing tasks.
- The release of a large-scale, high-quality multi-layer dataset, containing $400k$ meticulously curated multi-layer images, covering multiple objects and scenes.

<a id="figure-3"></a>

![Framework_1](images/Framework_1.png)

> Figure 3: The DreamLayer Framework for Multi-Layer Image Generation: During the generation process, background and foreground prompts are combined via layer assign embeddings to form a global prompt $C_{t}^{k+1}$. In the attention phase, CACA extracts a context map from the global layer. Subsequently, the contextual information is fused across layers through LSSA, based on the global context map. Finally, IRH fuses the images using the latent image during the denoising process, achieving a harmonious result.

<a id="figure-4"></a>

![Framework_2](images/Framework_2.png)

> Figure 4: Overview of the Attention Mechanism in DreamLayer: (a) Context-Aware Cross-Attention for extracting the global context map and guiding the foreground layer layout; (b) Layer-Shared Self-Attention for establishing inter-layer connections and ensuring consistency.

<a id="section-2"></a>

## 2 Related Work

Diffusion based Image Generation. Diffusion models [9, 30] have shown leading performance in generative tasks, including image generation [47, 23, 40], editing [2, 11], inpainting [16], and video generation [46]. These models have evolved from early pixel-space denoising [27] to latent-space denoising [23], with architectures progressing from U-Net [24] to advanced designs like DiT [20, 4]. For multi-layer image generation, Text2Layer [44] uses a latent diffusion model to jointly denoise and reconstruct the RGB and alpha channels for two-layer images. LayerDiffusion [41] encodes the alpha channel in the latent manifold and shares attention between foreground and background layers to generate multi-layered images. LayerDiff [10] proposes to generate multi-layer composites with layer-collaborative attention. However, these approaches often neglect integrated effects such as shadows and other inter-layer interactions between multiple foreground and background layers.

Controllable Diffusion Model and Image Editing. To enhance controllability in image generation, a range of methods have been developed. Textual Inversion [5] and DreamBooth [25] enable personalized content generation from a small set of example images. ControlNet [42] and T2I-Adapter [38, 19] introduce conditional signals, using reference images as visual prompts for direct guidance. Other methods [34, 45] utilize bounding boxes to control image layout, while P2P [7] and PnP [32] condition attention layers to manage content and style. To enable more customized image content, some methods [6, 18, 11] leverage inversion techniques, converting the input image into a noise latent representation, which is then edited and generated in a controlled manner based on text prompts. DesignEdit [12] further segment the latent representations into multiple layers, allowing more flexible spatial editing. However, existing methods typically restrict layers to non-overlapping structures. In this work, we utilize a harmonious global layer to guide the generation of layers and employ independent layers to achieve seamless fusion in the latent space, enabling more harmonious editing.

<a id="section-3"></a>

## 3 Methodology

Definitions. Intuitively, a $k$-layer image consists of a background layer $I^{1}$, $k-1$ foreground layers $\{I^{i}\}_{i=2}^{k}$ and a global layer $I^{k+1}$. Each layer comprises a three-channel color image $I_{c}\in\mathbb{R}^{H\times W\times 3}$ and an alpha channel $I_{\alpha}\in\mathbb{R}^{H\times W\times 1}$, where the alpha channel indicates the visibility of the pixels within the color image. Formally, the global layer image $I^{k+1}$ can be expressed as

$$
I^{k+1}=\sum_{i=1}^{k}(I_{\alpha}^{i}\cdot I_{c}^{i}\cdot\prod^{k}_{f=i+1}(1-I _{\alpha}^{f})).(1)
$$

Each layer is associated with a corresponding textual description as a text prompt $\{C_{p}^{i}\}_{i=1}^{k+1}$.

Generation of Alpha Channel. For each layer image, we fill the image with a solid gray background based on its alpha channel to obtain an RGB layer. This RGB layer is then encoded into a latent image $z\in\mathbb{R}^{hw\times D}$ and perturbed with noise for $t$ timesteps to produce a noisy latent image $z_{t}$. With the timestep $t$ and a text prompt $C$ as conditions, the diffusion model trains a network $\epsilon_{\theta}$ to predict the noise added to the noisy latent image $z_{t}$ with

$$
\mathcal{L}_{noise}=\mathbb{E}_{z_{t},t,C,\epsilon\sim\mathcal{N}(0,1)}\left[| |\epsilon-\epsilon_{\theta}(z_{t},t,C)||_{2}^{2}\right],(2)
$$

where $\mathcal{L}_{noise}$ represents the learning objective of the diffusion model. After $T$ denoising steps, the latent image $z_{0}$ is decoded by a layer decoder to generate the final transparent layer image with alpha channel. The layer decoder can be diverse, some methods [44, 10] train a 4-channel VAE decoder, while others [41] utilize a VAE decoder combined with a gray-background segmentation model. In this work, we adopt the same layer decoder as used in LayerDiffusion [41]. Notably, we primarily focuses on the layout coherence and overall harmony in multi-layer generation, rather than the accuracy of alpha channel generation.

Overview. As shown in [Fig. 3](#figure-3), for multi-layer generation, we simultaneously encode the prompt of background layer and each foreground layer with the text encoder to obtain text embedding $\{C_{t}^{i}\in\mathbb{R}^{S\times D}\}_{i=1}^{k}$, where $S$ denotes the sequence length after tokenization. A learnable layer assign embedding is then added to each text embedding. We extract the portion between the [SOS] and [EOS] tokens from each text embedding and concatenate them to form a global embedding $C_{t}^{k+1}\in\mathbb{R}^{S\times D}$, which captures the essential information of all layers and guides the generation of the global layer. All layers are processed in a batch-wise manner during the attention computation of the diffusion model. To fully utilize the information from the global layer, we design three key components: Context-Aware Cross-Attention (CACA), Layer-Shared Self-Attention (LSSA), and Information Retained Harmonization (IRH). These components leverage guidance from the global layer to ensure consistency across background layer and foreground layers, facilitating the generation of harmonious multi-layer images.

<a id="figure-5"></a>

![Data_Pipeline](images/Data_Pipeline.png)

> Figure 5: The pipeline of multi-layer data preparation. We utilize GPT-4 to process a randomly selected base prompt, structuring it into a background prompt and multiple foreground prompts. After generating the image using a diffusion model, we apply an open-set detection model GroundingDINO to identify the positions of the foreground objects and use the DepthAnything model to obtain a depth map. Based on the depth order, we sequentially extract the foreground layers and fill in the missing areas with an inpainting model.

<a id="section-3-1"></a>

### 3.1 Context-Aware Cross-Attention

The key to multi-layer generation is maintaining consistency in layout and proportions across all layers. During the generation process, we align the layout positions of other layers with the global layer. Utilizing the text embeddings of each layer, we extract relevant information from the cross-attention of the global layer. Formally, as shown in [Fig. 4](#figure-4) (a), the global noisy latent image $z_{t}^{j,k+1}$ in the $j^{th}$ cross-attention mechanism is projected to a query matrix $Q_{c}^{j}=\ell_{Q}(z_{t}^{j,k+1})$ and the attention map $\mathcal{M}_{j}\in\mathbb{R}^{hw\times S}$ is then calculated with global embedding as

$$
\mathcal{M}_{j}=Softmax(\frac{Q_{c}^{j}\ell_{K}(C_{t}^{k+1})^{T}}{\sqrt{d}}),(3)
$$

where $\ell_{Q},\ell_{K}$ are linear projections and $d$ is the latent dimension. The attention map preserves the spatial layout and geometry of the different foreground objects [7, 43]. Therefore, we extract the cross-attention maps corresponding to each foreground object from $J$ layers in the diffusion model. These maps are combined to create $f$ initial spatial-aware global attention maps $\mathcal{M}_{G}^{f}$:

$$
\mathcal{M}_{G}^{f}=Norm(\sum_{s=1}^{S_{f}}\sum_{j=1}^{J}(\mathcal{M}_{j}^{s}) ),f=2,\cdots,k(4)
$$

where $Norm(\cdot)$ denotes the Min-Max Normalization and $S_{f}$ denotes the token length of each foreground’s text embedding within the global embedding. To enhance foreground layer context in the extracted attention map, we feed the initial map and the global noisy latent image of $J$ cross-attention mechanism into $N$ context-aware layers $CAL(\cdot,\cdot)$ to generate global context map $\mathcal{M}_{G}^{f,n+1}$ as:

$$
\mathcal{M}_{G}^{f,n+1}=CAL(\mathcal{M}_{G}^{f,n},\sum_{j=1}^{J}z_{t}^{j,k+1}).(5)
$$

Each context-aware layer consists of a multi-head attention [33] followed by a feed-forward network (FFN). The context map is supervised by the alpha channel of the foreground image with

$$
\mathcal{L}_{c}=\sum^{f}||\mathcal{R}(I^{f}_{\alpha})-\mathcal{M}_{G}^{f,N}||_ {2},(6)
$$

where $\mathcal{R}(\cdot)$ denotes the resize operation with interpolation.

After extracting the harmonious layout and geometric information of the foreground layer from the global layer $I^{k+1}$, we apply the same way to extract the corresponding spatial-aware attention maps $\mathcal{M}_{F}^{f}$ from the foreground-specific layers $(I^{i})_{i=2}^{k}$. Next, we implement a layout align loss $\mathcal{L}_{layout}$ to enable the global layer to supervise and guide the local foreground layers, facilitating alignment and coherence between them:

$$
\mathcal{L}_{layout}=\sum^{f}||\mathcal{M}_{G}^{f,N}-\mathcal{M}_{F}^{f}||_{2}.(7)
$$

The final objective can be jointly written as

$$
\mathcal{L}=\lambda_{noise}\mathcal{L}_{noise}+\lambda_{c}\mathcal{L}_{c}+ \lambda_{layout}\mathcal{L}_{layout},(8)
$$

where $\lambda_{noise},\lambda_{c}$ and $\lambda_{layout}$ are weight terms.

<a id="section-3-2"></a>

### 3.2 Layer-Shared Self-Attention

To further strengthen the connections between layers, we propose a layer-shared self-attention. This approach first integrates global layer information into the foreground layers through the attention map, then processes information from all layers simultaneously within the self-attention mechanism, reinforcing inter-layer relationships and ensuring consistency throughout the multi-layer generation.

Specifically, as shown in [Fig. 4](#figure-4) (b), given a layer batch of noisy latent images $\{z_{t}^{i}\}_{i=1}^{k+1}$ and the global context map $\{\mathcal{M}_{G}^{i,t}\}^{k}_{i=2}$ at time step $t$, we integrate global information into the foreground layers based on the global context map, which is expressed as:

$$
\tilde{z}_{t}^{i}=z_{t}^{k+1}\cdot\mathcal{M}_{G}^{i,t}+z_{t}^{i}\cdot(1- \mathcal{M}_{G}^{i,t}).(9)
$$

For a diffusion model with $T$ denoising steps, we execute the process during the first $T_{G}$ steps. Furthermore, to establish interaction between layers, we concatenate all noisy latent images along the sequence dimension to form a joint noise image at each denoising step:

$$
\tilde{z}_{t}^{c}=\mathrm{concat}(\tilde{z}_{t}^{1},\cdots,\tilde{z}_{t}^{k+1}).(10)
$$

We then perform linear projections to generate the joint key $K_{s}^{c}$ and value $V_{s}^{c}$, and apply attention with original layer query $Q_{s}^{i}$, formally as:

$$
O_{s}^{i}=Softmax(\frac{Q_{s}^{i}(K_{s}^{c})^{T}}{\sqrt{d}})V_{s}^{c},(11)
$$

where $d$ denotes the latent dimension. The weights of the linear projection are directly initialized from the original weights.

<a id="section-3-3"></a>

### 3.3 Information Retained Harmonization

In multi-layer fusion, simply blending layers based on the alpha channel often affects the overall visual quality, as adding foreground objects to the background typically introduces shadow variations in real-world scenarios. To achieve a more harmonious fusion of the composite image, we propose Information Retained Harmonization, which blends latent during the denoising process and incorporates additional denoising steps, resulting in a more coherent and visually consistent final composite image.

<a id="table-1"></a>

> Table 1: Quantitative comparison of multi-layer composite image generation. For SD v1.5, we generate the complete image from the global prompt as a composite image.

| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| ------------------- | -------------- | --------------- | ------------ | ------------- | -------------- | --------------- | ------ | ------------- | -------------- | --------------- | ------ |
| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| Methods             | Two Layers     |                 | Three Layers |               | Four Layers    |                 |        |               |                |                 |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| AES$\uparrow$       | Clip$\uparrow$ | FID$\downarrow$ |              | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |        |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| SD v1.5 [23]        | 6.930          | 34.678          | 53.950       |               | 6.363          | 34.222          | 55.198 |               | 6.367          | 35.000          | 59.149 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| LayerDiffusion [41] | 6.522          | 32.466          | 63.481       |               | 6.058          | 30.350          | 67.118 |               | 5.975          | 29.158          | 79.997 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer w/o IRH  | 6.967          | 34.587          | 51.957       |               | 6.351          | 34.696          | 58.812 |               | 6.340          | 34.993          | 57.073 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
| DreamLayer          | 7.013          | 34.835          | 50.761       |               | 6.441          | 35.267          | 54.508 |               | 6.422          | 35.723          | 53.598 |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |
|                     |                |                 |              |               |                |                 |        |               |                |                 |        |

Specifically, during the standard $T$ denoising steps, we retain the noisy latent images between step $T_{H}$ and $T_{H}^{\prime}$, denoted as $\{z_{t}\}_{t=T_{H}}^{T_{H}^{\prime}}$. After completing the $T$ denoising steps, we obtain the alpha channel $\{I_{\alpha}^{i}\}_{i=2}^{k}$ of the foreground-specific layer through the layer decoder. We then perform re-denoising for $T-T_{H}$ steps as a harmonization process, and between steps $T_{H}$ and $T_{H}^{\prime}$, we conduct latent-level layer fusion. The formulation is as follows:

$$
\hat{z}_{t}^{m}=\hat{z}_{t}^{1}\cdot\prod_{i=2}^{k}(1-I_{\alpha}^{i})+\sum_{i= 2}^{k}(z_{t}^{i}\cdot I_{\alpha}^{i}\cdot\prod_{f=i+1}^{k}(1-I_{\alpha}^{f})),(12)
$$

where $\hat{z}$ represents the noisy latent image obtained during the harmonization steps. During the IRH process, the fused latent is influenced by the foreground objects throughout the denoising steps, allowing for the generation of corresponding shadow details and enhancing the overall coherence of the image. Simultaneously, information from the foreground layers is gradually preserved during denoising, ensuring the consistency of the generated foreground layers.

Additionally, we can edit the layers within the latent space, ensuring a smoother, more harmonious fusion of the layers. This is expressed as follows:

$$
\displaystyle\hat{z}_{t}^{m}= \displaystyle\hat{z}_{t}^{1}\cdot\prod_{i=2}^{k}(1-op(I_{\alpha}^{i}))+ \displaystyle\sum_{i=2}^{k}(op(z_{t}^{i})\cdot op(I_{\alpha}^{i})\cdot\prod_{f =i+1}^{k}(1-op(I_{\alpha}^{f}))),(13)
$$

where $op(\cdot)$ represents the operations such as resizing, flipping, and moving.

<a id="table-2"></a>

> Table 2: Dataset comparison between MuLAn and DreamLayer.

| Dataset     | Images  | Resolutions   | Classes | Instances |
| ----------- | ------- | ------------- | ------- | --------- |
| Dataset     | Images  | Resolutions   | Classes | Instances |
| Dataset     | Images  | Resolutions   | Classes | Instances |
| Dataset     | Images  | Resolutions   | Classes | Instances |
| Dataset     | Images  | Resolutions   | Classes | Instances |
| MuLAn [31]  | 44,860  | 600$\sim$800  | 759     | 101,269   |
| MuLAn [31]  | 44,860  | 600$\sim$800  | 759     | 101,269   |
| MuLAn [31]  | 44,860  | 600$\sim$800  | 759     | 101,269   |
| MuLAn [31]  | 44,860  | 600$\sim$800  | 759     | 101,269   |
| MuLAn [31]  | 44,860  | 600$\sim$800  | 759     | 101,269   |
| DreamLayer  | 408,187 | 896$\sim$1152 | 1453    | 525,388   |
| DreamLayer  | 408,187 | 896$\sim$1152 | 1453    | 525,388   |
| DreamLayer  | 408,187 | 896$\sim$1152 | 1453    | 525,388   |
| DreamLayer  | 408,187 | 896$\sim$1152 | 1453    | 525,388   |
| DreamLayer  | 408,187 | 896$\sim$1152 | 1453    | 525,388   |
| -TwoLayer   | 305,801 | 896$\sim$1152 | 1379    | 305,801   |
| -TwoLayer   | 305,801 | 896$\sim$1152 | 1379    | 305,801   |
| -TwoLayer   | 305,801 | 896$\sim$1152 | 1379    | 305,801   |
| -TwoLayer   | 305,801 | 896$\sim$1152 | 1379    | 305,801   |
| -TwoLayer   | 305,801 | 896$\sim$1152 | 1379    | 305,801   |
| -ThreeLayer | 87,571  | 896$\sim$1152 | 1322    | 175,142   |
| -ThreeLayer | 87,571  | 896$\sim$1152 | 1322    | 175,142   |
| -ThreeLayer | 87,571  | 896$\sim$1152 | 1322    | 175,142   |
| -ThreeLayer | 87,571  | 896$\sim$1152 | 1322    | 175,142   |
| -ThreeLayer | 87,571  | 896$\sim$1152 | 1322    | 175,142   |
| -FourLayer  | 14,815  | 896$\sim$1152 | 1045    | 44,445    |
| -FourLayer  | 14,815  | 896$\sim$1152 | 1045    | 44,445    |
| -FourLayer  | 14,815  | 896$\sim$1152 | 1045    | 44,445    |
| -FourLayer  | 14,815  | 896$\sim$1152 | 1045    | 44,445    |
| -FourLayer  | 14,815  | 896$\sim$1152 | 1045    | 44,445    |
|             |         |               |         |           |
|             |         |               |         |           |
|             |         |               |         |           |
|             |         |               |         |           |
|             |         |               |         |           |

<a id="section-3-4"></a>

### 3.4 Dataset Preparation

[Fig. 5](#figure-5) illustrates the construction process of our multi-layer dataset. To manage complex layer relationships, we begin with the global layer and employ open-set object detection, depth maps, and inpainting to decompose it into multiple layers. First, we randomly sample a prompt from a large-scale prompt dataset [35] as the base prompt. This base prompt it then processed by the GPT-4 model, which breaks it down into a background prompt $C_{p}^{0}$, several foreground prompts, and a complete global prompt $C_{p}^{k+1}$. If the base prompt lacks sufficient foreground objects, GPT-4 selects a suitable category from the Object365 [29] dataset. Next, the global prompt is passed through a powerful image generation diffusion model, such as Flux [14], SD3 [15], or SDXL [21], to generate a complete image. We then use the foreground prompts and the open-set detection model, GroundingDINO [17], to match the text with objects in the image. Simultaneously, we generate a depth map of the complete image using the DepthAnything [37] model. Based on the depth map, we extract the object at the forefront using a matting model and fill in the missing areas with an inpainting model. Repeating this process, we determine the sequence of layers using the depth map and extract the corresponding foreground layers. We match the objects and text using segmentation masks and detection boxes, ultimately obtaining transparent images for multi-layers. Current generative models still struggle in generating a larger number of objects, resulting in low data retention. Therefore, we set the final output to 4 layers. Details of the pipeline are in the supplementary materials.

Following this pipeline, we generate a dataset containing millions of multi-layer images. We then conducted a manual review to filter and select images that met specific criteria, such as clear and complete foregrounds, harmonious backgrounds free of artifacts, and other quality standards. The final dataset comprises $300k$ two-layer images, $85k$ three-layer images, and $15k$ four-layer images. As shown in [Tab. 5](#table-5), our dataset contains more samples and encompasses a broader range of classes compared to existing datasets.

<a id="section-4"></a>

## 4 Experiments

<a id="figure-6"></a>

![Userstudy](images/Userstudy.png)

> Figure 6: The vote preference percentage in user study. We evaluate our method and LayerDiffusion on three aspects: multi-layer, foreground, and background quality.

<a id="figure-7"></a>

![Visual_LayerDiffusion](images/Visual_LayerDiffusion.png)

> Figure 7: Qualitative comparison of multi-layer image generation. We present the generation results of Layerdiffusion and our method with two-layer, three-layer and four-layer images.

<a id="section-4-1"></a>

### 4.1 Implementation Details

Training. We initialize training with the pre-trained weights of Stable Diffusion v1.5 [23] and employ the Custom Diffusion [13] strategy, fine-tuning the K&V linear layers in all attention layers. For foreground layers, additional K&V layers are trained separately. The Context-Aware Cross-Attention is applied in the downsampling layers at a resolution of 16, while Layer-Shared Self-Attention is used in all upsampling layers. Each layer batch is initialized with same timestep noise, and the Layer Embedding is zero-initialized to minimize interference with the original weights. The training is performed over 4 days on 2 A100 GPUs with a batch size of 4 and a learning rate of 2e-6. More details are available in the supplementary materials.

Evaluation. We evaluate DreamLayer on a test set of 3k multi-layer images from our proposed dataset. Aesthetic quality is assessed using the AES Score [28], text-image alignment with the CLIP-Score [22], and distribution similarity with the FID [8].

<a id="figure-8"></a>

![Visual_Ab_Cross](images/Visual_Ab_Cross.png)

> Figure 8: Ablation Study on Context-Aware Cross-Attention: Using $\mathcal{L}_{layout}$ supervision to extract layout information from the global image, guiding the generation of foreground layers and reducing overlapping placements.

<a id="table-3"></a>

> Table 3: Ablation study on LSSA and CACA.

| DreamLayer | Multi-Layers (Average) |               |                |                 |     |
| ---------- | ---------------------- | ------------- | -------------- | --------------- | --- |
| DreamLayer | Multi-Layers (Average) |               |                |                 |     |
| DreamLayer | Multi-Layers (Average) |               |                |                 |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
| LSSA       | CACA                   | AES$\uparrow$ | Clip$\uparrow$ | FID$\downarrow$ |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
|            |                        | 6.438         | 33.808         | 57.241          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
| ✓          |                        | 6.471         | 33.727         | 56.598          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
|            | ✓                      | 6.561         | 34.004         | 55.788          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
| ✓          | ✓                      | 6.625         | 35.275         | 52.956          |     |
|            |                        |               |                |                 |     |
|            |                        |               |                |                 |     |
|            |                        |               |                |                 |     |
|            |                        |               |                |                 |     |
|            |                        |               |                |                 |     |
|            |                        |               |                |                 |     |

<a id="section-4-2"></a>

### 4.2 Comparisons of Multi-Layer Image Generation

Quantitative Comparisons. As shown in [Tab. 1](#table-1), We compare DreamLayer’s performance in generating complete layers with results from Stable Diffusion [23] (SD15) and LayerDiffusion [41]. In this setup, SD15 generates a single complete image based solely on a global prompt. We use LayerDiffusion’s background-to-foreground approach for three-layer and four-layer images, sequentially adding foreground elements to simulate multi-layer composition. As shown in the table, our method outperforms LayerDiffusion across all three metrics for multi-layer generation, with a notable improvement of around 0.5 in aesthetic score. For composite multi-layer images, our approach also achieves higher aesthetic quality and better text alignment compared to direct full-image generation by SD15.

Qualitative Comparisons. In [Fig. 7](#figure-7), we present the multi-layer image generation results. Compared to Layerdiffusion [41], our method produces more coherent and appropriately sized foreground layers and achieves a more harmonious blending of the foreground and background.

User Study. As shown in [Fig. 6](#figure-6), we perform a user study with 20 subjects on 200 samples to evaluate the multi-layer generation quality of our method and LayerDiffusion [41] across three aspects: multi-layer, foreground, and background quality. The results show that our method achieves a preference percentage of 71.34%, 47.84%, 83.03% w.r.t the above three aspects. It indicates our method delivers more cohensive layouts and higher quality, particularly in background and multi-layer images.

<a id="section-4-3"></a>

### 4.3 Ablation Study

Context-Aware Cross-Attention. CACA extracts the context map information from the global layer and utilizes $\mathcal{L}_{layout}$ to guide the layout of the foreground layer. As shown in [Fig. 8](#figure-8), without the layout alignment loss w/o $\mathcal{L}_{l}$, foreground objects tend to generate in the same position, leading to overlap and occlusion. We report the qualitative results in [Tab. 3](#table-3). Removing CACA significantly degrades image quality, reducing the overall AES score of the multi-layer generation by 0.154.

Layer-Shared Self-Attention. LSSA is primarily used to maintain consistency across different image layers. As shown in [Tab. 3](#table-3), the absence of LSSA leads to a significant drop in the CLIP score, decreasing by approximately 1.27.

<a id="figure-9"></a>

![Visual_Ab_Harm](images/Visual_Ab_Harm.png)

> Figure 9: Ablation study on IRH. Our harmonization approach, unlike direct blending, generates appropriate shadows for foreground objects, resulting in a more cohesive overall composition.

<a id="table-4"></a>

> Table 4: Investigation of $T_{H}$ and $T_{H}^{\prime}$ in IBH.

| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| ----------------- | ----- | ----- | ----- | ----- | ----- |
| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| $T_{H}$           | 0     | 200   | 400   | 600   | 800   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| $T_{H}^{\prime}$  | 0     | 0     | 200   | 400   | 600   |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
| Avg AES$\uparrow$ | 6.553 | 6.568 | 6.600 | 6.625 | 6.615 |
|                   |       |       |       |       |       |
|                   |       |       |       |       |       |
|                   |       |       |       |       |       |
|                   |       |       |       |       |       |
|                   |       |       |       |       |       |
|                   |       |       |       |       |       |

Information Retained Harmonization. We further investigate the role of IRH in layer composition. As shown in [Fig. 9](#figure-9), simply stacking foreground and background layers (Blend) produces unrealistic composite images, lacking texture details like shadows. For example, the chair in [Fig. 9](#figure-9) appears to float without a shadow, disrupting visual harmony. With IRH, however, shadows and other details are generated in the background to reflect the presence of foreground objects, resulting in a more natural and cohesive layer composition. For quantitative results, as shown in [Tab. 1](#table-1), “DreamLayer w/o IRH” shows a noticeable decline in aesthetic scores, dropping by 0.1 without IRH.

$T_{H}$ and $T_{H}^{\prime}$ in IBH. We investigate the values of $T_{H}$ and $T_{H}^{\prime}$ in the IBH. We experiment with $T_{H}$ from 800 to 0 steps. As [Tab. 4](#table-4) shows, when $T_{H}^{\prime}<600$, IBH is applied near the end of the denoising process, resulting in poor harmonization and low AES score. Conversely, when $T_{H}$ is large (e.g., $T_{H}=800$), IBH over-modifies the background, reducing the AES score. Based on these observations, we selected $T_{H}=600$ and $T_{H}^{\prime}=400$.

<a id="section-4-4"></a>

### 4.4 Further Application

Image to Layer Within the DreamLayer framework, we can extend it to Image-to-Layer task in a training-free manner. Specifically, we encode the input image into a latent representation as the global latent, then progressively add noise up to the $T$ step latent using an inversion technique [18], which serves as the initial latent for all layers in DreamLayer. To obtain a more accurate initial latent during this inversion process, we isolate global image information using a mask, minimizing the influence of other layers. As shown in [Fig. 10](#figure-10), this approach enables us to decompose the input image into separate layers based on text prompts. Detailed steps are provided in the supplementary materials.

<a id="figure-10"></a>

![Visual_Img2Layer](images/Visual_Img2Layer.png)

> Figure 10: Image to Layer Visualization: By leveraging inversion to transfer the input image as the initial noise latent for all layers, DreamLayer can decompose the input with the text prompt.

<a id="figure-11"></a>

![Visual_Edit](images/Visual_Edit.png)

> Figure 11: Layer Editing Visualization: Compared to DesignEdit, DreamLayer can complement objects at the image edges and create more cohesive results when they are flipped or moved.

Layer Editing In practical applications, DreamLayer can generate multi-layer images and allow users to make harmonious edits to the layers. As described in [Eq. 13](https://arxiv.org/html/2503.12838v1#S3.E13), we perform these edits within IRH at the latent level, ensuring more cohesive adjustments. For instance, in [Fig. 11](#figure-11), when the chair is flipped and moved, the floor shadow is updated to align with its new position, enhancing overall realism. Moreover, when parts of a foreground object extend beyond the image boundary, DreamLayer has the ability of foreground object amodal completion, which can complete the missing sections as needed when repositioned. As shown in [Fig. 11](#figure-11), compared to existing methods like DesignEdit [12], DreamLayer successfully restores the out-of-frame areas of objects such as the robot and blue vase after they are moved.

<a id="section-5"></a>

## 5 Conclusion

In this paper, we introduce a large-scale, high-quality multi-layer dataset featuring diverse foreground objects and backgrounds. Building on this, we propose DreamLayer, a framework for simultaneously generating multi-layer images. To address layout consistency among foreground layers, we introduce Context-Aware Cross-Attention, which guides foreground generation using the harmonious layout of a global image. To enhance inter-layer connections, we present Layer-Shared Self-Attention, enabling effective information exchange between layers. Finally, to generate a cohesive composite image, we propose Information Retained Harmonization, which merges layers at the latent level to achieve seamless fusion. DreamLayer support not only multi-layer generation but also layer decomposition for image-to-layer task with inversion, enabling flexible editing within the latent space for harmonious adjustments. Experimental results demonstrate the effectiveness of DreamLayer in multi-layer generation.
