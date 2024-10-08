**Thank you for the comprehensive review and acknowledging the strengths of our work!**


* *The superiority of random walks over a simpler undirected graph of motifs is not adequately addressed. Specifically, it is unclear what advantages random walks offer compared to applying Graph Neural Networks (GNNs) directly to motif graphs, like junction-tree graphs [1]. Potential drawbacks such as loops and duplicates arising from random walks warrant further discussion.*

In our experiments, we actually did use $H_M$, the bi-directionally connected (i.e. undirected) graph of motifs as the input to the GNN. We apologize for not making this clear in the main text.

To answer your inquiry, we ran an additional ablation study that compares across the four possible combinations between 1) whether to use directed/undirected edges, 2) whether to attribute edge weights for the task of property prediction. Our findings are summarized in the table below:

|      |                  | $H_M$           | $H_M+w_M$       | $\hat{H_M}-w_M$  | $\hat{H_M}$     |
|------|------------------|-----------------|-----------------|------------------|-----------------|
| GC   | MAE $\downarrow$ | $0.24 \pm 0.08$ | $0.24 \pm 0.09$ | $0.26 \pm 0.11$  | $0.27 \pm 0.10$ |
|      | $R^2$ $\uparrow$ | $0.79 \pm 0.03$ | $0.80 \pm 0.02$ | $0.83 \pm 0.10$  | $0.79 \pm 0.12$ |
| HOPV | MAE $\downarrow$ | $0.30 \pm 0.05$ | $0.34 \pm 0.04$ | $0.39 \pm 0.06 $ | $0.38 \pm 0.03$ |
|      | $R^2$ $\uparrow$ | $0.80 \pm 0.11$ | $0.75 \pm 0.11$ | $0.61 \pm 0.25$  | $0.62 \pm 0.23$ |
| PTC  | Acc $\uparrow$   | $0.70 \pm 0.01$ | $0.68 \pm 0.01$ | $0.71 \pm 0.02$  | $0.70 \pm 0.02$ |
|      | AUC $\uparrow$   | $0.71 \pm 0.02$ | $0.65 \pm 0.01$ | $0.67 \pm 0.03$  | $0.67 \pm 0.05$ |

In this table, we use notations consistent with their definitions in the paper. $H_M$ is the simple bidirectional graph of motifs, $H_M+w_M$ is $H_M$ with the attribution of edge weights, $\hat{H_M}-w_M$ is the directed random walk, and $\hat{H_M}$ is the random walk of directed, weighted edges.

We make some observations:

1) Directed graph ($H_M$ --> $\hat{H_M}-w_M$) does not affect performance on GC and PTC but hurts performance on HOPV.

2) Adding edge weights ($H_M$ --> $H_M+w_M$) does not affect performance on GC but hurts performance on HOPV and PTC.

3) The standard deviation is higher for the directed representation ($H_M$ --> $\hat{H_M}-w_M$) but not for the weighted representation.

The directed representation does not improve discriminative ability and incurs the cost of poor generalization. On GC, monomers follow the IUPAC convention that reads left-to-right. The "directedness" inductive bias is consistent with the inherent directedness in the data. The same is not true for HOPV and PTC. The performance drop is especially pronounced on HOPV, which has larger molecules hence more nodes in $H_M$. Continuous edge weights also enhance the representation space but does not necessarily lead to stronger generalization. 

Based on these findings, we confirm your intuition that the undirected representation is to be preferred for property prediction tasks. Random walks are more interesting in the context of the design space graph, where the random walk concept allows for sampling molecules in a generation setting.

Regarding the difference of our method with [1]'s junction tree representation: As the author remarks himself in his follow-up work [2], the junction tree representation encounters difficulties when decoding larger molecules that require more assembly steps due to the combinatorial complexity of assembling the nodes. Meanwhile, our representation features a random walk procedure that enables a direct derivation sequence of motifs. We included additional ablation studies below comparing our method against [1, 2] and they demonstrate our efficacy in data-efficient settings.

* *The paper contains ambiguous points within the experimental results. For instance, the performance metrics in Table 4, utilizing a bag of motifs, appear superior to those achieved by the proposed method in Table 1. The explanation provided, attributing this to generalization capabilities, lacks empirical support (e.g., training Mean Absolute Error (MAE) for the proposed method is not shown)....*

We just realized that there is a typo in Table 4: the metrics for PTC should be Acc and AUC (the higher the better). We apologize for the confusion this has caused. Further, we report the seemingly good results of Bag-of-Motifs on the Train set for demonstration purposes, to show that it is considerably overfitting. Yet the actually important test performance is significantly worse across all three datasets. For clarity, Table 4 should have been as follows:

| Ablation/Dataset        |          HOPV          |                        |                       |                       | PTC                  |                      |                     |                     | Group Contribution     |                        |                       |                       |
|-------------------------|:----------------------:|:----------------------:|:---------------------:|:---------------------:|----------------------|----------------------|---------------------|---------------------|------------------------|------------------------|-----------------------|-----------------------|
|                         | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ | Train Acc $\uparrow$ | Train AUC $\uparrow$ | Test Acc $\uparrow$ | Test AUC $\uparrow$ | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ |
| Bag-of-Motifs           | $0.014 \pm 0.002$      | $0.997 \pm 0.060$      | $0.486 \pm 0.025$     | $0.489 \pm 0.062$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.529 \pm 0.031$   | $0.609 \pm 0.031$   | $0.000 \pm 0.000$      | $1.000 \pm 0.000$      | $0.481 \pm 0.174$     | $0.257 \pm 0.453$     |
| Bag-of-Motifs (+expert) | $0.011 \pm 0.004$      | $1.000 \pm 0.000$      | $0.521 \pm 0.031$     | $0.446 \pm 0.125$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.581 \pm 0.018$   | $0.612 \pm 0.029$   | $0.000 \pm 0.000$      | $1.000 \pm 0.000$      | $0.493 \pm 0.143$     | $0.214 \pm 0.404$     |
| Ours                    | $0.045 \pm 0.003$      | $0.996 \pm 0.001$      | $0.295 \pm 0.049$     | $0.796 \pm 0.105$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.705 \pm 0.007$   | $0.711 \pm 0.018$   | $0.028 \pm 0.007$      | $0.998 \pm 0.002$      | $0.222 \pm 0.079$     | $0.819 \pm 0.137$     |

In this table, we show the performance of Bag-of-Motifs with both heuristic and expert motifs (note: we use +expert so that the default setting is to apply heuristic motifs). We also ran our method again to evaluate the metrics on the training set. This provides empirical support for our explanation regarding our method's greater generalization capabilities. 

* *...Questions also arise regarding the reported RS values exceeding those of the training data and the absence of novelty metrics for molecule generation.*

We assure you that our method does achieve the reported RS scores. For transparency, we publish our generated molecules and the RS result at the following (anonymous) link: https://docs.google.com/document/d/e/2PACX-1vRs2Jx4_0GNihD3QMIUU1n1m_ZMr3niCz1fSvl9240Q5EblKOKeGdlSmcdpsYyoue58e3zPM_B73ehH/pub

We have added two additional metrics: 1) Novelty, the percentage of novel molecules, and 2) Membership, the percentage of molecules that belong to a chemical class. 

The Membership metric is included after further consultation with experts, who identify the presence of Thiophene as a proxy for Membership to HOPV, and the presence of Chloride/Bromide Halides (a key indicator of toxicity) for PTC. In the case of both datasets, the Membership metric is only a sanity check that the method can produce a non-trivial number of characteristic compounds (see our response to Reviewer QHNR).

Please see the updated Table 3:

|      | Methods            | Valid | Unique | Novel | Diversity | RS  | Memb. |
|------|--------------------|-------|--------|-------|-----------|-----|-------|
| HOPV | Train Data         | 100%  | 100%   | N/A   | 0.86      | 51% | 100%  |
|      | DEG                | 100%  | 98%    | 99%   | 0.93      | 19% | 46%   |
|      | JT-VAE             | 100%  | 11%    | 100%  | 0.77      | 99% | 84%   |
|      | Hier-VAE           | 100%  | 43%    | 96%   | 0.87      | 79% | 76%   |
|      | Hier-VAE (+expert) | 100%  | 29%    | 92%   | 0.86      | 84% | 82%   |
|      | Ours               | 100%  | 100%   | 100%  | 0.89      | 58% | 71%   |
| PTC  | Train Data         | 100%  | 100%   | N/A   | 0.94      | 87% | 30%   |
|      | DEG                | 100%  | 88%    | 87%   | 0.95      | 38% | 27%   |
|      | JT-VAE             | 100%  | 8%     | 80%   | 0.83      | 96% | 27%   |
|      | Hier-VAE           | 100%  | 20%    | 85%   | 0.91      | 92% | 25%   |
|      | Hier-VAE (+expert) | 100%  | 28%    | 75%   | 0.93      | 90% | 17%   |
|      | Ours               | 100%  | 100%   | 100%  | 0.93      | 60% | 22%   |


* *Incorporating ablation studies that compare the proposed method with other motif-based approaches, potentially utilizing vocabularies crafted by domain experts, would strengthen the validation of the experimental results.*

We weren't sure whether you meant motif-based *generative* models or motif-based property *prediction*, as the literature for the two tasks are different (most works focus on one or the other, whereas our method handles both). For completeness, we performed separate ablation studies with respective SOTA motif-based methods.

**Comparison with Other Motif-based Generative Models**

As mentioned previously, we include JT-VAE [1] as an additional baseline. Since JT-VAE is limited to only rings and bonds, we also include the author's follow-up work, Hier-VAE [2] which adopts larger structural motifs. Furthermore, we modified the implementation of Hier-VAE to incorporate *our* epert motifs. For all three cases, we follow the default settings, train until convergence, and use the checkpoint with the lowest loss to sample 1000 molecules. We include the results in the updated Table 3.

We observe that both VAE-based methods [1, 2] struggle to generate sufficiently unique molecules, with only 11%-43% (HOPV) and 8%-28% (PTC) of the 1000 generated molecules being unique. This is despite sampling from a Gaussian noise distribution. Meanwhile, our model generates 100% unique and novel molecules, while ensuring a high internal diversity second only to DEG. 

For reference, [1, 2] trained and evaluated their models on ZINC containing ~250K drug-like molecules and a polymer dataset containing 86K polymers. Meanwhile, our datasets contain only ~100-300 molecules and, in the case of HOPV, feature much larger molecules. Rather than using an encoder-decoder setup which requires significantly more data to learn the mapping to and from a latent space, our generative model explicitlys capture the transition probabilities over traversing the symbolic space of structural motifs. Our grammar derivation can easily be conditioned by a set-based context to apply a diverse set of transition rules (see the response to Reviewer 1 for a further discussion). This leads to more unique, diverse, and, most importantly, synthesizable structures. Additionally, our generative process is explainable to a chemist, and lead to scientific insights that may be difficult to interpret from black-box models.

**Comparison with Other Motif-based Property Predictors**

For property prediction, we ran a comparison study HM-GNN, a SOTA motif-based property predictor that explicitly models motif-molecule and motif-motif relationships using a hetereogenous graph. Just like the previous comparison study, we also endowed the method with *our* expert motifs since the vanilla version only considers bonds and rings. We plan to add the comparison to Table 4, so the final version is as follows:

| Ablation/Dataset        |          HOPV          |                        |                       |                       | PTC                  |                      |                     |                     | Group Contribution     |                        |                       |                       |
|-------------------------|:----------------------:|:----------------------:|:---------------------:|:---------------------:|----------------------|----------------------|---------------------|---------------------|------------------------|------------------------|-----------------------|-----------------------|
|                         | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ | Train Acc $\uparrow$ | Train AUC $\uparrow$ | Test Acc $\uparrow$ | Test AUC $\uparrow$ | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ |
| Bag-of-Motifs           | $0.014 \pm 0.002$      | $0.997 \pm 0.060$      | $0.486 \pm 0.025$     | $0.489 \pm 0.062$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.529 \pm 0.031$   | $0.609 \pm 0.031$   | $0.000 \pm 0.000$      | $1.000 \pm 0.000$      | $0.481 \pm 0.174$     | $0.257 \pm 0.453$     |
| Bag-of-Motifs (+expert) | $0.011 \pm 0.004$      | $1.000 \pm 0.000$      | $0.521 \pm 0.031$     | $0.446 \pm 0.125$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.581 \pm 0.018$   | $0.612 \pm 0.029$   | $0.000 \pm 0.000$      | $1.000 \pm 0.000$      | $0.493 \pm 0.143$     | $0.214 \pm 0.404$     |
| HM-GNN                  | $0.366 \pm 0.035$      | $0.686 \pm 0.066$      | $0.473 \pm 0.019$     | $0.441 \pm 0.065$     | $0.915 \pm 0.033$    | $0.966 \pm 0.016$    | $0.71 \pm 0.023$    | $0.678 \pm 0.040$   | $0.281 \pm 0.064$      | $0.717 \pm 0.137$      | $0.362 \pm 0.113$     | $0.592 \pm 0.202$     |
| HM-GNN (+expert)        | $0.201 \pm 0.009$      | $0.895 \pm 0.019$      | $0.451 \pm 0.025$     | $0.408 \pm 0.095$     | $0.999 \pm 0.002$    | $1.000 \pm 0.000$    | $0.681 \pm 0.024$   | $0.587 \pm 0.075$   | $0.185 \pm 0.016$      | $0.926 \pm 0.039$      | $0.345 \pm 0.149$     | $0.547 \pm 0.295$     |
| Ours                    | $0.045 \pm 0.003$      | $0.996 \pm 0.001$      | $0.295 \pm 0.049$     | $0.796 \pm 0.105$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.705 \pm 0.007$   | $0.711 \pm 0.018$   | $0.028 \pm 0.007$      | $0.998 \pm 0.002$      | $0.222 \pm 0.079$     | $0.819 \pm 0.137$     |



On both regression datasets, HM-GNN avoids overfitting but does not catch up to our method's generalization capability. Endowing HM-GNN with our expert motifs enables better fitting of the training data but further hinders generalization. On PTC, HM-GNN is competitive our method in accuracy but shows a discrepancy in terms of AUC. This is concerning as a lower AUC may imply higher sensitivity to class imbalance (in PTC, there are 45% more negatives than positives) and classification thresholds. Meanwhile, our method can both 1) completely fit the training data (>0.99 $R^2$, >99% Acc/AUC), and 2) generalize to the test data, with further regularization potentially leading to even better results. We believe our motif-based representation carries better inductive biases, integrates better with expert motifs, and demonstrates stronger empirical performance.


* *A minor typographical error is noted: "As shown in Table 4.3.1” in Section 4.3.2 should be corrected to "As shown in Table 4."*

We fixed it in the manuscript.

We will update our manuscript with these additional results for the camera-ready version. 

**Questions:**

* *The description of the attachment points is unclear. Is it right that the contexts of attachment (red parts) is attached to the black parts of the other motif as described in Figure 1 (c)? Then how can motif 71 and 18 can be attached?*

Thank you for the question. The matching between two motifs A and B occurs iff there is a *subgraph* of A's black part that satisfies B's red part AND a *subgraph* of B's black part that satisfies A's red part. For 71 and 18, this is rather straightforward: both 71 and 18 seek a context of a single-bonded carbon. 71's context is satisfied once by 18, using the only carbon atom of 18 which has three neighbors. 18's context is satisfied four times by 71, using any of 71's four carbon atoms which reside on an aromatic ring and are the base of a red single bond.

The mathematical formulation is given at the start at 3.1. There's also an additional example we include in the Appendix (end of page 24) to elucidate this matching process. 

An updated Figure 1 (with minor correction) is here: https://ibb.co/TcbVJ2H

* *The choice of a graph heat diffusion process for the random walk mechanism prompts inquiry. A discussion on the relationship between random walk dynamics and graph heat diffusion, including the advantages of this approach over autoregressive models, would enhance understanding. Given that autoregressive models could effectively model the sequence of probabilities produced by a random walk, their comparative advantages should be explored.*

Thank you for the question! We share your inquiry into the random walk mechanism. The relationship between graph heat diffusion and random walk has been studied before [4], but we integrate two new ideas: 1) making the Laplacian (edge weights) learnable and dynamically adjustable, and 2) parameterizing the adjustment on an order-invariant context. The reason as to why we don't just use autoregressive models is part of a larger discussion on the respective merits of autoregressive models vs grammar-based approaches (which we include in the Related Works, but can expand more on if necessary). In data-efficient settings, previous works [3, 4] show grammar (esp. context-free grammar) work well due to the relatively small (tens/hundreds) number of examples needed to learn valid rules and derivation sequences. Meanwhile, the number of possible hidden states that autoregressive models are parameterized to learn is exponential (to the length of the sequence), and learning a good parameterization requires a large amount of data. We take a middle ground, combining the data-efficient advantages of context-free grammar and the expressivity of autoregressive models, by introducing a *context-sensitive* grammar which utilizes a set-based memory during the random walk.

* *The term "context sensitivity" requires a more detailed definition. An explicit description of what is meant by context sensitivity, especially given its highlighted importance as a significant contribution of the proposed grammar, would provide clarity on this aspect.*

Thank you for the question! The description of "context sensitivity" in our grammar boils down to how we enumerate all the "context-sensitive" production rules in our grammar. We use the notations defined at the start of 3.1. There exists exactly one production rule in our grammar for each edge $(v_1, v_2, e_{l_1, l_2})$ on $G$. If this edge connects motifs $v_1 \rightarrow v_2$, then the application of the production rule equates to attaching the fragment of $v_2$ to the fragment of $v_1$, at the attachment points keyed with $l_1, l_2$. The "context" of this production rule is hence $(v_1, l_1)$. If the LHS molecule has the fragment of $v_1$ with $l_1$ being an open attachment point, the rule can be applied.

We will add this definition in the manuscript. We realize we overloaded the term "context" -- "in the context of all nodes visited thus far" and "context-sensitive adjustment" actually refer to the memory mechanism of our random walk (which prior literature would refer to as being "history-aware" or "context-aware"). You can refer to our response to Reviewer WVcM for further details. We will revise our manuscript to clearly separate the two usages.

Thank you for your valuable suggestions, which we have taken to improve our manuscript at length. Should there be further suggestions, please let us know!

[1] Jin, W., et al. Junction tree variational autoencoder for molecular graph generation. ICML 2018.

[2] Jin, W., et al. Hierarchical Generation of Molecular Graphs using Structural Motifs. ICML 2020.

[3] Guo, M., et al. Hierarchical Grammar-Induced Geometry for
Data-Efficient Molecular Property Prediction. ICML 2023.

[4] Guo, M., et al. Data-Efficient Graph Grammar Learning for Molecular Generation. ICLR 2022.