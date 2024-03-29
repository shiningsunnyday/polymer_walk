*<strong>Summary:</strong>*

*The paper proposes a novel representation of molecular structures as random walks over a design space. The design space is represented as a multigraph, where the nodes contains the motifs and the possible contexts it can occur. The edges represent a transition rule. The main contribution of the paper is to model the molecules as random walks in this graph, which is formulated as a heat diffusion. It shows that the method outperform state of the art methods for molecular property prediction in 3 tasks. They show that the method provides interpretable representations, and show that it is able to propose novel, synthesizable molecules.*

*<strong>Strengths And Weaknesses:</strong>*

*Strengths*

*The paper proposes a novel framing for designing molecules. The design and construction of the graph is novel, as well as the representation of molecules as a random walk on this graph The paper shows state-of-the-art performance for different problems and in different datasets, both for property prediction and novel molecule generation. It also shows an improvement in running time over other available methods for molecule representation using grammars. The paper also proposes a way to incorporate external expertise and feedback.*

*Obs: section 4.4.1 shows an example on how meaningful the rules extracted are for experts outside of the machine learning field, yet this is outside my area of expertise and I can not provide any feedback or assesement on this topic.*

*Weaknesses* 

*One of the claims of the papers is about interpretability, yet to answer this question they analyse a 2D t-SNE of the embeddings in comparison with pre-trained representations. t-SNEs should not be used for interpretability and are not an indication of this. It is only a tool for visualisation and these are not meaningful by any means. I understand that this is used to identify visual clusters that show meaning, but this is not necesarily generalizable.*


The visual clusters in our 2D t-SNE was an unexpected finding that enhanced our own interpretation of what our model is learning. We did not intend to claim it is a universal way to test for interpretability, and apologize that our writing made it appear so. We will write a disclaimer and move this subsection to the Appendix. We will re-emphasize that the interpretability of our method is provided by the fact our design space spans explicit functional motifs that are well-understood by experts, and the learnt parameters tell experts about their importance. This interpretation is done in 4.4.1.

We also acknolwedge a 2D t-SNE is only a tool for visualization, so we present an alternative way to analyze our model's learnt embeddings:

<a href="https://ibb.co/sJPhHPL"><img src="https://i.ibb.co/h1MkfMp/image.png" alt="image" border="0"></a>

Link: https://ibb.co/sJPhHPL

There are 64 molecules in this test set indexed from lowest to highest HOMO value. The grid visualizes the distance between each pair of molecules as a cosine distance between the final layer embeddings of our model, with darker color representing lower distance (higher similarity). We use 4 quantiles, and refer to their ranges as low, medium-low, medium-high, and high similarity. Since the final layer embedding is used for prediction, we expect molecules with similar properties to have similar embeddings. What is more interesting is to analyze is the agreement between embedding similarity and structural similarity.

Several groups of trends stand out. Particularly, highlighted in green are cases where the embedding similarity is high despite dissimilar HOMO property values; blue marks cases where the embedding similarity is low, and red marks sections that are similar in property, structure, and embedding. We detail each of these, basing comparison against molecule 50 for illustration:
* For the topmost green section, molecules in the range 1-4 have similar components as those with higher HOMO values, though are much smaller in size and relatively disordered. For instance, molecules 3 and 4 each share key subcomponents (thiophene groups) with molecule 50, despite having quite different overall structure. The embedding similarity between (50, 4) and (50, 3) is thus medium-low and medium-high.
* For the red sections along the diagonal, molecules in the ranges 14-16 and 17-26 cluster together. These tend to have an over-representation of electron-withdrawing groups in in non-symmetric locations in the structure, particularly methoxy, cyano, and carbonyl groups, without sufficient electron donating groups. Molecules 15 and 20 are shown as examples, and their embedding similarity is high. Blue outlines mark similar sub-groups between 15 and 20. 
*  For the second-from-top green section, we again consider molecules in the range 18-26, where they show high similarity to the highest band in the range 47-63. These share many component structures, for instance thiophenes groups and derivatives. Molecule 23 is shown as an example, and has a barbituric acid core on one side, an electron withdrawing group, with methoxy groups on benzene rings on the other side, with a nitrogen atom between benzene rings, contributing to electron delocalization. The most likely explanation is that similar high-sterics groups have developed similar embeddings in this case.
* For the bottom-right red section, molecules in the range 47-63 generally cluster together, reflecting the model's ability to agree on both structural and property similarity. They tend to have an alternating pattern of electron-donating and electron-withdrawing groups which can increase the HOMO and provide a more direct pathway for charge transport. Yellow outlines mark matching and similar groups with molecule 50. In these cases, more than simply thiophene shows similar or the same structure. The embedding similarity between (50, 52), (50, 57), (52, 57) are all medium-high.

These insights show how complex molecule structure affects the measured property in this application, and how both structure and property are captured in the embedding.
We hope the above provides more insights into how structural priors in our representation facilitates learning and generalization.



*Minor* 

*While I understand that the method combines chemically meaningful groups, showing the Membership metric at least in the appendix would help validate this....*

Thanks for your suggestion. The Membership metric is indeed a good metric to test whether the method can generate molecules within the same chemical class. However, the Membership criteria isn't straightforward to define for our datasets, due to a greater diversity of characteristic motifs (unlike, say, Isocyanates which has the defining motif O=C=N). After consulting with domain experts, we learned the following:

* Thiophene, a 5-member ring with one sulfur group, is the most common motif in the HOPV dataset, making it the best choice for a single-motif membership criterion for HOPV. More broadly, thiophene and its derivatives are arguably the most common chemical substructure in photovoltaics due to their ability to donate electrons, resulting in particularly high highest occupied molecular orbital (HOMO) levels, along with stability, tunability of energy levels, and compatibility with film forming techniques. While not every suitable organic photovoltaic compound will contain it, the vast majority will.

* A chloroalkane (Cl-C) is the most common motif in the PTC dataset. Yet, it is still not present in a majority of structures, making the broader class of alkyl halides (Cl-C, Br-C-C) the best choice for a membership criterion. Their prevalence is attributed to their reactivity and ability to undergo metabolic activation [1], leading to the formation of highly reactive intermediates that can interact with DNA and other cellular components, potentially initiating carcinogenic processes. Although not all carcinogenic compounds will necessarily contain this class of motifs, their presence contributes a strong likelihood.

In both cases, our method can easily achieve 100% membership with a slight modification to the sampling procedure: instead of iterating through every possible starting motif node, we always initialize our random walk at the membership motif. We choose not to modify our sampling procedure, and instead include this metric for completeness, since it is still a good sanity check for other methods to show they generate a non-trivial fraction of candidates with those motif(s). You can refer to our updated Table 3 below. It also includes additional motif-based baselines after incorporating Reviewer ETCQ's suggestions.

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


* ...*The paper cites a SOTA by Schimunek et al, 2022, yet the reference provided is for a rejected paper. Either provide another reference or remove it*

Their paper actually got accepted to ICLR 2023. We will update the reference.


Should there be further suggestions, please let us know!


[1] Louis Leung, Amit S. Kalgutkar & R. Scott Obach (2012) Metabolic activation in drug-induced liver injury, Drug Metabolism Reviews, 44:1, 18-33, DOI: 10.3109/03602532.2011.605791