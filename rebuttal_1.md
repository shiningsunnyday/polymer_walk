**Thank you for your incredibly positive review!**

**Questions**
* *how does the performance change with the size and complexity of the grammar? i.e. what is the dependency on the number of fragments?*

Thank you for this question! The number of fragments is not a parameter we directly control, so we prefer to think about how the performance depends on the motif collection strategy. In our paper, we tried three strategies:
* Fragments from literature (GC)
* Fragments from expert segmentation (HOPV, PTC)
* Fragments from heuristic algorithm (GC, HOPV, PTC)

| Grammar Complexity (\|V\|, \|E\|) | HOPV         | PTC          | GC         |
|-----------------------------------|--------------|--------------|------------|
| Literature                        | N/A          | N/A          | (96, 3656) |
| Expert                            | (329, 37273) | (407, 23145) | N/A        |
| Heuristic                         | (208, 16880) | (279, 37968) | (90, 4095) |

In the above table, we show the number of fragments and number of edges in the grammar from respective strategies. We notice that strategies 1 and 2 create more fragments, but not necessarily more edges (transition rules). This is due to the expert-defined context usually being more specific, so despite a higher number of fragments, fewer pairs of fragments may get matched.

| Ablation/Dataset        |          HOPV          |                        |                       |                       | PTC                  |                      |                     |                     | Group Contribution     |                        |                       |                       |
|-------------------------|:----------------------:|:----------------------:|:---------------------:|:---------------------:|----------------------|----------------------|---------------------|---------------------|------------------------|------------------------|-----------------------|-----------------------|
|                         | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ | Train Acc $\uparrow$ | Train AUC $\uparrow$ | Test Acc $\uparrow$ | Test AUC $\uparrow$ | Train MAE $\downarrow$ | Train $R^2$ $\uparrow$ | Test MAE $\downarrow$ | Test $R^2$ $\uparrow$ |
| Ours                    | $0.045 \pm 0.003$      | $0.996 \pm 0.001$      | $0.295 \pm 0.049$     | $0.808 \pm 0.076$     | $0.996 \pm 0.000$    | $1.000 \pm 0.000$    | $0.705 \pm 0.007$   | $0.711 \pm 0.018$   | $0.028 \pm 0.007$      | $0.998 \pm 0.002$      | $0.222 \pm 0.079$     | $0.819 \pm 0.137$     |
| Ours (-expert)          | $0.075 \pm 0.003$      | $0.990 \pm 0.001$      | $0.288 \pm 0.048$     | $0.765 \pm 0.146$     | $0.994 \pm 0.001$    | $0.999 \pm 0.000$    | $0.671 \pm 0.020$   | $0.659 \pm 0.047$   | $0.044 \pm 0.015$      | $0.995 \pm 0.004$      | $0.268 \pm 0.084$     | $0.738 \pm 0.148$     |

In the above table, we include an ablation comparing the overfitting vs generalization abilities between strategies 1 and 2 (GC) and between 2 and 3 (HOPV, PTC). Interestingly, the expert-based strategy is better at *both* overfitting and generalizing. Thus, we believe the *quality* of motifs is the dimension that matters. As you mentioned, this is how we integrate domain expert knowledge. Our experts generally have their own, rigid criteria for determining what constitutes a fragment, hence controlling how many fragments there are in the end.



* *the notion of the context for the expansion application is not clear; is it the previous fragment? i.e. an order 1 Markov process? ...*


Thank you for the question! You actually touch on two notions of "context", which we will separate for clarification. 

The first notion, the context of expansion equates to the previous fragment and open attachment point, i.e. $(u, r_{l_1})$. If the context is satisfied in the LHS, any edge $(u, v, e_{l_1, l_2}) \in E$ can be traversed, and the production rule attaches the fragment of $v$ (insertion at $v_{r_{l_2}}$) to the LHS.

The second notion actually describes the memory mechanism of our (non-Markovian) random walk. Without the "context-aware" memory mechanism, our model is indeed an order 1 Markov process. Previous literature show that higher-order random walks are required to capture temporal correlations in edge activations [1, 2], with a tradeoff of complexity and practicality. In the design of complex and modular structures, the order 1 Markov assumption is not sufficient (see footnote 2 in the paper). Meanwhile, higher-order models make it difficult to scale our grammar to larger motif graphs. We take a middle ground by introducing a set-based memory state, replacing the entire visit history with a summary of node visit counts. If $p^{(t)}$ is the current state of the random walk, the set-based memory, $c^{({t+1})}$, is updated as follows: $c^{(t+1)} \leftarrow \frac{t}{t+1} \cdot c^{(t)}+\frac{1}{t+1}\cdot p^{(t)}$. This set-based memory mechanism has precedents in graph theory literature. In particular, prior works study how memory mechanisms in random walks affect exploration efficiency [3, 4] and enable negative/positive feedback [3, 5].

* *How does the performance change with the size of possible contexts or size of contexts?*

This is a very interesting question! Our ablation study (-expert) reveals some preliminary insight into this matter. In our heuristic segmentation strategy, the size of each context is always a single atom. This is because when ablating the expert, a heuristic strategy cannot infer whether a motif should require a context of a single atom, bond, ring, or larger substructure. Meanwhile, the expert provides us dataset-specific rules for automatically determining the context (see Appendix A) and inform us of exceptions to the rule, if any. When implementing our workflows, we found this to be a good tradeoff between keeping the workflow semi-automated and allowing a few different types of contexts. Thus, we especially instructed our experts to keep the context to single/double atoms and rings, even in cases where they feel strongly a larger context must be present to attach the functional group. We acknowledge that this somewhat limits the degree to which we integrate domain expert knowledge, but this may be desirable from a computational perspective the motif graph is dense (lots of matches) rather than sparse (on which learning may prove more difficult). 

Incorporating larger contexts identified by the expert is definitely worth investigating into. One idea we have for future research is to use *Matryoshka* contexts, where each motif allows a hierarchy of attachment contexts. This makes our production rule set hierarchical and can encode stronger priors from domain experts. We are also looking at problems in domain areas beyond chemistry where such hierarichial priors are desirable.

Should there be further suggestions, please let us know!


[1] Rosvall, M., Esquivel, A., Lancichinetti, A. et al. Memory in network flows and its effects on spreading dynamics and community detection. Nat Commun 5, 4630 (2014).

[2] Masuda, Naoki, et al. ‘Random Walks and Diffusion on Networks’. Physics Reports, vol. 716–717, 2017, pp. 1–58.

[3] Fang, Guanhua, Gennady Samorodnitsky, and Zhiqiang Xu. "A Cover Time Study of a non-Markovian Algorithm." arXiv preprint arXiv:2306.04902 (2023).

[4] Gąsieniec, Leszek, and Tomasz Radzik. ‘Memory Efficient Anonymous Graph Exploration’. Graph-Theoretic Concepts in Computer Science, edited by Hajo Broersma et al., Springer Berlin Heidelberg, 2008, pp. 14–29.

[5] Pemantle, Robin. "A survey of random processes with reinforcement." (2007): 1-79.
