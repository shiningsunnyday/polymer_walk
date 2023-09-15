import argparse

def main(args):
    # in: n walks, m groups, features F, init model f
    # embed F(g_j) for j = 1, ..., m

    # for each walk i =: w_i
        # truncate steps that cause cycles, form w_i's DAG =: DAG_i
        # F(w_i) := embed w_i using F, e.g. POOL({F(w_ij)})
        # attach DAG_i to meta-grammar with features F(w_i)

    # for each w_i
        # W_i = combine some other walks
        # L_i = construct explicit directed multigraph normalized L from W_i
        # L_i <- f(L_i) correction
        # optimize f with -log_prob(w_i ; L_i)
        # get hat(E)_i
    
    
    # GRAND(phi) on meta-grammar
    # decode property values X on the n leaves
    # optimize phi with MSE(X, Y)
    
    return # out: f and phi

    # inference: n good walks, phi*, f*
    # L = construct explicit normalized L from [w_1, ..., w_n]

    # for k = 1, ..., K
        # w_new_k = walk with f*(L) until hit cycle
        # truncate each w_new_k's last step to form DAG_new_k
        # attach DAG_new_k to meta-grammar

    # attach {DAG_1, ..., DAG_n} to meta-grammar
    # GRAND(phi) on meta-grammar
    # decode property values X on the K leaves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motifs_folder')
    parser.add_argument('--extra_label_path')    
    parser.add_argument('--data_file')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--graph_vis_file')
    args = parser.parse_args()    
    main(args)
