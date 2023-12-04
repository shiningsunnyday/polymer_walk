# Polymer walk

## Requirements

Conda is used to create the environment for running polymer_walk.

```bash
# Install environment from file
conda env create -f polymer_walk.yml
```

## Data preparation

Create a directory for the motifs. 
```
data/datasets/{dataset}/
└───walks_new/
│   │   *.edgelist
└───all_groups/
│   │   *.mol
│   │   all_extra.txt
│   └───with_label/
│       │   *.png
```
A dataset of random walks on the motif graph can be provided as a text file. 

For example, data/polymer_walks_v2_preprocess.txt
has examples of the following form:
```bash
PIM-1 L3-2>S32-2>S20[-1>S1,-4>S1]-2>S32-2>L3 (1300,92,370,125,2300) # property vals
```
where L3, S32, S20 are names of groups under all_groups/ and -{j}> is the j-th edge on the motif graph, which is a digraph with multiedges.

It can also be simultaneously extracted along with the dataset of random walks (will upload script soon). The output should be placed in walks_new/.


## Data preprocessing

Set the directory to your motifs:


```bash
export dataset=data/datasets/group-contrib
```

Go to the top of utils/graph.py and utils/preprocess.py and toggle between group folders, as needed.

Build the motif graph. 
```bash
python walk_grammar.py --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt --out_path $dataset/red_graph.adjlist
```

Preprocess existing random walks.

```bash
python construct_graph.py --data data/polymer_walks_v2_preprocess.txt --graph_vis_file $dataset/polymer_walk.png --predefined_graph_file $dataset/red_graph.edgelist --extra_label_path $dataset/all_groups/all_extra.txt --motifs_folder $dataset/all_groups/ --out_path $dataset/dags.pkl --extract_edges
```

Remove --extract_edges if your .txt file has no edge info (e.g. L3->S32->S20[->S1,->S1]->S32->L3). The algorithm will guess the edges, but will not guarantee validity.

Train graph diffusion grammar. 
```bash
python diffusion.py --data_file data/polymer_walks_v2_preprocess.txt --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt --dags_file $dataset/dags.pkl --predefined_graph_file $dataset/red_graph.edgelist --num_epochs 10000 --alpha 1e-1 --context_L
```
The artifacts are saved under logs/, e.g. logs-1695308013.8280082/.

Train property predictor. 
```bash
python main.py --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt --grammar_folder logs/logs-1695308013.8280082 --predefined_graph_file $dataset/red_graph.edgelist --dags_file $dataset/dags.pkl --num_epochs 1000 --walks_file data/polymer_walks_v2_preprocess.txt
```

The predictor is saved under grammar_folder, e.g. predictor_1696440075.4187388. 

Sample novel ones from grammar checkpoint.
```bash
python diffusion.py --data_file data/polymer_walks_v2_preprocess.txt --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt --dags_file $dataset/dags.pkl --predefined_graph_file $dataset/red_graph.edgelist --num_epochs 2000 --context_L --alpha 1e-1 --log_folder logs/logs-1695308013.8280082
```

Do property-driven sampling.
```bash
python main.py --motifs_folder $dataset/all_groups/ --extra_label_path $dataset/all_groups/all_extra.txt --grammar_folder logs/logs-1695308013.8280082 --predefined_graph_file $dataset/red_graph.edgelist --dags_file $dataset/dags.pkl --num_epochs 1000 --walks_file data/polymer_walks_v2_preprocess.txt --predictor_file logs/logs-1695308013.8280082/predictor_1696440075.4187388/predictor_ckpt_0.007912156573847579.pt --grammar_file logs/logs-1695308013.8280082/predictor_1696440075.4187388/grammar_ckpt_0.007912156573847579.pt
```
