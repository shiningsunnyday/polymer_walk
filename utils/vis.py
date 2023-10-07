import argparse
import pickle
import json


def visualize_dag(args, dag):

    breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--E_file')
    parser.add_argument('--predefined_graph_file')
    parser.add_argument('--walks_file')
    parser.add_argument('--dags_file')
    parser.add_argument('--out_path')
    args = parser.parse_args()

    dags = json.load(open(args.dags_file, 'r'))
    for dag in dags['old']:
        visualize_dag(args, dag)
