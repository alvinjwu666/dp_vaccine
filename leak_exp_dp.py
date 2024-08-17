import time
import networkx as nx
import sys
import os
import json
from lp_utils import *
from data_utils import *
from dp_utils import *
import random

# setup stuff
graph_seeds = [int(sys.argv[1])]
eps = float(sys.argv[2])
delt = float(sys.argv[3])
rand_seeds = range(100)
# the target max degree
target=int(sys.argv[4])
vertex_size=10000

for graph_seed in graph_seeds:
    G = load_graph()
    G = find_neighborhood(G, size=vertex_size, seed=graph_seed)
    vertices = G.nodes
    print("Number of Vertices:", len(vertices))

    # List of edges
    edges = G.edges
    print("Number of Edges:", len(edges))
    
    start_setup = time.time()

    ans, vert_deg = set_dp_multisets(vertices, edges, target)
    
    max_deg_acc = max_deg(vertices, edges)
    with open(f"runs/size_{vertex_size}_graph_{graph_seed}_maxDeg.txt", "w") as f:
        f.write(str(max_deg_acc))
        f.close()

    end_setup = time.time()
    print("Time to Setup LP:", end_setup-start_setup)

    filename = f"runs/dp_size_{vertex_size}_graph_{graph_seed}_eps_{eps}_delt_{delt}_target_{target}.json"
    # This is the seed for the randomness
    for t in rand_seeds:
        ans.reseed(t)
        
        st_trial = time.time()
        
        output = ans.cover(eps / 6, delt / 6)
        realOut, counts = ans.realCover()
        
        if not bool(counts):
            counts = {0: 0}
        
        end_trial = time.time()
        print("trial time: ", end_trial-st_trial)
        
        
        file_data = [{
                    "lp_type": "dp",
                    "rand_seed": t,
                    "graph_seed": graph_seed,
                    "num_vertices": len(G.nodes),
                    "num_edges": len(G.edges),
                    "solution": output,
                    "realSol": realOut,
                    "budget": len(realOut),
                    "residue": max(counts.values()),
                    "total_time": end_trial-st_trial
        }]
        
    
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                file_data = json.load(f) + file_data
        with open(filename, 'w') as f:
            json.dump(file_data, f)