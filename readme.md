# Differentially Private Vaccination Experiments

This project is based on code that Ann Li and Jiayi Wu wrote for Preventing Epidemic Spread Using Leaky Vaccines.

## Multi-set Multi-Cover
The majority of the multi-set multi-cover is in ```dp_utils.py```. Allows for the creation of an instance of the problem and methods to output a cover using private and non-private greedy algorithms.

## Data
We utilize the network in ```montogmery_labels_all.txt``` for the data, and use BFS on three seeds (```42```, ```379427824```, ```3345638259```) to find three smaller sub-networks of ```1000``` and ```10000```` nodes.

```data_utils.py```: python script to extract small subnetworks
    - ```load_graph(seed, unvacc_rate)```: function to load to the social network from the file ```montomgery_labels_all.txt```
        - ```seed``` is a random seed used to specify the network
        - ```unvacc_rate``` is a value between 0 and 1 indicating the proportion of people unvaccinated in a network 
        - Both parameters mentioned above are unused in current experiments
    - ```find_neighborhood(G, size, seed)```: function to extract small subnetworks from the graph using BFS produced by ```load_graph```
        - ```G```: graph from ```load_graph```
        - ```size```: number of nodes in subnetwork
        - ```seed```: seed value for randomly choosing the initial starting node of the subnetwork

## Evaluation
### Setup
```lp_utils.py```: python script to setup the experiments (also contains a lot of methods from the leaky experiments)
    - ```set_dp_multisets(vertices, edges, target)```: function to set up the multisets given a graph
### Experiments
Are done in ```leak_exp_dp.py```, takes in 4 command line arguments:
    - ```graph_seed```: the seed specifying the subgraph of interest
    - ```eps```: the epsilon to be used in privacy guarentees
    - ```delt```: delta
    - ```target```: the target max degree to be achieved
