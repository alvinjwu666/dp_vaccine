import sys
import multiset_multicover as mm
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import math
import networkx as nx
import time
import json
import os
from dp_utils import *

powerk=32

def generate_test_samples_file(filename, test_sample_size = 250, vertex_count = 1000, leak_probabilities=[]):
    
    if not os.path.isfile(filename):
        
        test_samples = {}
        
        for leak in leak_probabilities:
            test_samples[leak] = [list(sample) for sample in generate_samples(vertex_count, test_sample_size, leak)]
        
        with open(filename, 'w') as f:
            json.dump(test_samples, f)
    
    with open(filename, 'r') as f:
        file_data = json.load(f)
    
    return file_data

def generate_infection_sets_file(filename, G, initial_infection_size, infection_trials):
    
    if not os.path.isfile(filename):
        
        initial_infection_sets = [list(infection_set) for infection_set in generate_infection_sets(G, initial_infection_size, infection_trials)]
        
        with open(filename, 'w') as f:
            json.dump(initial_infection_sets, f)
    
    with open(filename, 'r') as f:
        file_data = json.load(f)
    
    return file_data

def generate_samples_file(filename, sample_size = 250, vertex_count = 1000, trials = 15, leak_probabilities=[]):
    
    if not os.path.isfile(filename):
        
        samples = {}
        
        for t in range(trials):
        
            samples[t] = {}

            for leak in leak_probabilities:
                samples[t][leak] = [list(sample) for sample in generate_samples(vertex_count, sample_size, leak)]

        with open(filename, 'w') as f:
            json.dump(samples, f)
    
    with open(filename, 'r') as f:
        file_data = json.load(f)
    
    return file_data

def generate_samples(number_of_vertices, number_of_samples, leak_probability):
    samples = np.zeros((number_of_samples, number_of_vertices))
    for i in range(number_of_samples):
        samples[i, :] = np.random.choice([0, 1], size=number_of_vertices, p=[leak_probability, 1-leak_probability])
    return samples

# reset the vertices to be the consequtive integers
def graph_isomorphism(vertices, edges):
    i = 0
    nv = range(len(vertices))
    dic = {k:v for k,v in zip(vertices, nv)}
    ne = [(dic[e[0]], dic[e[1]]) for e in edges]
    return nv, dic, ne


def set_lp_constraints(vertices, edges, samples):
    try:
        start_lp_setup = time.time()
        
        # Create a new model
        m = gp.Model()

        # Create vertex-selection variables
        x = m.addVars(len(vertices), lb=[0]*len(vertices), ub=[1]*len(vertices))
        print("Created X Variables")

        # Create edge-coverage indicator variables
        # 0 = not covered, 1 = covered
        y = m.addMVar((len(samples), len(edges)), lb=0, ub=1)
        #y = m.addMVar(len(edges), lb=0, ub=len(samples))
        print("Created Y Variables")

        # Add budget constraint
        # m.addConstr(x.sum() <= budget, "budget_constraint")
        # print("Set Budget Constraint")

        # Add coverage constraint of edges
        vertex_edge_dict = {v:set() for v in range(len(vertices))}
        for i, edge in enumerate(edges):

            v1 = edge[0]
            v2 = edge[1]
            
            vertex_edge_dict[v1].add(i)
            vertex_edge_dict[v2].add(i)
            
            #m.addConstr(gp.quicksum(sample[v1]*x[v1] + sample[v2]*x[v2] for sample in samples) >= y[i])
            
            for j in range(len(samples)):
                m.addConstr(samples[j][v1]*x[v1] + samples[j][v2]*x[v2] >= y[j][i])
        print("Set Coverage Constraint")
        
        end_lp_setup = time.time()
        print("Total Setup Time:", end_lp_setup-start_lp_setup)
        
        return (m, x, y), vertex_edge_dict
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

def set_greedy_multisets(vertices, edges, samples):
    start_mm_setup = time.time()
    # Create a new model, initializes the number of multisets needed (number of vertices)
    m = mm.GreedyCoverInstance(len(vertices) * len(samples))
    vert, dic, edg = graph_isomorphism(vertices, edges)

    # Budget constraint is added when the function is called so none is added right here

    # Adds the multisets to the problem
    # Do NOT do anything to the dict until find out what it does
    vertex_edge_dict = {v:set() for v in range(len(vertices))}
    # this set is the multisets for ONE sample assuming non-leak need to expand to all samples
    # Note that the first element is the vertex itself, which should be covered for d_v times in the multiset
    vertex_adj_list = [[v] for v in vertices]
    for i, edge in enumerate(edg):

        v1 = edge[0]
        v2 = edge[1]
        
        vertex_edge_dict[v1].add(i)
        vertex_edge_dict[v2].add(i)

        vertex_adj_list[v1].append(v2)
        vertex_adj_list[v2].append(v1)
        
    vertex_degree = [len(vert) for vert in vertex_adj_list]
    # For each vertex, sample pair, add a multiset if and only if the vertex is not leaky in the sample
    max_degree = 0
    for se in vertex_adj_list:
        v = se[0]
        i = 0
        non_leaks = 0
        mset = []
        dummy_count = [1] * len(se)
        dummy_count[0] = len(dummy_count) - 1
        if max_degree < dummy_count[0]:
            max_degree = dummy_count[0]
        for lis in samples:
            if lis[v] == 1:
                mset = mset + [ver + i * len(vertices) for ver in se]
                non_leaks = non_leaks + 1
            i = i + 1
        mcounts = dummy_count * non_leaks
        m.add_multiset(mset, mcounts)
    print("Set Coverage Constraint")
    
    end_mm_setup = time.time()
    print("Total Mutliset Setup Time:", end_mm_setup-start_mm_setup)
    
    return m, vertex_edge_dict, max_degree, vertex_degree

def set_dp_multisets(vertices, edges, target):
    start_mm_setup = time.time()
    vertex_edge_dict = {v:set() for v in vertices}
    vertex_adj_list = [[] for v in vertices]
    ans = MultiSetDP(42)
    for i, edge in enumerate(edges):

        v1 = edge[0]
        v2 = edge[1]
        
        vertex_edge_dict[v1].add(i)
        vertex_edge_dict[v2].add(i)

        vertex_adj_list[v1].append(v2)
        vertex_adj_list[v2].append(v1)
        ans.addToSet(v1, v2, 1)
        ans.addToSet(v2, v1, 1)
    vertex_degree = {vert: len(vertex_adj_list[vert]) for vert in vertices}
    for i in vertex_degree:
        ans.addToSet(i, i, vertex_degree[i])
        ans.addItem(i, max(0, vertex_degree[i] - target))
    return ans, vertex_degree
    
def max_deg(vertices, edges):
    vertex_adj_list = {v:set() for v in vertices}
    for i, edge in enumerate(edges):
        v1 = edge[0]
        v2 = edge[1]
        vertex_adj_list[v1].add(v2)
        vertex_adj_list[v2].add(v1)
    ans = 0
    for v in vertex_adj_list:
        if ans < len(vertex_adj_list[v]):
            ans = len(vertex_adj_list[v])
    return ans

def set_spectral_mat(vertices, edges):
    start_mm_setup = time.time()
    # Create a new model, initializes the number of multisets needed (number of vertices)
    vert, dic, edg = graph_isomorphism(vertices, edges)

    # Budget constraint is added when the function is called so none is added right here

    # Adds the multisets to the problem
    # Do NOT do anything to the dict until find out what it does
    vertex_edge_dict = {v:set() for v in range(len(vertices))}
    # this set is the multisets for ONE sample assuming non-leak need to expand to all samples
    # Note that the first element is the vertex itself, which should be covered for d_v times in the multiset
    vertex_adj_list = [[v] for v in vertices]
    vertex_adj_mat = np.zeros((len(vert), len(vert)), np.longlong)
    for i, edge in enumerate(edg):

        v1 = edge[0]
        v2 = edge[1]
        
        vertex_edge_dict[v1].add(i)
        vertex_edge_dict[v2].add(i)

        vertex_adj_list[v1].append(v2)
        vertex_adj_list[v2].append(v1)
        vertex_adj_mat[v1, v2] = 1
        vertex_adj_mat[v2, v1] = 1
        
    vertex_degree = [len(vert) for vert in vertex_adj_list]
    
    end_mm_setup = time.time()
    print("Total spectral Setup Time:", end_mm_setup-start_mm_setup)
    
    return vertex_edge_dict, vertex_degree, vertex_adj_mat

def set_lp_budget(lp, budget):
    
    m, x, y = lp
    
    # Add budget constraint
    m.addConstr(x.sum() <= budget, "budget_constraint")
    print(f"Set Budget Constraint to {budget}")
    
    m.update()
    return m, x, y
    
def set_lp_objective(lp, vertex_edge_dict, sample_size, objective="avg_degree"):
    
    m, x, y = lp
    
    if objective=="avg_degree":
        # Maximize the number of edges covered
        m.setObjective(y.sum(), GRB.MAXIMIZE)
    elif objective=="max_degree":
        num_vertices = len(vertex_edge_dict.keys())
        z = m.addMVar((sample_size, num_vertices), lb=0, ub=len(vertex_edge_dict.keys()))
        z_max = m.addVars(sample_size, lb=[0]*sample_size, ub=[len(vertex_edge_dict.keys())]*sample_size)
        
        for sample in range(sample_size):
            for i in range(num_vertices):
                m.addConstr(len(vertex_edge_dict[i]) - gp.quicksum(y[sample][e] for e in vertex_edge_dict[i]) == z[sample][i])
            # Calculate maximum degree per sample
            m.addConstr(z_max[sample] == gp.max_(z[sample].tolist()))
        
        # Minimize the maximum number of uncovered edges
        m.setObjective(z_max.sum(), GRB.MINIMIZE)
        
    m.update()
    return m, x, y

def reset_lp(lp, keep_budget=True):
    m, x, y = lp
    m.reset()
    
    if not keep_budget:
        constr = m.getConstrByName("budget_constraint")
        if constr:
            m.remove(constr)
    
    return (m, x, y)

def get_lp_solution(lp):
    m, x, y = lp
    
    start_lp_solution = time.time()
        
    # Optimize model
    m.optimize()
    print('Obj: %g' % m.ObjVal)

    end_lp_solution = time.time()
    print("Time to Solve LP:", end_lp_solution-start_lp_solution)

    return {"given_solution": list([float(x[i].X) for i in range(len(x.keys()))]),
            "lp_objective": m.ObjVal}

def get_greedy_solution(mm, vertex_degree, max_degree, len_samples, budget):
    start_greedy_solution = time.time()

    l = 0
    r = max_degree
    cov_req = vertex_degree * len_samples


    solution = mm.cover(0)
    while l < r:
        m = (l + r) // 2
        coverage_req = [max(0, vd - m) for vd in cov_req]
        solution = mm.cover(coverage_req)
        #print(solution)
        if len(mm.multisets_incomplete_cover_) == 0 and len(solution) <= budget:
            r = m
        else:
            l = m + 1
    end_greedy_solution = time.time()
    print("time to solve greedy: ", end_greedy_solution - start_greedy_solution)
    return r, solution

def get_spectral_solution(adj_mat, samples, budget):
    start_greedy_solution = time.time()

    ans = []
    for i in range(0, budget):
        goodness = np.zeros(adj_mat.shape[0])
        for samp in samples:
            nadj = np.copy(adj_mat)
            for an in ans:
                if samp[an] == 1:
                    nadj[an] = 0
                    nadj[:, an] = 0
            eigvals, eigvects = np.linalg.eigh(nadj)
            neigvals = np.diag([e ** powerk for e in eigvals])
            nadj = np.real(np.matmul(eigvects, np.matmul(neigvals, np.matrix(eigvects).getH())))
            for ind in range(adj_mat.shape[0]):
                if samp[ind] == 1:
                    goodness[ind] += nadj[ind, ind]
        ans.append(np.argmax(goodness))
    rans = [0] * (adj_mat.shape[0])
    for an in ans:
        rans[an] = 1
    
    return rans,ans


def subsets_eq_k(A,K):
    subsets = []
    N = len(A)

    # iterate over subsets of size K
    mask = (1<<K)-1     # 2^K - 1 is always a number having exactly K 1 bits
    masks = []
    while mask < (1<<N):
        subset = []
        for n in range(N):
            if ((mask>>n)&1) == 1:
                subset.append(A[n])
 
        subsets.append(subset)
        masks.append(mask)
 
        # catch special case
        if mask == 0:
            break
 
        # determine next mask with Gosper's hack
        a = mask & -mask                # determine rightmost 1 bit
        b = mask + a                    # determine carry bit
        mask = int(((mask^b)>>2)/a) | b # produce block of ones that begins at the least-significant bit

    return subsets,masks

def get_exact_spec(adj_mat, budget, leak_prob):
    start_greedy_solution = time.time()
    n = adj_mat.shape[0]
    goodness = []
    subs = [subsets_eq_k(range(n), b) for b in range(budget+1)]
    fulsubs, fulmasks = subsets_eq_k(range(n), budget)
    actgoodness = [0 for f in fulsubs]
    b = 0
    for bud in subs:
        bs = []
        nadj = np.copy(adj_mat)
        subs, mat = bud
        fctr = 0
        for va in subs:
            for v in va:
                nadj[va] = 0
                nadj[:,va] = 0
            eigvals, eigvects = np.linalg.eigh(nadj)
            bs.append(np.max(np.real(eigvals)))
            ctr = 0
            for vacc in fulmasks:
                if (vacc | mat[fctr]) == vacc:
                    actgoodness[ctr] = actgoodness[ctr] + np.max(np.real(eigvals)) * ((1 - leak_prob) ** b) * (leak_prob ** (budget - b))
                ctr = ctr + 1
            fctr = fctr + 1
        goodness.append(bs)
    ans = fulsubs[np.argmax(actgoodness)]
    rans = [0] * (adj_mat.shape[0])
    for an in ans:
        rans[an] = 1
    return rans,ans


def lp_max_round(x):
    
    cover = set()
    for i in range(math.ceil(2 * math.log2(len(x)))):
        for j, x_var in enumerate(x):
            if random.random() <= x_var:
                cover.add(j)
    return cover


# TODO: Round differently for max v. avg degree LP
def lp_round(epsilon, x):
    
    lmbda = 2*(1-epsilon)
    
    cover = set()
    for i, x_var in enumerate(x):
        if (x_var) >= 1/lmbda:
            cover.add(i)
        else:
            # round to 1 with probability lmbda * x
            if random.random() <= lmbda*(x_var):
                cover.add(i)
    return cover

def mask_to_cover(x):
    cover = set()
    for i, x_var in enumerate(x):
        if x_var >= 0.1:
            cover.add(i)
    return cover

def calculate_avg_degree(G):
    return 2*len(G.edges)/len(G.nodes)

def calculate_max_degree(G):
    return max(G.degree(), key=lambda x:x[1])[1]


"""
Returns the average degree of the resulting vaccination graph
"""
def evaluate_avg_degree(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    samples = np.array(samples)
    
    total_edges = len(G.edges)
    removed_edges = 0
    for s in samples:
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges += len(G.edges(successful_vaccinations))
    return 2*(total_edges - (removed_edges/len(samples)))/len(G.nodes)

"""
Returns the max degree of the resulting vaccination graph
"""
def evaluate_max_degree(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    samples = np.array(samples)
    
    total_max = 0
    for s in samples:
        
        G_copy = G.copy()
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges = G.edges(successful_vaccinations)
        G_copy.remove_edges_from(removed_edges)
        
        max_degree = max(G_copy.degree(), key=lambda x:x[1])[1]
        total_max += max_degree
        
    return total_max/len(samples)

def generate_infection_sets(G, infection_set_size, number_of_sets):
    
    infection_set_list = []
    for trial in range(number_of_sets):
        infected_set = set([])
        while len(infected_set)<infection_set_size:
            chosen = random.randint(0, len(G.nodes)-1)
            infected_set.add(chosen)
        infection_set_list.append(infected_set)

    return infection_set_list

def evaluate_infection_spread(G, vaccinated_vertices, samples, infection_set_trials = [set([])], transmission_probability=1, trials=10):
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    samples = np.array(samples)
    
    infected_size_list = []
    
    for s in samples:
        
        if len(vaccinated_vertices)>0:
            G_copy = G.copy()
            successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
            removed_edges = G.edges(successful_vaccinations)
            G_copy.remove_edges_from(removed_edges)
        else:
            G_copy = G
        
        for i in range(trials):
            
            for initial_infection in infection_set_trials:
                
                infected = set(initial_infection)
                queue = list(initial_infection)
                
                while len(queue)>0:
                    
                    infected_v = queue.pop(0)
                    infected.add(infected_v)

                    for edge in G_copy.edges([infected_v]):

                        v1, v2 = edge

                        if v1 not in infected and random.random() <= transmission_probability:
                            queue.append(v1)
                        if v2 not in infected and random.random() <= transmission_probability:
                            queue.append(v2)
            
                infected_size_list.append(len(infected))
    
    return sum(infected_size_list)/len(infected_size_list)

"""
Returns the spectral radius of the resulting vaccination graph
"""
def evaluate_spectral_radius(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = np.array(list(vaccinated_vertices))
    vertices = list(G.nodes)
    samples = np.array(samples)
    
    total_spectral_radius = 0
    for s in samples:
        G_copy = G.copy()
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges = G.edges(successful_vaccinations)
        G_copy.remove_edges_from(removed_edges)
        spectral_radius = max(nx.adjacency_spectrum(G_copy))
        total_spectral_radius += spectral_radius.real
    return total_spectral_radius/len(samples)

def calculate_spectral_radius(G):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    return np.max(np.real(np.linalg.eigvals(G)))
    

def evaluate_spectral_rad(adj_mat, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    ans = 0
    for lis in samples:
        adj = np.copy(adj_mat)
        for i in vaccinated_vertices:
            if lis[i] == 1:
                adj[i] = 0
                adj[:, i] = 0
        ans += np.max(np.real(np.linalg.eigvals(adj)))
    return ans/len(samples)
# ---------------   Define variables    ------------------ #

# B is the budget on the number of vertices that can be vaccinated
'''budget = 5
num_vertices = 50
edge_connectivity = 0.1
sample_size = 100
leak_probability = 0.2
epsilon = 0.5
test_sample_size = 1000

num_vertices = int(sys.argv[1]) if len(sys.argv)>1 else num_vertices
edge_connectivity = float(sys.argv[2]) if len(sys.argv)>2 else edge_connectivity
sample_size = int(sys.argv[3]) if len(sys.argv)>3 else sample_size
leak_probability = float(sys.argv[4]) if len(sys.argv)>4 else leak_probability

# ---------------   Generate Samples    ------------------ #

start_setup = time.time()

# Generate Erdos Renyi graph
G = nx.erdos_renyi_graph(num_vertices, edge_connectivity)

vertices = G.nodes
print("Number of Vertices:", len(vertices))

# List of edges
edges = G.edges
print("Number of Edges:", len(edges))

# Generate samples for leaky vaccine on vertices (1 for successful vaccination, 0 for leak)
samples = generate_samples(len(vertices), sample_size, leak_probability)
print("Number of Samples:", len(samples))

end_setup = time.time()
print("Time to Setup Samples:", end_setup-start_setup)

# --------------------   Solve LP    --------------------- #

start_lp = time.time()
lp, vertex_edge_dict = set_lp_constraints(vertices, edges, samples, budget)
lp = set_lp_objective(lp, vertex_edge_dict)
lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
end_lp = time.time()

vaccinated_vertices = lp_solution["rounded_solution"]

print("Total LP Time:", end_lp - start_lp)

# --------------------   Evaluate LP    --------------------- #

new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
#spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
print("Simulated Average Degree:", avg_degree_obj)
print("Simulated Max Degree:", max_degree_obj)
#print("Simulated Spectral Radius:", spectral_radius_obj)

avg_degree_run = {"vertices": list(vertices),
               "edges": list(edges),
               "budget": budget,
               "given_solution": lp_solution["given_solution"],
               "rounded_solution": lp_solution["rounded_solution"],
               "lp_objective": lp_solution["lp_objective"],
               "evaluated_avg_degree_objective": avg_degree_obj,
               "evaluated_max_degree_objective": max_degree_obj
              }

start_lp = time.time()
lp = reset_lp(lp)
lp = set_lp_objective(lp, vertex_edge_dict, sample_size, objective="max_degree")
lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
end_lp = time.time()

vaccinated_vertices = lp_solution["rounded_solution"]

print("Total LP Time:", end_lp - start_lp)

new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
print("Simulated Average Degree:", avg_degree_obj)
print("Simulated Max Degree:", max_degree_obj)
print("Simulated Spectral Radius:", spectral_radius_obj)

max_degree_run = {"vertices": list(vertices),
               "edges": list(edges),
               "budget": budget,
               "given_solution": lp_solution["given_solution"],
               "rounded_solution": lp_solution["rounded_solution"],
               "lp_objective": lp_solution["lp_objective"],
               "evaluated_avg_degree_objective": avg_degree_obj,
               "evaluated_max_degree_objective": max_degree_obj
              }

#with open("avg_max_degree_run_test.json", 'w') as f:
#   json.dump([avg_degree_run, max_degree_run], f)'''