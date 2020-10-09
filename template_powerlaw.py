import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import random


####################################################################################
#
# Test functions that might be helpful while implementing your models
#
####################################################################################

def get_alpha(G, k_min):
    ''' Returns the best fitting power law exponent for the in degree distribution '''
    return get_alpha_from_data([d for n, d in G.in_degree()], k_min)

def get_alpha_from_data(data, min_val):
    ''' Returns the best fitting power law exponent for the data '''
    data_sorted = sorted(data)
    min_idx = np.searchsorted(data_sorted, min_val)
    data2 = data_sorted[min_idx:]
    denom = np.log(min_val - 1 / 2)
    log_data = [np.log(x) - denom for x in data2]
    alpha = 1 + len(log_data) / sum(log_data)

    print('fit alpha on # points', len(data2))

    return alpha

def plot_degrees(graph, scale='log', colour='#40a6d1', alpha=.8):
    '''Plots the log-log degree distribution of the graph'''
    plot.close()
    num_nodes = graph.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in graph.nodes():
        if graph.degree(n) > max_degree:
            max_degree = graph.degree(n)
    # X-axis and y-axis values
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree+1):
        x.append(i)
        y_tmp.append(0)
        for n in graph.nodes():
            if graph.degree(n) == i:
                y_tmp[i] += 1
        y = [i/num_nodes for i in y_tmp]
    # Plot the graph
    deg, = plot.plot(x, y,label='Degree distribution',linewidth=0, marker='o',markersize=8, color=colour, alpha=alpha)
    # Check for the lin / log parameter and set axes scale
    if scale == 'log':
        plot.xscale('log')
        plot.yscale('log')
        plot.title('Degree distribution (log-log scale)')
    else:
        plot.title('Degree distribution (linear scale)')


    plot.ylabel('P(k)')
    plot.xlabel('k')
    plot.show()

##########################################################################
#
# Functions that your final implementation will call
#
##########################################################################


def test_vertex_copy():
    ''' Runs the vertex copy model and plots the result '''
    G = vertex_copy_model(1, 1/2, 100)
    nx.draw(G, with_labels=True)
    plot.show()


def test_lfna():
    ''' Runs the lnfa model and plots the result '''
    G = lnfa_model(1, 1/2, 100)
    nx.draw(G, with_labels=True)
    plot.show()


def run_vertex_copy(out_dir_name):
    ''' Creates the three vertex copy model networks and exports them in gexf format'''
    c = 1
    num = 100
    outfile_template = out_dir_name + '/vertex-copy-{0}-{1}-{2}.gexf'

    for gamma in [1/9,1/2,8/9]:
        G = vertex_copy_model(c, gamma, num)

        outfile = outfile_template.format(num,c,gamma)
        print('vertex copy model writing to ' + outfile)
        nx.write_gexf(G, outfile)


def run_lnfa(out_dir_name):
    ''' Creates the three LNFA model networks and exports them in gexf format'''
    c = 1
    num=100
    outfile_template = out_dir_name + '/lnfa-{0}-{1}-{2}.gexf'


    for sigma in [1/4, 2, 16]:
        G = lnfa_model(c, sigma, num)

        outfile = outfile_template.format(num, c, sigma)
        print('lnfa model writing to ' + outfile)
        nx.write_gexf(G, outfile)

##########################################################################
#
# Helper functions for your implementation
#
##########################################################################

def get_seed_multidigraph(c):
    ''' Returns a complete digraph on c+1 vertices'''
    graph = nx.gnp_random_graph(c + 1, p=1, directed=True)
    seed_graph = nx.MultiDiGraph()
    seed_graph.add_nodes_from(graph)
    seed_graph.add_edges_from(graph.edges())

    return seed_graph


def get_fitness(sigma):
    ''' Samples from the standard lognormal distribution with std dev = sigma'''
    return np.random.lognormal(0, sigma)


def get_random_node(G):
    ''' Returns a random node from the graph G '''
    return random.choice(list(G.nodes))

def get_random_index_by_fitness(fitness_list):
    ''' Returns a random index chosen according to the distribution corresponding to the fitness list. '''

    fitness_sum = sum(fitness_list)
    fitness_dist = [ fit/ fitness_sum for fit in fitness_list]
    node_list = range(len(fitness_dist))

    return random.choices(node_list, weights=fitness_dist, k=1)[0]

##########################################################################
#
# Your implementation starts here
#
##########################################################################



def vertex_copy_model(c, gamma, num_steps):
    """
    Returns a directed multigraph on c + 1 + num_steps nodes created by the vertex copy model.

    :param c: The number of links to create for each new node.
    :param gamma: The probability of attaching preferentially
    :param num_steps: The number of nodes to add to the initial seed graph
    :return: A directed multigraph with c + 1+ num_steps nodes.


    The vertex copy model works as follows:
    - Start with a complete directed graph on c+1 nodes.
    - Add num_steps nodes, one at a time. For each new_node, create c links as follows
       Pick an existing node x (not equal to new_node).
       For each out-link of x:
          copy the out-link with probability 0 < gamma < 1,
          otherwise link to a random vertex.

    """

    G = get_seed_multidigraph(c)
    
    prev_node = c

    for _ in range(num_steps):  # Add num_steps nodes
        new_node = prev_node + 1
        G.add_node(new_node)    # Add the new node
        x = random.randint(0, prev_node)   # Pick an existing node x
        for n in G.out_edges(x):    # Loop through c out-edges
            probability = random.random()
            out_n = n[1] 
            if probability < gamma: # Copy the citation
                G.add_edge(new_node, out_n)
            else:   # Cite paper at random
                rand_n = random.randint(0, prev_node)
                G.add_edge(new_node, rand_n)
        prev_node += 1

    ############
    #
    # You provide the implementation that adds num_steps vertices
    # according to the Vertex Copy model
    #
    ###########

    return G





def lnfa_model(c, sigma, num_steps):
    """
    Returns a directed multigraph on c + 1 + num_steps nodes created by the vertex copy model.

    :param c: The number of links to create for each new node.
    :param sigma: The variance for the fitness distribution
    :param num_steps: The number of nodes to add to the initial seed graph
    :return: A directed multigraph with c + 1+ num_steps nodes.

    The LNFA model works as follows:
    - Start with a complete directed graph on c+1 nodes.
    - Create fitness_list of c+1 fitness values corresponding to these nodes.
    - Add num_steps nodes, one at a time. For each one create c links as follows:
        Create c links from the new node to existing nodes chosen according to fitness.
        Assign a fitness to the new node and add it to the fitness_list.


    """

    G = get_seed_multidigraph(c)

    ############
    #
    # You provide the implementation that adds num_steps vertices
    # according to the LNFA model
    #
    ###########


    return G


###############################################################################
#
# MAIN EXECUTION
#
###############################################################################


#### replace this code with your own code to test your models

test_vertex_copy()

#test_lfna()

#### once your code is working, use these two methods to generate the 6 networks
#### that will be part of your written report
#run_vertex_copy('/output_dir_name')
#run_lnfa('/output_dir_name')

