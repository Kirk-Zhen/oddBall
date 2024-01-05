
import numpy as np
import networkx as nx
import pandas as pd

#load data, a weighted undirected graph
def load_data(path):
    data = np.loadtxt(path).astype('int32')
    G = nx.Graph()
    for ite in data:
        G.add_edge(ite[0], ite[1], weight=ite[2])
    return G

def load_csv_data(path, directed:bool=False):
    df = pd.read_csv(path)
    G = nx.Graph() if not directed else nx.DiGraph()
    for index, edge in df.iterrows():
        G.add_edge(edge['u'], edge['v'], weight=edge['weight'])
    return G


def get_feature(G):
    '''
    feature dictionary which format is { node i's id : [Ni, Ei, Wi, λw,i] }
    Ni:     number of neighbors (degress) of ego i
    Ei:     number of edges in egonet i
    Wi:     total weight of egonet i
    λw,i:   the principal eigenvalue(the maximum eigenvalue with abs) of egonet i's weighted adjacency matrix
    '''
    featureDict = {}
    nodes = list(G.nodes)
    # iterate all nodes in the graph 
    featureDict = {}
    nodelist = list(G.nodes)
    for node in nodelist:
        featureDict[node] = []
        #the number of node i's neighbor
        Ni = G.degree(node)

        egonet = nx.ego_graph(G, node)
        Wi = sum([egonet[u][v]['weight'] for u,v in egonet.edges()])
        Ei = len(egonet.edges())

        egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
        eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
        eigenvalue.sort()

        # In case that the matrix is not positive definite, we need to consider the negative lambda
        # Positive Definite: 
        #   M is positive definite if z*Mz is positive for every nonzero complex column vector z
        #   where z* is the conjugate transpose of z
        Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        featureDict[node].append(Ni)
        featureDict[node].append(Ei)
        featureDict[node].append(Wi)
        featureDict[node].append(Lambda_w_i)
    return featureDict
