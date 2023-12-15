
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

def load_csv_data(path):
    df = pd.read_csv(path)
    G = nx.Graph()
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
        featureDict[node].append(Ni)

        # initialization of features:
        Ei, Wi, Lambda_w_i = 0, 0, 0
        Ei += Ni

        iNeighbor = list(G.neighbors(node)) # the set of node i's neighbor
        egonet = nx.Graph()
        for nei in iNeighbor:
            Wi += G[nei][node]['weight']
            egonet.add_edge(node, nei, weight=G[nei][node]['weight'])
        iNeighborLen = len(iNeighbor) # number of i's neighbor
        for it1 in range(iNeighborLen):
            for it2 in range(it1+1, iNeighborLen):
                #if it1 in it2's neighbor list
                if iNeighbor[it1] in list(G.neighbors(iNeighbor[it2])):
                    Ei += 1
                    Wi += G[iNeighbor[it1]][iNeighbor[it2]]['weight']
                    egonet.add_edge(iNeighbor[it1], iNeighbor[it2], weight=G[iNeighbor[it1]][iNeighbor[it2]]['weight'])
        egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
        eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
        eigenvalue.sort()

        # In case that the matrix is not positive definite, we need to consider the negative lambda
        # Positive Definite: 
        #   M is positive definite if z*Mz is positive for every nonzero complex column vector z
        #   where z* is the conjugate transpose of z
        Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        featureDict[node].append(Ei)
        featureDict[node].append(Wi)
        featureDict[node].append(Lambda_w_i)
    return featureDict
