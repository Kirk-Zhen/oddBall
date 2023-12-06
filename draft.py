import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.loadtxt(path).astype('int32')
    G = nx.Graph()
    for edge in data:
        G.add_edge(edge[0], edge[1], weight=edge[2])
    return G

def get_feature(G):
    '''
    feature dictionary which format is { node i's id : [Ni, Ei, Wi, λw,i] }
    Ni:     number of neighbors (degress) of ego i
    Ei:     number of edges in egonet i
    Wi:     total weight of egonet i
    λw,i:   principal egenvalue of the weighted adjacency matrix of egonet i
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
        #the set of node i's neighbor
        iNeighbor = list(G.neighbors(node))
        #the number of edges in egonet i
        Ei = 0
        #sum of weights in egonet i
        Wi = 0
        #the principal eigenvalue(the maximum eigenvalue with abs) of egonet i's weighted adjacency matrix
        Lambda_w_i = 0
        Ei += Ni
        egonet = nx.Graph()
        for nei in iNeighbor:
            Wi += G[nei][node]['weight']
            egonet.add_edge(node, nei, weight=G[nei][node]['weight'])
        iNeighborLen = len(iNeighbor)
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
        Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        featureDict[node].append(Ei)
        featureDict[node].append(Wi)
        featureDict[node].append(Lambda_w_i)
    return featureDict


path = 'data/sample.txt'
G = load_data(path)
fea = get_feature(G)

print(fea)