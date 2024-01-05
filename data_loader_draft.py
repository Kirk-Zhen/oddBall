import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd

def load_csv_data(path, directed:bool=False):
    df = pd.read_csv(path)
    G = nx.Graph() if not directed else nx.DiGraph()
    for index, edge in df.iterrows():
        G.add_edge(edge['u'], edge['v'], weight=edge['weight'])
    return G


# path = 'data/enron_cnt.csv'
path = 'data/sample.csv'
G =  load_csv_data(path, directed=False)

print(G.nodes())

featureDict = {}
nodelist = list(G.nodes)
for node in nodelist:
    #the number of node i's neighbor
    Ni = G.degree(node)

    # build egonet
    egonet = nx.ego_graph(G, node)
    Wi = sum([egonet[u][v]['weight'] for u,v in egonet.edges()])
    Ei = len(egonet.edges())

    # build adjacency matrix
    egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
    eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
    eigenvalue.sort()

    Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
    featureDict[node] = [Ni, Ei, Wi, Lambda_w_i]
print(featureDict)