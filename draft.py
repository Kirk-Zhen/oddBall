import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
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


path = 'data/sample.txt'
G = load_data(path)
featureDict = get_feature(G)

# print(featureDict)

W = []
E = []
for key in featureDict.keys():
    W.append(featureDict[key][2])
    E.append(featureDict[key][1])
#W=CE^β => log on both sides => logW=logC+βlogE
#regard as y=b+wx to do linear regression
#here the base of log is 2
y_train = np.array(np.log2(W)).reshape(-1, 1)
x_train = np.array(np.log2(E)).reshape(-1, 1)


model = LinearRegression()
model.fit(x_train, y_train)
# w = model.coef_[0][0]
# b = model.intercept_[0]
# C = 2**b
# beta = w
# outlineScoreDict = {}
# for key in featureDict.keys():
#     yi = featureDict[key][2]
#     xi = featureDict[key][1]
#     outlineScore = (max(yi, C*(xi**beta))/min(yi, C*(xi**beta)))*np.log(abs(yi-C*(xi**beta))+1)
#     outlineScoreDict[key] = outlineScore

print(model.coef_)