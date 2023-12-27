import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path):
    data = np.loadtxt(path).astype('int32')
    G = nx.Graph()
    for edge in data:
        G.add_edge(edge[0], edge[1], weight=edge[2])
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


def outlierness_score(xi, yi, C, theta):
    return (max(yi, C*(xi**theta))/min(yi, C*(xi**theta))) * np.log(abs(yi-C*(xi**theta))+1)
    # a_max = np.max(np.concatenate([yi, C*(xi**theta)], axis=1), axis=1).reshape(-1,1)
    # a_min = np.min(np.concatenate([yi, C*(xi**theta)], axis=1), axis=1).reshape(-1,1)
    # a_log = np.log(abs(yi-C*(xi**theta))+1)
    # return ( a_max / a_min ) * a_log

# path = 'data/sample.txt'
# G = load_data(path)
# featureDict = get_feature(G)

path = 'data/sample.csv'
G = load_csv_data(path)
featureDict = get_feature(G)


N = [featureDict[node][0] for node in featureDict.keys()]
E = [featureDict[node][1] for node in featureDict.keys()]

#E=CN^α => log on both sides => logE=logC+αlogN
#regard as y=b+wx to do linear regression
#here the base of log is 2
y_train = np.array(np.log2(E)).reshape(-1, 1)
x_train = np.array(np.log2(N)).reshape(-1, 1) # the order in x_train and y_train is the same as which in featureDict.keys() now

#prepare data for LOF
xAndyForLOF = np.concatenate([x_train, y_train], axis=1)

model = LinearRegression()
model.fit(x_train, y_train)
w = model.coef_[0][0]
b = model.intercept_[0]
C = 2**b
alpha = w
print('alpha={}'.format(alpha))

#LOF algorithm
clf = LocalOutlierFactor(n_neighbors=10)
clf.fit(xAndyForLOF)
LOFScoreArray = -clf.negative_outlier_factor_

outScoreDict = {}
count = 0   #Used to take LOFScore in sequence from LOFScoreArray

#get the maximum outLine
maxOutLine = 0
for node in featureDict.keys():
    yi = featureDict[node][1]
    xi = featureDict[node][0]
    outlineScore = outlierness_score(xi, yi, C, alpha)
    if outlineScore > maxOutLine:
        maxOutLine = outlineScore

print('maxOutLine={}'.format(maxOutLine))

#get the maximum LOFScore
maxLOFScore = 0
for ite in range(len(N)):
    if LOFScoreArray[ite] > maxLOFScore:
        maxLOFScore = LOFScoreArray[ite]

print('maxLOFScore={}'.format(maxLOFScore))

for node in featureDict.keys():
    yi = featureDict[node][1]
    xi = featureDict[node][0]
    outlineScore = outlierness_score(xi, yi, C, alpha)
    LOFScore = LOFScoreArray[count]
    count += 1
    outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
    outScoreDict[node] = outScore
print(outScoreDict)