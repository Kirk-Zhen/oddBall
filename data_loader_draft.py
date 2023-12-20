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


path = 'data/enron_cnt.csv'
G =  load_csv_data(path, directed=True)

