import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd

def load_csv_data(path):
    df = pd.read_csv(path)
    G = nx.Graph()
    for index, edge in df.iterrows():
        G.add_edge(edge['u'], edge['v'], weight=edge['weight'])
    return G



# def data_cleaning_enron():

email_data = pd.read_csv('data/emails.csv')
print(email_data.head())