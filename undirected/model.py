

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
# feature dictionary which format is  { node i's id : [Ni, Ei, Wi, λw,i] }


def outlierness_score(xi, yi, C, theta):
    return (max(yi, C*(xi**theta))/min(yi, C*(xi**theta))) * np.log(abs(yi-C*(xi**theta))+1)
    # a_max = np.max(np.concatenate([yi, C*(xi**theta)], axis=1), axis=1).reshape(-1,1)
    # a_min = np.min(np.concatenate([yi, C*(xi**theta)], axis=1), axis=1).reshape(-1,1)
    # a_log = np.log(abs(yi-C*(xi**theta))+1)
    # return ( a_max / a_min ) * a_log
    

# Observation 1: EDPL
def star_or_clique(featureDict):
    E = [featureDict[node][1] for node in featureDict.keys()]
    N = [featureDict[node][0] for node in featureDict.keys()]
    
    #E=CN^α => log on both sides => logE=logC+αlogN
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.array(np.log2(E)).reshape(-1, 1)
    x_train = np.array(np.log2(N)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    alpha = w

    outlineScoreDict = {}
    for node in featureDict.keys():
        yi = featureDict[node][1]
        xi = featureDict[node][0]
        outlineScoreDict[node] =  outlierness_score(xi, yi, C, alpha)
    
    print(f'C:{2**b}, alpha={w}')
    return outlineScoreDict


# Observation 2: EWPL
def heavy_vicinity(featureDict):
    W = [featureDict[node][2] for node in featureDict.keys()]
    E = [featureDict[node][1] for node in featureDict.keys()]

    #W=CE^β => log on both sides => logW=logC+βlogE
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.array(np.log2(W)).reshape(-1, 1)
    x_train = np.array(np.log2(E)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    beta = w
    outlineScoreDict = {}
    for node in featureDict.keys():
        yi = featureDict[node][2]
        xi = featureDict[node][1]
        outlineScoreDict[node] =  outlierness_score(xi, yi, C, beta)
    
    print(f'C:{2**b}, beta={w}')
    return outlineScoreDict


# Observation 3: ELWPL
def dominant_edge(featureDict):
    Lambda_w_i = [featureDict[node][3] for node in featureDict.keys()]
    W = [featureDict[node][2] for node in featureDict.keys()]
    
    #λ=CW^γ => log on both sides => logλ=logC+γlogW
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.array(np.log2(Lambda_w_i)).reshape(len(Lambda_w_i), 1)
    x_train = np.array(np.log2(W)).reshape(len(W), 1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2 ** b
    gamma = w
    outlineScoreDict = {}
    for node in featureDict.keys():
        yi = featureDict[node][3]
        xi = featureDict[node][2]
        outlineScoreDict[node] =  outlierness_score(xi, yi, C, gamma)
    
    print(f'C:{2**b}, gamma={w}')
    return outlineScoreDict


# Observation 1: EDPL with LOF (Local Outlier Factor)
def star_or_clique_withLOF(featureDict):
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
    return outScoreDict



# Observation 2: EWPL with LOF (Local Outlier Factor)
def heavy_vicinity_withLOF(featureDict):

    W = [featureDict[node][2] for node in featureDict.keys()]
    E = [featureDict[node][1] for node in featureDict.keys()]

    #W=CE^β => log on both sides => logW=logC+βlogE
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.array(np.log2(W)).reshape(-1, 1)
    x_train = np.array(np.log2(E)).reshape(-1, 1) #the order in x_train and y_train is the same as which in featureDict.keys() now


    #prepare data for LOF
    xAndyForLOF = np.concatenate([x_train, y_train], axis=1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    beta = w
    print('beta={}'.format(beta))

    #LOF algorithm
    clf = LocalOutlierFactor(n_neighbors=10)
    clf.fit(xAndyForLOF)
    LOFScoreArray = -clf.negative_outlier_factor_

    outScoreDict = {}
    count = 0   #Used to take LOFScore in sequence from LOFScoreArray

    #get the maximum outLine
    maxOutLine = 0
    for node in featureDict.keys():
        yi = featureDict[node][2]
        xi = featureDict[node][1]
        outlineScore =  outlierness_score(xi, yi, C, beta)
        if outlineScore > maxOutLine:
            maxOutLine = outlineScore

    print('maxOutLine={}'.format(maxOutLine))

    #get the maximum LOFScore
    maxLOFScore = 0
    for ite in range(len(W)):
        if LOFScoreArray[ite] > maxLOFScore:
            maxLOFScore = LOFScoreArray[ite]

    print('maxLOFScore={}'.format(maxLOFScore))

    for node in featureDict.keys():
        yi = featureDict[node][2]
        xi = featureDict[node][1]
        outlineScore = outlierness_score(xi, yi, C, beta)
        LOFScore = LOFScoreArray[count]
        count += 1
        outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
        outScoreDict[node] = outScore
    return outScoreDict


# Observation 3: ELWPL with LOF (Local Outlier Factor)
def dominant_edge_withLOF(featureDict):
    Lambda_w_i = [featureDict[node][3] for node in featureDict.keys()]
    W = [featureDict[node][2] for node in featureDict.keys()]

    #λ=CW^γ => log on both sides => logλ=logC+γlogW
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.array(np.log2(Lambda_w_i)).reshape(-1, 1)
    x_train = np.array(np.log2(W)).reshape(-1, 1) #the order in x_train and y_train is the same as which in featureDict.keys() now


    #prepare data for LOF
    xAndyForLOF = np.concatenate([x_train, y_train], axis=1)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    gamma = w
    print('gamma={}'.format(gamma))

    #LOF algorithm
    clf = LocalOutlierFactor(n_neighbors=10)
    clf.fit(xAndyForLOF)
    LOFScoreArray = -clf.negative_outlier_factor_

    outScoreDict = {}
    count = 0   #Used to take LOFScore in sequence from LOFScoreArray

    #get the maximum outLine
    maxOutLine = 0
    for node in featureDict.keys():
        yi = featureDict[node][3]
        xi = featureDict[node][2]
        outlineScore = outlierness_score(xi, yi, C, gamma)
        if outlineScore > maxOutLine:
            maxOutLine = outlineScore

    print('maxOutLine={}'.format(maxOutLine))

    #get the maximum LOFScore
    maxLOFScore = 0
    for ite in range(len(W)):
        if LOFScoreArray[ite] > maxLOFScore:
            maxLOFScore = LOFScoreArray[ite]

    print('maxLOFScore={}'.format(maxLOFScore))

    for node in featureDict.keys():
        yi = featureDict[node][3]
        xi = featureDict[node][2]
        outlineScore = outlierness_score(xi, yi, C, gamma)
        LOFScore = LOFScoreArray[count]
        count += 1
        outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
        outScoreDict[node] = outScore
    return outScoreDict