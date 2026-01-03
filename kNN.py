from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = iris.data
Y = iris.target

N = 30      # number of holdout iter
RN = 10     # number of repeated ter
B = 200     # number of bootstrap iter


# holdout 1
acc_list = []
f1_list = []
scaler = StandardScaler()
for i in range(N):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=i)
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
    knn.fit(trainX, trainY)

    predY = knn.predict(testX)

    accuracy = accuracy_score(testY, predY)
    macro_f1 = f1_score(testY, predY, average='macro')
    acc_list.append(accuracy)
    f1_list.append(macro_f1)
    
acc_statistics = pd.Series(acc_list).describe(percentiles=[0.05, 0.95])
f1_statistics = pd.Series(f1_list).describe(percentiles=[0.05, 0.95])

print("Holdout 1")
print(f"accuracy    {pd.Series(acc_statistics)}")
print(f"macro-f1    {pd.Series(f1_statistics)}")
print()

# Repeated Straitified K-Fold
acc_list = []
f1_list = []
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=RN, random_state=36851234)
for i, (train_index, test_index) in enumerate(rskf.split(X, Y), 1):
    trainX, testX, trainY, testY = X[train_index], X[test_index], Y[train_index], Y[test_index]
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
    knn.fit(trainX, trainY)

    predY = knn.predict(testX)

    accuracy = accuracy_score(testY, predY)
    macro_f1 = f1_score(testY, predY, average='macro')
    acc_list.append(accuracy)
    f1_list.append(macro_f1)
    
acc_statistics = pd.Series(acc_list).describe(percentiles=[0.05, 0.95])
f1_statistics = pd.Series(f1_list).describe(percentiles=[0.05, 0.95])

print("k-fold")
print(f"accuracy    {pd.Series(acc_statistics)}")
print(f"macro-f1    {pd.Series(f1_statistics)}")
print()


# Bootstrap(OOB)
acc_list = []
f1_list = []

num_samples, num_features = iris.data.shape
idx_data = [i for i in range(num_samples)]

for i in range(B):
    bootstrap_train = []
    oob_test = []
    
    bootstrap_train.extend(np.random.choice(idx_data, size=num_samples, replace=True))
    bootstrap_train = np.array(bootstrap_train).tolist()

    oob_test = np.setdiff1d(idx_data, np.unique(bootstrap_train))
    oob_test = np.array(oob_test).tolist()
    if (len(oob_test)==0):
        continue
    
    trainX, testX, trainY, testY = [X[i] for i in bootstrap_train], [X[i] for i in oob_test], \
                                    [Y[i] for i in bootstrap_train], [Y[i] for i in oob_test]
    trainX = scaler.fit_transform(trainX)
    testX = scaler.transform(testX)

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2)
    knn.fit(trainX, trainY)

    predY = knn.predict(testX)

    accuracy = accuracy_score(testY, predY)
    macro_f1 = f1_score(testY, predY, average='macro')
    acc_list.append(accuracy)
    f1_list.append(macro_f1)
    
acc_statistics = pd.Series(acc_list).describe(percentiles=[0.05, 0.95])
f1_statistics = pd.Series(f1_list).describe(percentiles=[0.05, 0.95])

print("Bootstrap(OOB)")
print(f"accuracy    {pd.Series(acc_statistics)}")
print(f"macro-f1    {pd.Series(f1_statistics)}")
print()