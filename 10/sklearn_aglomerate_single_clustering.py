from packaging import version
import sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


np.random.seed(123)

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10
df = pd.DataFrame(X, columns=variables, index=labels)


if version.parse(sklearn.__version__) > version.parse("1.2"):
    ac = AgglomerativeClustering(n_clusters=3,
                                 metric="euclidean",
                                 linkage="complete"
                                )
else:
    ac = AgglomerativeClustering(n_clusters=3,
                                 affinity="euclidean",
                                 linkage="complete"
                                )

labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')




if version.parse(sklearn.__version__) > version.parse("1.2"):
    ac = AgglomerativeClustering(n_clusters=2,
                                 metric="euclidean",
                                 linkage="complete"
                                )
else:
    ac = AgglomerativeClustering(n_clusters=2,
                                 affinity="euclidean",
                                 linkage="complete"
                                )

labels = ac.fit_predict(X)
print(f'Cluster labels: {labels}')