from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_blobs
from joblib import dump
from clearml import Task

task = Task.init(project_name="serving examples", task_name="train model ensemble", output_uri=True)

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X, y)

rf = RandomForestRegressor(n_estimators=50)
rf.fit(X, y)

estimators = [("knn", knn), ("rf", rf), ]
ensemble = VotingRegressor(estimators)
ensemble.fit(X, y)

dump(ensemble, filename="ensemble-vr.pkl", compress=9)
