from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from joblib import dump
from clearml import Task

task = Task.init(project_name="serving examples", task_name="custom train model", output_uri=True)

# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=3, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)

dump(model, filename="custom-model.pkl", compress=9)

