import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from clearml import Task

task = Task.init(project_name="serving examples", task_name="train lightgbm model", output_uri=True)

iris = load_iris()
y = iris['target']
X = iris['data']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
dtrain = lgb.Dataset(X_train, label=y_train)

params = {
    'objective': 'multiclass',
    'metric': 'softmax',
    'num_class': 3
}
lgb_model = lgb.train(params=params, train_set=dtrain)

lgb_model.save_model("lgbm_model")
