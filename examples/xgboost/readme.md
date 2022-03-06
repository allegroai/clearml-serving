# Train and Deploy XGBoost model

## training iris classifier model

Run the mock python training code
```bash
pip install -r examples/xgboost/requirements.txt 
python examples/xgboost/train_model.py
```

The output will be a model created on the project "serving examples", by the name "train xgboost model"

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Create model endpoint: 

3. `clearml-serving --id <service_id> model add --engine xgboost --endpoint "test_model_xgb" --preprocess "examples/xgboost/preprocess.py" --name "train xgboost model" --project "serving examples"`
Or auto update 
`clearml-serving --id <service_id> model auto-update --engine xgboost --endpoint "test_model_xgb_auto" --preprocess "examples/xgboost/preprocess.py" --name "train xgboost model" --project "serving examples" --max-versions 2`
Or add Canary endpoint
`clearml-serving --id <service_id> model canary --endpoint "test_model_xgb_auto" --weights 0.1 0.9 --input-endpoint-prefix test_model_xgb_auto`

4. Run the clearml-serving container `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`
5. Test new endpoint: `curl -X POST "http://127.0.0.1:8080/serve/test_model_xgb" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2, "x2": 3, "x3": 4}'`

> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
