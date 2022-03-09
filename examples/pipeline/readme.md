# Deploy a model inference pipeline 

## prerequisites 

Training a scikit-learn model (see example/sklearn) 

## setting up the serving service

1. Create serving Service (if not already running): 
`clearml-serving create --name "serving example"` (write down the service ID)

2. Create model base two endpoints: 
`clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn_a" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --project "serving examples"`

`clearml-serving --id <service_id> model add --engine sklearn --endpoint "test_model_sklearn_b" --preprocess "examples/sklearn/preprocess.py" --name "train sklearn model" --project "serving examples"`

3. Create pipeline model endpoint: 
`clearml-serving --id <service_id> model add --engine custom --endpoint "test_model_pipeline" --preprocess "examples/pipeline/preprocess.py"`

4. Run the clearml-serving container `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`

5. Test new endpoint: `curl -X POST "http://127.0.0.1:8080/serve/test_model_pipeline" -H "accept: application/json" -H "Content-Type: application/json" -d '{"x0": 1, "x1": 2}'`


> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
