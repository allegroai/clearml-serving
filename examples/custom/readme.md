# Train and Deploy custom model

## training mock custom model

Run the mock python training code
```bash
pip install -r examples/custom/requirements.txt 
python examples/custom/train_model.py
```

The output will be a model created on the project "serving examples", by the name "custom train model"

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Make sure to add any required additional packages (for your custom model) to the [docker-compose.yml](https://github.com/allegroai/clearml-serving/blob/826f503cf4a9b069b89eb053696d218d1ce26f47/docker/docker-compose.yml#L97) (or as environment variable to the `clearml-serving-inference` container), by defining for example: `CLEARML_EXTRA_PYTHON_PACKAGES="scikit-learn numpy"`
3. Create model endpoint: 
`clearml-serving --id <service_id> model add --engine custom --endpoint "test_model_custom" --preprocess "examples/custom/preprocess.py" --name "custom train model" --project "serving examples"`

Or auto update 

`clearml-serving --id <service_id> model auto-update --engine custom --endpoint "test_model_custom_auto" --preprocess "examples/custom/preprocess.py" --name "custom train model" --project "serving examples" --max-versions 2`

Or add Canary endpoint

`clearml-serving --id <service_id> model canary --endpoint "test_model_custom_auto" --weights 0.1 0.9 --input-endpoint-prefix test_model_custom_auto`

4. If you already have the `clearml-serving` docker-compose running, it might take it a minute or two to sync with the new endpoint.

Or you can run the clearml-serving container independently `docker run -v ~/clearml.conf:/root/clearml.conf -p 8080:8080 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`

5. Test new endpoint (do notice the first call will trigger the model pulling, so it might take longer, from here on, it's all in memory): `curl -X POST "http://127.0.0.1:8080/serve/test_model_custom" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [1, 2, 3]}'`


> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
by default new endpoints/models will be automatically updated after 1 minute
