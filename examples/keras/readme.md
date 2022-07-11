# Train and Deploy Keras model with Nvidia Triton Engine

## training mnist digit classifier model

Run the mock python training code
```bash
pip install -r examples/keras/requirements.txt 
python examples/keras/train_keras_mnist.py
```

The output will be a model created on the project "serving examples", by the name "train keras model"

## setting up the serving service

Prerequisites, Keras/Tensorflow models require Triton engine support, please use `docker-compose-triton.yml` / `docker-compose-triton-gpu.yml` or if running on Kubernetes, the matching helm chart.

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Create model endpoint: 

 `clearml-serving --id <service_id> model add --engine triton --endpoint "test_model_keras" --preprocess "examples/keras/preprocess.py" --name "train keras model - serving_model" --project "serving examples" --input-size 1 784 --input-name "dense_input" --input-type float32 --output-size -1 10 --output-name "activation_2" --output-type float32   
`

Or auto update

`clearml-serving --id <service_id> model auto-update --engine triton --endpoint "test_model_auto" --preprocess "examples/keras/preprocess.py" --name "train keras model - serving_model" --project "serving examples" --max-versions 2
  --input-size 1 784 --input-name "dense_input" --input-type float32   
  --output-size -1 10 --output-name "activation_2" --output-type float32`

Or add Canary endpoint

`clearml-serving --id <service_id> model canary --endpoint "test_model_auto" --weights 0.1 0.9 --input-endpoint-prefix test_model_auto`
   
3. Run the Triton Engine `docker run -v ~/clearml.conf:/root/clearml.conf -p 8001:8001 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving-triton:latest`
4. Configure the Triton Engine IP on the Serving Service (if running on k8s, the gRPC ingest of the triton container)
`clearml-serving --id <service_id> config --triton-grpc-server <local_ip_here>:8001`
5. Run the clearml-serving container `docker run -v ~/clearml.conf:/root/clearml.conf -p 8001:8001 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`
6. Test new endpoint: `curl -X POST "http://127.0.0.1:8080/serve/test_model_keras" -H "accept: application/json" -H "Content-Type: application/json" -d '{"url": "https://camo.githubusercontent.com/8385ca52c9cba1f6e629eb938ab725ec8c9449f12db81f9a34e18208cd328ce9/687474703a2f2f706574722d6d6172656b2e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30372f6465636f6d707265737365642e6a7067"}'`

> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.
by default new endpoints/models will be automatically updated after 1 minute