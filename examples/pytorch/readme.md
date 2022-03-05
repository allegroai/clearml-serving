# Train and Deploy Keras model with Nvidia Triton Engine

## training mock model

Run the mock python training code
```bash
python3 train_pytorch_mnist.py
```

The output will be a model created on the project "serving examples", by the name "train pytorch model"
*Notice* Only TorchScript models are supported by Triton server

## setting up the serving service

1. Create serving Service: `clearml-serving create --name "serving example"` (write down the service ID)
2. Create model endpoint:
`clearml-serving --id <service_id> model add --engine triton --endpoint "test_model_pytorch" --preprocess "preprocess.py" --name "train pytorch model" --project "serving examples"
  --input-size 28 28 1 --input-name "INPUT__0" --input-type float32   
  --output-size -1 10 --output-name "OUTPUT__0" --output-type float32   
`
Or auto update 
`clearml-serving --id <service_id> model auto-update --engine triton --endpoint "test_model_pytorch_auto" --preprocess "preprocess.py" --name "train pytorch model" --project "serving examples" --max-versions 2
  --input-size 28 28 1 --input-name "INPUT__0" --input-type float32   
  --output-size -1 10 --output-name "OUTPUT__0" --output-type float32   
`
Or add Canary endpoint
`clearml-serving --id <service_id> model canary --endpoint "test_model_pytorch_auto" --weights 0.1 0.9 --input-endpoint-prefix test_model_pytorch_auto`
   
3. Run the Triton Engine `docker run -v ~/clearml.conf:/root/clearml.conf -p 8001:8001 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving-triton:latest`
4. Configure the Triton Engine IP on the Serving Service (if running on k8s, the gRPC ingest of the triton container)
`clearml-serving --id <service_id> config --triton-grpc-server <local_ip_here>:8001`
5. Run the clearml-serving container `docker run -v ~/clearml.conf:/root/clearml.conf -p 8001:8001 -e CLEARML_SERVING_TASK_ID=<service_id> clearml-serving:latest`
6. Test new endpoint: `curl -X POST "http://127.0.0.1:8080/serve/test_model_pytorch" -H "accept: application/json" -H "Content-Type: application/json" -d '{"url": "https://camo.githubusercontent.com/8385ca52c9cba1f6e629eb938ab725ec8c9449f12db81f9a34e18208cd328ce9/687474703a2f2f706574722d6d6172656b2e636f6d2f77702d636f6e74656e742f75706c6f6164732f323031372f30372f6465636f6d707265737365642e6a7067"}'`

> **_Notice:_**  You can also change the serving service while it is already running!
This includes adding/removing endpoints, adding canary model routing etc.


### Running / debugging the serving service manually
Once you have setup the Serving Service Task

```bash
$ pip3 install -r clearml_serving/serving/requirements.txt
$ CLEARML_SERVING_TASK_ID=<service_id> PYHTONPATH=$(pwd) python3 -m gunicorn \
    --preload clearml_serving.serving.main:app \ 
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080
```
