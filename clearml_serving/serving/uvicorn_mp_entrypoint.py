import uvicorn
from clearml_serving.serving.init import setup_task

if __name__ == "__main__":
    setup_task(force_threaded_logging=True)
    uvicorn.main()
