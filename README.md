# VisualBox
## Installation
### Models
Download all models on huggingface:
https://huggingface.co/Amazingldl/VisualBox

Place the `model` folder in the root directory of this repo before runing.

### Dependencies
```shell
conda create -n visualbox python=3.8
conda activate visualbox
pip install -r requirements.txt
```

## Quick start
### Web UI
```shell
python app.py
```

### Web API
```shell
uvicorn api:app --reload
```
Then you can read the user docs in http://127.0.0.1:8000/docs