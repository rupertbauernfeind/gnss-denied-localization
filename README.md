## -> [Link to Kaggle Page](https://www.kaggle.com/competitions/gnss-denied-localization-msc-hackathon-12-th-15th-feb/overview)

### Install pip package:
```sh
# pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118 
python3 -m venv .venv && source .venv/bin/activate && pip install .
```

### Resource Directory Structure

If the files cannot be downloaded automatically, please save them manually in the following locations:

```
data/
├── test_data
├── train_data
├── map.png
```



### Command to track the GPU usage
```sh
 nvidia-smi -i 0 --query-gpu=memory.used,memory.total --format=csv,noheader -l 20 
```

