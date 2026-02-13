### Install pip package:
```sh
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118 
```
### Install torch-scatter (PyTorch Geometric dependency):
```sh
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
or 
pip install -e '.[scat]' --extra-index-url https://download.pytorch.org/whl/cu118 -f https://data.pyg.org/whl/torch-2.7.1+cu118.html 
```

### Resource Directory Structure

If the files cannot be downloaded automatically, please save them manually in the following locations:

```
lfs_data/
├── wordnet
│   └── corpora
│       └── wordnet.zip
├── datasets
│   |── imagenette
│   |   └── imagenette2-320
│   |       └── imagenette2-320.tgz
│   └── cifar10
│       └── cifar-10-batches-py
│       └── cifar-10-python.tar.gz
└── models
    └── clip
        └── ViT-B-32.pt
```

- `wordnet.zip` should be placed in `lfs_data/wordnet/corpora/`
- `ViT-B-32.pt` should be placed in `lfs_data/models/clip/`
- `cifar-10-python.tar.gz` should be placed in `lfs_data/datasets/cifar10/`
- `imagenette2-320.tgz` should be placed in `lfs_data/datasets/imagenette/`


### Command to track the GPU usage
```sh
 nvidia-smi -i 0 --query-gpu=memory.used,memory.total --format=csv,noheader -l 20 
```

