
## DDP for transformer models

Educational project for distributed data-parallel training of transformer models.
- Runs on multiple CPUs and GPUs
- Uses a bare-bones implementation of HuggingFace datasets, backed by a pyarrow
table for reduced memory consumption and sharing across processes.

Based on:
- Bigram-model from [minGPT](https://github.com/karpathy/minGPT)
- Torch DDP from [minGPT-ddp](https://github.com/subramen/minGPT-ddp)
- ArrowDataset from [HuggingFace datasets](https://github.com/huggingface/datasets)
- Map-style custom dataset as in [Llama cookbook's ConcatDataset](https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/data/concatenator.py) from where I started exploring the architecture for the `dataset` used

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation

To install mintransformer from GitHub repository, do:

```console
git clone git@github.com:f-hafner/mintransformer.git
cd mintransformer
python -m pip install .
```

## Documentation

Include a link to your project's full documentation here.


## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
