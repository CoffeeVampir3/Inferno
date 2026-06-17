---
language:
- en
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 25058
  - name: test
    num_examples: 6265
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
