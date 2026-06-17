---
language:
- en
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_examples: 416
  - name: test
    num_examples: 104
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
