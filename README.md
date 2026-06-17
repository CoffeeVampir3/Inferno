Need linux. Threads stronger with linux.

Get model (needs python or download it yourself here: https://huggingface.co/google/gemma-4-26B-A4B)
Use script in checkpoint_downloader (cd in, it's setup for use with uv. Uses HF hub)
```
uv run download.py
```

Downloads to (relative to root)
checkpoints/gemma-4-26B-A4B

If you download it yourself, put it there.

Run bf16 model:
```
pixi run mojo run test_gemma4.mojo
```

Quantize bf16 model -> butterquant:
```
pixi run mojo run quantize_gemma.mojo
```

Run butterquant model:
```
pixi run mojo run test_gemma4_bq.mojo
```

If you want to build it for use many times:
```
pixi run mojo build test_gemma4_bq.mojo
./test_gemma4_bq
```
