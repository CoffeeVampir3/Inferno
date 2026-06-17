#!/usr/bin/env fish
set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
set REMOTE_PATH /home/blackroot/Desktop/gemma4mojo
set TARGET test_gemma4_load.mojo
set BINARY (string replace -r '\.mojo$' '' (basename $TARGET))
set REMOTE_DUMP $REMOTE_PATH/dump/embed_test
set LOCAL_DUMP dump/embed_test

rsync -av \
    --exclude='.*' \
    --exclude='pixi.lock' \
    --exclude='__pycache__' \
    --exclude='validation/.venv' \
    --exclude='test_smollm2_bin' \
    --exclude='test_smollm2_tp3_bin' \
    --exclude='test_tp3_bin' \
    --exclude='test_tp_bin' \
    --exclude='test_rings_bin' \
    --exclude='fence_experiment_bin' \
    --exclude='tp_param_bin' \
    --include='checkpoints/SmolLM2/model.safetensors' \
    --include='checkpoints/gemma-4-26B-A4B/*' \
    --exclude='checkpoints/**/*.safetensors' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Building and running $TARGET on $REMOTE_HOST"

ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && pixi run mojo build -I . $TARGET && ./$BINARY"

echo "→ Fetching dump files"
mkdir -p $LOCAL_DUMP
rsync -av $REMOTE_USER@$REMOTE_HOST:$REMOTE_DUMP/ $LOCAL_DUMP/

echo "→ Running validation"
cd validation && uv run python main.py
