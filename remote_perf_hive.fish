#!/usr/bin/env fish
set REMOTE_USER blackroot
set REMOTE_HOST 192.168.50.93
# Path is relative to the remote home; do not use a leading ~ here or fish would
# expand it to the LOCAL home before rsync/ssh ever see it.
set REMOTE_PATH Desktop/inferno
set DEFAULT_TARGET stub.mojo

if test (count $argv) -gt 0
    set TARGET $argv[1]
else
    set TARGET $DEFAULT_TARGET
end

if not test -f $TARGET
    echo "Target not found: $TARGET"
    exit 1
end

set BINARY (string replace -r '\.mojo$' '' (basename $TARGET))

if not set -q PERF_PROFILE
    set PERF_PROFILE numa
end

set PERF_EVENTS cycles

if set -q PERF_BASELINE
    set -a PERF_EVENTS instructions
end

switch $PERF_PROFILE
    case numa
        set -a PERF_EVENTS mem_load_retired.l3_miss
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
    case numa_wide
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
        set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram
    case lfb
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
    case stalls
        set -a PERF_EVENTS cycle_activity.stalls_total cycle_activity.stalls_l3_miss
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l2_miss mem_load_retired.l3_miss
    case pipeline
        set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
        set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
    case hierarchy
        set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
        set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
    case full
        set -a PERF_EVENTS assists.fp assists.sse_avx_mix machine_clears.count
        set -a PERF_EVENTS ld_blocks.store_forward mem_inst_retired.split_loads dtlb_load_misses.walk_completed
        set -a PERF_EVENTS mem_load_retired.l1_hit mem_load_retired.l1_miss mem_load_retired.fb_hit
        set -a PERF_EVENTS l1d_pend_miss.pending l1d_pend_miss.pending_cycles l1d_pend_miss.fb_full
        set -a PERF_EVENTS mem_load_retired.l2_hit mem_load_retired.l2_miss
        set -a PERF_EVENTS l2_rqsts.all_demand_data_rd l2_rqsts.all_demand_miss
        set -a PERF_EVENTS mem_load_retired.l3_hit mem_load_retired.l3_miss
        set -a PERF_EVENTS cycle_activity.stalls_l3_miss cycle_activity.stalls_total
        set -a PERF_EVENTS mem_load_l3_miss_retired.local_dram mem_load_l3_miss_retired.remote_dram
        set -a PERF_EVENTS mem_load_l3_miss_retired.remote_fwd mem_load_l3_miss_retired.remote_hitm
        set -a PERF_EVENTS ocr.reads_to_core.local_dram ocr.reads_to_core.remote_dram
    case '*'
        echo "Unknown PERF_PROFILE: $PERF_PROFILE"
        echo "Expected one of: numa, numa_wide, lfb, stalls, pipeline, hierarchy, full"
        exit 1
end

set PERF_EVENTS_CSV (string join , $PERF_EVENTS)

# Code-only sync. Honor .gitignore (checkpoints, models, quantized_models,
# steering data, build artifacts already live there) and drop the heavyweight
# data/reference trees. The distant end is assumed to already hold the
# checkpoints/models it needs.
rsync -av \
    --filter=':- .gitignore' \
    --exclude='.git' \
    --exclude='abliteration_data' \
    --exclude='Minimax-M3/references' \
    . \
    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/

echo "✓ Synced code to $REMOTE_HOST:$REMOTE_PATH"
echo "→ Building $TARGET on $REMOTE_HOST"

set PERF_DELAY 0
if test (count $argv) -gt 1
    set PERF_DELAY $argv[2]
end

ssh -t $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_PATH && env MOJO_ENABLE_RUNTIME=0 pixi run mojo build -I . -D ASSERT=all $TARGET && echo '=== PERF STAT ===' && perf stat -D $PERF_DELAY -e $PERF_EVENTS_CSV ./$BINARY"
