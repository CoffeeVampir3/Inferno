#!/usr/bin/env fish
# inspect.fish <file.mojo> [sym1 sym2 ...]
# Builds the inspectable for alderlake + sapphirerapids (ASSERT=none -g0 -O3),
# runs the alderlake binary (SIGILL = mis-lowered AVX-512), and disassembles the
# requested symbols on both arches. With no symbols, lists exported probe_* syms.
set FILE $argv[1]
set BASE (string replace -r '\.mojo$' '' $FILE)
set SYMS $argv[2..-1]

for pair in alderlake:ald sapphirerapids:spr
    set arch (string split ':' $pair)[1]
    set tag (string split ':' $pair)[2]
    pixi run mojo build -D ASSERT=none --march=$arch -g0 -O3 $FILE
    or begin; echo "BUILD FAILED ($arch)"; exit 1; end
    mv -f $BASE $BASE.$tag
end

echo "=== RUN alderlake (exit!=0 / SIGILL means trouble) ==="
./$BASE.ald
echo "RUN EXIT: $status"

if test (count $SYMS) -eq 0
    echo "=== exported probe symbols ==="
    nm $BASE.ald | grep -i probe
    exit 0
end

for tag in ald spr
    echo "################## $tag ##################"
    for sym in $SYMS
        echo "===== $sym ($tag) ====="
        objdump -d --no-show-raw-insn -M intel $BASE.$tag | awk "/<$sym>:/{f=1} f{print} f&&/\tret/{exit}"
        echo ""
    end
end
