from std.pathlib import Path
from std.memory import Span

from numa import NumaTopology
from threading.threading_traits import BurstThreadPool
from threading.topological_dispatch import with_topological_rank_dispatch

from tokenizer import load_tokenizer, BPETokenizer, AutoPreTokenizer, AutoByteTransform
from modeling_config import (
    Model, TOKENIZER_PATH, MODEL_DIR, stop_tokens,
    BOS_TOKEN_ID, TURN_START_TOKEN_ID, TURN_END_TOKEN_ID,
)
from modeling.gemma4_common import Gemma4BaseConfig
from inspectable_toolkit.probe import (
    ContrastSet, ProbeResult, build_probe, mean_row_norm,
)
from kernels.helpers import BF16Ptr, F32Ptr, W
from kernels.dot_products import dot_to_scalar
from kernels.flash_sample import SamplingParams
from continuous_batching.schedule import MAXIMUM_SAMPLING_LOGITS
from continuous_batching.scheduler import ContinuousBatchScheduler
from simd_math.ops import sqrt


comptime C = Gemma4BaseConfig
comptime DATA_DIR = "steering_data_big"
comptime STEP_BUDGET = Gemma4BaseConfig.SLIDING_WINDOW
comptime FIRST_TAP_LAYER = 5
comptime NUM_TAP_LAYERS = 20
comptime CB_RESIDENT = 5
comptime CB_BATCH_LEN = 8192
comptime WAVE_SIZE = CB_RESIDENT
comptime WAVE_STEP_GUARD = 64
comptime USER_CAP = 80
comptime MODEL_CAP = 112
comptime TOP_SINK_K = 8
comptime NAT_FRACTION = 0.5
comptime CURRENT_FRACTION = 0.6
comptime SAMPLE_CAP = 2048


def read_lines(path: Path) -> List[String]:
    var data: List[Byte]
    try:
        data = path.read_bytes()
    except:
        return List[String]()
    var lines = List[String]()
    var start = 0
    for i in range(len(data)):
        if data[i] == Byte(10):
            if i > start:
                lines.append(String(
                    unsafe_from_utf8=Span(data).unsafe_subspan(
                        offset=start, length=i - start)))
            start = i + 1
    if start < len(data):
        lines.append(String(
            unsafe_from_utf8=Span(data).unsafe_subspan(
                offset=start, length=len(data) - start)))
    return lines^


def append_enc(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    mut ids: List[Int32],
    read text: String,
    cap: Int,
):
    var enc = tok.encode(text)
    var n = min(len(enc), cap)
    for i in range(n):
        ids.append(Int32(enc[i]))


def encode_turn(
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read user_text: String,
    read model_text: String,
) -> List[Int32]:
    var ids = List[Int32]()
    ids.append(Int32(BOS_TOKEN_ID))
    ids.append(Int32(TURN_START_TOKEN_ID))
    append_enc(tok, ids, "user\n" + user_text, USER_CAP)
    ids.append(Int32(TURN_END_TOKEN_ID))
    append_enc(tok, ids, "\n", 4)
    ids.append(Int32(TURN_START_TOKEN_ID))
    append_enc(tok, ids, "model\n" + model_text, MODEL_CAP)
    return ids^


def extract_clean[
    P: BurstThreadPool, //,
](
    mut model: Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P],
    mut sched: ContinuousBatchScheduler[Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read inputs: List[String],
    read outputs: List[String],
    is_high: Bool,
    greedy: SamplingParams,
    mut dataset: List[ContrastSet[C.HIDDEN]],
) -> Bool:
    var limit = min(len(inputs), SAMPLE_CAP)
    var cursor = 0
    while cursor < limit:
        var count = min(WAVE_SIZE, limit - cursor)
        var wave_rids = List[Int]()
        var harvested = List[Bool]()
        for j in range(cursor, cursor + count):
            var rid = sched.submit(
                encode_turn(tok, inputs[j], outputs[j]), greedy, 1,
                no_share=True).value()
            wave_rids.append(rid)
            harvested.append(False)

        var guard = 0
        while True:
            var all_done = True
            for w in range(len(wave_rids)):
                if not sched.requests[wave_rids[w]].done:
                    all_done = False
            if all_done:
                break
            guard += 1
            if guard > WAVE_STEP_GUARD:
                return False
            if sched.step(model) == 0:
                return False
            for s in range(model.steer.last_num_slots):
                var rid = model.steer.last_step_requests[s]
                for w in range(len(wave_rids)):
                    if (
                        wave_rids[w] == rid
                        and not harvested[w]
                        and len(sched.requests[rid].generated) > 0
                    ):
                        for k in range(NUM_TAP_LAYERS):
                            dataset[k].add_row(
                                is_high, model.steer.captured_ptr(k, s))
                        harvested[w] = True

        for w in range(len(wave_rids)):
            if not harvested[w]:
                return False
            _ = sched.retire(wave_rids[w])
        cursor += count
    return True


@always_inline
def row_norm(p: BF16Ptr) -> Float64:
    return Float64(sqrt[DType.float32, 1](dot_to_scalar[C.HIDDEN](p, p))[0])


def accumulate_energy(rows: BF16Ptr, n: Int, acc: F32Ptr):
    for i in range(n):
        var p = rows + i * C.HIDDEN
        for off in range(0, C.HIDDEN, W):
            var x = (p + off).load[width=W]().cast[DType.float32]()
            var a = (acc + off).load[width=W]()
            (acc + off).store(x.fma(x, a))


def top_sink(energy: F32Ptr, k: Int) -> List[Int]:
    var used = List[Bool](length=C.HIDDEN, fill=False)
    var idxs = List[Int]()
    for _ in range(k):
        var best = -1
        var bestv = Float32(-1)
        for j in range(C.HIDDEN):
            if not used[j] and energy[j] > bestv:
                bestv = energy[j]
                best = j
        if best < 0:
            break
        used[best] = True
        idxs.append(best)
    return idxs^


def energy_fraction(energy: F32Ptr, read sink: List[Int]) -> Float64:
    var total = Float64(0)
    for j in range(C.HIDDEN):
        total += Float64(energy[j])
    var part = Float64(0)
    for t in range(len(sink)):
        part += Float64(energy[sink[t]])
    if total <= Float64(0):
        return Float64(0)
    return part / total


def mean_semantic_norm(rows: BF16Ptr, n: Int, read sink: List[Int]) -> Float64:
    if n <= 0:
        return Float64(0)
    var total = Float64(0)
    for i in range(n):
        var p = rows + i * C.HIDDEN
        var full_sq = Float64(dot_to_scalar[C.HIDDEN](p, p))
        var s = Float64(0)
        for t in range(len(sink)):
            var v = Float64((p + sink[t]).load[width=1]().cast[DType.float32]()[0])
            s += v * v
        var rem = full_sq - s
        if rem < Float64(0):
            rem = Float64(0)
        total += sqrt[DType.float64, 1](rem)[0]
    return total / Float64(n)


@always_inline
def chord(pa: BF16Ptr, pb: BF16Ptr) -> Float64:
    var na = row_norm(pa)
    var nb = row_norm(pb)
    if na <= Float64(0) or nb <= Float64(0):
        return Float64(0)
    var cos = Float64(dot_to_scalar[C.HIDDEN](pa, pb)) / (na * nb)
    if cos > Float64(1):
        cos = Float64(1)
    if cos < Float64(-1):
        cos = Float64(-1)
    var v = Float64(2) * (Float64(1) - cos)
    if v < Float64(0):
        v = Float64(0)
    return sqrt[DType.float64, 1](v)[0]


def mean_drift(
    a_high: BF16Ptr, a_low: BF16Ptr, b_high: BF16Ptr, b_low: BF16Ptr,
    n_high: Int, n_low: Int,
) -> Float64:
    var total = Float64(0)
    var count = 0
    for i in range(n_high):
        total += chord(a_high + i * C.HIDDEN, b_high + i * C.HIDDEN)
        count += 1
    for i in range(n_low):
        total += chord(a_low + i * C.HIDDEN, b_low + i * C.HIDDEN)
        count += 1
    if count == 0:
        return Float64(0)
    return total / Float64(count)


def diff_norm(mh: F32Ptr, ml: F32Ptr) -> Float64:
    var acc = SIMD[DType.float32, W](0)
    for off in range(0, C.HIDDEN, W):
        var d = (mh + off).load[width=W]() - (ml + off).load[width=W]()
        acc = d.fma(d, acc)
    return Float64(sqrt[DType.float32, 1](acc.reduce_add())[0])


def analyze_trait[
    P: BurstThreadPool, //,
](
    mut model: Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P],
    mut sched: ContinuousBatchScheduler[Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P].POSITIONS_PER_PAGE],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
    read trait_name: String,
    greedy: SamplingParams,
):
    var prefix = String(DATA_DIR) + "/" + trait_name
    var hi_in = read_lines(Path(prefix + "_high_train_in.txt"))
    var hi_out = read_lines(Path(prefix + "_high_train_out.txt"))
    var lo_in = read_lines(Path(prefix + "_low_train_in.txt"))
    var lo_out = read_lines(Path(prefix + "_low_train_out.txt"))
    if len(hi_in) == 0 or len(lo_in) == 0:
        print(t"  missing data for {trait_name}")
        return

    var cap = max(len(hi_in), len(lo_in))
    var dataset = List[ContrastSet[C.HIDDEN]](capacity=NUM_TAP_LAYERS)
    for _ in range(NUM_TAP_LAYERS):
        dataset.append(ContrastSet[C.HIDDEN](cap))

    model.steer.clear_inject()
    if not extract_clean(model, sched, tok, hi_in, hi_out, True, greedy, dataset):
        print("  high extraction failed")
        return
    if not extract_clean(model, sched, tok, lo_in, lo_out, False, greedy, dataset):
        print("  low extraction failed")
        return

    var n_full = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var n_sem = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var drift = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var fisher = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var sep = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var par = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))
    var sink_frac = List[Float64](length=NUM_TAP_LAYERS, fill=Float64(0))

    var mean_high = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var mean_low = List[Float32](length=C.HIDDEN, fill=Float32(0))
    var direction = List[BFloat16](length=C.HIDDEN, fill=BFloat16(0))
    var mh_ptr: F32Ptr = mean_high.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var ml_ptr: F32Ptr = mean_low.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var dir_ptr: BF16Ptr = direction.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    var best_k = -1
    var best_fr = Float64(0)
    for k in range(NUM_TAP_LAYERS):
        var n = dataset[k].n_high + dataset[k].n_low
        n_full[k] = mean_row_norm[C.HIDDEN](dataset[k])

        var energy = List[Float32](length=C.HIDDEN, fill=Float32(0))
        var e_ptr: F32Ptr = energy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        accumulate_energy(dataset[k].high_ptr(0), dataset[k].n_high, e_ptr)
        accumulate_energy(dataset[k].low_ptr(0), dataset[k].n_low, e_ptr)
        var sink = top_sink(e_ptr, TOP_SINK_K)
        sink_frac[k] = energy_fraction(e_ptr, sink)
        _ = energy^
        var sem_hi = mean_semantic_norm(
            dataset[k].high_ptr(0), dataset[k].n_high, sink)
        var sem_lo = mean_semantic_norm(
            dataset[k].low_ptr(0), dataset[k].n_low, sink)
        if n > 0:
            n_sem[k] = (
                sem_hi * Float64(dataset[k].n_high)
                + sem_lo * Float64(dataset[k].n_low)) / Float64(n)

        var r = build_probe[C.HIDDEN](dataset[k], FIRST_TAP_LAYER + k, mh_ptr, ml_ptr, dir_ptr)
        fisher[k] = r.fr
        sep[k] = r.separation
        var raw = diff_norm(mh_ptr, ml_ptr)
        if raw > Float64(0):
            var ratio = (r.separation * r.separation) / (raw * raw)
            if ratio > Float64(1):
                ratio = Float64(1)
            par[k] = Float64(1) - ratio
        if r.fr > best_fr:
            best_fr = r.fr
            best_k = k
    _ = mean_high^
    _ = mean_low^
    _ = direction^

    for k in range(NUM_TAP_LAYERS - 1):
        var a_high = dataset[k].high_ptr(0)
        var a_low = dataset[k].low_ptr(0)
        var dn_high = dataset[k].n_high
        var dn_low = dataset[k].n_low
        var b_high = dataset[k + 1].high_ptr(0)
        var b_low = dataset[k + 1].low_ptr(0)
        drift[k] = mean_drift(a_high, a_low, b_high, b_low, dn_high, dn_low)

    print(t"=== {trait_name} ===")
    print(t"  samples: {dataset[0].n_high} high / {dataset[0].n_low} low")
    print("  L | n_full | n_sem | sem/full | sink8% | fisher | sep | par% | drift")
    for k in range(NUM_TAP_LAYERS):
        var ratio = (n_sem[k] / n_full[k]) if n_full[k] > Float64(0) else Float64(0)
        print(t"  {FIRST_TAP_LAYER + k} | {n_full[k]} | {n_sem[k]} | {ratio} "
              t"| {sink_frac[k]} | {fisher[k]} | {sep[k]} | {par[k]} | {drift[k]}")

    print()
    print("  injection-point safe-magnitude bracket (downstream taps only)")
    print("  L | sum1/Nf | natRot | a_eq_full | a_eq_sem | a_safe_f | a_safe_sem | 0.6*Nf | overdrive")
    for lk in range(NUM_TAP_LAYERS):
        var s1f = Float64(0)
        var s2f = Float64(0)
        var s1s = Float64(0)
        var nat = Float64(0)
        for m in range(lk + 1, NUM_TAP_LAYERS):
            if n_full[m] > Float64(0):
                s1f += Float64(1) / n_full[m]
                s2f += Float64(1) / (n_full[m] * n_full[m])
            if n_sem[m] > Float64(0):
                s1s += Float64(1) / n_sem[m]
            nat += drift[m - 1]
        var a_eq_full = (nat / s1f) if s1f > Float64(0) else Float64(0)
        var a_eq_sem = (nat / s1s) if s1s > Float64(0) else Float64(0)
        var a_safe_f = Float64(NAT_FRACTION) * a_eq_full
        var s2root = sqrt[DType.float64, 1](s2f)[0] if s2f > Float64(0) else Float64(0)
        var a_safe_sem = (
            Float64(NAT_FRACTION) * nat / s2root) if s2root > Float64(0) else Float64(0)
        var cur = Float64(CURRENT_FRACTION) * n_full[lk]
        var over = (cur / a_safe_f) if a_safe_f > Float64(0) else Float64(0)
        var mark = "  *" if lk == best_k else ""
        print(t"  {FIRST_TAP_LAYER + lk} | {s1f} | {nat} | {a_eq_full} | {a_eq_sem} "
              t"| {a_safe_f} | {a_safe_sem} | {cur} | {over}{mark}")

    if best_k >= 0:
        print(t"  selected (max fisher) layer {FIRST_TAP_LAYER + best_k} "
              t"| fisher {best_fr}")
    print()


def run[
    P: BurstThreadPool, //,
](
    topo: NumaTopology,
    var pools: List[P],
    mut tok: BPETokenizer[AutoPreTokenizer, AutoByteTransform],
):
    var model_opt = Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P].load(Path(MODEL_DIR), topo, pools^)
    if not model_opt:
        print("model load failed")
        return
    var model = model_opt.take()
    print(t"loaded (degree {model.degree})")

    var tap = List[Int](capacity=NUM_TAP_LAYERS)
    for k in range(NUM_TAP_LAYERS):
        tap.append(FIRST_TAP_LAYER + k)
    model.steer.arm(tap^)

    var greedy = SamplingParams(
        Float32(1.0), Float32(0.0), 0, 0, MAXIMUM_SAMPLING_LOGITS, True)
    var sched = ContinuousBatchScheduler[
        Model[batching_seq_len=CB_BATCH_LEN, max_resident_seqs=CB_RESIDENT, steer_vectors=16, Pool=P].POSITIONS_PER_PAGE,
    ](model.batch_geometry(), STEP_BUDGET, stop_tokens())

    var traits = List[String]()
    traits.append("openness")
    traits.append("conscientiousness")
    traits.append("extraversion")
    traits.append("agreeableness")
    traits.append("neuroticism")

    print()
    print(t"sink dims = top {TOP_SINK_K} by mean-square | nat_fraction = {NAT_FRACTION}")
    print()
    for ti in range(len(traits)):
        analyze_trait(model, sched, tok, traits[ti], greedy)

    model.steer.disarm()


def main():
    print("steer geometry analysis")
    var tok_opt = load_tokenizer(Path(TOKENIZER_PATH))
    if not tok_opt:
        print(t"failed to load tokenizer from {TOKENIZER_PATH}")
        return
    var tok = tok_opt.take()

    var topo = NumaTopology()
    var nodes = topo.num_nodes()
    print(t"{nodes} NUMA nodes")

    @parameter
    def dispatch_tp[
        P: BurstThreadPool, //,
    ](var selected_pools: List[P]):
        run(topo, selected_pools^, tok)

    with_topological_rank_dispatch[
        dispatch=dispatch_tp,
    ](
        topo, "mode: isolated (spin-only)", "mode: cold (spin-backoff)")
