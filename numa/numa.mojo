import linux.sys as linux
from std.pathlib import Path
from .cpumask import CpuMask

def get_current_cpu_and_node() -> Tuple[Int, Int]:
    var sys = linux.linux_sys()
    return sys.sys_getcpu()

def read_sysfs(path: String) raises -> String:
    var p = Path(path)
    if not p.exists():
        return String("")
    var content = p.read_text()
    var newline_pos = content.find("\n")
    if newline_pos >= 0:
        return String(content[byte=:newline_pos])
    return content

def parse_cpulist(cpulist: String) raises -> List[Int]:
    var cpus = List[Int]()
    if cpulist.byte_length() == 0:
        return cpus^
    var parts = cpulist.split(",")
    for part in parts:
        var dash_pos = part.find("-")
        if dash_pos >= 0:
            var start = atol(part[byte=:dash_pos])
            var end = atol(part[byte=dash_pos + 1:])
            for cpu in range(start, end + 1):
                cpus.append(cpu)
        else:
            cpus.append(atol(part))
    return cpus^


def primary_sibling(cpu: Int) -> Int:
    """Return the lowest-numbered SMT sibling of cpu (its physical-core primary).

    Every SMT sibling of a core reports the same thread_siblings_list; the first
    entry is the primary. If the topology file is unreadable, fall back to cpu
    itself so the CPU is kept (safer than dropping it on exotic systems).
    """
    try:
        var raw = read_sysfs(
            "/sys/devices/system/cpu/cpu" + String(cpu) + "/topology/thread_siblings_list")
        if raw.byte_length() == 0:
            return cpu
        var siblings = parse_cpulist(raw)
        if len(siblings) == 0:
            return cpu
        return siblings[0]
    except:
        return cpu


def filter_to_physical_primaries(cpus: List[Int]) -> List[Int]:
    """Keep only the primary SMT sibling of each physical core."""
    var primaries = List[Int]()
    for cpu in cpus:
        if primary_sibling(cpu) == cpu:
            primaries.append(cpu)
    return primaries^

def parse_distances(s: String) raises -> List[Int]:
    var distances = List[Int]()
    var parts = s.split(" ")
    for part in parts:
        if part.byte_length() > 0:
            distances.append(atol(part))
    return distances^

def parse_meminfo(path: String, field: String) raises -> Int:
    var p = Path(path)
    if not p.exists():
        return 0
    var content = p.read_text()
    var lines = content.split("\n")
    for line in lines:
        if field in line:
            var key_pos = line.find(field)
            if key_pos == -1:
                continue
            var bytes = line.as_bytes()
            var value_start = key_pos + field.byte_length()

            while value_start < len(bytes):
                var b = bytes[value_start]
                if b == Byte(32) or b == Byte(58) or b == Byte(9):
                    value_start += 1
                else:
                    break

            var value = 0
            var saw_digit = False
            var value_end = value_start
            while value_end < len(bytes):
                var b = bytes[value_end]
                if b >= Byte(48) and b <= Byte(57):
                    value = value * 10 + Int(b - Byte(48))
                    saw_digit = True
                    value_end += 1
                else:
                    break
            if saw_digit:
                return value
    return 0

struct NumaNode(Copyable, Writable):
    var id: Int
    var cpu_ids: List[Int]
    var logical_cpu_ids: List[Int]
    var distances: List[Int]
    var mem_total_kb: Int
    var mem_free_kb: Int

    def __init__(out self, id: Int):
        self.id = id
        self.cpu_ids = List[Int]()
        self.logical_cpu_ids = List[Int]()
        self.distances = List[Int]()
        self.mem_total_kb = 0
        self.mem_free_kb = 0


struct NumaTopology(Movable, Sized):
    """Unified NUMA description: discovery of the box's nodes plus the
    ring-ordered placement plan.

    Constructed in one shot: __init__ reads sysfs and orders every node it
    finds into a nearest-neighbor ring. tp is whatever the system has —
    callers do not pick a subset.

    Rank-indexed accessors (node, mask, worker_mask, cpus_on, worker_count)
    are the canonical interface for downstream consumers; system-level
    accessors (distance, num_nodes, has_isolation) remain available for
    inspection."""
    var nodes: List[NumaNode]
    var isolated_cpus: List[Int]
    var rank_to_node: List[Int]

    def __init__(out self):
        self.nodes = List[NumaNode]()
        self.isolated_cpus = List[Int]()
        self.rank_to_node = List[Int]()
        try:
            self.isolated_cpus = parse_cpulist(
                read_sysfs("/sys/devices/system/cpu/isolated"))
            var online_str = read_sysfs("/sys/devices/system/node/online")
            if online_str.byte_length() == 0:
                return
            var node_ids = parse_cpulist(online_str)
            for node_id in node_ids:
                var base = "/sys/devices/system/node/node" + String(node_id)
                var node = NumaNode(node_id)
                var all_cpus = parse_cpulist(read_sysfs(base + "/cpulist"))
                node.logical_cpu_ids = all_cpus.copy()
                node.cpu_ids = filter_to_physical_primaries(all_cpus)
                node.distances = parse_distances(read_sysfs(base + "/distance"))
                node.mem_total_kb = parse_meminfo(base + "/meminfo", "MemTotal")
                node.mem_free_kb = parse_meminfo(base + "/meminfo", "MemFree")
                self.nodes.append(node^)
        except:
            print("NumaTopology failed to read system numa information or it was not present on the system.")
        self.plan()

    def plan(mut self):
        """Order every discovered node into a nearest-neighbor ring for
        tensor-parallel placement. Seed with the most central node (min
        total distance to all others), then walk the nearest unvisited
        neighbour at each step."""
        var n = len(self.nodes)
        self.rank_to_node = List[Int]()
        if n == 0:
            return
        if n == 1:
            self.rank_to_node.append(self.nodes[0].id)
            return

        var best_centrality = Int.MAX
        var seed = 0
        for i in range(n):
            var total = 0
            for j in range(n):
                total += self.distance(self.nodes[i].id, self.nodes[j].id)
            if total < best_centrality:
                best_centrality = total
                seed = i

        var visited = List[Bool](length=n, fill=False)
        visited[seed] = True
        self.rank_to_node.append(self.nodes[seed].id)
        var last_idx = seed

        for _ in range(1, n):
            var best_next = -1
            var best_d = Int.MAX
            for k in range(n):
                if visited[k]:
                    continue
                var d = self.distance(self.nodes[last_idx].id, self.nodes[k].id)
                if d < best_d:
                    best_d = d
                    best_next = k
            if best_next < 0:
                break
            visited[best_next] = True
            self.rank_to_node.append(self.nodes[best_next].id)
            last_idx = best_next

    def __len__(self) -> Int:
        return len(self.rank_to_node)

    def __getitem__(self, rank: Int) -> Int:
        return self.rank_to_node[rank]

    def node(self, rank: Int) -> Int:
        return self.rank_to_node[rank]

    def mask[mask_size: Int = 128](self, rank: Int) -> CpuMask[mask_size]:
        """Full logical CPU mask for the rank's node (includes HT siblings)."""
        var mask = CpuMask[mask_size]()
        var node_idx = self.find_node_index(self.rank_to_node[rank])
        if node_idx < 0:
            return mask
        for cpu in self.nodes[node_idx].logical_cpu_ids:
            mask.set(cpu)
        return mask

    def worker_mask[mask_size: Int = 128](self, rank: Int) -> CpuMask[mask_size]:
        """Worker CPU mask for the rank's node (isolated ∩ node, or all
        physical primaries on the node when no isolation is configured)."""
        var mask = CpuMask[mask_size]()
        for cpu in self.worker_cpus(rank):
            mask.set(cpu)
        return mask

    def worker_cpus(self, rank: Int) -> List[Int]:
        var node_idx = self.find_node_index(self.rank_to_node[rank])
        if node_idx < 0:
            return List[Int]()
        var node_cpus = self.nodes[node_idx].cpu_ids.copy()
        if not self.has_isolation():
            return node_cpus^
        var workers = List[Int]()
        for cpu in node_cpus:
            for iso in self.isolated_cpus:
                if cpu == iso:
                    workers.append(cpu)
                    break
        return workers^

    def cpus_on(self, rank: Int) -> Int:
        var node_idx = self.find_node_index(self.rank_to_node[rank])
        if node_idx < 0:
            return 0
        return len(self.nodes[node_idx].cpu_ids)

    def worker_count(self, rank: Int) -> Int:
        return len(self.worker_cpus(rank))

    def has_isolation(self) -> Bool:
        return len(self.isolated_cpus) > 0

    def num_nodes(self) -> Int:
        return len(self.nodes)

    def find_node_index(self, node_id: Int) -> Int:
        for i in range(len(self.nodes)):
            if self.nodes[i].id == node_id:
                return i
        return -1

    def distance(self, from_node: Int, to_node: Int) -> Int:
        var i = self.find_node_index(from_node)
        if i < 0:
            return -1
        var j = self.find_node_index(to_node)
        if j < 0 or j >= len(self.nodes[i].distances):
            return -1
        return self.nodes[i].distances[j]

    def print_debug(self):
        print(t"NUMA Topology: {self.num_nodes()} nodes, tp = {len(self)}")
        print()
        for i in range(self.num_nodes()):
            ref n = self.nodes[i]
            print(t"Node {n.id}: {len(n.cpu_ids)} cpus, "
                  t"{n.mem_total_kb // 1024} MB total, "
                  t"{n.mem_free_kb // 1024} MB free")
        print()
        print("Ring order (rank -> node):")
        for r in range(len(self)):
            print(t"  rank {r} -> node {self.rank_to_node[r]}")
        print()
        print("Distance matrix:")
        for i in range(self.num_nodes()):
            for j in range(self.num_nodes()):
                print(self.distance(self.nodes[i].id, self.nodes[j].id), end=" ")
            print()
