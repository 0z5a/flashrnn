// Exhaustive CPU check of the schedule tables in schedule.h (no GPU, no MPI).
//
// For every P in [2, PMAX], node size L in {1, 2, 4, 8} (P a multiple of L for hier), every schedule and every stride
// in [1, P-1]: the facts of docs/flashrnn2/scale-out-plan.md section 2, checked on the tables the benchmark runs.
//   complete      every rank forwards only blocks it holds and ends with all P blocks
//   rounds        flat 1, cyclic P-1, ring P-1, doubling log2 P, hier 2
//   permutation   cyclic, ring: every round every rank sends exactly one message and receives exactly one
//   exact cover   cyclic: over all rounds every ordered pair (i, j), i != j, carries exactly one message
//   matching      doubling: every round, i sends to j iff j sends to i (a perfect matching)
//   coprime       flat/cyclic apply iff gcd(s, P) = 1; for P prime every s in [1, P-1] applies
// Per (schedule, P, L, s) it also records the per-round link loads the cost model prices: the most network messages /
// blocks leaving one NIC (one NIC per GPU) and the most NVLink messages / blocks on one directed GPU pair.
// Usage: schedule_check [PMAX=64] [--json FILE]; build with -DSCHED_MAX_ROUNDS=64 -DSCHED_MAX_SENDS=64 to cover PMAX = 64.
#include "schedule.h"
#include <map>
#include <set>
#include <cstring>

static int gcd(int a, int b) { while (b) { int c = a % b; a = b; b = c; } return a; }
static bool prime(int p) { if (p < 2) return false; for (int d = 2; d * d <= p; ++d) if (p % d == 0) return false; return true; }

struct Load { int rounds; int net_msgs, net_blocks, nv_msgs, nv_blocks; int net_msgs_total, net_blocks_total; };

int main(int argc, char** argv) {
  int PMAX = 64; const char* json = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--json") && i + 1 < argc) json = argv[++i];
    else PMAX = atoi(argv[i]);
  }
  const char* scheds[] = {"flat", "cyclic", "ring", "doubling", "hier"};
  long cases = 0, failures = 0;
  std::map<std::string, long> per_check;
  FILE* out = json ? fopen(json, "w") : nullptr;
  if (out) fprintf(out, "[\n");
  bool first = true;
  auto fail = [&](const char* what, const char* s, int P, int L, int st) {
    ++failures; per_check[what]++;
    if (failures <= 20) fprintf(stderr, "FAIL %s: %s P=%d L=%d s=%d\n", what, s, P, L, st);
  };
  for (int P = 2; P <= PMAX; ++P) {
    for (int L : {1, 2, 4, 8}) {
      if (L > P) continue;
      for (const char* sched : scheds) {
        bool strided = !strcmp(sched, "flat") || !strcmp(sched, "cyclic");
        for (int st = 1; st <= (strided ? P - 1 : 1); ++st) {
          Topo t0 = make_topo(P, L, 0);
          Table tb0 = build(sched, st, t0, 0);
          bool applies = tb0.nround >= 0;
          // Applicability: coprime stride rule; doubling for powers of two; hier for >= 2 full nodes.
          bool expect = true;
          if (strided) expect = gcd(st, P) == 1;
          if (!strcmp(sched, "doubling")) { int lg = 0; while ((1 << lg) < P) ++lg; expect = (P & (P - 1)) == 0 && lg <= MAX_ROUNDS; }
          if (!strcmp(sched, "hier")) expect = P % L == 0 && P / L >= 2 && (P / L - 1) * L <= MAX_SENDS;
          if (!strcmp(sched, "cyclic") || !strcmp(sched, "ring")) expect = expect && P - 1 <= MAX_ROUNDS;
          if (!strcmp(sched, "flat")) expect = expect && P - 1 <= MAX_SENDS;
          if (applies != expect) { fail("applicability", sched, P, L, st); continue; }
          if (strided && prime(P) && !applies && P - 1 <= MAX_ROUNDS && P - 1 <= MAX_SENDS) fail("prime_all_strides", sched, P, L, st);
          if (!applies) continue;
          ++cases;
          if (!schedule_complete(sched, st, t0)) fail("complete", sched, P, L, st);
          int q = 0; while ((1 << q) < P) ++q;
          int want_rounds = !strcmp(sched, "flat") ? 1 : !strcmp(sched, "doubling") ? q : !strcmp(sched, "hier") ? 2 : P - 1;
          if (tb0.nround != want_rounds) fail("rounds", sched, P, L, st);
          std::vector<Table> tabs; for (int r = 0; r < P; ++r) tabs.push_back(build(sched, st, make_topo(P, L, r), r));
          Load ld{tb0.nround, 0, 0, 0, 0, 0, 0};
          std::set<std::pair<int, int>> pairs; long pair_msgs = 0;
          for (int k = 0; k < tb0.nround; ++k) {
            std::vector<int> recv(P, 0);
            std::map<std::pair<int, int>, int> nv_msgs, nv_blocks;
            for (int src = 0; src < P; ++src) {
              const Round& rd = tabs[src].round[k];
              int nm = 0, nb = 0;
              for (int i = 0; i < rd.nsend; ++i) {
                const Send& s = rd.send[i];
                recv[s.peer]++;
                pairs.insert({src, s.peer}); ++pair_msgs;
                if (s.peer / L != src / L) { ++nm; nb += s.nblocks; }
                else { nv_msgs[{src, s.peer}]++; nv_blocks[{src, s.peer}] += s.nblocks; }
              }
              ld.net_msgs = std::max(ld.net_msgs, nm); ld.net_blocks = std::max(ld.net_blocks, nb);
              ld.net_msgs_total += nm; ld.net_blocks_total += nb;
              if (!strcmp(sched, "cyclic") || !strcmp(sched, "ring"))
                if (rd.nsend != 1) fail("permutation_send", sched, P, L, st);
              if (!strcmp(sched, "doubling")) {
                if (rd.nsend != 1) fail("matching", sched, P, L, st);
                else {
                  int j = rd.send[0].peer; const Round& back = tabs[j].round[k];
                  if (back.nsend != 1 || back.send[0].peer != src) fail("matching", sched, P, L, st);
                }
              }
            }
            for (auto& kv : nv_msgs) ld.nv_msgs = std::max(ld.nv_msgs, kv.second);
            for (auto& kv : nv_blocks) ld.nv_blocks = std::max(ld.nv_blocks, kv.second);
            if (!strcmp(sched, "cyclic") || !strcmp(sched, "ring"))
              for (int r = 0; r < P; ++r) if (recv[r] != 1) fail("permutation_recv", sched, P, L, st);
          }
          if (!strcmp(sched, "cyclic") && ((long)pairs.size() != (long)P * (P - 1) || pair_msgs != (long)P * (P - 1)))
            fail("exact_cover", sched, P, L, st);
          // Arrival counts used by the kernel must sum to the messages addressed to each rank.
          for (int r = 0; r < P; ++r) {
            Topo tr = make_topo(P, L, r); Table m = tabs[r]; arrivals(sched, st, tr, m);
            long got = 0; for (int k = 0; k < m.nround; ++k) got += m.round[k].arrive_local + m.round[k].arrive_remote;
            long want = 0; for (int src = 0; src < P; ++src) for (int k = 0; k < tabs[src].nround; ++k)
              for (int i = 0; i < tabs[src].round[k].nsend; ++i) want += tabs[src].round[k].send[i].peer == r;
            if (got != want) { fail("arrivals", sched, P, L, st); break; }
          }
          if (out) {
            fprintf(out, "%s{\"schedule\":\"%s\",\"P\":%d,\"L\":%d,\"stride\":%d,\"rounds\":%d,\"max_net_msgs_per_nic_round\":%d,"
                    "\"max_net_blocks_per_nic_round\":%d,\"max_nv_msgs_per_pair_round\":%d,\"max_nv_blocks_per_pair_round\":%d,"
                    "\"net_msgs_total\":%d,\"net_blocks_total\":%d}", first ? "" : ",\n", sched, P, L, st, ld.rounds, ld.net_msgs,
                    ld.net_blocks, ld.nv_msgs, ld.nv_blocks, ld.net_msgs_total, ld.net_blocks_total);
            first = false;
          }
        }
      }
    }
  }
  if (out) { fprintf(out, "\n]\n"); fclose(out); }
  printf("{\"event\":\"schedule_check\",\"PMAX\":%d,\"cases\":%ld,\"failures\":%ld", PMAX, cases, failures);
  for (auto& kv : per_check) printf(",\"%s\":%ld", kv.first.c_str(), kv.second);
  printf(",\"pass\":%s}\n", failures ? "false" : "true");
  return failures ? 1 : 0;
}
