// All-gather schedule tables shared by exchange_bench.cu and schedule_check.cpp (host only).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef SCHED_MAX_ROUNDS
#define SCHED_MAX_ROUNDS 8
#endif
#ifndef SCHED_MAX_SENDS
#define SCHED_MAX_SENDS 32
#endif
constexpr int MAX_ROUNDS = SCHED_MAX_ROUNDS;   // the kernel keeps 2 * MAX_ROUNDS arrival counters
constexpr int MAX_SENDS = SCHED_MAX_SENDS;     // per round

struct Send { int peer; int block; int nblocks; };
struct Round { int nsend; Send send[MAX_SENDS]; int arrive_local; int arrive_remote; };
struct Table { int nround; Round round[MAX_ROUNDS]; };

// P ranks, L ranks per node (block distribution: rank = node * L + local).
struct Topo { int P, rank, L, node, local, nodes; };

static inline Topo make_topo(int P, int L, int rank) { return Topo{P, rank, L, rank / L, rank % L, (P + L - 1) / L}; }

// Rounds of the schedule; a rank sends only blocks it holds at the start of the round. Returns nround = -1 when the
// schedule does not apply to (P, L, stride) or needs more than MAX_ROUNDS / MAX_SENDS.
static inline Table build(const std::string& sched, int stride, const Topo& t, int rank) {
  Table tb{}; int P = t.P;
  bool overflow = false;
  auto add = [&](int r, int peer, int block, int n) {
    if (r >= MAX_ROUNDS) { overflow = true; return; }
    Round& rd = tb.round[r]; if (rd.nsend >= MAX_SENDS) { overflow = true; return; }
    rd.send[rd.nsend++] = Send{peer, block, n};
  };
  auto gcd = [](int a, int b) { while (b) { int c = a % b; a = b; b = c; } return a; };
  if (sched == "flat") {
    if (gcd(stride, P) != 1) { tb.nround = -1; return tb; }
    tb.nround = 1;
    for (int k = 1; k < P; ++k) add(0, (rank + k * stride) % P, rank, 1);
  } else if (sched == "cyclic") {
    if (gcd(stride, P) != 1) { tb.nround = -1; return tb; }
    tb.nround = P - 1;
    for (int k = 1; k < P; ++k) add(k - 1, (rank + k * stride) % P, rank, 1);
  } else if (sched == "ring") {
    tb.nround = P - 1;
    for (int k = 0; k < P - 1; ++k) add(k, (rank + 1) % P, ((rank - k) % P + P) % P, 1);
  } else if (sched == "doubling") {
    if (P & (P - 1)) { tb.nround = -1; return tb; }
    int q = 0; while ((1 << q) < P) ++q;
    tb.nround = q;
    for (int k = 0; k < q; ++k) { int low = rank & ~((1 << k) - 1); add(k, rank ^ (1 << k), low, 1 << k); }
  } else if (sched == "hier") {
    if (P % t.L || t.nodes < 2) { tb.nround = -1; return tb; }
    tb.nround = 2;
    int local = rank % t.L, node = rank / t.L;
    for (int n = 0; n < t.nodes; ++n) if (n != node) add(0, n * t.L + local, rank, 1);
    for (int j = 0; j < t.L; ++j) if (j != local)
      for (int n = 0; n < t.nodes; ++n) add(1, node * t.L + j, n * t.L + local, 1);
  } else if (sched == "repeat") {
    // Microbenchmark: R = stride rounds, each sending the own block to rank + 1 (an all-gather only for P = 2);
    // the per-step time against R separates the per-step overhead from the per-round cost.
    if (stride < 1) { tb.nround = -1; return tb; }
    tb.nround = stride;
    for (int k = 0; k < stride && k < MAX_ROUNDS; ++k) add(k, (rank + 1) % P, rank, 1);
  } else if (sched == "fanout") {
    // Microbenchmark, not an all-gather: own block to k = stride ranks of the next node, one round (identifies the
    // per-put injection gap of one NIC).
    if (P % t.L || t.nodes < 2 || stride < 1 || stride > t.L) { tb.nround = -1; return tb; }
    tb.nround = 1;
    int local = rank % t.L, nn = (rank / t.L + 1) % t.nodes;
    for (int j = 0; j < stride; ++j) add(0, nn * t.L + (local + j) % t.L, rank, 1);
  } else { fprintf(stderr, "unknown schedule %s\n", sched.c_str()); exit(4); }
  if (overflow || tb.nround > MAX_ROUNDS) tb.nround = -1;
  return tb;
}

// Expected arrivals per round at t.rank, split by same node / other node: simulate every rank's table.
static inline void arrivals(const std::string& sched, int stride, const Topo& t, Table& mine) {
  for (int r = 0; r < MAX_ROUNDS; ++r) { mine.round[r].arrive_local = 0; mine.round[r].arrive_remote = 0; }
  for (int src = 0; src < t.P; ++src) {
    Table o = build(sched, stride, t, src);
    for (int r = 0; r < o.nround; ++r)
      for (int i = 0; i < o.round[r].nsend; ++i)
        if (o.round[r].send[i].peer == t.rank) {
          if (src / t.L == t.node) mine.round[r].arrive_local += 1; else mine.round[r].arrive_remote += 1;
        }
  }
}

// Every rank only forwards blocks it holds at the start of the round, and after the last round holds every block.
static inline bool schedule_complete(const std::string& sched, int stride, const Topo& t) {
  std::vector<std::vector<char>> have(t.P, std::vector<char>(t.P, 0));
  for (int r = 0; r < t.P; ++r) have[r][r] = 1;
  std::vector<Table> tabs; for (int src = 0; src < t.P; ++src) tabs.push_back(build(sched, stride, t, src));
  int rounds = tabs[0].nround;
  if (rounds < 0) return false;
  for (int k = 0; k < rounds; ++k) {
    auto next = have;
    for (int src = 0; src < t.P; ++src) {
      const Round& rd = tabs[src].round[k];
      for (int i = 0; i < rd.nsend; ++i) {
        const Send& s = rd.send[i];
        for (int b = s.block; b < s.block + s.nblocks; ++b) { if (!have[src][b]) return false; next[s.peer][b] = 1; }
      }
    }
    have = next;
  }
  for (auto& v : have) for (char c : v) if (!c) return false;
  return true;
}
