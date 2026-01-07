#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_set>
#include <functional>
#include "ad/ag_all.hpp"


using namespace ag;
using namespace OwnTensor;


// Heap-based topo_from (for comparison)
std::vector<Node*> topo_from_heap(Node* root) {
   std::vector<Node*> order;
   order.reserve(256);
  
   std::unordered_set<Node*> vis;
   vis.reserve(256);
  
   std::function<void(Node*)> dfs = [&](Node* n) {
       if (!n || vis.count(n)) return;
       vis.insert(n);
       for (auto& p : n->inputs) dfs(p.get());
       order.push_back(n);
   };
  
   dfs(root);
   return order;
}


// Helper to build test graphs
std::vector<Value> build_chain(int length) {
   std::vector<Value> nodes;
   auto x = make_tensor(Tensor::randn(Shape{{1, 1}}, TensorOptions().with_req_grad(true)), "x");
   nodes.push_back(x);
  
   Value current = x;
   for (int i = 0; i < length - 1; ++i) {
       current = current + current;
       nodes.push_back(current);
   }
   return nodes;
}


// Timing helper
struct Timer {
   using Clock = std::chrono::high_resolution_clock;
   Clock::time_point start;
  
   void reset() { start = Clock::now(); }
   double elapsed_ms() const {
       auto end = Clock::now();
       return std::chrono::duration<double, std::milli>(end - start).count();
   }
};


// Benchmark function
void benchmark_comparison(const std::string& name, Node* root, int node_count, int iterations = 1000) {
   // Warmup
   for (int i = 0; i < 100; ++i) {
       auto order1 = topo_from(root);
       auto order2 = topo_from_heap(root);
   }
  
   // Benchmark heap version
   Timer timer;
   timer.reset();
   for (int i = 0; i < iterations; ++i) {
       auto order = topo_from_heap(root);
   }
   double heap_time = timer.elapsed_ms();
  
   // Benchmark arena version (current implementation)
   timer.reset();
   for (int i = 0; i < iterations; ++i) {
       auto order = topo_from(root);
   }
   double arena_time = timer.elapsed_ms();
  
   double speedup = heap_time / arena_time;
  
   std::cout << std::left << std::setw(20) << name
             << std::right << std::setw(8) << node_count
             << std::setw(12) << std::fixed << std::setprecision(3)
             << heap_time
             << std::setw(12) << arena_time
             << std::setw(12) << std::setprecision(3) << speedup << "x";
  
   if (speedup > 1.0) {
       std::cout << "  FASTER";
   } else if (speedup < 1.0) {
       std::cout << "  SLOWER";
   } else {
       std::cout << "  ≈ SAME";
   }
   std::cout << std::endl;
}


int main() {
   std::cout << "========================================" << std::endl;
   std::cout << "ARENA vs HEAP COMPARISON" << std::endl;
   std::cout << "Testing topo_from Performance" << std::endl;
   std::cout << "========================================\n" << std::endl;
  
   std::cout << "Building test graphs..." << std::endl;
   auto small = build_chain(100);
   std::cout << "   Small chain (100 nodes)" << std::endl;
  
   auto medium = build_chain(500);
   std::cout << "   Medium chain (500 nodes)" << std::endl;
  
   auto large = build_chain(2000);
   std::cout << "   Large chain (2000 nodes)" << std::endl;
  
   auto very_large = build_chain(5000);
   std::cout << "   Very large chain (5000 nodes)" << std::endl;
  
   std::cout << "\n========================================" << std::endl;
   std::cout << "Running Benchmarks (1000 iterations each)" << std::endl;
   std::cout << "========================================\n" << std::endl;
  
   std::cout << std::left << std::setw(20) << "Test Case"
             << std::right << std::setw(8) << "Nodes"
             << std::setw(12) << "Heap (ms)"
             << std::setw(12) << "Arena (ms)"
             << std::setw(12) << "Speedup"
             << std::endl;
   std::cout << std::string(64, '-') << std::endl;
  
   benchmark_comparison("Small Chain", small.back().node.get(), 100);
   benchmark_comparison("Medium Chain", medium.back().node.get(), 500);
   benchmark_comparison("Large Chain", large.back().node.get(), 2000);
   benchmark_comparison("Very Large", very_large.back().node.get(), 5000);
  
   std::cout << std::string(64, '-') << std::endl;
  
   std::cout << "\n========================================" << std::endl;
   std::cout << "SUMMARY" << std::endl;
   std::cout << "========================================" << std::endl;
   std::cout << "Arena: 2MB reusable thread-local buffer" << std::endl;
   std::cout << "Heap: Standard std::vector + std::unordered_set" << std::endl;
   std::cout << "Speedup > 1.0 means arena is faster" << std::endl;
   std::cout << "========================================" << std::endl;
  
   return 0;
}
