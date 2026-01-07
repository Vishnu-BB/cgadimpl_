#include "ad/core/arena.hpp" // Assuming arena.hpp is in the ad/ directory
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <memory>




// A simple counter to track destructor calls for testing purposes.
static int g_destructor_call_count = 0;




// A non-trivially destructible class for testing.
// It increments a global counter when it is destroyed.
struct TestObject {
  int id;
  std::string name;




  TestObject(int i, const std::string& n) : id(i), name(n) {
      // std::cout << "TestObject " << id << " constructed.\n";
  }




  ~TestObject() {
      g_destructor_call_count++;
      // std::cout << "TestObject " << id << " destroyed. Total destroyed: " << g_destructor_call_count << "\n";
  }
};








// A helper to run a test and print its status.
void run_test(const std::string& test_name, void (*test_func)()) {
  std::cout << "--- Running: " << test_name << " ---\n";
  g_destructor_call_count = 0; // Reset counter for each test
  try {
      test_func();
      std::cout << "Pass\n" << std::endl;
  } catch (const std::exception& e) {
      std::cout << "Fail: " << e.what() << "\n" << std::endl;
  }
}








// ==========================================================
// TEST CASES
// ==========================================================




void test_basic_allocation_and_reset() {
  ad::memory::Arena arena(1024); // 1 KB arena
  assert(arena.capacity() == 1024);
  assert(arena.used() == 0);




  // Allocate a simple integer
  int* my_int = arena.create<int>(42);
  assert(*my_int == 42);
  assert(arena.used() >= sizeof(int));




  // Allocate a more complex object
  TestObject* obj = arena.create<TestObject>(1, "Test");
  assert(obj->id == 1);
  assert(obj->name == "Test");
  assert(arena.used() > sizeof(int));
  assert(g_destructor_call_count == 0);




  // Reset should call the destructor for TestObject
  arena.reset();
  assert(g_destructor_call_count == 1);
  assert(arena.used() == 0);
}




void test_alignment() {
  ad::memory::Arena arena(1024);




  // Allocate a char (alignment 1)
  char* p1 = arena.create<char>('a');
  assert(reinterpret_cast<uintptr_t>(p1) % alignof(char) == 0);




  // Allocate an int (alignment 4) after the char
  int* p2 = arena.create<int>(123);
  assert(reinterpret_cast<uintptr_t>(p2) % alignof(int) == 0);




  // Allocate a double (alignment 8)
  double* p3 = arena.create<double>(3.14);
  assert(reinterpret_cast<uintptr_t>(p3) % alignof(double) == 0);




  // Check that the head has advanced correctly, including padding for alignment
  size_t expected_used = sizeof(char);
  expected_used = (expected_used + alignof(int) - 1) & ~(alignof(int) - 1);
  expected_used += sizeof(int);
  expected_used = (expected_used + alignof(double) - 1) & ~(alignof(double) - 1);
  expected_used += sizeof(double);




  assert(arena.used() == expected_used);
}




void test_mark_and_shrink() {
  ad::memory::Arena arena(1024);
  arena.create<TestObject>(1, "Permanent"); // Stays




  ad::memory::Mark m = arena.mark(); // Mark after the first object
  size_t used_at_mark = arena.used();




  arena.create<TestObject>(2, "Temp1");
  arena.create<TestObject>(3, "Temp2");




  assert(arena.used() > used_at_mark);
  assert(g_destructor_call_count == 0);




  // Shrinking should destroy Temp1 and Temp2
  arena.shrink_to(m);
  assert(g_destructor_call_count == 2); // Temp1 and Temp2 destroyed
  assert(arena.used() == used_at_mark);




  // Reset should destroy the final Permanent object
  arena.reset();
  assert(g_destructor_call_count == 3);
  assert(arena.used() == 0);
}




void test_arena_scope() {
  ad::memory::Arena arena(1024);
  arena.create<TestObject>(1, "Outer");
  size_t used_before_scope = arena.used();




  {
      ad::memory::ArenaScope scope(arena);
      arena.create<TestObject>(2, "Inner1");
      arena.create<TestObject>(3, "Inner2");
      assert(arena.used() > used_before_scope);
  } // ArenaScope destructor is called here, automatically shrinking the arena




  assert(g_destructor_call_count == 2); // Inner1 and Inner2 destroyed
  assert(arena.used() == used_before_scope);
}




void test_pmr_integration() {
  ad::memory::Arena arena(4096);




  // Create a PMR vector that uses our arena
  ad::memory::pmr_vec<int> vec(arena.pmr());
  size_t used_before = arena.used();




  // The vector should allocate its memory from the arena
  for(int i = 0; i < 100; ++i) {
      vec.push_back(i);
  }




  assert(arena.used() > used_before);
  assert(vec.size() == 100);
  assert(vec[99] == 99);
  assert(vec.get_allocator().resource() == &arena);




  // Resetting the arena while the vector is live is dangerous,
  // but this tests that the memory is indeed reclaimed.
  arena.reset();
  assert(arena.used() == 0);
  // Accessing `vec` now would be undefined behavior.
}




void test_allocation_failure() {
  ad::memory::Arena arena(100); // Small arena




  try {
      // This allocation should be too large and throw
      arena.allocate_raw(200);
      // If we reach here, the test has failed
      assert(false && "std::bad_alloc was not thrown on overflow");
  } catch (const std::bad_alloc& e) {
      // This is the expected outcome
  }
}




void test_zero_size_allocation() {
  ad::memory::Arena arena(1024);
  size_t used_before = arena.used();
   // Allocating zero bytes should return a valid, non-null pointer
  // and potentially advance the head slightly due to alignment.
  void* p = arena.allocate_raw(0);
  assert(p != nullptr);
  assert(arena.used() >= used_before);
}




void test_untracked_create() {
  ad::memory::Arena arena(1024);
  arena.create_untracked<TestObject>(1, "Untracked");
   // Reset should NOT call the destructor for the untracked object.
  arena.reset();
  assert(g_destructor_call_count == 0);
}




// ==========================================================
// MAIN FUNCTION
// ==========================================================




int main() {
  std::cout << "================================\n";
  std::cout << "--- Starting Arena Test Suite ---\n";
  std::cout << "================================\n" << std::endl;




  run_test("Basic Allocation and Reset", &test_basic_allocation_and_reset);
  run_test("Alignment Guarantees", &test_alignment);
  run_test("Mark and Shrink Lifecycle", &test_mark_and_shrink);
  run_test("RAII ArenaScope Helper", &test_arena_scope);
  run_test("PMR Container Integration", &test_pmr_integration);
  run_test("Allocation Failure on Overflow", &test_allocation_failure);
  run_test("Zero-Size Allocation", &test_zero_size_allocation);
  run_test("Untracked Object Creation", &test_untracked_create);




  std::cout << "--- All Arena tests completed. ---\n" << std::endl;




  return 0;
}
