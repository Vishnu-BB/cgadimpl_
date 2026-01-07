

#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>
#include <memory_resource>




// Minimal, safe arena allocator for CPU-side phase-scoped allocations.
// Features:
//  - bump-pointer allocation with proper alignment
//  - O(1) reset and rewind-to-mark
//  - optional destructor tracking for non-trivial types
//  - PMR adapter so std::pmr containers can draw from the arena
//  - RAII scope helper (auto-shrink-to-mark on scope exit)




namespace ad::memory {


struct Mark { std::size_t offset = 0; }; // snapshot of memory usage at a given instant




class Arena : public std::pmr::memory_resource {
public:
explicit Arena(std::size_t capacity_bytes,
               std::pmr::memory_resource* upstream = std::pmr::get_default_resource());
~Arena();




Arena(const Arena&) = delete;
Arena& operator=(const Arena&) = delete;




// Capacity/usage
std::size_t capacity()  const noexcept;
std::size_t used()      const noexcept;
std::size_t remaining() const noexcept;




// Lifetime controls ()
Mark mark() const noexcept;        // low-watermark
void shrink_to(Mark m) noexcept;   // rewind to mark (runs dtors for region >= mark)
void reset() noexcept;




// Raw allocation (unconstructed) - force-inlined for performance
void* allocate_raw(std::size_t bytes,
                   std::size_t align = alignof(std::max_align_t));




// Construct T in-place; two flavors:
template <class T, class... Args>
T* create_untracked(Args&&... args) {
  void* p = allocate_raw(sizeof(T), alignof(T));
  return ::new (p) T(static_cast<Args&&>(args)...);
}




template <class T, class... Args>
T* create(Args&&... args) {
  void* p = allocate_raw(sizeof(T), alignof(T));
  T* obj = ::new (p) T(static_cast<Args&&>(args)...);
  if constexpr (!std::is_trivially_destructible_v<T>) {
    register_dtor(p, &destroy<T>);
  }
  return obj;
}




template <class T>
T* allocate_array(std::size_t count) {
  void* p = allocate_raw(sizeof(T) * count, alignof(T));
  return static_cast<T*>(p);
}




// PMR gateway: hand this to pmr containers: std::pmr::vector<int> v{ arena.pmr() };
std::pmr::memory_resource* pmr() noexcept { return this; }








protected:
// std::pmr::memory_resource virtuals
void* do_allocate(std::size_t bytes, std::size_t alignment) override;


void  do_deallocate(void*, std::size_t, std::size_t) override;


bool  do_is_equal(const std::pmr::memory_resource& other) const noexcept override;




private:
// internal helpers
static std::size_t align_up(std::size_t x, std::size_t a) noexcept {
  const std::size_t mask = a - 1;
  return (x + mask) & ~mask;
}




template <class T>
static void destroy(void* p) noexcept {
  static_cast<T*>(p)->~T();
}




using DtorFn = void(*)(void*);
struct DtorEntry {
  std::size_t offset; // relative to base
  DtorFn      fn;
};




void register_dtor(void* p, DtorFn fn);
void run_dtors_from(std::size_t from) noexcept;




private:
std::pmr::memory_resource* upstream_;
std::vector<std::byte>     buffer_; // owns the bytes
std::byte*                 base_ = nullptr;
std::size_t                cap_  = 0;
std::size_t                head_ = 0;
std::vector<DtorEntry>     dtors_; // LIFO stack
};




// RAII scope that rewinds to a mark on destruction
struct ArenaScope {
explicit ArenaScope(Arena& a) : arena(a), m(a.mark()), active(true) {}
~ArenaScope() { if (active) arena.shrink_to(m); }
void cancel() noexcept { active = false; } // keep allocations
private:
Arena& arena;
Mark   m;
bool   active;
};




//pmr convenience alias
template <class T>
using pmr_vec = std::pmr::vector<T>;




} // namespace ad::memory
