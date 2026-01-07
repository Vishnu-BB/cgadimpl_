#include "ad/core/arena.hpp"
#include <cassert>




namespace ad::memory {




Arena::Arena(std::size_t capacity_bytes, std::pmr::memory_resource* upstream)
: upstream_(upstream) {
buffer_.resize(capacity_bytes);
base_ = buffer_.data();
cap_  = buffer_.size();
head_ = 0;
}




Arena::~Arena() {
// Run all tracked dtors to be polite if users placed non-trivial types
run_dtors_from(0);
}




std::size_t Arena::capacity()  const noexcept { return cap_;  }
std::size_t Arena::used()      const noexcept { return head_; }
std::size_t Arena::remaining() const noexcept { return cap_ - head_; }




Mark Arena::mark() const noexcept { return Mark{ head_ }; }




void Arena::shrink_to(Mark m) noexcept {
run_dtors_from(m.offset);
head_ = m.offset;






 // Drop any stale dtor entries for regions >= head_
 while (!dtors_.empty() && dtors_.back().offset >= head_) {
   dtors_.pop_back();
 }
}




void Arena::reset() noexcept {
 run_dtors_from(0);
 head_ = 0;
 dtors_.clear();
}




void* Arena::allocate_raw(std::size_t bytes, std::size_t align) {
const std::size_t aligned = align_up(head_, align);
if (aligned > cap_ || bytes > cap_ - aligned) {
  throw std::bad_alloc{};
}
void* p = base_ + aligned;
head_   = aligned + bytes;
return p;
}




void* Arena::do_allocate(std::size_t bytes, std::size_t alignment) {
return allocate_raw(bytes, alignment);
}




void Arena::do_deallocate(void*, std::size_t, std::size_t) {
// no-op; reclamation happens via reset()/shrink_to()
}




bool Arena::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
return this == &other;
}

void Arena::register_dtor(void* p, DtorFn fn) {
const auto off = static_cast<std::size_t>(static_cast<std::byte*>(p) - base_);
dtors_.push_back({ off, fn });
}

void Arena::run_dtors_from(std::size_t from) noexcept {
// Run dtors in reverse construction order for objects at offsets >= from
while (!dtors_.empty() && dtors_.back().offset >= from) {
  void* ptr = base_ + dtors_.back().offset;
  dtors_.back().fn(ptr);
  dtors_.pop_back();
}
}




} // namespace ad::memory
