#pragma once

#ifdef _USE_TBB_
#include <tbb/scalable_allocator.h>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <source_location>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef _DEBUG
struct DebugInfo {
  std::string file;
  std::string function;
  int line;
};
#endif

// Base class using CRTP
template <typename T, typename Derived> struct Allocator {
  T *allocate(std::size_t n,
              std::source_location loc = std::source_location::current()) {
    return static_cast<Derived *>(this)->allocate_impl(n, loc);
  }
  void deallocate(void *ptr, std::size_t n) {
    static_cast<Derived *>(this)->deallocate_impl(ptr, n);
  }
  T *reallocate(T *ptr, std::size_t n, std::size_t new_size) {
    return static_cast<Derived *>(this)->reallocate_impl(ptr, n, new_size);
  }
  std::shared_ptr<Allocator<T, Derived>> copy() const {
    return std::make_shared<Derived>(*static_cast<const Derived *>(this));
  }
};

// StackAllocator using CRTP
template <typename T> struct StackAllocator : Allocator<T, StackAllocator<T>> {
  std::size_t size, used, shift;
  T *data;

  StackAllocator(T *ptr, std::size_t max_size)
      : size(max_size), used(0), shift(0), data(ptr) {}
  StackAllocator() : size(0), used(0), shift(0), data(nullptr) {}

  T *allocate_impl(std::size_t n,
                   std::source_location loc = std::source_location::current()) {
    assert(shift == 0);
    if (used + n > size) {
      std::cout << "exceeding allowed memory (size=" << size
                << ", trying to allocate " << n << ") "
                << (std::is_same<T, uint32_t>::value
                        ? " (uint32)"
                        : (std::is_same<T, float>::value ? " (float)"
                                                         : " (double)"))
                << std::endl;
      return nullptr;
    } else {
      return data + (used += n) - n;
    }
  }

  void deallocate_impl(void *ptr, std::size_t n) {
    if (n == 0)
      return;
    if (used < n || ptr != data + used - n) {
      std::cout << "deallocation not happening in reverse order" << '\n';
    } else {
      used -= n;
    }
  }

  T *reallocate_impl(T *ptr, std::size_t n, std::size_t new_n) {
    ptr += shift;
    shift += new_n - n;
    used = used + new_n - n;
    if (ptr == data + used - new_n)
      shift = 0;
    return ptr;
  }

  friend std::ostream &operator<<(std::ostream &os, const StackAllocator &c) {
    os << "SIZE=" << c.size << " PTR=" << c.data << " USED=" << c.used
       << " SHIFT=" << (long)c.shift << std::endl;
    return os;
  }
};

// VectorAllocator using CRTP
template <typename T>
struct VectorAllocator : Allocator<T, VectorAllocator<T>> {
#ifdef _USE_TBB_
  std::vector<
      std::vector<T, tbb::scalable_allocator<T>>,
      tbb::scalable_allocator<std::vector<T, tbb::scalable_allocator<T>>>>
      data; //!< The data blocks allocated using TBB for better threading
            //!< performance.
#else
  std::vector<std::vector<T>> data;
#endif
  std::unordered_map<void *, size_t> ptr_size_map;

#ifdef _DEBUG
  std::unordered_map<void *, DebugInfo> ptr_debug_map;
#endif

  VectorAllocator() = default;

  ~VectorAllocator() {
    if (!ptr_size_map.empty()) {
      std::cerr << "[VectorAllocator] Memory leak detected:\n";
      std::cerr << *this;

      for (auto &kv : ptr_size_map) {
        std::cerr << "  -> ptr=" << kv.first << ", size=" << kv.second << '\n';
#ifdef _DEBUG
        const DebugInfo &dbg = ptr_debug_map[kv.first];
        std::cerr << "    allocated at " << dbg.file << ":" << dbg.line
                  << " in " << dbg.function << "\n";
#endif
      }
    }
  }

  T *allocate_impl(std::size_t n,
                   std::source_location loc = std::source_location::current()) {
    data.emplace_back(n);
    T *ptr = data.back().data();
    ptr_size_map[ptr] = n;
#ifdef _DEBUG
    ptr_debug_map[ptr] = DebugInfo{.file = loc.file_name(),
                                   .function = loc.function_name(),
                                   .line = (int)loc.line()};
#endif
    return ptr;
  }

  void deallocate_impl(void *ptr, std::size_t n) {
#ifdef _DEBUG
    ptr_debug_map.erase(ptr);
#endif
    auto it = ptr_size_map.find(ptr);
    if (it == ptr_size_map.end() || it->second != n) {
      std::cout << "deallocation of unallocated or mismatched address" << '\n';
      std::terminate();
    }
    ptr_size_map.erase(it);
    for (int i = (int)data.size() - 1; i >= 0; i--) {
      if (data[i].data() == ptr) {
        data.erase(data.begin() + i);
        return;
      }
    }
    std::cout << "internal error: ptr not found in data after map check"
              << '\n';
    std::terminate();
  }

  T *reallocate_impl(T *ptr, std::size_t n, std::size_t new_n) {
    auto it = ptr_size_map.find(ptr);
    if (it == ptr_size_map.end() || it->second != n) {
      std::cout << "reallocation of unallocated or mismatched address" << '\n';
      std::terminate();
    }
    for (int i = (int)data.size() - 1; i >= 0; i--) {
      if (data[i].data() == ptr) {
        std::cout << "warning: reallocation in vector allocator may cause "
                     "undefined behavior!"
                  << '\n';
        data[i].resize(new_n);
        ptr_size_map[ptr] = new_n;
        return data[i].data();
      }
    }
    std::cout << "reallocation internal error: ptr not found in data" << '\n';
    std::terminate();
  }

  std::shared_ptr<Allocator<T, VectorAllocator<T>>> copy() const {
    return std::make_shared<VectorAllocator<T>>();
  }

  friend std::ostream &operator<<(std::ostream &os, const VectorAllocator &c) {
    os << "N-ALLOCATED=" << c.data.size() << " USED="
       << std::accumulate(c.data.begin(), c.data.end(), (size_t)0,
                          [](size_t i, const auto &j) { return i + j.size(); })
       << std::endl;
    return os;
  }
};
