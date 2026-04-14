#pragma once

#include <memory>

#include "allocator.hpp"

template <typename T>
struct Parameter {
  std::shared_ptr<VectorAllocator<T>> allocator;
  T* ptr = nullptr;
  std::size_t size = 0;

  // 我们用 mdspan 作为访问接口
  // 对于一维 bias，它是 mdspan<T, extents<size_t, dynamic_extent>>
  // 对于二维 weights，我们可以根据需要特化，或者使用更通用的存储方式
  Parameter(std::shared_ptr<VectorAllocator<T>> alloc, size_t n)
      : allocator(std::move(alloc)), size(n) {
    ptr = allocator->allocate(size);
  }
  ~Parameter() {
    if (ptr && allocator) {
      allocator->deallocate(ptr, size);
    }
  }
  // 禁止拷贝，防止双重释放
  Parameter(const Parameter&) = delete;
  Parameter& operator=(const Parameter&) = delete;

  // 允许移动
  Parameter(Parameter&& other) noexcept
      : allocator(std::move(other.allocator)),
        ptr(other.ptr),
        size(other.size) {
    other.ptr = nullptr;
  }

};  //