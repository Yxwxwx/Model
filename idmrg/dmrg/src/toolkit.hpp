#pragma once
#ifndef TOOLKIT_HPP
#define TOOLKIT_HPP

#include <cassert>
#include <fmt/core.h>
#include <fmt/format.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace DMRG::Toolkit {

// Print formatted values
/**
 * 打印格式化的值。
 *
 * 这个模板函数根据传入的值的类型进行格式化打印。如果是整数类型，它将打印一个宽度为
 * 12 的零填充整数。如果是浮点类型，它将打印一个宽度为 0，精度为 10
 * 的浮点数格式。
 *
 * @tparam T 要打印的值的类型，可以是整数类型或浮点类型。
 * @param value 要打印的值。
 */
template <typename T>
void print_formatted(const T& value)
{
    if constexpr (std::is_integral_v<T>) {
        fmt::print("{: 12}", value); // Zero-padded integers
    }
    else if constexpr (std::is_floating_point_v<T>) {
        fmt::print("{: 0.10f}", value);
    }
}

// Recursive tensor printing
/**
 * 递归打印张量的内容
 *
 * @param tensor 要打印的张量对象
 * @param shape 张量的形状，一个整数向量，表示每个维度的大小
 * @param dim 当前递归到的维度
 * @param indices 当前维度的索引向量
 * @param indent 缩进字符串，用于表示递归层次
 * @param is_last 是否是当前维度的最后一个元素
 */
template <typename TensorType>
void print_recursive(const TensorType& tensor,
    const typename TensorType::Dimensions& shape, size_t dim,
    std::vector<size_t> indices, const std::string& indent,
    bool is_last)
{
    if (dim == shape.size()) {
        Eigen::array<Eigen::Index, TensorType::NumDimensions> eigen_indices;
        std::copy(indices.begin(), indices.end(), eigen_indices.begin());
        print_formatted(tensor(eigen_indices));
    }
    else {
        fmt::print("[");
        std::string new_indent = indent + " ";
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices.emplace_back(i);
            print_recursive(tensor, shape, dim + 1, indices, new_indent,
                i == shape[dim] - 1);
            indices.pop_back();
            if (i < shape[dim] - 1) {
                fmt::print(" ");
            }
        }
        fmt::print("]");
        if (!is_last) {
            fmt::print("\n{}", indent);
        }
    }
}

/**
 * 打印张量的内容
 *
 * 这个模板函数递归地打印张量的内容。它首先打印一个空格，然后调用`print_recursive`函数来打印张量的内容。`print_recursive`函数根据张量的维度和当前维度的索引，递归地打印张量中的每个元素。最后，`print_tensor`函数在打印完张量内容后，打印一个换行符。
 *
 * @tparam TensorType 张量的类型，需要支持.dimensions()成员函数来获取维度信息
 * @param tensor 要打印的张量对象
 */
template <typename TensorType>
void print_tensor(const TensorType& tensor)
{
    const auto& shape = tensor.dimensions();
    fmt::print(" ");
    print_recursive(tensor, shape, 0, std::vector<size_t>(), "", true);
    fmt::print("\n");
}

template <typename T = double>
inline void assign_block(Eigen::Tensor<T, 4>& tensor, const Eigen::Tensor<T, 2>& block,
    const int idx1, const int idx2)
{
    assert(block.dimension(0) == 2 && block.dimension(1) == 2);

    Eigen::TensorMap<const Eigen::Tensor<T, 4>> block_4d(
        block.data(), 1, 1, block.dimension(0), block.dimension(1));

    Eigen::array<Eigen::Index, 4> offsets {
        idx1,
        idx2,
        0, 0
    };
    Eigen::array<Eigen::Index, 4> extents { 1, 1, 2, 2 };

    tensor.slice(offsets, extents) = block_4d;
}
} // namespace DMRG::Toolkit

#endif // TOOLKIT_HPP