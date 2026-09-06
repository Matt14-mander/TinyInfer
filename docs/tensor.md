# Tensor fundamentals

The current Tensor owns contiguous row-major storage. Its shape describes the logical dimensions, while its strides map a multidimensional coordinate to a flat storage offset.

For a Tensor with shape `[2, 3, 4]`, contiguous strides are `[12, 4, 1]`. Coordinate `[1, 2, 3]` maps to:

```text
offset = 1 * 12 + 2 * 4 + 3 * 1 = 23
```

The API supports flat and multidimensional access:

```cpp
tensor.at(23);          // Flat storage index
tensor.at({1, 2, 3});   // Logical multidimensional index

tinyinfer::Shape index{1, 2, 3};
tensor.at(index) = 42.0F;
```

`offset(indices)` exposes the coordinate conversion for learning and debugging. It rejects a coordinate with the wrong rank, a negative index, or an index outside its dimension.

This first version intentionally supports only contiguous storage. Negative indexing, slicing, views, and non-contiguous strides will be introduced separately so their ownership and layout semantics remain explicit.

## Value semantics

Copying a Tensor performs a deep copy: shape, strides, dtype, and data are copied into independent storage. Changing the copy therefore does not affect the original. Copy assignment uses copy-and-swap so allocation failure cannot leave the destination half-updated.

Moving a Tensor transfers its shape metadata and storage pointer without copying the data buffer. As with standard C++ moved-from objects, the source remains destructible and assignable but its contents should not be inspected.

## Printing

`to_string()` and stream output expose metadata and nested row-major data:

```cpp
std::cout << tensor << '\n';
```

```text
Tensor(shape=[2, 2], dtype=float32, data=[[1, 2], [3.5, -4]])
```

Printing supports every currently declared dtype: Float32, Float16, Int8, and Int32. Float16 storage is converted to readable Float32 values for display.
