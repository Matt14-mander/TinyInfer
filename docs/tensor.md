# Tensor fundamentals

The current Tensor owns row-major storage. A Tensor starts contiguous but metadata-only operations such as transpose can produce a non-contiguous layout. Its shape describes the logical dimensions, while its strides map a multidimensional coordinate to a flat storage offset.

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

## Reshape

`reshape(new_shape)` changes the logical shape and recomputes contiguous strides without allocating or copying the data buffer:

```cpp
auto tensor = tinyinfer::Tensor::from_vector({2, 3}, values);
tensor.reshape({3, 2});
tensor.reshape({-1});       // Infer 6 and flatten the Tensor
tensor.reshape({2, -1, 1}); // Infer the middle dimension as 3
```

The new shape must preserve `numel()`. At most one dimension may be `-1`; all other dimensions must be non-negative. Inference is rejected when the known dimensions multiply to zero because the result would be ambiguous.

This operation mutates the Tensor itself and returns `Tensor&` for chaining. It is not a shared-storage View: introducing explicit View semantics remains a separate step.

## Transpose

`transpose(dimension0, dimension1)` swaps two dimensions in place by exchanging their shape and stride entries. It does not allocate or move tensor data:

```cpp
auto matrix = tinyinfer::Tensor::from_vector({2, 3}, values);
matrix.transpose(0, 1);
```

```text
before: shape=[2, 3], strides=[3, 1], data=[[1, 2, 3], [4, 5, 6]]
after:  shape=[3, 2], strides=[1, 3], data=[[1, 4], [2, 5], [3, 6]]
```

The transposed Tensor is normally non-contiguous. Multidimensional indexing, printing, and the Phase 0 reference operators respect its strides. Calling `reshape` on a non-contiguous Tensor is rejected because merely replacing its strides would change the logical element order; transpose it back or introduce an explicit contiguous copy in a later step.

## Contiguous conversion

`contiguous()` materializes the current logical order into standard row-major storage:

```cpp
tensor.transpose(0, 1); // Metadata only; usually non-contiguous
tensor.contiguous();    // Reorders data into a new contiguous buffer
tensor.reshape({-1});   // Reshape is now valid
```

For the transposed matrix `[[1, 4], [2, 5], [3, 6]]`, the physical buffer changes from `[1, 2, 3, 4, 5, 6]` to `[1, 4, 2, 5, 3, 6]`, and strides change from `[1, 3]` to `[2, 1]`. Shape, dtype, and logical values remain unchanged.

The operation mutates the Tensor and returns `Tensor&`. Calling it on an already-contiguous Tensor is a no-op and preserves the existing data pointer. All currently declared dtypes are supported because elements are reordered as raw byte ranges using their dtype size.

## Type-safe data access

The templated access API checks both the supported C++ storage type and the Tensor dtype:

```cpp
auto tensor = tinyinfer::Tensor::from_vector<std::int32_t>(
    {2, 2}, std::vector<std::int32_t>{1, 2, 3, 4});

std::int32_t* storage = tensor.data<std::int32_t>();
auto value = tensor.at<std::int32_t>({1, 0});
```

The current mapping is:

| C++ storage type | TinyInfer dtype |
| --- | --- |
| `float` | Float32 |
| `std::uint16_t` | Float16 raw IEEE 754 binary16 bits |
| `std::int8_t` | Int8 |
| `std::int32_t` | Int32 |

Unsupported C++ types fail at compile time. A supported but mismatched type, such as `data<float>()` on an Int32 Tensor, throws `std::logic_error` at runtime. The untyped `data()` remains available as a low-level escape hatch, while `data_f32()` and the non-template `at()` overloads remain Float32 convenience APIs.

## Tensor views

`view(new_shape)` returns a new Tensor metadata object that shares the source Buffer. It requires contiguous input, preserves the element count, and supports one inferred `-1` dimension:

```cpp
auto tensor = tinyinfer::Tensor::from_vector({2, 3}, values);
auto view = tensor.view({3, -1});
```

Mutating either Tensor changes the shared data. The two Tensor objects can have different shapes and strides, but their `storage().buffer()` values are equal. The Buffer is reference-counted, so a view remains valid after the source Tensor is destroyed.

`narrow(dimension, start, length)` creates a bounded subrange view using the same Buffer, original strides, and a Storage byte offset. It works with contiguous and non-contiguous tensors.

Normal C++ copy construction remains a deep copy. Copying any view creates an independent, contiguous Buffer in logical element order. Calling `contiguous()` on a non-contiguous view also detaches it into a new Buffer; calling it on an already-contiguous view keeps sharing the existing Buffer.
