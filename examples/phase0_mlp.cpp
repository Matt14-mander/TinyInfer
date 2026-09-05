#include <iomanip>
#include <iostream>

#include "tinyinfer/tinyinfer.h"

int main() {
    // y = softmax(linear(relu(linear(x, W1, b1)), W2, b2))
    const auto x = tinyinfer::Tensor::from_vector({1, 2}, {1.0F, -2.0F});
    const auto w1 = tinyinfer::Tensor::from_vector({2, 3}, {1.0F, 0.0F, -1.0F,
                                                             1.0F, -1.0F, 1.0F});
    const auto b1 = tinyinfer::Tensor::from_vector({3}, {0.0F, 0.0F, 1.0F});
    const auto w2 = tinyinfer::Tensor::from_vector({3, 2}, {1.0F, 0.0F,
                                                             1.0F, 2.0F,
                                                             0.0F, 1.0F});
    const auto b2 = tinyinfer::Tensor::from_vector({2}, {0.0F, -1.0F});

    const auto hidden = tinyinfer::ops::relu(tinyinfer::ops::linear(x, w1, b1));
    const auto logits = tinyinfer::ops::linear(hidden, w2, b2);
    const auto probabilities = tinyinfer::ops::softmax(logits);

    std::cout << std::fixed << std::setprecision(6)
              << "logits: [" << logits.at(0) << ", " << logits.at(1) << "]\n"
              << "probabilities: [" << probabilities.at(0) << ", " << probabilities.at(1) << "]\n";
    return 0;
}
