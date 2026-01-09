#pragma once

#include "tensor.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

namespace ag::utils {

inline OwnTensor::Tensor load_csv(const std::string& filename, bool has_header = true) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    if (has_header) {
        std::getline(file, line); // Skip header
    }

    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int current_cols = 0;
        while (std::getline(ss, value, ',')) {
            try {
                data.push_back(std::stof(value));
                current_cols++;
            } catch (...) {
                // Skip non-numeric values if any
            }
        }
        if (rows == 0) {
            cols = current_cols;
        } else if (current_cols != cols) {
            // Handle inconsistent column counts if necessary
        }
        rows++;
    }

    OwnTensor::Tensor tensor(OwnTensor::Shape{{rows, cols}}, OwnTensor::Dtype::Float32);
    tensor.set_data(data);
    return tensor;
}

} // namespace ag::utils
