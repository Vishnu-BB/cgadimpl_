#include "ad/utils/csv_loader.hpp"
#include <iostream>
#include <cassert>

int main() {
    try {
        std::string filename = "advertising.csv";
        // Assuming we are running from the root or the directory containing the csv
        // For testing, we might need to provide the full path or ensure it's in the CWD
        
        // Let's try to load it
        auto tensor = ag::utils::load_csv(filename, true);
        
        std::cout << "Loaded tensor shape: ";
        for (auto d : tensor.shape().dims) {
            std::cout << d << " ";
        }
        std::cout << std::endl;

        assert(tensor.shape().dims[0] == 200);
        assert(tensor.shape().dims[1] == 4);
        
        std::cout << "First row: ";
        const float* data = tensor.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "CSV Loader Test Passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
