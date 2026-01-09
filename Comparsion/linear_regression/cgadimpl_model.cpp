#include "ad/ag_all.hpp"
#include "ad/utils/csv_loader.hpp"
#include "ad/optimizer/optim.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

int main() {
    try {
        // 1. Load Data
        std::cout << "Loading data..." << std::endl;
        auto data_tensor = ag::utils::load_csv("advertising.csv", true);
        int num_samples = data_tensor.shape().dims[0];
        int num_cols = data_tensor.shape().dims[1];

        // 2. Split X and y
        // X: TV, Radio, Newspaper (cols 0, 1, 2)
        // y: Sales (col 3)
        std::cout << "Splitting X and y..." << std::endl;
        std::vector<float> x_data;
        std::vector<float> y_data;
        const float* raw_data = data_tensor.data<float>();

        for (int i = 0; i < num_samples; ++i) {
            for (int j = 0; j < 3; ++j) {
                x_data.push_back(raw_data[i * num_cols + j]);
            }
            y_data.push_back(raw_data[i * num_cols + 3]);
        }

        // 3. Manual Train/Test Split (80/20)
        std::cout << "Splitting train and test..." << std::endl;
        int train_size = static_cast<int>(num_samples * 0.8);
        int test_size = num_samples - train_size;

        std::vector<float> x_train_data(x_data.begin(), x_data.begin() + train_size * 3);
        std::vector<float> y_train_data(y_data.begin(), y_data.begin() + train_size);
        std::vector<float> x_test_data(x_data.begin() + train_size * 3, x_data.end());
        std::vector<float> y_test_data(y_data.begin() + train_size, y_data.end());

        // 4. Manual Scaling (StandardScaler)
        std::cout << "Scaling data..." << std::endl;
        std::vector<float> means(3, 0.0f);
        std::vector<float> stds(3, 0.0f);

        for (int j = 0; j < 3; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < train_size; ++i) {
                sum += x_train_data[i * 3 + j];
            }
            means[j] = sum / train_size;

            float sq_sum = 0.0f;
            for (int i = 0; i < train_size; ++i) {
                float diff = x_train_data[i * 3 + j] - means[j];
                sq_sum += diff * diff;
            }
            stds[j] = std::sqrt(sq_sum / train_size);
            if (stds[j] == 0) stds[j] = 1.0f;
        }

        for (int i = 0; i < train_size; ++i) {
            for (int j = 0; j < 3; ++j) {
                x_train_data[i * 3 + j] = (x_train_data[i * 3 + j] - means[j]) / stds[j];
            }
        }
        for (int i = 0; i < test_size; ++i) {
            for (int j = 0; j < 3; ++j) {
                x_test_data[i * 3 + j] = (x_test_data[i * 3 + j] - means[j]) / stds[j];
            }
        }

        // Convert to Tensors
        auto x_train = OwnTensor::Tensor(OwnTensor::Shape{{train_size, 3}}, OwnTensor::Dtype::Float32);
        x_train.set_data(x_train_data);
        auto y_train = OwnTensor::Tensor(OwnTensor::Shape{{train_size, 1}}, OwnTensor::Dtype::Float32);
        y_train.set_data(y_train_data);

        auto x_test = OwnTensor::Tensor(OwnTensor::Shape{{test_size, 3}}, OwnTensor::Dtype::Float32);
        x_test.set_data(x_test_data);
        auto y_test = OwnTensor::Tensor(OwnTensor::Shape{{test_size, 1}}, OwnTensor::Dtype::Float32);
        y_test.set_data(y_test_data);

        // 5. Model Training
        std::cout << "Training model..." << std::endl;
        ag::nn::Linear model(3, 1);
        ag::Adam optimizer(model.parameters(), 0.1f);

        ag::Value x_train_val = ag::make_tensor(x_train, "x_train");
        ag::Value y_train_val = ag::make_tensor(y_train, "y_train");

        for (int epoch = 0; epoch < 1000; ++epoch) {
            optimizer.zero_grad();
            ag::Value pred = model(x_train_val);
            ag::Value loss = ag::mse_loss(pred, y_train_val);
            ag::backward(loss);
            optimizer.step();

            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss.val().to_cpu().data<float>()[0] << std::endl;
            }
        }

        // 6. Evaluation
        std::cout << "Evaluating..." << std::endl;
        ag::Value x_test_val = ag::make_tensor(x_test, "x_test");
        ag::Value y_test_val = ag::make_tensor(y_test, "y_test");
        
        ag::Value test_pred = model(x_test_val);
        ag::Value test_mse = ag::mse_loss(test_pred, y_test_val);
        float mse_val = test_mse.val().to_cpu().data<float>()[0];
        float rmse_val = std::sqrt(mse_val);

        std::cout << "Final Test RMSE: " << rmse_val << std::endl;

        // Baseline (Mean of y_train)
        float y_train_mean = 0.0f;
        for (float val : y_train_data) y_train_mean += val;
        y_train_mean /= train_size;

        float baseline_mse = 0.0f;
        for (float val : y_test_data) {
            float diff = val - y_train_mean;
            baseline_mse += diff * diff;
        }
        baseline_mse /= test_size;
        float baseline_rmse = std::sqrt(baseline_mse);

        std::cout << "Baseline RMSE: " << baseline_rmse << std::endl;

        if (rmse_val < baseline_rmse) {
            std::cout << "Model is good!" << std::endl;
        } else {
            std::cout << "Model is not good." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
