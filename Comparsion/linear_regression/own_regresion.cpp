#include<iostream>
#include "ad/ag_all.hpp"
#include "ad/utils/csv_loader.hpp"
#include "ad/optimizer/optim.hpp"
#include<fstream>
#include<sstream>
#include<vector>
 #include<limits>
int main() {
    std::vector<float> X_data;
    std::vector<float> Y_data;
    std::string line;
    std::ifstream file("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/Salary_dataset.csv");
    std::getline(file, line); // Skip header

    while(std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // Index - ignore
        std::getline(ss, cell, ',');  // YearsExperience
        X_data.push_back(std::stof(cell));
        std::getline(ss, cell, ','); // Salary
        Y_data.push_back(std::stof(cell));
    }
    
    size_t n = X_data.size();
    auto opts = OwnTensor::TensorOptions().with_device(OwnTensor::Device::CUDA).with_dtype(OwnTensor::Dtype::Float32);

    // Create tensors
    OwnTensor::Tensor X(OwnTensor::Shape{{(int64_t)n, 1}}, opts);
    X.set_data(X_data);
    std::cout << "X (YearsExperience):" << std::endl;
    X.to_cpu().display();

    OwnTensor::Tensor Y(OwnTensor::Shape{{(int64_t)n, 1}}, opts);
    Y.set_data(Y_data);
    std::cout << "\nY (Salary):" << std::endl;
    Y.to_cpu().display();

    // ========== NORMALIZATION ==========
    float X_min = OwnTensor::reduce_min(X).to_cpu().data<float>()[0];
    float X_max = OwnTensor::reduce_max(X).to_cpu().data<float>()[0];
    std::cout << "\nX min: " << X_min << ", max: " << X_max << std::endl;
    OwnTensor::Tensor X_norm = (X - X_min) / (X_max - X_min);

    float Y_min = OwnTensor::reduce_min(Y).to_cpu().data<float>()[0];
    float Y_max = OwnTensor::reduce_max(Y).to_cpu().data<float>()[0];
    std::cout << "Y min: " << Y_min << ", max: " << Y_max << std::endl;
    OwnTensor::Tensor Y_norm = (Y - Y_min) / (Y_max - Y_min);

    std::cout << "\nX normalized:" << std::endl;
    X_norm.to_cpu().display();
    std::cout << "\nY normalized:" << std::endl;
    Y_norm.to_cpu().display();

    // Create ag::Values
    ag::Value X_Value = ag::make_tensor(X_norm, "X");
    ag::Value Y_Value = ag::make_tensor(Y_norm, "Y");


    ag::nn::Linear layer1(1, 4,OwnTensor::Device::CUDA); //Input(1) --->Hidden(4)
    ag::nn::Linear layer2(4, 8,OwnTensor::Device::CUDA); //Hidden1(4) --->Hidden2(8)
    ag::nn::Linear layer3(8, 1,OwnTensor::Device::CUDA); //Hidden2(8) --->Output(1)

    float learning_rate = 0.1f;
    std::vector<ag::Value> all_params;
    for (auto& p : layer1.parameters()) all_params.push_back(p);
    for (auto& p : layer2.parameters()) all_params.push_back(p);
    for(auto& p : layer3.parameters()) all_params.push_back(p);
    ag::SGDOptimizer optimizer(all_params, learning_rate);
    int epochs = 1000;
float convergence= 0.0000001;
float ploss=std::numeric_limits<float>::infinity();
bool converged = false;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        ag::Value hidden1 = layer1(X_Value);
        ag::Value hidden_relu1=ag::relu(hidden1);
        ag::Value hidden2 = layer2(hidden_relu1);
        ag::Value hidden_relu2=ag::relu(hidden2);
        ag::Value Y_pred = layer3(hidden_relu2);

        

        ag::Value loss = ag::mse_loss(Y_pred, Y_Value);
        float closs= loss.val().to_cpu().data<float>()[0];
        if (epoch % 100 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.val().to_cpu().data<float>()[0] << std::endl;
        }
        if(closs<convergence || (std::abs(ploss-closs)<convergence && closs < 0.005)){
            std::cout<<"loss converged at "<<epoch<<std::endl;
            std::cout<<"loss value at "<<epoch<<" epoch is "<<closs<<std::endl;
            converged=true;
            ag::backward(loss);
            optimizer.step();
            optimizer.zero_grad();
            break;
        }
        ploss=closs;
        ag::backward(loss);
        optimizer.step();
        optimizer.zero_grad();

    }
if(!converged){
    std::cout<<"\n Training completed,loss not converged ,epochs reached,try with diff hyperparameters"<<std::endl;
}
    std::cout << "\n=== TRAINING COMPLETE ===" << std::endl;
int total_params=0;
for(auto& p : layer1.parameters()){
    total_params+=p.val().to_cpu().numel();
}
for(auto& p: layer2.parameters()){
    total_params+=p.val().to_cpu().numel();
}
for(auto& p:layer3.parameters()){
    total_params+=p.val().to_cpu().numel();
}
std::cout<<"Total parameters :" << total_params << std::endl;
 std::cout<<"Layer1 parameters : W1,b1"<<std::endl;
 std::cout<<"layer1 Weights tensor :"<<std::endl;
 layer1.parameters()[0].val().to_cpu().display();
 std::cout<<"layer1 Biases tensor :"<<std::endl;
 layer1.parameters()[1].val().to_cpu().display();
 std::cout<<"Layer2 parameters : W2,b2"<<std::endl;
 std::cout<<"layer2 Weights tensor :"<<std::endl;
 layer2.parameters()[0].val().to_cpu().display();
 std::cout<<"layer2 Biases tensor :"<<std::endl;
 layer2.parameters()[1].val().to_cpu().display();
 std::cout<<"Layer3 parameters : W3,b3"<<std::endl;
 std::cout<<"layer3 Weights tensor :"<<std::endl;
 layer3.parameters()[0].val().to_cpu().display();
 std::cout<<"layer3 Biases tensor :"<<std::endl;
 layer3.parameters()[1].val().to_cpu().display();
    // ========== PREDICTIONS ==========
    ag::Value hidden1=layer1(X_Value);
    ag::Value hidden_relu1=ag::relu(hidden1);
    ag::Value hidden2=layer2(hidden_relu1);
    ag::Value hidden_relu2=ag::relu(hidden2);
    ag::Value final_pred_norm = layer3(hidden_relu2);
    OwnTensor::Tensor Y_pred_norm = final_pred_norm.val();

    // Denormalize
    float Y_range = Y_max - Y_min;
    OwnTensor::Tensor Y_pred = Y_pred_norm * Y_range + Y_min;

    std::cout << "\n=== PREDICTIONS VS ACTUAL ===" << std::endl;
    std::cout << "Sample | Predicted | Actual | Error" << std::endl;
    for (size_t i = 0; i < 5; i++) {
        float pred = Y_pred.to_cpu().data<float>()[i];
        float actual = Y.to_cpu().data<float>()[i];
        float error = actual - pred;
        std::cout << i << "      | " << pred << " | " << actual << " | " << error << std::endl;
    }
    std::cout << "..." << std::endl;

    // ========== R² SCORE ==========
    float Y_mean = OwnTensor::reduce_mean(Y).to_cpu().data<float>()[0];
    std::cout << "\nY mean: " << Y_mean << std::endl;

    OwnTensor::Tensor residuals = Y - Y_pred;
    OwnTensor::Tensor residuals_sq = residuals * residuals;
    float SS_res = OwnTensor::reduce_sum(residuals_sq).to_cpu().data<float>()[0];

    OwnTensor::Tensor deviations = Y - Y_mean;
    OwnTensor::Tensor deviations_sq = deviations * deviations;
    float SS_tot = OwnTensor::reduce_sum(deviations_sq).to_cpu().data<float>()[0];

    float R2 = 1.0f - (SS_res / SS_tot);

    std::cout << "SS_res: " << SS_res << std::endl;
    std::cout << "SS_tot: " << SS_tot << std::endl;
    std::cout << "\n=============================\n";
    std::cout << "R² Score: " << R2 << std::endl;
    std::cout << "R² Percentage: " << (R2 * 100) << "%" << std::endl;
    std::cout << "=============================\n" << std::endl;

    return 0;
}