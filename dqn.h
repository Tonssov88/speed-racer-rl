#ifndef DQN_H
#define DQN_H

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <thread>

// Simple MLP policy network.
struct DQNNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    DQNNetImpl(int64_t state_size, int64_t action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(state_size, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, action_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

TORCH_MODULE(DQNNet);

class DQN {
public:
    DQN(int state_size, int action_size, float learning_rate = 0.001f, float gamma = 0.99f)
        : state_size_(state_size),
          action_size_(action_size),
          gamma_(gamma),
          device_(torch::kCPU),

          current_lr_(learning_rate) {

        std::cout << "Using CPU for training" << std::endl;
        torch::set_num_threads(std::thread::hardware_concurrency());
        std::cout << "Using " << torch::get_num_threads() << " CPU threads" << std::endl;

        policy_net_ = DQNNet(state_size_, action_size_);
        target_net_ = DQNNet(state_size_, action_size_);

        policy_net_->to(device_);
        target_net_->to(device_);

        copy_weights(policy_net_, target_net_);
        target_net_->eval();

        optimizer_ = std::make_unique<torch::optim::Adam>(
            policy_net_->parameters(),
            torch::optim::AdamOptions(current_lr_)
        );
    }

// Optional: change LR during training
    void set_learning_rate(float new_lr) {
        current_lr_ = new_lr;
        for (auto& group : optimizer_->param_groups()) {
// AdamOptions is the default group type.
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(current_lr_);
        }
    }

    float get_learning_rate() const { return current_lr_; }


// Gradually blend target network with policy network for stability.
    void soft_update_target(float tau = 0.005f) {
        torch::NoGradGuard no_grad;
        auto source_params = policy_net_->named_parameters();
        auto target_params = target_net_->named_parameters();

        for (auto& param : source_params) {
            auto& target = target_params[param.key()];
// θ' ← τθ + (1-τ)θ'.
            target.copy_(tau * param.value() + (1.0f - tau) * target);
        }
    }

    std::vector<float> predict(const std::vector<float>& state) {
        torch::NoGradGuard no_grad;

        if ((int)state.size() != state_size_) {
            std::cerr << "DQN::predict state size mismatch. got=" << state.size()
                      << " expected=" << state_size_ << std::endl;
        }

        auto state_tensor = torch::from_blob(
            const_cast<float*>(state.data()),
            {1, static_cast<long>(state.size())},
            torch::kFloat
        ).clone().to(device_);

        auto q_values = policy_net_->forward(state_tensor).to(torch::kCPU);

        std::vector<float> result(action_size_);
        auto accessor = q_values.accessor<float, 2>();
        for (int i = 0; i < action_size_; i++) {
            result[i] = accessor[0][i];
        }
        return result;
    }

// Train on a batch of experiences
    float train(const std::vector<std::vector<float>>& states,
                const std::vector<int>& actions,
                const std::vector<float>& rewards,
                const std::vector<std::vector<float>>& next_states,
                const std::vector<bool>& dones,
                int batch_size) {

// Flatten states
        std::vector<float> states_flat;
        states_flat.reserve(batch_size * state_size_);
        for (const auto& s : states) {
            states_flat.insert(states_flat.end(), s.begin(), s.end());
        }
        auto states_tensor = torch::from_blob(
            states_flat.data(),
            {batch_size, state_size_},
            torch::kFloat
        ).clone().to(device_);

// Actions
        std::vector<int64_t> actions_long(actions.begin(), actions.end());
        auto actions_tensor = torch::from_blob(
            actions_long.data(),
            {batch_size, 1},
            torch::kLong
        ).clone().to(device_);

// Rewards
        auto rewards_tensor = torch::from_blob(
            const_cast<float*>(rewards.data()),
            {batch_size, 1},
            torch::kFloat
        ).clone().to(device_);

// Flatten next states
        std::vector<float> next_states_flat;
        next_states_flat.reserve(batch_size * state_size_);
        for (const auto& s : next_states) {
            next_states_flat.insert(next_states_flat.end(), s.begin(), s.end());
        }
        auto next_states_tensor = torch::from_blob(
            next_states_flat.data(),
            {batch_size, state_size_},
            torch::kFloat
        ).clone().to(device_);

// Dones
        std::vector<float> dones_float(dones.begin(), dones.end());
        auto dones_tensor = torch::from_blob(
            dones_float.data(),
            {batch_size, 1},
            torch::kFloat
        ).clone().to(device_);

// Current Q(s,a).
        auto current_q = policy_net_->forward(states_tensor).gather(1, actions_tensor);

// ---- Double DQN target ----
        torch::Tensor next_q;
        {
            torch::NoGradGuard no_grad;

// action selection with policy net.
            auto next_q_policy = policy_net_->forward(next_states_tensor);
            auto next_actions = std::get<1>(next_q_policy.max(1, true)); // [B,1] long.

// action evaluation with target net.
            auto next_q_target = target_net_->forward(next_states_tensor);
            next_q = next_q_target.gather(1, next_actions); // [B,1].
        }

        auto target_q = rewards_tensor + (gamma_ * next_q * (1.0f - dones_tensor));

// Huber loss is typically more stable than MSE, but you asked only steps 1–5.
// Keeping MSE to match your request scope.
        auto loss = torch::mse_loss(current_q, target_q);

        optimizer_->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(policy_net_->parameters(), 1.0);
        optimizer_->step();


        soft_update_target(0.005f);


        return loss.item<float>();
    }

    void update_target_network() { copy_weights(policy_net_, target_net_); }

    void save_model(const std::string& path) {
        torch::save(policy_net_, path);
        std::cout << "Model saved to " << path << std::endl;
    }

    void load_model(const std::string& path) {
        torch::load(policy_net_, path);
        copy_weights(policy_net_, target_net_);
        std::cout << "Model loaded from " << path << std::endl;
    }

    void set_training_mode(bool training) {
        if (training) policy_net_->train();
        else policy_net_->eval();
    }

private:
    void copy_weights(DQNNet& source, DQNNet& target) {
        torch::NoGradGuard no_grad;
        auto source_params = source->named_parameters();
        auto target_params = target->named_parameters();

        for (auto& param : source_params) {
            target_params[param.key()].copy_(param.value());
        }
    }

    int state_size_;
    int action_size_;
    float gamma_;

    DQNNet policy_net_{nullptr};
    DQNNet target_net_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;

    torch::Device device_;

// int update_counter_;.
// int target_update_frequency_ = 1000;.

    float current_lr_;
};

#endif // DQN_H