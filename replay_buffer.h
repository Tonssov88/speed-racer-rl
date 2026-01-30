#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include <vector>
#include <random>
#include <algorithm>

// Experience structure
struct Experience {
    std::vector<float> state;
    int action;
    float reward;
    std::vector<float> next_state;
    bool done;
    
    Experience(const std::vector<float>& s, int a, float r, 
               const std::vector<float>& ns, bool d)
        : state(s), action(a), reward(r), next_state(ns), done(d) {}
};

// Experience Replay Buffer
class ReplayBuffer {
public:
    ReplayBuffer(int capacity) : capacity_(capacity) {
        buffer_.reserve(capacity);
    }
    
    // Add experience to buffer
    void add(const std::vector<float>& state, int action, float reward,
             const std::vector<float>& next_state, bool done) {
        if (buffer_.size() >= capacity_) {
            buffer_.erase(buffer_.begin());
        }
        buffer_.emplace_back(state, action, reward, next_state, done);
    }
    
    // Sample random batch
    void sample(int batch_size,
                std::vector<std::vector<float>>& states,
                std::vector<int>& actions,
                std::vector<float>& rewards,
                std::vector<std::vector<float>>& next_states,
                std::vector<bool>& dones) {
        
        states.clear();
        actions.clear();
        rewards.clear();
        next_states.clear();
        dones.clear();
        
        states.reserve(batch_size);
        actions.reserve(batch_size);
        rewards.reserve(batch_size);
        next_states.reserve(batch_size);
        dones.reserve(batch_size);
        
        // Random sampling
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, buffer_.size() - 1);
        
        for (int i = 0; i < batch_size; i++) {
            int idx = dis(gen);
            const auto& exp = buffer_[idx];
            
            states.push_back(exp.state);
            actions.push_back(exp.action);
            rewards.push_back(exp.reward);
            next_states.push_back(exp.next_state);
            dones.push_back(exp.done);
        }
    }
    
    int size() const { return buffer_.size(); }
    bool can_sample(int batch_size) const { return buffer_.size() >= batch_size; }
    
private:
    int capacity_;
    std::vector<Experience> buffer_;
};

#endif // REPLAY_BUFFER_H