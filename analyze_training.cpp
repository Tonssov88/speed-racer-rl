#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>

struct EpisodeData {
    int episode;
    float reward;
    int length;
    float avgLoss;
    int laps;
};

std::vector<EpisodeData> loadCSV(const std::string& filename) {
    std::vector<EpisodeData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        EpisodeData episode;
        
        std::getline(ss, token, ',');
        episode.episode = std::stoi(token);
        
        std::getline(ss, token, ',');
        episode.reward = std::stof(token);
        
        std::getline(ss, token, ',');
        episode.length = std::stoi(token);
        
        std::getline(ss, token, ',');
        episode.avgLoss = std::stof(token);
        
        std::getline(ss, token, ',');
        episode.laps = std::stoi(token);
        
        data.push_back(episode);
    }
    
    file.close();
    return data;
}

float calculateMean(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : values) sum += v;
    return sum / values.size();
}

float calculateStdDev(const std::vector<float>& values, float mean) {
    if (values.size() <= 1) return 0.0f;
    float sumSquares = 0.0f;
    for (float v : values) {
        float diff = v - mean;
        sumSquares += diff * diff;
    }
    return std::sqrt(sumSquares / (values.size() - 1));
}

std::vector<float> calculateMovingAverage(const std::vector<float>& values, int window) {
    std::vector<float> result;
    if (values.size() < window) return result;
    
    for (size_t i = 0; i <= values.size() - window; i++) {
        float sum = 0.0f;
        for (int j = 0; j < window; j++) {
            sum += values[i + j];
        }
        result.push_back(sum / window);
    }
    return result;
}

void printSeparator() {
    std::cout << std::string(70, '=') << std::endl;
}

void printSection(const std::string& title) {
    std::cout << "\n";
    printSeparator();
    std::cout << title << std::endl;
    printSeparator();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: analyze_training <stats_file.csv>" << std::endl;
        std::cout << "Example: analyze_training models/training_stats_50.csv" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    
    std::cout << "\n=== Racing DQN Training Analysis ===" << std::endl;
    std::cout << "Loading data from: " << filename << std::endl;
    
    std::vector<EpisodeData> data = loadCSV(filename);
    
    if (data.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << data.size() << " episodes" << std::endl;
    
    // Extract metrics
    std::vector<float> rewards, losses;
    std::vector<int> lengths, laps;
    
    for (const auto& ep : data) {
        rewards.push_back(ep.reward);
        losses.push_back(ep.avgLoss);
        lengths.push_back(ep.length);
        laps.push_back(ep.laps);
    }
    
    // Calculate statistics
    float meanReward = calculateMean(rewards);
    float stdReward = calculateStdDev(rewards, meanReward);
    float minReward = *std::min_element(rewards.begin(), rewards.end());
    float maxReward = *std::max_element(rewards.begin(), rewards.end());
    
    float meanLoss = calculateMean(losses);
    float meanLength = calculateMean(std::vector<float>(lengths.begin(), lengths.end()));
    float meanLaps = calculateMean(std::vector<float>(laps.begin(), laps.end()));
    
    int maxLaps = *std::max_element(laps.begin(), laps.end());
    int totalCompletedLaps = 0;
    int episodesCompletingRace = 0;
    
    for (int lap : laps) {
        totalCompletedLaps += lap;
        if (lap >= 3) episodesCompletingRace++;
    }
    
    // Overall Statistics
    printSection("OVERALL STATISTICS");
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Episodes:              " << data.size() << std::endl;
    std::cout << "Mean Reward:           " << meanReward << " ± " << stdReward << std::endl;
    std::cout << "Reward Range:          [" << minReward << ", " << maxReward << "]" << std::endl;
    std::cout << "Mean Episode Length:   " << meanLength << " steps" << std::endl;
    std::cout << "Mean Loss:             " << meanLoss << std::endl;
    std::cout << "Mean Laps Completed:   " << meanLaps << std::endl;
    std::cout << "Max Laps in Episode:   " << maxLaps << std::endl;
    std::cout << "Total Laps Completed:  " << totalCompletedLaps << std::endl;
    std::cout << "Episodes Finishing:    " << episodesCompletingRace << " (" 
              << (100.0f * episodesCompletingRace / data.size()) << "%)" << std::endl;
    
    // Moving averages
    int windowSizes[] = {10, 50, 100};
    
    for (int window : windowSizes) {
        if (data.size() >= window) {
            std::vector<float> ma = calculateMovingAverage(rewards, window);
            if (!ma.empty()) {
                printSection("MOVING AVERAGE (Window = " + std::to_string(window) + ")");
                
                // Show first, middle, and last values
                std::cout << "First " << window << " episodes avg:  " << ma[0] << std::endl;
                if (ma.size() > 1) {
                    std::cout << "Middle avg:                    " << ma[ma.size() / 2] << std::endl;
                    std::cout << "Last " << window << " episodes avg:   " << ma[ma.size() - 1] << std::endl;
                    std::cout << "Improvement:                   " 
                              << (ma[ma.size() - 1] - ma[0]) << " ("
                              << ((ma[ma.size() - 1] - ma[0]) / std::abs(ma[0]) * 100) << "%)" << std::endl;
                }
            }
        }
    }
    
    // Top performing episodes
    printSection("TOP 10 EPISODES");
    
    std::vector<EpisodeData> sortedData = data;
    std::sort(sortedData.begin(), sortedData.end(), 
              [](const EpisodeData& a, const EpisodeData& b) {
                  return a.reward > b.reward;
              });
    
    std::cout << std::setw(10) << "Episode" 
              << std::setw(15) << "Reward"
              << std::setw(12) << "Steps"
              << std::setw(10) << "Laps" << std::endl;
    std::cout << std::string(47, '-') << std::endl;
    
    for (int i = 0; i < std::min(10, (int)sortedData.size()); i++) {
        const auto& ep = sortedData[i];
        std::cout << std::setw(10) << ep.episode
                  << std::setw(15) << ep.reward
                  << std::setw(12) << ep.length
                  << std::setw(10) << ep.laps << std::endl;
    }
    
    // Progress by quarters
    if (data.size() >= 40) {
        printSection("PROGRESS BY QUARTER");
        
        int quarterSize = data.size() / 4;
        std::vector<std::string> quarterNames = {"First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"};
        
        for (int q = 0; q < 4; q++) {
            int start = q * quarterSize;
            int end = (q == 3) ? data.size() : (q + 1) * quarterSize;
            
            std::vector<float> quarterRewards;
            std::vector<float> quarterLaps;
            
            for (int i = start; i < end; i++) {
                quarterRewards.push_back(data[i].reward);
                quarterLaps.push_back(data[i].laps);
            }
            
            float qMeanReward = calculateMean(quarterRewards);
            float qMeanLaps = calculateMean(quarterLaps);
            
            std::cout << "\n" << quarterNames[q] << " (Episodes " << start + 1 << "-" << end << "):" << std::endl;
            std::cout << "  Avg Reward: " << qMeanReward << std::endl;
            std::cout << "  Avg Laps:   " << qMeanLaps << std::endl;
        }
    }
    
    // Learning progress indicators
    printSection("LEARNING INDICATORS");
    
    // Compare first and last 20%
    int compareWindow = std::max(10, (int)(data.size() * 0.2));
    
    std::vector<float> earlyRewards, lateRewards;
    for (int i = 0; i < compareWindow; i++) {
        earlyRewards.push_back(data[i].reward);
        lateRewards.push_back(data[data.size() - compareWindow + i].reward);
    }
    
    float earlyMean = calculateMean(earlyRewards);
    float lateMean = calculateMean(lateRewards);
    float improvement = lateMean - earlyMean;
    float improvementPct = (improvement / std::abs(earlyMean)) * 100;
    
    std::cout << "First " << compareWindow << " episodes avg:  " << earlyMean << std::endl;
    std::cout << "Last " << compareWindow << " episodes avg:   " << lateMean << std::endl;
    std::cout << "Improvement:                 " << improvement << " (" << improvementPct << "%)" << std::endl;
    
    if (improvement > 0) {
        std::cout << "\n✓ Agent is learning! Positive improvement detected." << std::endl;
    } else {
        std::cout << "\n⚠ Agent may need more training or hyperparameter tuning." << std::endl;
    }
    
    // Recommendations
    printSection("RECOMMENDATIONS");
    
    if (episodesCompletingRace == 0) {
        std::cout << "• Agent has not completed any races yet" << std::endl;
        std::cout << "• Recommendation: Train for more episodes (aim for 200-500)" << std::endl;
    } else if (episodesCompletingRace < data.size() * 0.1) {
        std::cout << "• Agent rarely completes races" << std::endl;
        std::cout << "• Recommendation: Continue training to improve consistency" << std::endl;
    } else if (episodesCompletingRace < data.size() * 0.5) {
        std::cout << "• Agent is learning but not yet consistent" << std::endl;
        std::cout << "• Recommendation: Train for 100-200 more episodes" << std::endl;
    } else {
        std::cout << "• Agent is performing well!" << std::endl;
        std::cout << "• Recommendation: Fine-tune with more training or adjust rewards" << std::endl;
    }
    
    if (maxLaps < 3) {
        std::cout << "• Agent has not completed a full race (3 laps)" << std::endl;
    } else {
        std::cout << "• Agent has completed at least one full race!" << std::endl;
    }
    
    if (improvementPct > 50) {
        std::cout << "• Strong learning progress!" << std::endl;
    } else if (improvementPct > 0) {
        std::cout << "• Moderate learning progress" << std::endl;
    } else {
        std::cout << "• Limited learning - may need more episodes or hyperparameter tuning" << std::endl;
    }
    
    // Save summary to file
    std::string summaryPath = filename.substr(0, filename.find_last_of('.')) + "_summary.txt";
    std::ofstream summaryFile(summaryPath);
    
    if (summaryFile.is_open()) {
        summaryFile << "=== Training Summary ===" << std::endl;
        summaryFile << "File: " << filename << std::endl;
        summaryFile << "Episodes: " << data.size() << std::endl;
        summaryFile << "Mean Reward: " << meanReward << std::endl;
        summaryFile << "Mean Laps: " << meanLaps << std::endl;
        summaryFile << "Races Completed: " << episodesCompletingRace << std::endl;
        summaryFile << "Improvement: " << improvementPct << "%" << std::endl;
        summaryFile.close();
        
        std::cout << "\n✓ Summary saved to: " << summaryPath << std::endl;
    }
    
    printSeparator();
    std::cout << "\nAnalysis complete!" << std::endl;
    
    return 0;
}