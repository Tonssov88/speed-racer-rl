// racing_trainer.cpp.
#include "raylib.h"
#include "dqn.h"
#include "replay_buffer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <csignal>
#include <chrono>
#include <cstdlib>
#include <algorithm>

// Ctrl+C support.
volatile sig_atomic_t interrupted = 0;

void signal_handler(int signal) {
    (void)signal;
    interrupted = 1;
}

// Track helpers.
bool IsWall(Color color)  { return (color.r == 15 && color.g == 15 && color.b == 15); }
bool IsTrack(Color color) { return (color.r == 35 && color.g == 35 && color.b == 35); }
bool IsGrass(Color color) { return (color.r == 34 && color.g == 177 && color.b == 76); }

float GetFrictionMultiplier(Color color) {
    if (IsWall(color))  return 999.0f;
    if (IsGrass(color)) return 3.0f;
    if (IsTrack(color)) return 1.0f;
    return 1.0f;
}

struct Checkpoint {
    Vector2 start;
    Vector2 end;
    bool crossed;

    bool CheckCrossing(Vector2 prevPos, Vector2 currentPos) {
        float x1 = prevPos.x, y1 = prevPos.y;
        float x2 = currentPos.x, y2 = currentPos.y;
        float x3 = start.x,  y3 = start.y;
        float x4 = end.x,    y4 = end.y;

        float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (fabs(denom) < 0.001f) return false;

        float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
        float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

        return (t >= 0 && t <= 1 && u >= 0 && u <= 1);
    }
};

// LIDAR ray cast
float CastLIDARRay(const Image& trackImage, Vector2 position, float angle, float maxDistance) {
    float distance = 0.0f;
    const float step = 2.0f;

    while (distance < maxDistance) {
        float checkX = position.x + cos(angle) * distance;
        float checkY = position.y + sin(angle) * distance;

        int pixelX = (int)checkX;
        int pixelY = (int)checkY;

        if (pixelX < 0 || pixelX >= trackImage.width ||
            pixelY < 0 || pixelY >= trackImage.height) {
            return distance;
        }

        Color pixel = GetImageColor(trackImage, pixelX, pixelY);
        if (IsWall(pixel)) return distance;

        distance += step;
    }

    return maxDistance;
}

// State: 5 base + 13 short-range lidar(danger) + 5 long-range anticipation (distance) = 23 dims.
std::vector<float> GetState(const Image& trackImage, Vector2 position, float angle, float speed) {
    std::vector<float> state;
    state.reserve(5 + 13 + 5);

    const float MAX_SPEED = 300.0f;
    state.push_back(speed / MAX_SPEED);

    state.push_back(sin(angle));
    state.push_back(cos(angle));

    state.push_back(position.x / (float)trackImage.width);
    state.push_back(position.y / (float)trackImage.height);

// Short-range rays 
    const float LIDAR_RANGE = 200.0f;
    const float REFERENCE_DIST = 50.0f; // Distance at which danger ~= 1.0.

    const float angleOffsets[] = {
        -PI/2, // -90
        -5*PI/12, // -75
        -PI/3, // -60
        -PI/4, // -45
        -PI/6, // -30
        -PI/12, // -15
        0.0f, // 0
        PI/12, // +15
        PI/6, // +30
        PI/4, // +45
        PI/3, // +60
        5*PI/12, // +75
        PI/2 // +90
    };

    for (float offset : angleOffsets) {
        float d = CastLIDARRay(trackImage, position, angle + offset, LIDAR_RANGE);

// Inverse normalization: close walls = high value.
        float danger = 1.0f / ((d / REFERENCE_DIST) + 0.1f);
        float normalized = std::min(1.0f, danger);

        state.push_back(normalized);
    }

// Long-range anticipation rays 
// These help the network "see" a turn earlier without changing your short-range "danger" behavior.
    const float LONG_RANGE = 900.0f; // tune 700..1200 depending on track scale.

    const float anticipateOffsets[] = {
        -PI/6, // -30
        -PI/12, // -15
        0.0f, // 0
        PI/12, // +15
        PI/6 // +30
    };

    for (float offset : anticipateOffsets) {
        float d = CastLIDARRay(trackImage, position, angle + offset, LONG_RANGE);
        float norm = d / LONG_RANGE; // 0..1 where 1 means far/clear.
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        state.push_back(norm);
    }

    return state; // 23.
}

// Training statistics (kept in memory; CSVs are written per milestone window).
struct TrainingStats {
    std::vector<float> episode_rewards;
    std::vector<int>   episode_lengths;
    std::vector<float> episode_losses;
    std::vector<int>   episode_laps;
    std::vector<int>   episode_finishes; // 1 if finished all laps.
};

static inline float DistToCheckpointMid(const std::vector<Checkpoint>& checkpoints, int cpIndex, Vector2 p) {
    const Checkpoint& cp = checkpoints[cpIndex];
    Vector2 mid = { (cp.start.x + cp.end.x) * 0.5f, (cp.start.y + cp.end.y) * 0.5f };
    float dx = mid.x - p.x;
    float dy = mid.y - p.y;
    return sqrtf(dx*dx + dy*dy);
}

struct EvalResult {
    int episodes = 0;

    int finishes = 0; // number of full-race finishes.
    double finish_rate = 0.0; // finishes / episodes.

    double avg_laps = 0.0;

    double avg_steps_finish = 0.0; // among finished episodes.
    double avg_steps_all = 0.0; // all episodes.

    double avg_wall_hits = 0.0; // all episodes.
    double avg_grass_frames = 0.0; // all episodes.

    double avg_score = 0.0; // all episodes.
};

// Greedy evaluation (epsilon=0) – used for best-model selection.
// IMPORTANT: action mapping MUST match training (case 1 is reverse, no braking hack).
static EvalResult EvaluateGreedy(
    DQN& dqn,
    const Image& trackImage,
    const std::vector<Checkpoint>& checkpointsTemplate,
    int evalEpisodes,
    int max_steps,
    float DT
) {
// Physics constants (must match training).
    const float MAX_SPEED = 300.0f;
    const float ACCELERATION = 150.0f;
    const float FRICTION = 50.0f;
    const float TURN_SPEED_BASE = 3.0f;
    const float TURN_SPEED_FACTOR = 0.3f;

// Scoring weights (tune later if you want).
    const float FINISH_BONUS = 100000.0f;
    const float STEP_PENALTY = 1.0f; // per step.
    const float WALL_HIT_PENALTY = 200.0f; // per hit.
    const float GRASS_PENALTY = 50.0f; // per frame on grass.

    EvalResult out;
    out.episodes = evalEpisodes;

    dqn.set_training_mode(false);

    long long sumLaps = 0;
    long long sumStepsAll = 0;

    int finishedCount = 0;
    long long sumStepsFinished = 0;

    long long sumWallHits = 0;
    long long sumGrassFrames = 0;

    double sumScore = 0.0;

    for (int ep = 0; ep < evalEpisodes; ep++) {
        std::vector<Checkpoint> checkpoints = checkpointsTemplate;
        for (auto& cp : checkpoints) cp.crossed = false;

        Vector2 position = {430, 92};
        Vector2 velocity = {0, 0};
        float angle = 0.0f;
        float speed = 0.0f;

        int currentLap = -1;
        const int totalLaps = 3;
        int nextCheckpoint = 0;
        bool raceFinished = false;

        int wallHits = 0;
        int grassFrames = 0;

        std::vector<float> state = GetState(trackImage, position, angle, speed);

        int steps = 0;
        while (!raceFinished && steps < max_steps) {
            Vector2 prevPosition = position;

            auto q_values = dqn.predict(state);
            int action = (int)(std::max_element(q_values.begin(), q_values.end()) - q_values.begin());

            float accelerationInput = 0.0f;
            float steeringInput = 0.0f;

            switch (action) {
                case 0: accelerationInput = 1.0f; break;
                case 1: accelerationInput = -0.4f; break;
                case 2: steeringInput = -1.0f; break;
                case 3: steeringInput =  1.0f; break;
                case 4: accelerationInput = 1.0f; steeringInput = -1.0f; break;
                case 5: accelerationInput = 1.0f; steeringInput =  1.0f; break;
                case 6: break;
            }

            int checkPixelX = (int)position.x;
            int checkPixelY = (int)position.y;
            float surfaceFriction = 1.0f;

            if (checkPixelX >= 0 && checkPixelX < trackImage.width &&
                checkPixelY >= 0 && checkPixelY < trackImage.height) {
                Color surfaceColor = GetImageColor(trackImage, checkPixelX, checkPixelY);
                surfaceFriction = GetFrictionMultiplier(surfaceColor);
            }

            if (surfaceFriction > 2.0f) grassFrames++;

            speed += accelerationInput * ACCELERATION * DT;

            float frictionToApply = FRICTION;
            if (accelerationInput == 0.0f) frictionToApply = FRICTION * surfaceFriction;

            if (speed > 0) {
                speed -= frictionToApply * DT;
                if (speed < 0) speed = 0;
            } else if (speed < 0) {
                speed += frictionToApply * DT;
                if (speed > 0) speed = 0;
            }

            float maxSpeedOnSurface = MAX_SPEED;
            if (surfaceFriction > 2.0f) maxSpeedOnSurface = MAX_SPEED * 0.5f;

            if (speed > maxSpeedOnSurface) speed = maxSpeedOnSurface;
            if (speed < -maxSpeedOnSurface * 0.5f) speed = -maxSpeedOnSurface * 0.5f;

            float speedFactor = 1.0f / (1.0f + fabs(speed) / MAX_SPEED * TURN_SPEED_FACTOR);
            float turnRate = TURN_SPEED_BASE * speedFactor;

            if (fabs(speed) > 1.0f) angle += steeringInput * turnRate * DT * (speed / fabs(speed));

            velocity.x = cos(angle) * speed;
            velocity.y = sin(angle) * speed;

            position.x += velocity.x * DT;
            position.y += velocity.y * DT;

            int pixelX = (int)position.x;
            int pixelY = (int)position.y;

            if (pixelX >= 0 && pixelX < trackImage.width &&
                pixelY >= 0 && pixelY < trackImage.height) {
                Color currentColor = GetImageColor(trackImage, pixelX, pixelY);
                if (IsWall(currentColor)) {
                    wallHits++;
                    position = prevPosition;
                    speed *= -0.3f;
                }
            } else {
                wallHits++;
                position = prevPosition;
                speed *= -0.3f;
            }

            Checkpoint& cp = checkpoints[nextCheckpoint];
            if (cp.CheckCrossing(prevPosition, position)) {
                if (nextCheckpoint == 0) {
                    if (currentLap > 0) {
                        bool allCrossed = true;
                        for (int i = 1; i < (int)checkpoints.size(); i++) {
                            if (!checkpoints[i].crossed) { allCrossed = false; break; }
                        }
                        if (allCrossed) {
                            cp.crossed = true;
                            currentLap++;

                            for (auto& c : checkpoints) c.crossed = false;
                            nextCheckpoint = 1;

                            if (currentLap >= totalLaps) raceFinished = true;
                        } else {
                            cp.crossed = false;
                        }
                    } else {
                        currentLap = 1;
                        cp.crossed = false;
                        nextCheckpoint = 1;
                    }
                } else {
                    if (currentLap > 0 && nextCheckpoint != 0) {
                        cp.crossed = true;
                        nextCheckpoint = (nextCheckpoint + 1) % (int)checkpoints.size();
                    } else {
                        cp.crossed = false;
                    }
                }
            }

            steps++;
            state = GetState(trackImage, position, angle, speed);
        }

        double score = 0.0;
        if (raceFinished) score += FINISH_BONUS;
        score -= (double)steps * STEP_PENALTY;
        score -= (double)wallHits * WALL_HIT_PENALTY;
        score -= (double)grassFrames * GRASS_PENALTY;

        sumScore += score;
        sumLaps += currentLap;
        sumStepsAll += steps;

        sumWallHits += wallHits;
        sumGrassFrames += grassFrames;

        if (raceFinished) {
            finishedCount++;
            sumStepsFinished += steps;
        }
    }

    out.finishes = finishedCount;
    out.finish_rate = (evalEpisodes > 0) ? ((double)finishedCount / (double)evalEpisodes) : 0.0;

    out.avg_laps = (evalEpisodes > 0) ? ((double)sumLaps / (double)evalEpisodes) : 0.0;

    out.avg_steps_all = (evalEpisodes > 0) ? ((double)sumStepsAll / (double)evalEpisodes) : 0.0;
    out.avg_steps_finish = (finishedCount > 0) ? ((double)sumStepsFinished / (double)finishedCount) : 0.0;

    out.avg_wall_hits = (evalEpisodes > 0) ? ((double)sumWallHits / (double)evalEpisodes) : 0.0;
    out.avg_grass_frames = (evalEpisodes > 0) ? ((double)sumGrassFrames / (double)evalEpisodes) : 0.0;

    out.avg_score = (evalEpisodes > 0) ? (sumScore / (double)evalEpisodes) : 0.0;
    return out;
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);

    int MILESTONE_FREQUENCY = 50;
    const int BATCH_SIZE = 32;
    const int REPLAY_BUFFER_SIZE = 50000;

    float LEARNING_RATE = 0.001f;

    const float GAMMA = 0.99f;
    const float EPSILON_START = 1.0f;
    const float EPSILON_END = 0.005f;
    const float EPSILON_DECAY = 0.995f;
    const int WARMUP_EPISODES = 5;

    const int TRAIN_EVERY_N_STEPS = 3;
    const int max_steps = 7500;

    if (argc > 1) {
        MILESTONE_FREQUENCY = std::atoi(argv[1]);
    }

    std::cout << "=== Racing DQN Training (CPU Optimized) ===\n";
    std::cout << "Milestone frequency: " << MILESTONE_FREQUENCY << " episodes\n";
    std::cout << "Batch size: " << BATCH_SIZE << "\n";
    std::cout << "Press Ctrl+C to save and exit gracefully\n";
    std::cout << "==========================================\n\n";

    SetTraceLogLevel(LOG_ERROR);

    Image trackImage = LoadImage("assets/raceTrackFullyWalled.png");
    if (trackImage.data == NULL) {
        std::cerr << "Failed to load track image!\n";
        return 1;
    }

    std::vector<Checkpoint> checkpointsTemplate;
    checkpointsTemplate.push_back({{450,35},  {450,150}, false});
    checkpointsTemplate.push_back({{719,260}, {850,260}, false});
    checkpointsTemplate.push_back({{850,665}, {723,665}, false});
    checkpointsTemplate.push_back({{523,482}, {625,517}, false});
    checkpointsTemplate.push_back({{409,438}, {295,413}, false});
    checkpointsTemplate.push_back({{160, 730}, {220, 815}, false});
    checkpointsTemplate.push_back({{138, 600}, {49, 600}, false});
    checkpointsTemplate.push_back({{138,205}, {49,205},  false});

    const float MAX_SPEED = 300.0f;
    const float ACCELERATION = 150.0f;
    const float FRICTION = 50.0f;
    const float TURN_SPEED_BASE = 3.0f;
    const float TURN_SPEED_FACTOR = 0.3f;
    const float DT = 1.0f / 60.0f;

// UPDATED STATE SIZE: 5 + 13 + 5 = 23.
    const int STATE_SIZE = 23;
    const int ACTION_SIZE = 7;

    DQN dqn(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA);
    ReplayBuffer replay_buffer(REPLAY_BUFFER_SIZE);

// Resume from a checkpoint
    dqn.load_model("models/best_time.pt");
    dqn.set_learning_rate(1e-4f);

float epsilon = EPSILON_START;
    TrainingStats stats;

#ifdef _WIN32
    system("if not exist models mkdir models");
#else
    system("mkdir -p models");
#endif

    auto training_start = std::chrono::steady_clock::now();

    double best_finish_rate = -1.0;
    double best_time_avg_steps_finish = 1e18;
    double best_score = -1e18;

    const int EVAL_EPISODES = 20;
    const int FINISH_RATE_MIN_IMPROVEMENT = 2;
    const double SCORE_MIN_IMPROVEMENT = 500.0;

    bool lr_dropped_once = false;
    bool lr_dropped_twice = false;

    for (int episode = 1; !interrupted; episode++) {
        std::vector<Checkpoint> checkpoints = checkpointsTemplate;
        for (auto& cp : checkpoints) cp.crossed = false;

        Vector2 position = {430, 92};
        Vector2 velocity = {0, 0};
        float angle = 0.0f;
        float speed = 0.0f;

        int currentLap = -1;
        const int totalLaps = 3;
        int nextCheckpoint = 0;
        bool raceFinished = false;

        float episode_reward = 0.0f;
        int episode_steps = 0;
        float total_loss = 0.0f;
        int loss_count = 0;

        int stuckCounter = 0;
        Vector2 lastCheckPosition = position;

        int idleCounter = 0;
        const float V_IDLE = 8.0f;
        const int IDLE_GRACE_FRAMES = 30;
        const float IDLE_PENALTY = 0.02f;

        const int STUCK_CHECK_INTERVAL = 75;
        const float STUCK_DIST_THRESHOLD = 30.0f;
        const int STUCK_STRIKES_MAX = 3;
        const float STUCK_BREAK_PENALTY = 50.0f;

        std::vector<float> state = GetState(trackImage, position, angle, speed);
        if ((int)state.size() != STATE_SIZE) {
            std::cerr << "STATE SIZE MISMATCH at init: got=" << state.size()
                      << " expected=" << STATE_SIZE << "\n";
            UnloadImage(trackImage);
            return 1;
        }

        while (!raceFinished && episode_steps < max_steps && !interrupted) {
            Vector2 prevPosition = position;

            if (episode_steps % STUCK_CHECK_INTERVAL == 0 && episode_steps > 0) {
                float dx = position.x - lastCheckPosition.x;
                float dy = position.y - lastCheckPosition.y;
                float distMoved = sqrtf(dx*dx + dy*dy);

                if (distMoved < STUCK_DIST_THRESHOLD) {
                    stuckCounter++;
                    if (stuckCounter >= STUCK_STRIKES_MAX) {
                        episode_reward -= STUCK_BREAK_PENALTY;
                        break;
                    }
                } else {
                    stuckCounter = 0;
                }
                lastCheckPosition = position;
            }

            int action = 0;
            if ((float)rand() / (float)RAND_MAX < epsilon) {
                action = rand() % ACTION_SIZE;
            } else {
                auto q_values = dqn.predict(state);
                action = (int)(std::max_element(q_values.begin(), q_values.end()) - q_values.begin());
            }

            float accelerationInput = 0.0f;
            float steeringInput = 0.0f;

            switch (action) {
                case 0: accelerationInput = 1.0f; break;
                case 1: accelerationInput = -0.4f; break;
                case 2: steeringInput = -1.0f; break;
                case 3: steeringInput =  1.0f; break;
                case 4: accelerationInput = 1.0f; steeringInput = -1.0f; break;
                case 5: accelerationInput = 1.0f; steeringInput =  1.0f; break;
                case 6: break;
            }

            int checkPixelX = (int)position.x;
            int checkPixelY = (int)position.y;
            float surfaceFriction = 1.0f;

            if (checkPixelX >= 0 && checkPixelX < trackImage.width &&
                checkPixelY >= 0 && checkPixelY < trackImage.height) {
                Color surfaceColor = GetImageColor(trackImage, checkPixelX, checkPixelY);
                surfaceFriction = GetFrictionMultiplier(surfaceColor);
            }

            speed += accelerationInput * ACCELERATION * DT;

            float frictionToApply = FRICTION;
            if (accelerationInput == 0.0f) frictionToApply = FRICTION * surfaceFriction;

            if (speed > 0) {
                speed -= frictionToApply * DT;
                if (speed < 0) speed = 0;
            } else if (speed < 0) {
                speed += frictionToApply * DT;
                if (speed > 0) speed = 0;
            }

            float maxSpeedOnSurface = MAX_SPEED;
            if (surfaceFriction > 2.0f) maxSpeedOnSurface = MAX_SPEED * 0.5f;

            if (speed > maxSpeedOnSurface) speed = maxSpeedOnSurface;
            if (speed < -maxSpeedOnSurface * 0.5f) speed = -maxSpeedOnSurface * 0.5f;

            float speedFactor = 1.0f / (1.0f + fabs(speed) / MAX_SPEED * TURN_SPEED_FACTOR);
            float turnRate = TURN_SPEED_BASE * speedFactor;

            if (fabs(speed) > 1.0f) angle += steeringInput * turnRate * DT * (speed / fabs(speed));

            velocity.x = cos(angle) * speed;
            velocity.y = sin(angle) * speed;

            position.x += velocity.x * DT;
            position.y += velocity.y * DT;

            int pixelX = (int)position.x;
            int pixelY = (int)position.y;
            bool hitWall = false;

            if (pixelX >= 0 && pixelX < trackImage.width &&
                pixelY >= 0 && pixelY < trackImage.height) {
                Color currentColor = GetImageColor(trackImage, pixelX, pixelY);
                if (IsWall(currentColor)) {
                    hitWall = true;
                    position = prevPosition;
                    speed *= -0.3f;
                }
            } else {
                hitWall = true;
                position = prevPosition;
                speed *= -0.3f;
            }

            float reward = 0.0f;

            float distToNextCP = DistToCheckpointMid(checkpoints, nextCheckpoint, position);
            float prevDistToNextCP = DistToCheckpointMid(checkpoints, nextCheckpoint, prevPosition);
            float progress = prevDistToNextCP - distToNextCP;
            reward += progress * 0.1f;

            if (progress > 0.0f){
                reward += fabs(speed) * DT * 0.0075f;
            }

            if (hitWall) reward -= 10.0f;
            if (surfaceFriction > 2.0f) reward -= 2.0f * DT;

            reward -= 0.005f;

            if (fabs(speed) < V_IDLE && progress <= 0.0f) {
                idleCounter++;
                if (idleCounter > IDLE_GRACE_FRAMES) reward -= IDLE_PENALTY;
            } else {
                idleCounter = 0;
            }

            Checkpoint& cp = checkpoints[nextCheckpoint];
            if (cp.CheckCrossing(prevPosition, position)) {
                if (nextCheckpoint == 0) {
                    if (currentLap > 0) {
                        bool allCrossed = true;
                        for (int i = 1; i < (int)checkpoints.size(); i++) {
                            if (!checkpoints[i].crossed) { allCrossed = false; break; }
                        }

                        if (allCrossed) {
                            cp.crossed = true;
                            reward += 50.0f;
                            currentLap++;
                            reward += 200.0f;

                            for (auto& c : checkpoints) c.crossed = false;
                            nextCheckpoint = 1;

                            if (currentLap >= totalLaps) {
                                raceFinished = true;
                                reward += 500.0f;
                            }
                        } else {
                            cp.crossed = false;
                        }
                    } else {
                        currentLap = 1;
                        cp.crossed = false;
                        nextCheckpoint = 1;
                    }
                } else {
                    if (currentLap > 0 && nextCheckpoint != 0) {
                        cp.crossed = true;
                        reward += 50.0f;
                        nextCheckpoint = (nextCheckpoint + 1) % (int)checkpoints.size();
                    } else {
                        cp.crossed = false;
                    }
                }
            }

            if (nextCheckpoint != 0) {
                Checkpoint& finishLine = checkpoints[0];
                if (finishLine.CheckCrossing(prevPosition, position)) reward -= 10.0f;
            }

            episode_reward += reward;
            episode_steps++;

            std::vector<float> next_state = GetState(trackImage, position, angle, speed);
            bool done = raceFinished || episode_steps >= max_steps;

            replay_buffer.add(state, action, reward, next_state, done);

            if (episode >= WARMUP_EPISODES &&
                replay_buffer.can_sample(BATCH_SIZE) &&
                (episode_steps % TRAIN_EVERY_N_STEPS == 0)) {

                std::vector<std::vector<float>> batch_states, batch_next_states;
                std::vector<int> batch_actions;
                std::vector<float> batch_rewards;
                std::vector<bool> batch_dones;

                replay_buffer.sample(BATCH_SIZE, batch_states, batch_actions,
                                    batch_rewards, batch_next_states, batch_dones);

                float loss = dqn.train(batch_states, batch_actions, batch_rewards,
                                    batch_next_states, batch_dones, BATCH_SIZE);
                total_loss += loss;
                loss_count++;
            }

            state = std::move(next_state);
            if (done) break;
        }

        epsilon = std::max(EPSILON_END, epsilon * EPSILON_DECAY);

        stats.episode_rewards.push_back(episode_reward);
        stats.episode_lengths.push_back(episode_steps);
        stats.episode_losses.push_back(loss_count > 0 ? total_loss / loss_count : 0.0f);
        stats.episode_laps.push_back(currentLap);
        stats.episode_finishes.push_back(raceFinished ? 1 : 0);

        if (raceFinished && !lr_dropped_once) {
            dqn.set_learning_rate(3e-4f);
            lr_dropped_once = true;
            std::cout << "LR schedule: first finish detected. Lowering LR to " << dqn.get_learning_rate() << "\n";
        }

        if (!lr_dropped_twice && (int)stats.episode_finishes.size() >= 20) {
            int countFin = 0;
            for (int i = (int)stats.episode_finishes.size() - 20; i < (int)stats.episode_finishes.size(); i++) {
                countFin += stats.episode_finishes[i];
            }
            float finishRate20 = (float)countFin / 20.0f;
            if (finishRate20 >= 0.50f) {
                dqn.set_learning_rate(1e-4f);
                lr_dropped_twice = true;
                std::cout << "LR schedule: finishRate(last20)=" << finishRate20
                            << ". Lowering LR to " << dqn.get_learning_rate() << "\n";
            }
        }

        if (episode % 10 == 0) {
            float avg_reward = 0.0f;
            int window = std::min(10, (int)stats.episode_rewards.size());
            for (int i = 0; i < window; i++) {
                avg_reward += stats.episode_rewards[stats.episode_rewards.size() - 1 - i];
            }
            avg_reward /= window;

            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - training_start);

            std::cout << "Episode: " << episode
                        << " | Reward: " << std::fixed << std::setprecision(2) << episode_reward
                        << " | Avg(10): " << avg_reward
                        << " | Laps: " << currentLap
                        << " | ε: " << std::setprecision(3) << epsilon
                        << " | Steps: " << episode_steps
                        << " | LR: " << std::scientific << dqn.get_learning_rate()
                        << " | Time: " << std::fixed << duration.count() << "s"
                        << std::endl;
        }

        if (episode % MILESTONE_FREQUENCY == 0) {
            std::string model_path = "models/model_episode_" + std::to_string(episode) + ".pt";
            dqn.save_model(model_path);

            std::string stats_path = "models/training_stats_" + std::to_string(episode) + ".csv";
            std::ofstream stats_file(stats_path);

            stats_file << "episode,reward,length,avg_loss,laps,finished\n";

            int window = MILESTONE_FREQUENCY;
            int startEp = std::max(1, episode - window + 1);
            int endEp = episode;

            for (int ep = startEp; ep <= endEp; ep++) {
                int idx = ep - 1;
                if (idx < 0 || idx >= (int)stats.episode_rewards.size()) continue;

                stats_file << ep << ","
                            << stats.episode_rewards[idx] << ","
                            << stats.episode_lengths[idx] << ","
                            << stats.episode_losses[idx] << ","
                            << stats.episode_laps[idx] << ","
                            << stats.episode_finishes[idx] << "\n";
            }
            stats_file.close();

            const int EVAL_MAX_STEPS = max_steps;

            auto eval = EvaluateGreedy(dqn, trackImage, checkpointsTemplate, EVAL_EPISODES, EVAL_MAX_STEPS, DT);
            dqn.set_training_mode(true);

            std::cout << "\n✓ Milestone " << episode << " saved!\n";
            std::cout << "  Model: " << model_path << "\n";
            std::cout << "  Stats: " << stats_path << "\n";
            std::cout << "  Eval (greedy, " << EVAL_EPISODES << " eps)"
                        << " | finishes=" << eval.finishes << "/" << eval.episodes
                        << " (" << std::fixed << std::setprecision(1) << (eval.finish_rate * 100.0) << "%)"
                        << " | avg_laps=" << std::fixed << std::setprecision(2) << eval.avg_laps
                        << " | avg_steps_finish=" << std::fixed << std::setprecision(1) << eval.avg_steps_finish
                        << " | avg_wall_hits=" << std::fixed << std::setprecision(2) << eval.avg_wall_hits
                        << " | avg_grass_frames=" << std::fixed << std::setprecision(1) << eval.avg_grass_frames
                        << " | avg_score=" << std::fixed << std::setprecision(1) << eval.avg_score
                        << "\n\n";

            bool save_finish_rate = false;
            int best_finishes_int = (int)std::round(best_finish_rate * (double)EVAL_EPISODES);
            if (best_finish_rate < 0.0) {
                save_finish_rate = true;
            } else if (eval.finishes >= best_finishes_int + FINISH_RATE_MIN_IMPROVEMENT) {
                save_finish_rate = true;
            } else if (eval.finish_rate > best_finish_rate && eval.finishes > best_finishes_int) {
                save_finish_rate = true;
            }

            if (save_finish_rate) {
                best_finish_rate = eval.finish_rate;
                dqn.save_model("models/best_finish_rate.pt");
                std::cout << "★ Updated best_finish_rate.pt (finish_rate="
                        << std::fixed << std::setprecision(3) << best_finish_rate << ")\n";
            }

            bool save_time = false;
            if (eval.finishes > 0) {
                if (best_time_avg_steps_finish >= 1e17) {
                    save_time = true;
                } else if (eval.avg_steps_finish + 50.0 < best_time_avg_steps_finish) {
                    save_time = true;
                }
            }

            if (save_time) {
                best_time_avg_steps_finish = eval.avg_steps_finish;
                dqn.save_model("models/best_time.pt");
                std::cout << "★ Updated best_time.pt (avg_steps_finish="
                            << std::fixed << std::setprecision(1) << best_time_avg_steps_finish << ")\n";
            }

            bool save_score = false;
            if (best_score <= -1e17) {
                save_score = true;
            } else if (eval.avg_score > best_score + SCORE_MIN_IMPROVEMENT) {
                save_score = true;
            } else if (eval.avg_score > best_score && eval.finish_rate > best_finish_rate) {
                save_score = true;
            }

            if (save_score) {
                best_score = eval.avg_score;
                dqn.save_model("models/best_score.pt");
                std::cout << "★ Updated best_score.pt (avg_score="
                        << std::fixed << std::setprecision(1) << best_score << ")\n";
            }

            std::cout << "\n";
        }
    }

    if (interrupted) {
        std::cout << "\n\nInterrupted! Saving final model...\n";
        dqn.save_model("models/model_final.pt");
        std::cout << "Final model saved. Safe to exit.\n";
    }

    UnloadImage(trackImage);
    return 0;
}
