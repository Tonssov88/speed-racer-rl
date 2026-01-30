// racing_replay.cpp.
#include "raylib.h"
#include "dqn.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>

// Track pixel helpers
bool IsWall(Color color) {
    return (color.r == 15 && color.g == 15 && color.b == 15);
}

bool IsTrack(Color color) {
    return (color.r == 35 && color.g == 35 && color.b == 35);
}

bool IsGrass(Color color) {
    return (color.r == 34 && color.g == 177 && color.b == 76);
}

float GetFrictionMultiplier(Color color) {
    if (IsWall(color)) return 999.0f;
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
        float x3 = start.x, y3 = start.y;
        float x4 = end.x, y4 = end.y;

        float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (fabs(denom) < 0.001f) return false;

        float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
        float u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

        return (t >= 0 && t <= 1 && u >= 0 && u <= 1);
    }
};

// LIDAR sensor with visualization
float CastRay(const Image& trackImage, Vector2 position, float angle, float maxDistance,
              Vector2* hitPoint = nullptr) {
    float step = 2.0f;
    for (float dist = 0; dist < maxDistance; dist += step) {
        float x = position.x + cos(angle) * dist;
        float y = position.y + sin(angle) * dist;

        int pixelX = (int)x;
        int pixelY = (int)y;

        if (pixelX < 0 || pixelX >= trackImage.width || pixelY < 0 || pixelY >= trackImage.height) {
            if (hitPoint) { hitPoint->x = x; hitPoint->y = y; }
            return dist;
        }

        Color surfaceColor = GetImageColor(trackImage, pixelX, pixelY);
        if (IsWall(surfaceColor)) {
            if (hitPoint) { hitPoint->x = x; hitPoint->y = y; }
            return dist;
        }
    }

    if (hitPoint) {
        hitPoint->x = position.x + cos(angle) * maxDistance;
        hitPoint->y = position.y + sin(angle) * maxDistance;
    }
    return maxDistance;
}

static inline const std::vector<float>& LidarOffsetsShort() {
    static const std::vector<float> kOffsets = {
        -PI/2.0f, // -90.
        -5.0f*PI/12.0f, // -75.
        -PI/3.0f, // -60.
        -PI/4.0f, // -45.
        -PI/6.0f, // -30.
        -PI/12.0f, // -15.
        0.0f, // 0.
        PI/12.0f, // +15.
        PI/6.0f, // +30.
        PI/4.0f, // +45.
        PI/3.0f, // +60.
        5.0f*PI/12.0f, // +75.
        PI/2.0f // +90.
    };
    return kOffsets;
}

static inline const std::vector<float>& LidarOffsetsAnticipation() {
    static const std::vector<float> kOffsets = {
        -PI/6.0f, // -30.
        -PI/12.0f, // -15.
        0.0f, // 0.
        PI/12.0f, // +15.
        PI/6.0f // +30.
    };
    return kOffsets;
}

// Get state representation 
std::vector<float> GetState(const Image& trackImage, Vector2 position, float angle, float speed) {
    std::vector<float> state;
    state.reserve(5 + (int)LidarOffsetsShort().size() + (int)LidarOffsetsAnticipation().size());

// Normalize speed
    const float MAX_SPEED = 300.0f;
    state.push_back(speed / MAX_SPEED);

// Angle as sin/cos
    state.push_back(sin(angle));
    state.push_back(cos(angle));

// Position (normalized by image dimensions).
    state.push_back(position.x / trackImage.width);
    state.push_back(position.y / trackImage.height);


    const float LIDAR_RANGE = 200.0f;
    const float REFERENCE_DIST = 50.0f;

    for (float offset : LidarOffsetsShort()) {
        float rayAngle = angle + offset;
        float d = CastRay(trackImage, position, rayAngle, LIDAR_RANGE, nullptr);

        float danger = 1.0f / ((d / REFERENCE_DIST) + 0.1f);
        float normalized = std::min(1.0f, danger);

        state.push_back(normalized);
    }

// ---- Long-range anticipation rays (distance-normalized, matches trainer) ----.
    const float LONG_RANGE = 900.0f;

    for (float offset : LidarOffsetsAnticipation()) {
        float rayAngle = angle + offset;
        float d = CastRay(trackImage, position, rayAngle, LONG_RANGE, nullptr);

        float norm = d / LONG_RANGE; // 0..1, where 1 = far/clear.
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;

        state.push_back(norm);
    }

    return state; // 5 + 13 + 5 = 23 dims.
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: racing_replay <model_path>\n";
        std::cout << "Example: racing_replay models/model_episode_450.pt\n";
        return 1;
    }

    std::string modelPath = argv[1];

    std::cout << "=== Racing DQN Replay ===\n";
    std::cout << "Loading model: " << modelPath << std::endl;

    const int screenWidth = 900;
    const int screenHeight = 900;
    InitWindow(screenWidth, screenHeight, "Speed Racer - AI Replay");

// Physics constants (match trainer).
    const float MAX_SPEED = 300.0f;
    const float ACCELERATION = 150.0f;
    const float FRICTION = 50.0f;
    const float TURN_SPEED_BASE = 3.0f;
    const float TURN_SPEED_FACTOR = 0.3f;

// Load assets
    Image trackImage = LoadImage("assets/raceTrackFullyWalled.png");
    Texture2D trackTexture = LoadTextureFromImage(trackImage);
    Texture2D carTexture = LoadTexture("assets/racecarTransparent.png");

// Setup checkpoints
    std::vector<Checkpoint> checkpoints;
    checkpoints.push_back({{450,35}, {450, 150}, false});
    checkpoints.push_back({{719, 260}, {850, 260}, false});
    checkpoints.push_back({{850, 665}, {723, 665}, false});
    checkpoints.push_back({{523, 482}, {625, 517}, false});
    checkpoints.push_back({{409, 438}, {295, 413}, false});
    checkpoints.push_back({{160, 730}, {220, 815}, false});
    checkpoints.push_back({{138, 600}, {49, 600}, false});
    checkpoints.push_back({{138, 205}, {49, 205}, false});

// Initialize DQN agent
    const int STATE_SIZE = 23; // MUST match trainer now
    const int ACTION_SIZE = 7;
    DQN agent(STATE_SIZE, ACTION_SIZE);

// Load trained model
    try {
        agent.load_model(modelPath);
        agent.set_training_mode(false);
        std::cout << "Model loaded successfully!\n\n";
        std::cout << "Controls:\n";
        std::cout << "  SPACE - Restart episode\n";
        std::cout << "  L     - Toggle LIDAR visualization\n";
        std::cout << "  ESC   - Exit\n";
        std::cout << "==========================================\n\n";
    } catch (const std::exception& e) {
        std::cout << "Error loading model: " << e.what() << std::endl;
        UnloadTexture(carTexture);
        UnloadTexture(trackTexture);
        UnloadImage(trackImage);
        CloseWindow();
        return 1;
    }

// Car state
    Vector2 position = {430, 92};
    Vector2 velocity = {0, 0};
    float angle = 0.0f;
    float speed = 0.0f;

// Race state
    int currentLap = -1;
    int totalLaps = 3;
    float currentLapTime = 0.0f;
    float bestLapTime = 999999.0f;
    std::vector<float> lapTimes;
    bool raceStarted = true;
    bool raceFinished = false;
    int nextCheckpoint = 0;

    bool showLidar = true;

    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        float dt = GetFrameTime();
        Vector2 prevPosition = position;

        if (raceStarted && !raceFinished) currentLapTime += dt;

        if (IsKeyPressed(KEY_SPACE)) {
            position = {430, 92};
            velocity = {0, 0};
            speed = 0;
            angle = 0;
            currentLap = 0;
            currentLapTime = 0;
            raceStarted = true;
            raceFinished = false;
            nextCheckpoint = 0;
            lapTimes.clear();
            bestLapTime = 999999.0f;
            for (auto& checkpoint : checkpoints) checkpoint.crossed = false;
        }

        if (IsKeyPressed(KEY_L)) showLidar = !showLidar;


        if (!raceFinished) {
            std::vector<float> state = GetState(trackImage, position, angle, speed);
            auto qValues = agent.predict(state);
            int action = (int)(std::max_element(qValues.begin(), qValues.end()) - qValues.begin());

            float accelerationInput = 0.0f;
            float steeringInput = 0.0f;

// MUST match trainer mapping exactly:.
// 0 forward, 1 reverse, 2 left, 3 right, 4 fwd+left, 5 fwd+right, 6 nothing.
            switch(action) {
                case 0: accelerationInput = 1.0f; break;
                case 1: accelerationInput = -0.4f; break; // IMPORTANT: no braking hack.
                case 2: steeringInput = -1.0f; break;
                case 3: steeringInput =  1.0f; break;
                case 4: accelerationInput = 1.0f; steeringInput = -1.0f; break;
                case 5: accelerationInput = 1.0f; steeringInput =  1.0f; break;
                case 6: break;
            }

// Surface detection
            int checkPixelX = (int)position.x;
            int checkPixelY = (int)position.y;
            float surfaceFriction = 1.0f;

            if (checkPixelX >= 0 && checkPixelX < trackImage.width &&
                checkPixelY >= 0 && checkPixelY < trackImage.height) {
                Color surfaceColor = GetImageColor(trackImage, checkPixelX, checkPixelY);
                surfaceFriction = GetFrictionMultiplier(surfaceColor);
            }

// Physics update (match trainer).
            speed += accelerationInput * ACCELERATION * dt;

            float frictionToApply = FRICTION;
            if (accelerationInput == 0.0f) frictionToApply = FRICTION * surfaceFriction;

            if (speed > 0) {
                speed -= frictionToApply * dt;
                if (speed < 0) speed = 0;
            } else if (speed < 0) {
                speed += frictionToApply * dt;
                if (speed > 0) speed = 0;
            }

            float maxSpeedOnSurface = MAX_SPEED;
            if (surfaceFriction > 2.0f) maxSpeedOnSurface = MAX_SPEED * 0.5f;

            if (speed > maxSpeedOnSurface) speed = maxSpeedOnSurface;
            if (speed < -maxSpeedOnSurface * 0.5f) speed = -maxSpeedOnSurface * 0.5f;

            float speedFactor = 1.0f / (1.0f + fabs(speed) / MAX_SPEED * TURN_SPEED_FACTOR);
            float turnRate = TURN_SPEED_BASE * speedFactor;

            if (fabs(speed) > 1.0f) angle += steeringInput * turnRate * dt * (speed / fabs(speed));

            velocity.x = cos(angle) * speed;
            velocity.y = sin(angle) * speed;

            position.x += velocity.x * dt;
            position.y += velocity.y * dt;
        }

// Collision detection
        int pixelX = (int)position.x;
        int pixelY = (int)position.y;

        if (pixelX >= 0 && pixelX < trackImage.width &&
            pixelY >= 0 && pixelY < trackImage.height) {
            Color currentColor = GetImageColor(trackImage, pixelX, pixelY);
            if (IsWall(currentColor)) {
                position = prevPosition;
                speed *= -0.3f;
            }
        } else {
            position = prevPosition;
            speed *= -0.3f;
        }

// Checkpoint detection (expected checkpoint only).
        if (raceStarted && !raceFinished) {
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
                            lapTimes.push_back(currentLapTime);
                            if (currentLapTime < bestLapTime) bestLapTime = currentLapTime;

                            currentLap++;
                            currentLapTime = 0.0f;

                            for (auto& checkpoint : checkpoints) checkpoint.crossed = false;
                            nextCheckpoint = 1;

                            if (currentLap >= totalLaps) raceFinished = true;
                        } else {
                            cp.crossed = false;
                        }
                    } else {
                        currentLap = 1;
                        currentLapTime = 0.0f;
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
        }

        BeginDrawing();
            ClearBackground(RAYWHITE);
            DrawTexture(trackTexture, 0, 0, WHITE);

// Draw checkpoints
            for (int i = 0; i < (int)checkpoints.size(); i++) {
                Color cpColor = (i == 0) ? RED : YELLOW;
                if (checkpoints[i].crossed) cpColor = GREEN;
                if (i == nextCheckpoint) cpColor = BLUE;

                DrawLineEx(checkpoints[i].start, checkpoints[i].end, 3, cpColor);

                Vector2 mid = {
                    (checkpoints[i].start.x + checkpoints[i].end.x)/2,
                    (checkpoints[i].start.y + checkpoints[i].end.y)/2
                };
                DrawText(TextFormat("%d", i), (int)mid.x - 10, (int)mid.y - 10, 20, cpColor);
            }

// Draw LIDAR rays (short + long).
            if (showLidar) {
// Short-range (danger rays).
                const float shortRange = 200.0f;
                for (float off : LidarOffsetsShort()) {
                    float rayAngle = angle + off;
                    Vector2 hitPoint;
                    CastRay(trackImage, position, rayAngle, shortRange, &hitPoint);
                    DrawLineV(position, hitPoint, Fade(ORANGE, 0.35f));
                    DrawCircleV(hitPoint, 3, ORANGE);
                }

// Long-range (anticipation rays).
                const float longRange = 900.0f;
                for (float off : LidarOffsetsAnticipation()) {
                    float rayAngle = angle + off;
                    Vector2 hitPoint;
                    CastRay(trackImage, position, rayAngle, longRange, &hitPoint);
                    DrawLineV(position, hitPoint, Fade(BLUE, 0.25f));
                    DrawCircleV(hitPoint, 3, BLUE);
                }
            }

            DrawText("AI Racing!", 10, 10, 20, RED);
            DrawText(TextFormat("Speed: %.0f", fabs(speed)), 10, 30, 20, DARKGRAY);
            DrawText(TextFormat("Lap: %d / %d", currentLap, totalLaps), 10, 50, 20, DARKGRAY);
            DrawText(TextFormat("Time: %.2fs", currentLapTime), 10, 70, 20, DARKGRAY);

            if (bestLapTime < 999999.0f) {
                DrawText(TextFormat("Best: %.2fs", bestLapTime), 10, 90, 20, GOLD);
            }

            DrawText("SPACE - Restart | L - Toggle LIDAR | ESC - Exit", 10, screenHeight - 30, 16, DARKGRAY);

            if (raceFinished && lapTimes.size() >= 3) {
                DrawRectangle(0, 0, screenWidth, screenHeight, Fade(BLACK, 0.7f));
                DrawText("RACE FINISHED!", screenWidth/2 - 120, screenHeight/2 - 40, 30, GOLD);
                DrawText(TextFormat("Total Time: %.2fs", lapTimes[0] + lapTimes[1] + lapTimes[2]),
                         screenWidth/2 - 100, screenHeight/2, 20, WHITE);
                DrawText("Press SPACE to restart", screenWidth/2 - 100, screenHeight/2 + 40, 20, WHITE);
            }

            float carTextureScale = 0.15f;
            Rectangle source = {0, 0, (float)carTexture.width, (float)carTexture.height};
            Rectangle dest = { position.x, position.y, carTexture.width * carTextureScale, carTexture.height * carTextureScale };
            Vector2 origin = { carTexture.width * carTextureScale / 2.0f, carTexture.height * carTextureScale / 2.0f };
            DrawTexturePro(carTexture, source, dest, origin, angle * RAD2DEG, WHITE);
        EndDrawing();
    }

    UnloadTexture(carTexture);
    UnloadTexture(trackTexture);
    UnloadImage(trackImage);
    CloseWindow();
    return 0;
}
