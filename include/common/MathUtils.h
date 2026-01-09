#pragma once
#include <vector>

inline float ParallelImpedance(const std::vector<float>& imps, const std::vector<int>& idx) {
    float sum = 0.0f;
    for(int i : idx) {
        sum += 1.0f / imps[i];
    }

    return 1.0f / sum;
}