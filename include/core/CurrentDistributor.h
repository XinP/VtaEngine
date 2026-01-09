#pragma once
#include <vector>
#include <string>
#include "common/Types.h"

struct CurrentResult {
    std::vector<float> amp;
    float maxCurrent;
};

class CurrentDistributor {
public:
    static CurrentResult Compute(
        LeadType leadType, 
        const std::vector<float>& polarity,
        Mode mode
    );
};