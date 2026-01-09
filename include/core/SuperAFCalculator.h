#pragma once
#include "data/StdAFLoader.h"
#include "common/Types.h"

class SuperAFCalculator {
public:
    static std::vector<float> Compute(LeadType leadType, 
        const std::vector<std::vector<float>>* af,
        const std::vector<float>& amp,
        int nx, int ny, int nz,
        int laxon
    );
};