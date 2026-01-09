#pragma once
#include <vector>
#include "CNNModel.h"

class Threshold {
public:
    static std::vector<float> PredictCNN(
        CNNModel& cnnModel,
        float pw,
        const std::vector<float>& superAF,
        int numAxon
    ) {
        std::vector<float> pw4cnn(numAxon, pw / 500.0f);
        
        return cnnModel.RunBatch(superAF, pw4cnn);
    }

    static std::vector<float> CATransform(const std::vector<float>& cathode) {
        std::vector<float> anode(cathode.size());
        for(size_t i = 0; i < cathode.size(); ++i) {
            anode[i] = 4.454f * cathode[i] + 0.05018f;
        }
        return anode;
    }
};