#pragma once
#include "data/StdAFLoader.h"
#include "core/Threshold.h"
#include "core/CurrentDistributor.h"
#include "core/SuperAFCalculator.h"

class VTAEngine {
public:
    void Init();
    void Compute();

    void SetLeadType(LeadType leadType);
    void SetMiccType(MiccType miccType);
    void SetPulse(float pw);
    void SetFrequency(float freq);

    ~VTAEngine();

private:
    std::vector<std::vector<float>>* af_ = nullptr;
    CNNModel* cnnModel_ = nullptr;

    LeadType leadType_ = LeadType::PINS301;

    MiccType miccType_ = MiccType::INDEPENDENT;

    Mode mode_ = Mode::VOLTAGE;

    float pw_ = 100;    // 脉宽
    float freq_ = 130;  // 频率

    int nx_ = 33;
    int ny_ = 33;
    int axon_ = 21;
    std::string dataPath_ = "data/StdAF";
};
