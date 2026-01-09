#include "app/VTAEngine.h"

#include <iostream>

VTAEngine::~VTAEngine()
{
    if (af_ != nullptr) {
        delete af_;
        af_ = nullptr;
    }

    if (cnnModel_ != nullptr) {
        delete cnnModel_;
        cnnModel_ = nullptr;
    }
}

void VTAEngine::Init()
{
    af_ = StdAFLoader::Load(dataPath_, nx_, ny_, axon_);

    cnnModel_ = new CNNModel("model.onnx");
}

void VTAEngine::SetLeadType(LeadType leadType)
{
    leadType_ = leadType;
}

int sign(float x) 
{
    if(x > 0.0) return 1;
    if(x < 0.0) return -1;
    return 0;
}

void VTAEngine::SetMiccType(MiccType miccType)
{
    miccType_ = miccType;
}

void VTAEngine::SetPulse(float pw)
{
    pw_ = pw;
}

void VTAEngine::SetFrequency(float freq)
{
    freq_ = freq;
}


int GetNzByLeadType(LeadType lead) {
    switch (lead)
    {
        case LeadType::PINS301:
        case LeadType::PINS302:
        case LeadType::PINS305:
        case LeadType::PINS306:
            return 64;

        case LeadType::PINS307:
        case LeadType::PINS308:
            return 88;

        default:
            return -1; // 或抛异常
    }
}

bool IsAllNonNegative(const std::vector<float>& vec) {
    for (float v : vec) {
        if (v < 0.0f) return false;
    }
    return true;
}

void VTAEngine::Compute()
{
    std::vector<float> currentRatio;
    std::vector<float> contPol;
    
    if (miccType_ == MiccType::NONINDEPENDENT) {
        currentRatio = {1.0f};
        contPol = {-1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    } else {
        currentRatio = {0.3333f, 0.0f, 0.3333f, 0.3333f, 0.0f, 0.0f, 0.0f, 0.0f};
        contPol.resize(currentRatio.size());
        for (size_t i = 0; i < currentRatio.size(); ++i)
        {
            contPol[i] = sign(currentRatio[i]);
        }
    }

    int nz = GetNzByLeadType(leadType_);
    if (nz == -1) {
        std::cerr << "Invalid lead type." << std::endl;
        return;
    }

    std::vector<float> amp4SuperPosition(currentRatio.size());

    if(miccType_ == MiccType::NONINDEPENDENT) {
        CurrentResult result = CurrentDistributor::Compute(leadType_, contPol, mode_);
    } else {
        bool allNonNegative = IsAllNonNegative(contPol);
        for(size_t i = 0; i < currentRatio.size(); ++i)
        {
            amp4SuperPosition[i] = allNonNegative ? -currentRatio[i] : currentRatio[i];
        }
    }

    SuperAFCalculator::Compute(leadType_, af_, amp4SuperPosition, 0.0f, 1.0f);
        std::vector<float> input_af = SuperAFCalculator::Compute(leadType_, af_, 0.0f, 1.0f);
        std::vector<float> input_pw = std::vector<float>(input_af.size(), 1.0f);
        std::vector<float> output = cnnModel_->RunBatch(input_af, input_pw);
    
}