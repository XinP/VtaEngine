#pragma once
#include <vector>

#include "../common/Types.h"

// 间距0.5mm的电极（L301、L305）单触点阻抗  
const float Imp05_R = 0.8703f;
const float Imp05_D = 1.8614f;
const float Imp05_Rcase = 0.2398f;
const float Imp05_Dcase = 0.4308f;
const float Imp05_RDcase = 0.2846f;

// 间距1.5mm的电极（L302、L306）单触点阻抗
const float Imp15_R = 0.9533f;
const float Imp15_D = 1.9200f;
const float Imp15_Rcase = 0.1589f;
const float Imp15_Dcase = 0.3631f;
const float Imp15_RDcase = 0.2101f;


struct LeadImpedance {
    std::vector<float> contactImp;
};

class LeadConfig {
    public:
        static LeadImpedance GetLeadImpedance(LeadType leafType);
        static int GridNz(LeadType leadType);
};