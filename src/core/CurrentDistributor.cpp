#include "core/CurrentDistributor.h"
#include "config/LeadConfig.h"
#include "common/MathUtils.h"

CurrentResult CurrentDistributor::Compute(
    LeadType leadType, 
    const std::vector<float>& polarity,
    int mode
) {
    CurrentResult result;
    result.amp.resize(polarity.size(), 0.0f);

    // 1. 读取配置
    auto cfg = LeadConfig::GetLeadImpedance(leadType);

    // 2. 找阴极/阳极
    std::vector<int> cathode, anode;
    for(size_t i = 0; i < polarity.size(); ++i) {
        if(polarity[i] < 0.0f) cathode.push_back(i);
        if(polarity[i] > 0.0f) anode.push_back(i);
    }

    // 3. 计算并联阻抗
    float Zc = ParallelImpedance(cfg.contactImp, cathode);
    float Za = anode.empty() ? 0.0f : ParallelImpedance(cfg.contactImp, anode);

    // 4. 计算配置总抗阻
    float ZConfig = 0.0f;

    if(!anode.empty()) {
        // 双极刺激
        ZConfig = Zc + Za;
    } else {
        // 单极刺激
        switch(leadType) {
        case LeadType::PINS301:
        case LeadType::PINS307:
            ZConfig = Zc + Imp05_Rcase;
            break;
        case LeadType::PINS302:
        case LeadType::PINS308:
            ZConfig = Zc + Imp15_Rcase;
            break;
        case LeadType::PINS305:
            bool onlyDirectional = std::abs(polarity[0]) == 0.0f &&
                std::abs(polarity[7]) == 0.0f;
            bool onlyRing = true;
            for (int i = 1; i <= 6; ++i) {
                if (std::abs(polarity[i]) != 0.0f) {
                    onlyRing = false;
                    break;
                }
            }

            if(onlyDirectional) {
                ZConfig = Zc + Imp05_Dcase;
            } else if (onlyRing) {
                ZConfig = Zc + Imp05_Rcase;
            } else {
                ZConfig = Zc + Imp05_RDcase;
            }
            break;
        case LeadType::PINS306:
            bool onlyDirectional = std::abs(polarity[0]) == 0.0f &&
                std::abs(polarity[7]) == 0.0f;
            bool onlyRing = true;
            for (int i = 1; i <= 6; ++i) {
                if (std::abs(polarity[i]) != 0.0f) {
                    onlyRing = false;
                    break;
                }
            }

            if(onlyDirectional) {
                ZConfig = Zc + Imp15_Dcase;
            } else if (onlyRing) {
                ZConfig = Zc + Imp15_Rcase;
            } else {
                ZConfig = Zc + Imp15_RDcase;
            }
            break;
        default:
            ZConfig = Zc;
            break;
        }
    }

    // 5. 电压/电流模式
    float totalCurrent = 0.0f;

    if(mode == 1) {
        // 电压模式计算电压，转换为电流
        totalCurrent = 1.0f / ZConfig;
        result.maxCurrent = 0.0f;
    } else if(mode == 2) {
        // 电流模式，非独立电流源，同极性均分电流
        totalCurrent = 1.0f;
        result.maxCurrent = 10.0f / ZConfig;
    }

    // 6. 计算电流分配
    if(!anode.empty()) {
        float voltageAnode = totalCurrent * Za;
        for(int idx : anode) {
            result.amp[idx] = voltageAnode / cfg.contactImp[idx];
        }
    }

    float voltageCathode = totalCurrent * Zc;
    for(int idx : cathode) {
        result.amp[idx] = -voltageCathode / cfg.contactImp[idx];
    }

    return result;
}
