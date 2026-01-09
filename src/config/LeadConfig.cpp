#include <stdexcept>
#include "config/LeadConfig.h"

LeadImpedance LeadConfig::GetLeadImpedance(LeadType lead) {
    LeadImpedance cfg;

    switch (lead) {
    case LeadType::PINS301:
        cfg.contactImp = std::vector<float>(4, Imp05_R);
        break;
    case LeadType::PINS302:
        cfg.contactImp = std::vector<float>(4, Imp15_R);
        break;
    case LeadType::PINS305:
        cfg.contactImp = { Imp05_R, Imp05_D, Imp05_D, Imp05_D, Imp05_D, Imp05_D, Imp05_D, Imp05_R};
        break;
    case LeadType::PINS306:
        cfg.contactImp = { Imp15_R, Imp15_D, Imp15_D, Imp15_D, Imp15_D, Imp15_D, Imp15_D, Imp15_R};
        break;
    case LeadType::PINS307:
        cfg.contactImp = std::vector<float>(8, Imp05_R);
        break;
    case LeadType::PINS308:
        cfg.contactImp = std::vector<float>(8, Imp15_R);
        break;
    default:
        throw std::runtime_error("LeadType not implemented");
    }

    return cfg;
}

int LeadConfig::GridNz(LeadType leadType) {
    if(leadType == LeadType::PINS307 || leadType == LeadType::PINS308) {
        return 88;
    }

    return 64;
}