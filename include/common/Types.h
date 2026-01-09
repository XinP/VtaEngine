#pragma once
#include <vector>
#include <string>
#include <cstdint>

enum class LeadType {
    PINS301,
    PINS302,
    PINS305,
    PINS306,
    PINS307,
    PINS308
};

enum class MiccType {
    NONINDEPENDENT = 0,
    INDEPENDENT = 1
};

enum class Mode {
    VOLTAGE = 1,
    CURRENT = 2
};

inline LeadType LeafFromString(const std::string& str) {
    if (str == "PINS301") return LeadType::PINS301;
    if (str == "PINS302") return LeadType::PINS302;
    if (str == "PINS305") return LeadType::PINS305;
    if (str == "PINS306") return LeadType::PINS306;
    if (str == "PINS307") return LeadType::PINS307;
    if (str == "PINS308") return LeadType::PINS308;
    return LeadType::PINS301;
}
