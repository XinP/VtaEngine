#pragma once
#include <vector>
#include <string>


class StdAFLoader {
    public: 
        static std::vector<std::vector<float>>* Load(const std::string& filePath, int Nx, int Ny, int Laxon);
};