#include "data/StdAFLoader.h"
#include <fstream>

std::vector<std::vector<float>>* StdAFLoader::Load(const std::string& filePath, int nx, int ny, int axon) {
    std::vector<std::vector<float>>* table = new std::vector<std::vector<float>>();

    size_t blockSize = (2*axon-1) * nx * ny * 2;

    std::vector<float> blk;
    blk.resize(blockSize);
    
    std::ifstream f(filePath, std::ios::binary);
    f.read(reinterpret_cast<char*>(blk.data()),
           blockSize * sizeof(float));

    table->push_back(std::move(blk));
    return table;
}