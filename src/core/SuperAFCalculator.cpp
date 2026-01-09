#include "core/SuperAFCalculator.h"

std::vector<float> SuperAFCalculator::Compute(LeadType leadType, 
    const std::vector<std::vector<float>>* af,
    const std::vector<float>& amp,
    int nx, int ny, int nz,
    int laxon
) {
    std::vector<float> superAF;
    size_t voxelSize = nx * ny * nz;
    size_t axonSize = 2 * laxon - 1;


    superAF.assign(axonSize * voxelSize, 0.0f); 

    // 遍历z block
    for(int iFile = 0; iFile < nz/2; ++iFile) {
        // 当前z block 的临时叠加
        std::vector<float> block(axonSize * nx * ny * 2, 0.0f);

        // 遍历激活触点
        for(size_t c = 0; c < amp.size(); ++c) {
            if(amp[c] == 0.0f) continue;

            int afIndex = -1;
            int ic = static_cast<int>(c) + 1;

            switch(leadType) {
            case LeadType::PINS301:
            case LeadType::PINS307:
                afIndex = iFile + 21 + (ic - 1) * (-2);
                break;
            case LeadType::PINS302:
            case LeadType::PINS308:
                afIndex = iFile + 21 + (ic - 1) * (-3);
                break; 
            case LeadType::PINS305:
                if(ic == 1) {
                    afIndex = iFile + 21;
                } else if(ic >= 2 && ic <= 4) {
                    afIndex = 65 + 48 * (ic - 2) + iFile + 4;
                } else if(ic >= 5 && ic <= 7) {
                    afIndex = 65 + 48 * (ic - 5) + iFile + 2;
                } else if(ic == 8) {
                    afIndex = iFile + 15;
                }
                break;
            case LeadType::PINS306:
                if (ic == 1)
                    afIndex = iFile + 21;
                else if (ic >= 2 && ic <= 4)
                    afIndex = 65 + 48 * (ic - 2) + iFile + 3;
                else if (ic >= 5 && ic <= 7)
                    afIndex = 65 + 48 * (ic - 5) + iFile;
                else if (ic == 8)
                    afIndex = iFile + 12;
                break;
            default:
                afIndex = -1;
                break;    
            }

            if(afIndex < 0) continue;

            const auto& tempAF = af->at(afIndex);
            float weight = amp[c];
            for(size_t i = 0; i < tempAF.size(); ++i) {
                block[i] += tempAF[i] * weight;
            }
        }

        int sliceVoxel = nx * ny * 2;
        int dstOffset = iFile * sliceVoxel;
        for(int a = 0; a < axonSize; ++a) {
            memcpy(
                &superAF[a * voxelSize + dstOffset],
                &block[a * sliceVoxel],
                sliceVoxel * sizeof(float)
            );
        }
    }

    return superAF;
}