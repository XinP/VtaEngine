#include <iostream>

#include "OnnxInfer.h"
#include <Windows.h>

std::wstring GetExeDir()
{
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring path(buf);
    size_t pos = path.find_last_of(L"\\/");
    return path.substr(0, pos);
}

int main()
{
    std::wstring model_path = GetExeDir() + L"\\models\\test_model.onnx";
    OnnxInfer infer(model_path);

    // 单样本示例（B=1）：af 41 个元素，pw 1 个元素
    std::vector<float> af;
    af.reserve(3 * 41);


	float af_raw[41][3] = {
		{1.53e-05, -6.66e-06, -1.52e-05},
		{-1.00e-05, -1.61e-05, 1.39e-05},
		{-7.31e-06, -4.83e-06, -4.53e-06},
		{-1.16e-05, 1.23e-05, -2.77e-06},
		{3.25e-06, -1.89e-05, 1.71e-05},
		{-9.84e-06, 1.63e-05, -6.45e-06},
		{-7.17e-07, -4.08e-06, -2.43e-05},
		{-3.98e-06, -1.93e-05, 1.03e-05},
		{-1.35e-05, -2.20e-05, -2.84e-05},
		{-3.26e-05, -5.58e-06, -2.94e-05},
		{8.96e-06, -5.78e-06, 5.68e-06},
		{-1.28e-05, -2.24e-05, -1.98e-05},
		{-2.95e-05, -1.42e-05, -1.89e-05},
		{3.92e-06, -1.26e-05, -1.74e-05},
		{-2.81e-05, -1.70e-05, -2.40e-05},
		{-9.59e-06, -3.01e-05, -4.11e-05},
		{-3.40e-05, -3.61e-05, -1.33e-05},
		{-2.02e-05, -1.02e-05, -3.67e-05},
		{-6.27e-06, -2.94e-05, -1.90e-06},
		{-2.56e-05, -2.77e-05, -4.53e-05},
		{-2.31e-05, -1.26e-05, 8.61e-06},
		{-1.98e-05, -1.55e-05, -4.78e-05},
		{-1.13e-05, -2.28e-05, -2.60e-05},
		{-3.60e-05, -2.93e-05, -1.73e-05},
		{-2.48e-05, -3.42e-05, -1.69e-05},
		{-2.29e-05, -2.71e-05, -3.96e-05},
		{-6.40e-06, 1.01e-05, -4.31e-05},
		{-2.09e-05, -4.31e-05, -1.38e-05},
		{-7.39e-06, -1.76e-05, -4.88e-06},
		{-1.21e-05, 2.30e-06, -4.38e-05},
		{-1.65e-05, -1.81e-05, 3.98e-06},
		{-1.18e-05, -1.25e-05, -1.30e-05},
		{-1.72e-05, -1.54e-05, -1.02e-05},
		{-8.97e-06, -7.86e-07, 7.57e-07},
		{-3.99e-07, -1.12e-05, -6.76e-06},
		{1.61e-06, -1.62e-05, -8.84e-06},
		{-8.00e-06, -8.45e-06, -1.37e-05},
		{3.99e-06, -2.90e-06, -1.02e-05},
		{-2.84e-05, 9.57e-06, 9.43e-06},
		{1.38e-05, 5.34e-06, -7.49e-06},
		{5.37e-06, 7.52e-07, -6.55e-06},
	};

	// 重新排布成 batch major
	for (int b = 0; b < 3; b++) {
		for (int i = 0; i < 41; i++) {
			af.push_back(af_raw[i][b]);
		}
	}

    std::vector<float> pw = { 0.3f, 0.3f, 0.3f };

    std::vector<float> output = {};
    try {
        output = infer.RunBatch(af, pw); // 单样本
    } catch (const std::exception& e) {
        std::cerr << "Run failed: " << e.what() << std::endl;
        return -1;
    }
    
    for (size_t i = 0; i < output.size(); ++i)
    {
        std::cout << "ONNX Output = " << output[i] << std::endl;
    }
    return 0;
}
