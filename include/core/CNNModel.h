#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class CNNModel
{
public:
	CNNModel(const std::string& model_path);
	~CNNModel();

	std::vector<float> Run(const std::vector<float>& input_af, const std::vector<float>& input_pw);
	std::vector<float> RunBatch(const std::vector<float>& input_af, const std::vector<float>& input_pw);

private:
	Ort::Env env_;
	Ort::Session* session_;
	Ort::SessionOptions sessionOptions_;

	std::vector<std::string> inputNamesStr_;
	std::vector<std::string> outputNamesStr_;
	std::vector<std::vector<int64_t>> inputShapes_;
	std::vector<std::vector<int64_t>> outputShapes_;

	void InitIoInfo(); // 读取模型的真实 IO 名称与形状
	static size_t KnownElementCount(const std::vector<int64_t>& shape);
	static std::vector<int64_t> ResolveShape(const std::vector<int64_t>& model_shape, size_t element_count);
};
