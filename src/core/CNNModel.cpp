#include "core/CNNModel.h"

#include <iostream>

CNNModel::CNNModel(const std::string& model_path)
	: env_(ORT_LOGGING_LEVEL_WARNING, "onnx"),
	session_(nullptr)
{
	sessionOptions_.SetInterOpNumThreads(1);
	sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	try
	{

        std::wstring wModelPath = std::wstring(model_path.begin(), model_path.end());
		session_ = new Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
	}
	catch (const Ort::Exception& e)
	{
		std::cerr << "ONNXRuntime Exception: " << e.what() << std::endl;
		throw;
	}

    InitIoInfo();

	std::cout << "ONNX Model Loaded Successfully." << std::endl;
}

CNNModel::~CNNModel()
{
	if (session_)
	{
		delete session_;
	}
}

// 在文件中添加该方法，实现动态读取 IO 名称与形状
void CNNModel::InitIoInfo()
{
    inputNamesStr_.clear();
    outputNamesStr_.clear();
    inputShapes_.clear();
    outputShapes_.clear();

    Ort::AllocatorWithDefaultOptions allocator;

    // 读取输入
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_alloc = session_->GetInputNameAllocated(i, allocator);
        inputNamesStr_.push_back(name_alloc.get());

        auto ti = session_->GetInputTypeInfo(i);
        auto tensor_info = ti.GetTensorTypeAndShapeInfo();
        inputShapes_.push_back(tensor_info.GetShape());
    }

    // 读取输出
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
        outputNamesStr_.push_back(name_alloc.get());

        Ort::TypeInfo ti = session_->GetOutputTypeInfo(i);
        auto tensor_info = ti.GetTensorTypeAndShapeInfo();
        outputShapes_.push_back(tensor_info.GetShape());
    }

    // 形状校验（仅非批量维）
    if (inputShapes_.size() != 2 || outputShapes_.size() < 1)
        throw std::runtime_error("Unexpected IO count. Expect 2 inputs and at least 1 output.");

    // input_af: [-1,41,1]
    if (inputShapes_[0].size() != 3 ||
        (inputShapes_[0][1] > 0 && inputShapes_[0][1] != 41) ||
        (inputShapes_[0][2] > 0 && inputShapes_[0][2] != 1)) {
        throw std::runtime_error("Input 0 shape must be [B,41,1] on non-batch dims.");
    }
    // input_pw: [-1,1]
    if (inputShapes_[1].size() != 2 ||
        (inputShapes_[1][1] > 0 && inputShapes_[1][1] != 1)) {
        throw std::runtime_error("Input 1 shape must be [B,1] on non-batch dims.");
    }
    // output: [-1,1]
    if (outputShapes_[0].size() != 2 ||
        (outputShapes_[0][1] > 0 && outputShapes_[0][1] != 1)) {
        throw std::runtime_error("Output shape must be [B,1] on non-batch dims.");
    }

    
    std::cout << "Model Inputs:\n";
    for (size_t i = 0; i < inputNamesStr_.size(); ++i) {
        std::cout << "   " << inputNamesStr_[i] << " shape=[";
        for (size_t j = 0; j < inputShapes_[i].size(); ++j) {
            std::cout << inputShapes_[i][j] << (j + 1 < inputShapes_[i].size() ? "," : "");
        }
        std::cout << "]\n";
    }
    std::cout << "Model Outputs:\n";
    for (size_t i = 0; i < outputNamesStr_.size(); ++i) {
        std::cout << "   " << outputNamesStr_[i] << " shape=[";
        for (size_t j = 0; j < outputShapes_[i].size(); ++j) {
            std::cout << outputShapes_[i][j] << (j + 1 < outputShapes_[i].size() ? "," : "");
        }
        std::cout << "]\n";
    }
}

size_t CNNModel::KnownElementCount(const std::vector<int64_t>& shape)
{
	size_t n = 1;
	for (auto d : shape) {
		if (d > 0) n *= static_cast<size_t>(d);
	}
	return n;
}

std::vector<int64_t> CNNModel::ResolveShape(const std::vector<int64_t>& model_shape, size_t element_count)
{
	std::vector<int64_t> s = model_shape;

	size_t known = 1;
	int first_dynamic = -1;
	for (int i = 0; i < static_cast<int>(s.size()); ++i) {
		if (s[i] > 0) {
			known *= static_cast<size_t>(s[i]);
		} else if (first_dynamic < 0) {
			first_dynamic = i;
		}
	}

	// 批量大小：元素数 / 已知维积，至少为 1
	size_t batch = known ? (element_count / known) : element_count;
	if (batch == 0) batch = 1;

	for (int i = 0; i < static_cast<int>(s.size()); ++i) {
		if (s[i] <= 0) {
			s[i] = (i == first_dynamic) ? static_cast<int64_t>(batch) : 1;
		}
	}
	return s;
}

std::vector<float> CNNModel::RunBatch(const std::vector<float>& input_af, const std::vector<float>& input_pw)
{
    if (!session_) throw std::runtime_error("ONNX session not initialized.");

    // 计算批量大小；要求 af 长度是 41 的倍数，pw 长度等于批量
    if (input_af.size() % 41 != 0)
        throw std::runtime_error("input_af length must be B*41.");
    size_t B = input_af.size() / 41;
    if (input_pw.size() != B)
        throw std::runtime_error("input_pw length must equal batch size B.");

    // 构造具体形状（将动态批量维设为 B）
    std::vector<int64_t> af_shape = { static_cast<int64_t>(B), 41, 1 };
    std::vector<int64_t> pw_shape = { static_cast<int64_t>(B), 1 };

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    Ort::Value af_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(input_af.data()),
        input_af.size(),
        af_shape.data(),
        af_shape.size()
    );

    Ort::Value pw_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(input_pw.data()),
        input_pw.size(),
        pw_shape.data(),
        pw_shape.size()
    );

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(af_tensor));
    inputs.emplace_back(std::move(pw_tensor));

    std::vector<const char*> input_names;
    for (auto& s : inputNamesStr_)
    {
        input_names.push_back(s.c_str());
    }

	std::vector<const char*> output_names;
	for (auto& s : outputNamesStr_)
		output_names.push_back(s.c_str());

    auto outputs = session_->Run(
        Ort::RunOptions{ nullptr },
        input_names.data(),
        inputs.data(),
        inputs.size(),
        output_names.data(),
        output_names.size()
    );

    // 读取输出 [B,1]
    Ort::TypeInfo out_ti = outputs[0].GetTypeInfo();
    auto out_tensor_info = out_ti.GetTensorTypeAndShapeInfo();
    auto out_shape = out_tensor_info.GetShape();
    size_t out_count = 1;
    for (auto d : out_shape) out_count *= static_cast<size_t>(d > 0 ? d : B); // 动态批量替换为 B

    float* out_ptr = outputs[0].GetTensorMutableData<float>();
    std::vector<float> result(out_ptr, out_ptr + out_count);
    return result;
}

std::vector<float> CNNModel::Run(const std::vector<float>& input_af, const std::vector<float>& input_pw)
{
	if (input_af.size() != 41 || input_pw.size() != 1)
		throw std::runtime_error("Single-sample Run expects af=41, pw=1.");
    return RunBatch(input_af, input_pw);
}