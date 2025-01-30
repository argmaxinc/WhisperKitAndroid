#pragma once

#include <string>
#include <optional>
#include <memory>
#include <unordered_map>
#include <any>

namespace WhisperKit {
namespace Delegates {

enum BackendType {
    WHISPERKIT_BACKEND_NPU_QCOM = 9,
    WHISPERKIT_BACKEND_GPU = 10,
    WHISPERKIT_BACKEND_CPU = 11,
    WHISPERKIT_BACKEND_EXPERIMENTAL = 12,
};

} // namespace Delegates
} // namespace WhisperKit

using namespace WhisperKit::Delegates;


// TODO: move in to separate file (ditto with enum above)
class BaseDelegateOptions {
public:
    virtual ~BaseDelegateOptions();
    BaseDelegateOptions() = default;
    virtual std::any get_options() = 0;
    virtual void set_value_for_option(const std::string& key, const std::string& value) = 0;
    virtual std::string get_value_for_option(const std::string& key) const = 0;
protected:
    std::any options_;
};

class NpuOptionsImpl : public BaseDelegateOptions {
    public:
        NpuOptionsImpl();
        ~NpuOptionsImpl();
        std::any get_options() override;
        void set_value_for_option(const std::string& key, const std::string& value) override;
        std::string get_value_for_option(const std::string& key) const override;
};

class GpuOptionsImpl : public BaseDelegateOptions {
    public: 
        GpuOptionsImpl();
        ~GpuOptionsImpl();  
        std::any get_options() override;
        void set_value_for_option(const std::string& key, const std::string& value) override;
        std::string get_value_for_option(const std::string& key) const override;
};

class CpuOptionsImpl : public BaseDelegateOptions {
    public:
        CpuOptionsImpl();
        ~CpuOptionsImpl();
        std::any get_options() override;
        void set_value_for_option(const std::string& key, const std::string& value) override;
        std::string get_value_for_option(const std::string& key) const override;
};


class DelegateManagerConfiguration {

    public:
        DelegateManagerConfiguration() = default;
        std::shared_ptr<BaseDelegateOptions> getDelegateOptionsForBackend(BackendType backend);

    private:
        std::unordered_map<BackendType, std::shared_ptr<BaseDelegateOptions>> delegate_options_;

};

// in common.h from tflite
struct TfLiteDelegate;

class DelegateManager {

public:

    DelegateManager();

    TfLiteDelegate* getDelegateForBackend(BackendType backend);
    BaseDelegateOptions* getDelegateOptionsForBackend(BackendType backend);

    void set_lib_dir(const std::string& lib_dir);
    void set_cache_dir(const std::string& cache_dir);
    void set_model_token(const std::string& model_token);

    void initialize(DelegateManagerConfiguration& config);

    ~DelegateManager();

    DelegateManager(const DelegateManager&) = delete;
    DelegateManager& operator=(const DelegateManager&) = delete;

private:

    void checkInitialization() const; 

    DelegateManagerConfiguration configuration;

    // TODO: these should be moved elsewhere and made pass through to the delegate options
    // prior to delegate creation.
    std::string _lib_dir;
    std::string _cache_dir;
    std::string _model_token;
    bool initialized_ = false;

    TfLiteDelegate* npu_delegate_ = nullptr;
    TfLiteDelegate* gpu_delegate_ = nullptr;
    TfLiteDelegate* cpu_delegate_ = nullptr;
    TfLiteDelegate* experimental_delegate_ = nullptr;

};

