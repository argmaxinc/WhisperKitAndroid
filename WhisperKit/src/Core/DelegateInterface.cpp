#include <string>
#include "DelegateInterface.hpp"

#include <memory>
#include <any>
#include <string>
#include <optional>
#include "tflite_msg.hpp"
#include "tensorflow/lite/c/common.h"

#if QNN_DELEGATE
#include "QnnTFLiteDelegate.h"
#endif 

#if GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

// XNNPack Delegate should always be available
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

using namespace WhisperKit::Delegates;


BaseDelegateOptions::~BaseDelegateOptions() = default;
/*
    Motivation:
    - We want to be able to configure the delegate options for each backend
    - Delegate options across different vendor owned delegates have no relationship
    - Delegate options have distinct member and types within their struct per each delegate
    - std::any is a better alternative to void*; and the string-string parsing logic 
      will be sufficiently flexible.  We can make it fancier later if needed.
*/

NpuOptionsImpl::NpuOptionsImpl() : BaseDelegateOptions() {
#if QNN_DELEGATE

    auto delegate_options = TfLiteQnnDelegateOptionsDefault();
    delegate_options.backend_type = kHtpBackend;
    delegate_options.htp_options.precision = kHtpFp16;
    delegate_options.htp_options.performance_mode = kHtpHighPerformance; 
    delegate_options.htp_options.useConvHmx = true;
    options_ = delegate_options;
#else
    options_ = std::nullopt;
#endif
}

std::any NpuOptionsImpl::get_options() {
    return options_;
}

NpuOptionsImpl::~NpuOptionsImpl() {

}

void NpuOptionsImpl::set_value_for_option(const std::string& key, const std::string& value) {
    // TODO: Implement this when delegate options are configurable
}

std::string NpuOptionsImpl::get_value_for_option(const std::string& key) const {
    // TODO: Implement this when delegate options are configurable
    return "";
}


GpuOptionsImpl::GpuOptionsImpl() : BaseDelegateOptions() {
#if GPU_DELEGATE

    auto delegate_options = TfLiteGpuDelegateOptionsV2Default();
    delegate_options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
    delegate_options.max_delegated_partitions = 3;
    options_ = delegate_options;
#else
    options_ = std::nullopt;
#endif
}

std::any GpuOptionsImpl::get_options() {
    return options_;
}

GpuOptionsImpl::~GpuOptionsImpl() {
    
}

void GpuOptionsImpl::set_value_for_option(const std::string& key, const std::string& value) {
    // TODO: Implement this when delegate options are configurable
}

std::string GpuOptionsImpl::get_value_for_option(const std::string& key) const {
    // TODO: Implement this when delegate options are configurable
    return "";
}



CpuOptionsImpl::CpuOptionsImpl() : BaseDelegateOptions() {
    auto delegate_options = TfLiteXNNPackDelegateOptionsDefault();
    options_ = delegate_options;
}

CpuOptionsImpl::~CpuOptionsImpl() {
    
}

std::any CpuOptionsImpl::get_options() {
    return options_;
}


void CpuOptionsImpl::set_value_for_option(const std::string& key, const std::string& value) {
    // TODO: Implement this when delegate options are configurable
}

std::string CpuOptionsImpl::get_value_for_option(const std::string& key) const {
    // TODO: Implement this when delegate options are configurable
    return "";
}



std::shared_ptr<BaseDelegateOptions> DelegateManagerConfiguration::getDelegateOptionsForBackend(BackendType backend) {

    if (delegate_options_.contains(backend)) {
        return delegate_options_[backend];
    }

    switch (backend) {
        case BackendType::WHISPERKIT_BACKEND_NPU_QCOM:
            delegate_options_[backend] = std::make_shared<NpuOptionsImpl>();
            break;
        case BackendType::WHISPERKIT_BACKEND_GPU:
            delegate_options_[backend] = std::make_shared<GpuOptionsImpl>();
            break;
        case BackendType::WHISPERKIT_BACKEND_CPU:
            delegate_options_[backend] = std::make_shared<CpuOptionsImpl>();
            break;
        case BackendType::WHISPERKIT_BACKEND_EXPERIMENTAL:
            // fallthrough
        default:
            return nullptr;
    }
    return delegate_options_[backend];
}

DelegateManager::DelegateManager() {

}

void DelegateManager::initialize(DelegateManagerConfiguration& config) {
    configuration = config;
}


TfLiteDelegate* DelegateManager::getDelegateForBackend(WhisperKit::Delegates::BackendType backend) {
    try {
        checkInitialization(); 
    } catch (const std::exception& e) {
        LOGI("DelegateManager::getDelegateForBackend: %s", e.what());
        return nullptr;
    }

    auto delegate_options = configuration.getDelegateOptionsForBackend(backend);

    if (!delegate_options) {
        LOGI("DelegateManager::getDelegateForBackend: No delegate options for backend %d available.", backend);
        return nullptr;
    }

    switch (backend) {
        case WhisperKit::Delegates::BackendType::WHISPERKIT_BACKEND_NPU_QCOM:
        {
#if QNN_DELEGATE
            if (!npu_delegate_) {

                if(_lib_dir.empty() || _cache_dir.empty()) {
                    LOGI("DelegateManager::NPU: lib_dir or cache_dir is not set");
                    return nullptr;
                }

                auto _options = std::any_cast<TfLiteQnnDelegateOptions>(delegate_options->get_options());

                _options.skel_library_dir = _lib_dir.c_str();
                _options.cache_dir = _cache_dir.c_str();
                _options.model_token = _model_token.c_str();

                npu_delegate_ = TfLiteQnnDelegateCreate(&_options);
            }

            return npu_delegate_;
#else
            return nullptr;
#endif
        }
        case WhisperKit::Delegates::BackendType::WHISPERKIT_BACKEND_GPU:
        {
#if GPU_DELEGATE
            if (!gpu_delegate_) {

                if(_cache_dir.empty()) {
                    LOGI("DelegateManager::GPU: cache_dir is not set");
                    return nullptr;
                }

                auto gpu_options = std::any_cast<TfLiteGpuDelegateOptionsV2>(delegate_options->get_options());
                gpu_options.serialization_dir = _cache_dir.c_str();

                gpu_delegate_ = TfLiteGpuDelegateV2Create(&gpu_options);
            }
            return gpu_delegate_;
#else
            return nullptr;
#endif
        }

        case WhisperKit::Delegates::BackendType::WHISPERKIT_BACKEND_CPU:
            {
                if (!cpu_delegate_) {
                    auto cpu_options = std::any_cast<TfLiteXNNPackDelegateOptions>(delegate_options->get_options());
                    cpu_delegate_ = TfLiteXNNPackDelegateCreate(&cpu_options);
                }
                return cpu_delegate_;
            }
        case WhisperKit::Delegates::BackendType::WHISPERKIT_BACKEND_EXPERIMENTAL:
            return nullptr;
        default:
            return nullptr;
    }
}

void DelegateManager::checkInitialization() const {
    if(_lib_dir.empty() || _cache_dir.empty()) {
        throw std::runtime_error("DelegateManager: lib_dir or cache_dir is not set");
    }
}

void DelegateManager::set_lib_dir(const std::string& lib_dir) {
    _lib_dir = lib_dir;
}

void DelegateManager::set_cache_dir(const std::string& cache_dir) {
    _cache_dir = cache_dir;
}

void DelegateManager::set_model_token(const std::string& model_token) {
    _model_token = model_token;
}

DelegateManager::~DelegateManager() {

#if QNN_DELEGATE
    if (npu_delegate_) {
        TfLiteQnnDelegateDelete(npu_delegate_);
    }
#endif 

#if GPU_DELEGATE
    if (gpu_delegate_) {
        TfLiteGpuDelegateV2Delete(gpu_delegate_);
    }
#endif 

    if (cpu_delegate_) {
        TfLiteXNNPackDelegateDelete(cpu_delegate_);
    }
}
