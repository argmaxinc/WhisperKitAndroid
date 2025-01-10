//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#if GPU_DELEGATE
#include "tflite_model.hpp"
#include "tensorflow/lite/delegates/gpu/delegate.h"

class TFLiteGPU: public TFLiteModel {
public:
    TFLiteGPU(const std::string& name);
    virtual ~TFLiteGPU();

    virtual bool initialize(
        std::string model_path, 
        std::string lib_dir,
        std::string cache_path, 
        int backend, 
        bool debug=false
    );
    virtual void uninitialize();

protected: 
    virtual bool create_interpreter_delegate(std::string model_path);
};
#endif
