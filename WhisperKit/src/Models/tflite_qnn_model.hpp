//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

#if QNN_DELEGATE
#include "tflite_model.hpp"
#include "QnnTFLiteDelegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

class TFLiteQNN: public TFLiteModel {
public:
    TFLiteQNN(const std::string& name);
    virtual ~TFLiteQNN();

    virtual bool initialize(
        std::string model_path, 
        std::string lib_dir,
        std::string cache_path, 
        int backend=kHtpBackend, 
        bool debug=false
    );
    virtual void uninitialize();

protected: 
    TfLiteQnnDelegateOptions _options;

    virtual bool create_interpreter_delegate(std::string model_path);
};
#endif
