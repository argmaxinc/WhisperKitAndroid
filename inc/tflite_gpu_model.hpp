//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <tflite_model.hpp>
#include "tensorflow/lite/delegates/gpu/delegate.h"

class TFLiteGPU: public TFLiteModel {
public:
    TFLiteGPU(const string& name);
    virtual ~TFLiteGPU();

    virtual bool initialize(
        string model_path, 
        string lib_dir,
        string cache_path, 
        int backend, 
        bool debug=false
    );
    virtual void uninitialize();

protected: 
    virtual bool create_interpreter_delegate(string model_path);
};
