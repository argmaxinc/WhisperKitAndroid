//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#include <tflite_model.hpp>
#include "QnnTFLiteDelegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

class TFLiteQNN: public TFLiteModel {
public:
    TFLiteQNN(const string& name);
    virtual ~TFLiteQNN();

    virtual bool initialize(
        string model_path, 
        string lib_dir,
        string cache_path, 
        int backend=kHtpBackend, 
        bool debug=false
    );
    virtual void uninitialize();

protected: 
    TfLiteQnnDelegateOptions _options;

    virtual bool create_interpreter_delegate(string model_path);
};
