//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once

enum ComputeBackend {
    None = 0,
    CPU = 1,
    GPU = 2,
    NPU = 3,
};

#if QNN_DELEGATE
#include "tflite_qnn_model.hpp"
#define MODEL_SUPER_CLASS TFLiteQNN
#else

#if GPU_DELEGATE

#include "tflite_gpu_model.hpp"
#define MODEL_SUPER_CLASS TFLiteGPU

#else

#include "tflite_model.hpp"
#define MODEL_SUPER_CLASS TFLiteModel

#endif
#endif
