//  For licensing see accompanying LICENSE file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.
#pragma once 

#if defined(QNN_DELEGATE)
#include <tflite_qnn_model.hpp>
#define MODEL_SUPER_CLASS     TFLiteQNN
#else 

#define kUndefinedBackend 0
#define kGpuBackend 1
#define kHtpBackend 2

#if defined(GPU_DELEGATE)   

#include <tflite_gpu_model.hpp>
#define MODEL_SUPER_CLASS     TFLiteGPU

#else

#include <tflite_model.hpp>
#define MODEL_SUPER_CLASS     TFLiteModel

#endif
#endif
