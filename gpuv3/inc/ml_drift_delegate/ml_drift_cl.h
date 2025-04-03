#ifndef THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_CL_H_
#define THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_CL_H_

#include "ml_drift_delegate/ml_drift_delegate.h"
#include "tensorflow_gpuv3/lite/core/c/common.h"

#ifdef __cplusplus
#include <memory>

extern "C" {
#endif  // __cplusplus

// TFLite ML Drift OpenCL delegate C API.
//
// Typical usage:
//
//   // Initialize.
//   MlDriftClDelegateOptions* options = MlDriftClDelegateDefaultOptions();
//   TfLiteDelegate* delegate = TfLiteCreateMlDriftClDelegate(options);
//   QCHECK(delegate != nullptr);
//
//   QCHECK_EQ(interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);
//
//   // Run inference.
//   QCHECK_EQ(interpreter->Invoke(), kTfLiteOk);
//
//   // Clean up.
//   interpreter = nullptr;
//   TfLiteDeleteMlDriftClDelegate(delegate);
//

typedef struct {
  MlDriftDelegatePrecision precision;

  // If true, only delegates the node range of `debug_first_delegate_node_index`
  // and `debug_last_delegate_node_index`.
  // Note: This is for debugging purpose.
  bool debug_delegate_partition;
  // This sets the index of the first node that could be delegated.
  int debug_first_delegate_node_index;
  // This sets the index of the last node that could be delegated.
  int debug_last_delegate_node_index;
  // Allows sharing of constant tensors between different subgraphs.
  bool enable_constant_tensors_sharing;
  // If true, the delegate will improve tuning time, but inference can be
  // slower.
  bool enable_fast_tuning;
  // If true, the delegate will enable op profiling.
  bool enable_op_profiling;
  // If true, the delegate will enable gpu op profiling detailed report.
  bool enable_op_profiling_detailed_report;
  // Set to enforce capping inf/-inf to max float values for the softmax input
  // and padding.
  bool enable_infinite_float_capping;

  // The nul-terminated directory to use for serialization.
  // Whether serialization actually happens or not is dependent on backend used
  // and validity of this directory.
  // Set to nullptr implies the delegate will not try serialization.
  //
  // NOTE: Users should ensure that this directory is private to the app to
  // avoid data access issues.
  // Delegate copies the string and doesn't take ownership of the memory.
  const char* serialization_dir;

  // The unique nul-terminated token string that acts as a 'namespace' for
  // all serialization entries.
  // Should be unique to a particular model (graph & constants).
  // For an example of how to generate this from a TFLite model, see
  // StrFingerprint() in lite/delegates/serialization.h.
  //
  // Set to nullptr implies the delegate will not try serialization.
  // Delegate copies the string and doesn't take ownership of the memory.
  const char* model_token;

  // Set to true to serialize immutable external tensors. By default only the
  // non-external tensors are serialized.
  bool serialize_external_tensors;

  // If true, the delegate will prefer to use textures rather than buffers for
  // weights. Use option when weights in texture has better performance.
  bool prefer_texture_weights;

  // Set to true to enable uploading tensor weights directly without processing.
  // This requires the model file to have pre-processed weights.
  //
  // WARNING: This differs from the typical serialization path because the
  // pre-processed immutable external tensors are stored in the model file
  // itself and not in a separate serialization directory. This option reduces
  // the disk space required and the memory usage on the first run. However, if
  // the prepacked weights become incompatible with the current ML Drift
  // kernels, there is no fallback path. Due to this risk, this option is
  // intended for ADVANCED USERS only.
  bool has_prepacked_external_tflite_tensors;

  // This enables dynamic range quantization of the input tensor for large
  // sized fully connected and convolution operations, if the device supports
  // it.
  // This will result in accuracy loss, since the input tensor will be
  // quantized to 8-bit.
  // Turning this on will also increase the initialization time to calculate
  // some extra constant tensor.
  // `enable_constant_tensors_sharing` must be true to use this.
  bool allow_src_quantized_fc_conv_ops;
} MlDriftClDelegateOptions;

// Returns default options for ML Drift OpenCL delegate.
// The current default is to use FP16 precision if available, otherwise FP32.
//
// This allocates a new object that should be freed using
// `MlDriftClFreeOptions`.
MlDriftClDelegateOptions* MlDriftClDelegateDefaultOptions();

// Deallocates an options object allocated with MlDriftClDelegateDefaultOptions.
void MlDriftClFreeOptions(MlDriftClDelegateOptions* options);

// Creates a new ML Drift OpenCL delegate object.
//
// This **takes ownership** of the options object.
TfLiteDelegate* TfLiteCreateMlDriftClDelegate(
    MlDriftClDelegateOptions* options);

// Deletes an existing ML Drift OpenCL delegate object incl. all its resources.
void TfLiteDeleteMlDriftClDelegate(TfLiteDelegate* delegate);

#ifdef __cplusplus
}  // extern "C"

namespace tflite {

// Definition from tensorflow/lite/core/interpreter.h
// Don't include the header since it's not a public header file.
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

namespace ml_drift {

// Additional ML Drift OpenCL Delegate C++ APIs.
//
// Typical usage:
//
//   // Initialize.
//   MlDriftClDelegateOptionsPtr options = MlDriftClDelegateDefaultOptionsPtr();
//   tflite::TfLiteDelegatePtr delegate =
//     TfLiteCreateMlDriftClDelegate(std::move(options));
//
//   QCHECK(delegate != nullptr);
//   QCHECK_EQ(interpreter->ModifyGraphWithDelegate(std::move(delegate)),
//             kTfLiteOk);
//
//   // Run inference.
//   QCHECK_EQ(interpreter->Invoke(), kTfLiteOk);

using MlDriftClDelegateOptionsPtr = std::unique_ptr<MlDriftClDelegateOptions>;

// Returns default options for ML Drift OpenCL delegate.
//
// This calls `MlDriftClDelegateDefaultOptions()` add return the result in an
// RAII wrapper.
MlDriftClDelegateOptionsPtr MlDriftClDelegateDefaultOptionsPtr();

// Creates a new ML Drift OpenCL delegate object.
TfLiteDelegatePtr TfLiteCreateMlDriftClDelegate(
    MlDriftClDelegateOptionsPtr options);

}  // namespace ml_drift
}  // namespace tflite

#endif  // __cplusplus

#endif  // THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_CL_H_
