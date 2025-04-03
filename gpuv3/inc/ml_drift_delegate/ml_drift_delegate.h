#ifndef THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_
#define THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_

#ifdef __cplusplus
extern "C" {
#endif

// Precision of the ML Drift OpenCl and WebGpu delegates.
// When the precision is `kDefault`, the delegate will check if FP16 is
// supported. If so, use Fp16. Otherwise, use Fp32.
typedef enum {
    // Use FP16 if available, otherwise use FP32.
    kDefault,
    // Use FP16; can result in wrong output.
    kFp16,
    // Use FP32; is slower than FP16.
    kFp32,
} MlDriftDelegatePrecision;

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_ODML_INFRA_ML_DRIFT_DELEGATE_ML_DRIFT_DELEGATE_H_
