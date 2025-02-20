package com.whispertflite;

public class WhisperKitNative {
    private long nativePtr;

    public WhisperKitNative(String modelPath, String audioPath, String reportPath, String nativeLibsDir, int concurrentWorkers) {
        nativePtr = init(modelPath, audioPath, reportPath, nativeLibsDir, false, concurrentWorkers);
    }

    public String transcribe(String audioPath) {
        return transcribe(nativePtr, audioPath);
    }

    public void release() {
        release(nativePtr);
    }

    private native long init(String modelPath, String audioPath, String reportPath, String nativeLibsDir, boolean enableReport, int concurrentWorkers);
    private native String transcribe(long nativePtr, String audioPath);
    private native void release(long nativePtr);
}
