package com.argmaxinc.whisperkit

/**
 * A simple class to test native code editing
 */
class NativeTest {

    /**
     * A native method that returns a string from C++
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the native library on application startup
        init {
            System.loadLibrary("whisperkit_jni")
        }
    }
}
