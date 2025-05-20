#pragma once

#include <jni.h>

#include "WhisperKit.h"  // Use our wrapper instead of direct inclusion

#define TAG "WhisperKitJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifdef __cplusplus
extern "C" {
#endif

// JNI method declarations for MainActivity
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_loadModels(JNIEnv *env, jobject thiz,
                                                                               jstring jsonstr);
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_init(JNIEnv *env, jobject thiz, jstring jsonstr);
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_close(JNIEnv *env, jobject thiz);
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_writeData(JNIEnv *env, jobject thiz,
                                                                              jbyteArray pcmbuffer);
JNIEXPORT jint JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_setBackend(JNIEnv *env, jobject thiz,
                                                                               jint encoder_backend,
                                                                               jint decoder_backend);
JNIEXPORT jstring JNICALL Java_com_argmaxinc_whisperkit_WhisperKitImpl_getPerfString(JNIEnv *env, jobject thiz);

enum class CallbackMsgType : int { INIT = 0, TEXT_OUT = 1, CLOSE = 2 };

// Callback helper function
void sendTextToJava(JNIEnv *env, jobject thiz, jint what, jfloat timestamp, const char *text);

#ifdef __cplusplus
}
#endif
