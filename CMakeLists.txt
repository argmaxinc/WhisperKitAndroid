# For licensing see accompanying LICENSE file.
# Copyright © 2024 Argmax, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.22.1)

# project name and language
project(axie_tflite LANGUAGES CXX)

set(TENSORFLOW_SOURCE_DIR $ENV{TENSORFLOW_SOURCE_DIR})
set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK_ROOT})

if (NOT DEFINED ANDROID_NDK_ROOT)
    message(FATAL_ERROR  "NDK root path is not defined. Please use cmake -ANDROID_NDK_ROOT=/path/to/android_ndk")
endif()

if (NOT DEFINED TENSORFLOW_SOURCE_DIR)
    message(FATAL_ERROR  "TensorFlow path is not defined. Please use cmake -DTENSORFLOW_SOURCE_DIR=/path/to/tensorflow")
endif()

set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake)
set(ANDROID_ABI arm64-v8a)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(AXIE_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(AXIE_INC_DIR ${CMAKE_SOURCE_DIR}/inc)
set(AXIE_LIB_DIR ${CMAKE_SOURCE_DIR}/libs)
set(JNI_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

if(${ANDROID})
  find_library(android_log log)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_library(qnn_delegate SHARED IMPORTED)

set_property(TARGET qnn_delegate PROPERTY IMPORTED_LOCATION 
  ${AXIE_LIB_DIR}/libQnnTFLiteDelegate.so
)

link_directories(
  ${AXIE_LIB_DIR}
)

# build CLI executable: axie_tflite
add_executable(axie_tflite 
  ${AXIE_SRC_DIR}/tflite_model.cpp
  ${AXIE_SRC_DIR}/post_proc.cpp
  ${AXIE_SRC_DIR}/audio_input.cpp
  ${AXIE_SRC_DIR}/tflite_runner.cpp
)

target_include_directories(axie_tflite PRIVATE
  ${AXIE_INC_DIR}
)

target_link_libraries(axie_tflite PRIVATE 
  qnn_delegate 
  tensorflow-lite 
  tensorflowlite_gpu_delegate
  ${android_log}
)
