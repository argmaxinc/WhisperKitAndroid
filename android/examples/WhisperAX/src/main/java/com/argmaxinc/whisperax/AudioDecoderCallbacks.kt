//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2025 Argmax, Inc. All rights reserved.
package com.argmaxinc.whisperax

interface AudioDecoderCallbacks {
    fun onAudioFormat(freq: Int, ch: Int, dur: Long)
    fun onOutputBuffer(pcmbuffer: ByteArray, timestamp: Long)
    fun onDecodeClose()
    fun onEndOfStream()
}
