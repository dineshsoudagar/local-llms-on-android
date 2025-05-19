package com.example.deen_translator

import android.content.Context
import ai.onnxruntime.*
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.LongBuffer

class OnnxModel(private val context: Context) {

    private lateinit var env: OrtEnvironment
    private lateinit var session: OrtSession

    init {
        initializeModel()
    }

    private fun initializeModel() {
        env = OrtEnvironment.getEnvironment()
        val modelFile = loadModelFile("qwen_25_05B_merged.onnx")
        Log.d("ONNX", "Starting model load")
        session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        Log.d("ONNX", "Model loaded")
    }

    private fun loadModelFile(filename: String): File {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val file = File(context.filesDir, filename)
        val outputStream = FileOutputStream(file)
        inputStream.copyTo(outputStream)
        inputStream.close()
        outputStream.close()
        return file
    }

    fun runInference(inputIds: IntArray, maxTokens: Int = 1024, endTokenId: Int = 151645): IntArray {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            val seqLen = generated.size.toLong()
            // val inputNameMap = session.inputNames.associateBy { it }

            // Create input_ids tensor
            val inputIdsArray = generated.map { it.toLong() }.toLongArray()
            val inputIdsBuffer = LongBuffer.wrap(inputIdsArray)
            val inputTensor = OnnxTensor.createTensor(env, inputIdsBuffer, longArrayOf(1, seqLen))

            // Create attention_mask tensor
            val attnMaskArray = LongArray(seqLen.toInt()) { 1L }
            val attnMaskBuffer = LongBuffer.wrap(attnMaskArray)
            val attnTensor = OnnxTensor.createTensor(env, attnMaskBuffer, longArrayOf(1, seqLen))

            // Create position_ids tensor
            val posIdsArray = LongArray(seqLen.toInt()) { it.toLong() }
            val posIdsBuffer = LongBuffer.wrap(posIdsArray)
            val posTensor = OnnxTensor.createTensor(env, posIdsBuffer, longArrayOf(1, seqLen))

            val inputs: Map<String, OnnxTensor> = mapOf(
                "input_ids" to inputTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            )

            val results = session.run(inputs)
            val output = results[0].value as Array<Array<FloatArray>>
            val logits = output[0].last()  // last token's logits
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            // Close tensors
            inputTensor.close()
            attnTensor.close()
            posTensor.close()
            results.close()

            // Break if end token
            if (nextTokenId == endTokenId) break
        }

        return generated.toIntArray()
    }
    // streaming the output
    fun runInferenceStreaming(
        inputIds: IntArray,
        maxTokens: Int = 1024,
        endTokenIds: Set<Int> = setOf(151645),
        shouldStop: () -> Boolean = { false },
        onTokenGenerated: (Int) -> Unit
    ) {
        val generated = inputIds.toMutableList()

        for (i in 0 until maxTokens) {
            if (shouldStop()) break

            val seqLen = generated.size.toLong()

            val inputIdsTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(generated.map { it.toLong() }.toLongArray()), longArrayOf(1, seqLen)
            )
            val attnTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(LongArray(seqLen.toInt()) { 1L }), longArrayOf(1, seqLen)
            )
            val posTensor = OnnxTensor.createTensor(
                env, LongBuffer.wrap(LongArray(seqLen.toInt()) { it.toLong() }), longArrayOf(1, seqLen)
            )

            val inputs = mapOf(
                "input_ids" to inputIdsTensor,
                "attention_mask" to attnTensor,
                "position_ids" to posTensor
            )

            val results = session.run(inputs)
            val output = results[0].value as Array<Array<FloatArray>>
            val logits = output[0].last()
            val nextTokenId = logits.indices.maxByOrNull { logits[it] } ?: 0
            generated.add(nextTokenId)

            inputIdsTensor.close()
            attnTensor.close()
            posTensor.close()
            results.close()

            onTokenGenerated(nextTokenId)
            if (nextTokenId in endTokenIds) break
        }
    }
}