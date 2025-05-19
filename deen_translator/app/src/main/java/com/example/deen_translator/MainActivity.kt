package com.example.deen_translator

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import com.example.deen_translator.BpeTokenizer as SimpleTokenizer

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: SimpleTokenizer
    private lateinit var onnxModel: OnnxModel
    private val inferenceScope = CoroutineScope(Dispatchers.IO)

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tokenizer = SimpleTokenizer(this)

        // Load the model in the background
        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity)
        }
        val systemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        val inputEditText: EditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val outputText: TextView = findViewById(R.id.outputView)

        sendButton.setOnClickListener {
            val inputText = inputEditText.text.toString()
            Log.d(inputText, "Input Text ")
            runOnUiThread {
                sendButton.isEnabled = false
                outputText.text = "⏳ Thinking..."
            }
            // Build the formatted prompt
            val chatPrompt = buildString {
                append("<|im_start|>system\n")
                append(systemPrompt)
                append("<|im_end|>\n")
                append("<|im_start|>user\n")
                append(inputText)
                append("<|im_end|>\n")
                append("<|im_start|>assistant\n")
            }

            /*
            inferenceScope.launch {
                try {
                    val tokenIds = tokenizer.tokenize(chatPrompt,  addSpecialTokens = false)
                    val promptLength = tokenIds.size
                    Log.d("MainActivity", "Token IDs: ${tokenIds.joinToString(",")}")  // Print token IDs nicely

                    val outputIds = onnxModel.runInference(tokenIds)
                    // Keep only generated tokens
                    val generatedIds = outputIds.drop(promptLength).toIntArray()
                    Log.d("MainActivity", "Output IDs: ${outputIds.joinToString(",")}")

                    val outputTokens = tokenizer.decode(generatedIds)
                    Log.d("MainActivity", "Output Text: $outputTokens")

                    withContext(Dispatchers.Main) {
                        outputText.text = outputTokens
                        sendButton.isEnabled = true
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        outputText.text = "Error: ${e.message}"
                    }
                }
            }
            */
            inferenceScope.launch {
                try {
                    val tokenIds = tokenizer.tokenize(chatPrompt, addSpecialTokens = false)
                    val promptLength = tokenIds.size
                    val builder = StringBuilder()

                    withContext(Dispatchers.Main) {
                        outputText.text = ""
                    }

                    onnxModel.runInferenceStreaming(tokenIds) { tokenId ->
                        val tokenStr = tokenizer.decode(intArrayOf(tokenId))

                        builder.append(tokenStr)
                        Log.d("Streaming", "Token: $tokenStr")

                        runOnUiThread {
                            outputText.text = builder.toString()
                        }
                    }

                    runOnUiThread {
                        sendButton.isEnabled = true
                    }

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        outputText.text = "Error: ${e.message}"
                        sendButton.isEnabled = true
                    }
                }
            }
        }

    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel() // Clean up coroutine scope
    }
}
