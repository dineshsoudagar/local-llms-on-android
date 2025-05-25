package com.example.local_llm

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var onnxModel: OnnxModel
    private val inferenceScope = CoroutineScope(Dispatchers.IO)
    private var inferenceJob: Job? = null

    private val END_TOKEN_IDS = setOf(151643, 151645) // <|endoftext|> and <|im_end|>

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tokenizer = BpeTokenizer(this)

        val inputEditText: EditText = findViewById(R.id.userInput)
        inputEditText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val clearButton: Button = findViewById(R.id.clearButton)
        val outputText: TextView = findViewById(R.id.outputView)
        // Removed progress bar and loading text references

        outputText.text = "⏳ Please wait, the model is still loading."
        sendButton.isEnabled = false

        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity)
            withContext(Dispatchers.Main) {
                outputText.text = "✅ Model is ready."
                sendButton.isEnabled = true
            }
        }

        val systemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

        sendButton.setOnClickListener {
            if (!::onnxModel.isInitialized) {
                outputText.text = "⏳ Please wait, the model is still loading."
                return@setOnClickListener
            }
            val inputText = inputEditText.text.toString()
            val chatPrompt = buildString {
                append("<|im_start|>system\n")
                append(systemPrompt)
                append("<|im_end|>\n")
                append("<|im_start|>user\n")
                append(inputText)
                append("<|im_end|>\n")
                append("<|im_start|>assistant\n")
            }

            sendButton.isEnabled = false
            stopButton.isEnabled = true
            outputText.text = "\u23F3 Thinking..."

            inferenceJob = inferenceScope.launch {
                try {
                    val tokenIds = tokenizer.tokenize(chatPrompt, addSpecialTokens = false)
                    val builder = StringBuilder()

                    withContext(Dispatchers.Main) {
                        outputText.text = ""
                    }

                    onnxModel.runInferenceStreamingWithPastKV(
                        inputIds = tokenIds,
                        endTokenIds = END_TOKEN_IDS,
                        shouldStop = { inferenceJob?.isActive != true },
                        onTokenGenerated = { tokenId ->
                            val tokenStr = tokenizer.decode(intArrayOf(tokenId))
                            builder.append(tokenStr)

                            // Check after appending
                            if (builder.contains("<|im_end|")) {
                                builder.replace(builder.indexOf("<|im_end|"), builder.indexOf("<|im_end|") + "<|im_end|".length, "")
                                runOnUiThread {
                                    outputText.text = builder.toString()
                                    sendButton.isEnabled = true
                                    stopButton.isEnabled = false
                                }
                                inferenceJob?.cancel()
                                return@runInferenceStreamingWithPastKV
                            }

                            runOnUiThread {
                                outputText.text = builder.toString()
                                val scrollView: ScrollView = findViewById(R.id.outputScroll)
                                scrollView.post {
                                    scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                                }
                            }
                        }
                    )

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        outputText.text = "Error: ${e.message ?: "Please wait for the model to load."}"
                        sendButton.isEnabled = true
                        stopButton.isEnabled = false
                    }
                } finally {
                    withContext(Dispatchers.Main) {
                        sendButton.isEnabled = true
                        stopButton.isEnabled = false
                    }
                }
            }
        }

        stopButton.setOnClickListener {
            inferenceJob?.cancel()
            outputText.append(" ⛔ Generation stopped.")
            sendButton.isEnabled = true
            stopButton.isEnabled = false
        }

        clearButton.setOnClickListener {
            inputEditText.text.clear()
            outputText.text = ""
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
    }
}
