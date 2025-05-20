package com.example.deen_translator

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
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

        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity)
        }

        // Tokens to suppress in output (decoded form)
        val tokensToFilter = setOf("<|im_end|>", "</im_end/>", "</endoftext/>")
        val systemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        val inputEditText: EditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val outputText: TextView = findViewById(R.id.outputView)

        sendButton.setOnClickListener {
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
            outputText.text = "⏳ Thinking..."

            inferenceJob = inferenceScope.launch {
                try {
                    val tokenIds = tokenizer.tokenize(chatPrompt, addSpecialTokens = false)
                    val builder = StringBuilder()

                    withContext(Dispatchers.Main) {
                        outputText.text = ""
                    }

                    onnxModel.runInferenceStreaming(
                        inputIds = tokenIds,
                        endTokenIds = END_TOKEN_IDS,
                        shouldStop = { inferenceJob?.isActive != true },
                        onTokenGenerated = { tokenId ->
                            // Only decode and append if not end token
                            if (tokenId !in END_TOKEN_IDS) {
                                val tokenStr = tokenizer.decode(intArrayOf(tokenId))
                                builder.append(tokenStr)

                                runOnUiThread {
                                    outputText.text = builder.toString()
                                }
                            }
                        }
                    )

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        outputText.text = "Error: ${e.message}"
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
            outputText.append("\n⛔ Generation stopped.")
            sendButton.isEnabled = true
            stopButton.isEnabled = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
    }
}
