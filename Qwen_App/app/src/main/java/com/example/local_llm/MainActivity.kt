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
import io.noties.markwon.Markwon
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: BpeTokenizer
    private lateinit var onnxModel: OnnxModel
    private lateinit var markwon: Markwon
    private val inferenceScope = CoroutineScope(Dispatchers.IO)
    private var inferenceJob: Job? = null

    private val END_TOKEN_IDS = setOf(151643, 151645) // <|endoftext|> and <|im_end|>

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tokenizer = BpeTokenizer(this)
        markwon = Markwon.create(this)

        val inputEditText: EditText = findViewById(R.id.userInput)
        inputEditText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val clearButton: Button = findViewById(R.id.clearButton)
        val outputText: TextView = findViewById(R.id.outputView)
        val scrollView: ScrollView = findViewById(R.id.outputScroll)
        // Extract role token IDs using tokenizer
        val roleTokens = RoleTokenIds(
            systemStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("system"), tokenizer.getTokenId("Ċ")),
            userStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("user"), tokenizer.getTokenId("Ċ")),
            assistantStart = listOf(tokenizer.getTokenId("<|im_start|>"), tokenizer.getTokenId("assistant"), tokenizer.getTokenId("Ċ")),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )
        val modelConfig = ModelConfig(
            modelName = "Qwen",
            promptStyle = PromptStyle.QWEN,
            eosTokenIds = setOf(151643, 151645),
            numLayers = 24,
            numKvHeads = 2,
            headDim = 64,
            batchSize = 1,
            defaultSystemPrompt =  "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            roleTokenIds = roleTokens
        )


        val promptBuilder = PromptBuilder(tokenizer, modelConfig)
        val intent = PromptIntent.QA(systemPrompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
        // Build the prompt tokens from user input + intent
        // val inputIds: IntArray = promptBuilder.buildPromptTokens(inputEditText.text.toString(), intent)
        val mapper = TokenDisplayMapper(this@MainActivity, modelConfig.modelName)
        markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
        sendButton.isEnabled = false

        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity, modelConfig)
            withContext(Dispatchers.Main) {
                markwon.setMarkdown(outputText, "✅ Model is ready.")
                sendButton.isEnabled = true
            }
        }


        sendButton.setOnClickListener {
            if (!::onnxModel.isInitialized) {
                markwon.setMarkdown(outputText, "⏳ Please wait, the model is still loading.")
                return@setOnClickListener
            }

            sendButton.isEnabled = false
            stopButton.isEnabled = true
            markwon.setMarkdown(outputText, "⏳ Thinking...")

            inferenceJob = inferenceScope.launch {
                try {
                    val inputIds: IntArray = promptBuilder.buildPromptTokens(inputEditText.text.toString(), intent)
                    val tokenIds = inputIds
                    Log.d("tokenized", "ID=${tokenIds}")
                    val builder = StringBuilder()
                    var tokenCounter = 0

                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, "")
                    }

                    onnxModel.runInferenceStreamingWithPastKV(
                        inputIds = tokenIds,
                        endTokenIds = END_TOKEN_IDS,
                        shouldStop = { inferenceJob?.isActive != true },
                        onTokenGenerated = { tokenId ->
                            //val tokenStr = tokenizer.decodeSingleToken(tokenId)
                            val tokenStr = mapper.map(tokenId)
                            builder.append(tokenStr)
                            tokenCounter++

                            runOnUiThread {
                                outputText.text = builder.toString()
                                scrollView.post {
                                    scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                                }
                            }

                        }
                    )

                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, builder.toString())
                        scrollView.post {
                            scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                        }
                    }

                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        markwon.setMarkdown(outputText, "❌ Error: ${e.message ?: "Unknown error."}")
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
            val current = outputText.text.toString()
            markwon.setMarkdown(outputText, "$current\n⛔ Generation stopped.")
            scrollView.post {
                scrollView.fullScroll(ScrollView.FOCUS_DOWN)
            }
            sendButton.isEnabled = true
            stopButton.isEnabled = false
        }

        clearButton.setOnClickListener {
            inputEditText.text.clear()
            markwon.setMarkdown(outputText, "")
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        inferenceScope.cancel()
    }
}
