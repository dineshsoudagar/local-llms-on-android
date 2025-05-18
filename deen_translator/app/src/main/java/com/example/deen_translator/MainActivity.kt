package com.example.deen_translator

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.*
import com.example.yourapp.SimpleTokenizer

class MainActivity : AppCompatActivity() {

    private lateinit var tokenizer: SimpleTokenizer
    private lateinit var onnxModel: OnnxModel
    private val inferenceScope = CoroutineScope(Dispatchers.IO)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tokenizer = SimpleTokenizer(this)

        // Load the model in the background
        inferenceScope.launch {
            onnxModel = OnnxModel(this@MainActivity)
        }

        val inputEditText: EditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val outputText: TextView = findViewById(R.id.outputView)

        sendButton.setOnClickListener {
            val inputText = inputEditText.text.toString()
            outputText.text = "Translating..."

            inferenceScope.launch {
                try {
                    val tokenIds = tokenizer.tokenize(inputText)
                    val outputIds = onnxModel.runInference(tokenIds)
                    val outputTokens = tokenizer.decode(outputIds)

                    withContext(Dispatchers.Main) {
                        outputText.text = outputTokens
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        outputText.text = "Error: ${e.message}"
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
