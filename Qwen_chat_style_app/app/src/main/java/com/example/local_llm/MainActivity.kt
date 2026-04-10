package com.example.local_llm

import android.os.Bundle
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var chatController: ChatController
    private var followOutput = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val thinkingToggle: CheckBox = findViewById(R.id.thinkingToggle)
        val inputEditText: EditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val clearButton: Button = findViewById(R.id.clearButton)
        val outputText: TextView = findViewById(R.id.outputView)
        val scrollView: ScrollView = findViewById(R.id.outputScroll)

        inputEditText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()
        outputText.movementMethod = android.text.method.ScrollingMovementMethod.getInstance()

        scrollView.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN,
                MotionEvent.ACTION_MOVE,
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> followOutput = isScrolledToBottom(scrollView)
            }
            false
        }
        scrollView.setOnScrollChangeListener { _, _, _, _, _ ->
            followOutput = isScrolledToBottom(scrollView)
        }

        chatController = ChatController(this, ModelRegistry.selected)

        lifecycleScope.launch {
            chatController.state.collect { state ->
                title = state.title
                thinkingToggle.visibility = if (state.supportsThinking) View.VISIBLE else View.GONE
                sendButton.isEnabled = state.isReady && !state.isGenerating
                stopButton.isEnabled = state.isGenerating
                clearButton.isEnabled = state.isReady && !state.isGenerating

                outputText.text = renderOutput(state)
                scrollOutputToBottomIfNeeded(scrollView, force = state.isGenerating)
            }
        }

        sendButton.setOnClickListener {
            followOutput = true
            chatController.sendPrompt(inputEditText.text.toString())
        }

        stopButton.setOnClickListener {
            chatController.cancelGeneration()
        }

        clearButton.setOnClickListener {
            inputEditText.text.clear()
            chatController.resetConversation()
        }

        thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
            chatController.setThinkingEnabled(isChecked)
        }

        chatController.initialize()
    }

    override fun onDestroy() {
        super.onDestroy()
        chatController.close()
    }

    private fun renderOutput(state: ChatUiState): String {
        if (state.transcript.isEmpty()) {
            return state.statusMessage
        }

        return buildString {
            if (state.statusMessage.isNotBlank()) {
                append(state.statusMessage)
                append("\n\n")
            }

            state.transcript.forEachIndexed { index, turn ->
                append(if (turn.role == ChatRole.USER) "You" else "Assistant")
                append(":\n")

                if (!turn.thinkingText.isNullOrBlank()) {
                    append("[Thinking]\n")
                    append(turn.thinkingText.trimEnd())
                    append("\n\n")
                }

                if (turn.text.isNotBlank()) {
                    append(turn.text.trimEnd())
                } else if (state.isGenerating && turn.role == ChatRole.ASSISTANT && index == state.transcript.lastIndex) {
                    append("⏳ Thinking...")
                }

                if (turn.stopped) {
                    if (turn.text.isNotBlank() || !turn.thinkingText.isNullOrBlank()) {
                        append("\n")
                    }
                    append("[Generation stopped]")
                }

                if (index < state.transcript.lastIndex) {
                    append("\n\n")
                }
            }
        }
    }

    private fun scrollOutputToBottomIfNeeded(scrollView: ScrollView, force: Boolean = false) {
        if (!force && !followOutput) return
        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    private fun isScrolledToBottom(scrollView: ScrollView): Boolean {
        val child = scrollView.getChildAt(0) ?: return true
        val bottomOffset = child.bottom - (scrollView.height + scrollView.scrollY)
        return bottomOffset <= 32
    }
}
