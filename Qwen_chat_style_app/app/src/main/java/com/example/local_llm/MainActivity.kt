package com.example.local_llm

import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var chatController: ChatController
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var inputEditText: EditText
    private var followOutput = true
    private var lastTurnCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val thinkingToggle: CheckBox = findViewById(R.id.thinkingToggle)
        inputEditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val statusView: TextView = findViewById(R.id.statusView)
        val chatRecyclerView: RecyclerView = findViewById(R.id.chatRecyclerView)

        chatAdapter = ChatAdapter()
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.itemAnimator = null
        chatRecyclerView.addOnScrollListener(
            object : RecyclerView.OnScrollListener() {
                override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                    followOutput = !recyclerView.canScrollVertically(1)
                }
            }
        )

        inputEditText.setOnFocusChangeListener { _, hasFocus ->
            if (hasFocus) {
                followOutput = true
                scrollChatToBottomIfNeeded(chatRecyclerView, force = true)
            }
        }

        chatController = ChatController(this, ModelRegistry.selected)

        lifecycleScope.launch {
            chatController.state.collect { state ->
                title = state.title
                thinkingToggle.visibility = if (state.supportsThinking) View.VISIBLE else View.GONE
                sendButton.visibility = if (state.isGenerating) View.GONE else View.VISIBLE
                stopButton.visibility = if (state.isGenerating) View.VISIBLE else View.GONE
                sendButton.isEnabled = state.isReady && !state.isGenerating
                stopButton.isEnabled = state.isGenerating

                statusView.text = state.statusMessage
                statusView.visibility = if (state.statusMessage.isBlank()) View.GONE else View.VISIBLE

                chatAdapter.submitTurns(state.transcript)

                val forceScroll = state.isGenerating || state.transcript.size != lastTurnCount
                lastTurnCount = state.transcript.size
                scrollChatToBottomIfNeeded(chatRecyclerView, force = forceScroll)
            }
        }

        sendButton.setOnClickListener {
            val prompt = inputEditText.text.toString()
            if (prompt.isBlank()) {
                return@setOnClickListener
            }

            followOutput = true
            chatController.sendPrompt(prompt)
            inputEditText.text.clear()
        }

        stopButton.setOnClickListener {
            chatController.cancelGeneration()
        }

        thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
            chatController.setThinkingEnabled(isChecked)
        }

        chatController.initialize()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.chat_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.menu_clear -> {
                if (!chatController.state.value.isGenerating) {
                    inputEditText.text.clear()
                    chatController.resetConversation()
                }
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
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

    private fun scrollChatToBottomIfNeeded(recyclerView: RecyclerView, force: Boolean = false) {
        if (!force && !followOutput) {
            return
        }

        val lastIndex = chatAdapter.itemCount - 1
        if (lastIndex < 0) {
            return
        }

        recyclerView.post {
            recyclerView.scrollToPosition(lastIndex)
        }
    }
}
