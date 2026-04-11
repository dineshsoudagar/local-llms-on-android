package com.example.local_llm

import android.os.Bundle
import android.text.format.DateUtils
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.launch

class PocketChatActivity : AppCompatActivity() {

    private lateinit var chatController: PersistentChatController
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var inputEditText: EditText
    private lateinit var chatRecyclerView: RecyclerView
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
        chatRecyclerView = findViewById(R.id.chatRecyclerView)

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
                scrollChatToBottomIfNeeded(force = true)
            }
        }

        chatController = PersistentChatController(this, ModelRegistry.selected)

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
                scrollChatToBottomIfNeeded(force = forceScroll)
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
            R.id.menu_new_chat -> {
                if (!chatController.state.value.isGenerating) {
                    inputEditText.text.clear()
                    chatController.startNewChat()
                }
                true
            }
            R.id.menu_sessions -> {
                showPreviousChats()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        chatController.close()
    }

    private fun showPreviousChats() {
        val sessions = chatController.listSavedSessions()
        if (sessions.isEmpty()) {
            AlertDialog.Builder(this)
                .setMessage(getString(R.string.no_saved_chats))
                .setPositiveButton(android.R.string.ok, null)
                .show()
            return
        }

        val items = sessions.map { session ->
            val updated = DateUtils.getRelativeTimeSpanString(
                session.updatedAtMillis,
                System.currentTimeMillis(),
                DateUtils.MINUTE_IN_MILLIS
            )
            buildString {
                append(session.title)
                append("\n")
                append(session.modelDisplayName)
                append(" - ")
                append(updated)
            }
        }.toTypedArray()

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.previous_chats))
            .setItems(items) { _, which ->
                followOutput = true
                chatController.loadSession(sessions[which].sessionId)
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun scrollChatToBottomIfNeeded(force: Boolean = false) {
        if (!force && !followOutput) {
            return
        }

        val lastIndex = chatAdapter.itemCount - 1
        if (lastIndex < 0) {
            return
        }

        chatRecyclerView.post {
            chatRecyclerView.scrollToPosition(lastIndex)
        }
    }
}
