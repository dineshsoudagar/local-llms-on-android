package com.example.local_llm

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.launch

open class PocketChatActivity : AppCompatActivity() {

    companion object {
        private const val TOOLBAR_LOGO_ASSET = "pocket_llm_logo.png"
    }

    private lateinit var settingsStore: AppSettingsStore
    private lateinit var currentSettings: AppSettings
    private lateinit var chatController: PersistentChatController
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var inputEditText: EditText
    private lateinit var toolbarSubtitleView: TextView
    private var autoScrollDuringGeneration = false
    private var autoScrollPendingFinalUpdate = false
    private var wasGenerating = false
    private val chatAdapterObserver = object : RecyclerView.AdapterDataObserver() {
        override fun onChanged() = scrollChatToBottomIfNeeded()

        override fun onItemRangeInserted(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int, payload: Any?) = scrollChatToBottomIfNeeded()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        settingsStore = AppSettingsStore(this)
        currentSettings = settingsStore.load()
        setTheme(currentSettings.theme.styleRes)
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val toolbar: MaterialToolbar = findViewById(R.id.topToolbar)
        val toolbarLogoView: ImageView = findViewById(R.id.toolbarLogo)
        toolbarSubtitleView = findViewById(R.id.toolbarSubtitle)
        val thinkingToggle: CheckBox = findViewById(R.id.thinkingToggle)
        inputEditText = findViewById(R.id.userInput)
        val sendButton: Button = findViewById(R.id.sendButton)
        val stopButton: Button = findViewById(R.id.stopButton)
        val statusView: TextView = findViewById(R.id.statusView)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)

        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbarSubtitleView.text = ModelRegistry.selected.displayName
        loadToolbarLogo(toolbarLogoView)

        chatAdapter = ChatAdapter(currentSettings.chatFontSizeSp)
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.itemAnimator = null
        chatAdapter.registerAdapterDataObserver(chatAdapterObserver)
        applyTypography(statusView, sendButton, stopButton)

        chatController = PersistentChatController(this, ModelRegistry.selected)
        chatRecyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                super.onScrollStateChanged(recyclerView, newState)
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING && chatController.state.value.isGenerating) {
                    autoScrollDuringGeneration = false
                    autoScrollPendingFinalUpdate = false
                }
            }
        })

        lifecycleScope.launch {
            chatController.state.collect { state ->
                val generationStarted = state.isGenerating && !wasGenerating
                val generationFinished = !state.isGenerating && wasGenerating

                if (generationStarted) {
                    autoScrollDuringGeneration = true
                    autoScrollPendingFinalUpdate = false
                }
                if (generationFinished) {
                    autoScrollPendingFinalUpdate = autoScrollDuringGeneration
                    autoScrollDuringGeneration = false
                }

                title = state.title
                toolbarSubtitleView.text = state.title.substringAfter("Pocket LLM - ", ModelRegistry.selected.displayName)
                thinkingToggle.visibility = if (state.supportsThinking) View.VISIBLE else View.GONE
                sendButton.visibility = if (state.isGenerating) View.GONE else View.VISIBLE
                stopButton.visibility = if (state.isGenerating) View.VISIBLE else View.GONE
                sendButton.isEnabled = state.isReady && !state.isGenerating
                stopButton.isEnabled = state.isGenerating

                statusView.text = state.statusMessage
                statusView.visibility = if (state.statusMessage.isBlank()) View.GONE else View.VISIBLE

                chatAdapter.submitTurns(state.transcript)
                wasGenerating = state.isGenerating
            }
        }

        sendButton.setOnClickListener {
            val prompt = inputEditText.text.toString()
            if (prompt.isBlank()) {
                return@setOnClickListener
            }

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
            R.id.menu_settings -> {
                showSettingsDialog()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onDestroy() {
        if (::chatAdapter.isInitialized) {
            chatAdapter.unregisterAdapterDataObserver(chatAdapterObserver)
        }
        super.onDestroy()
        chatController.close()
    }

    private fun showPreviousChats() {
        val sessions = chatController.listSavedSessions()
        if (sessions.isEmpty()) {
            MaterialAlertDialogBuilder(this)
                .setMessage(getString(R.string.no_saved_chats))
                .setPositiveButton(android.R.string.ok, null)
                .show()
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_saved_chats, null)
        val recyclerView: RecyclerView = dialogView.findViewById(R.id.savedChatsRecyclerView)
        recyclerView.layoutManager = LinearLayoutManager(this)

        val dialog = dialogBuilder
            .setView(dialogView)
            .setNegativeButton(android.R.string.cancel, null)
            .create()

        lateinit var adapter: SavedSessionsAdapter
        adapter = SavedSessionsAdapter(
            fontSizeSp = currentSettings.chatFontSizeSp,
            onSessionSelected = { session ->
                dialog.dismiss()
                chatController.loadSession(session.sessionId)
            },
            onDeleteRequested = { session ->
                MaterialAlertDialogBuilder(this)
                    .setTitle(getString(R.string.delete_chat))
                    .setMessage(getString(R.string.delete_chat_confirmation))
                    .setNegativeButton(android.R.string.cancel, null)
                    .setPositiveButton(getString(R.string.delete)) { _, _ ->
                        if (chatController.deleteSession(session.sessionId)) {
                            val remainingSessions = chatController.listSavedSessions()
                            adapter.submitList(remainingSessions)
                            if (remainingSessions.isEmpty()) {
                                dialog.dismiss()
                                MaterialAlertDialogBuilder(this)
                                    .setMessage(getString(R.string.no_saved_chats))
                                    .setPositiveButton(android.R.string.ok, null)
                                    .show()
                            }
                        }
                    }
                    .show()
            }
        )
        recyclerView.adapter = adapter
        adapter.submitList(sessions)

        dialog.show()
    }

    private fun showSettingsDialog() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_settings, null)
        val themeGroup: RadioGroup = dialogView.findViewById(R.id.themeGroup)
        val fontSizeValue: TextView = dialogView.findViewById(R.id.fontSizeValue)
        val fontSizePreview: TextView = dialogView.findViewById(R.id.fontSizePreview)
        val fontSizeSeekBar: SeekBar = dialogView.findViewById(R.id.fontSizeSeekBar)
        val cancelButton: Button = dialogView.findViewById(R.id.settingsCancelButton)
        val saveButton: Button = dialogView.findViewById(R.id.settingsSaveButton)

        when (currentSettings.theme) {
            AppThemeOption.OCEAN -> themeGroup.check(R.id.themeOcean)
            AppThemeOption.MIDNIGHT -> themeGroup.check(R.id.themeMidnight)
            AppThemeOption.FOREST -> themeGroup.check(R.id.themeForest)
            AppThemeOption.VIOLET -> themeGroup.check(R.id.themeViolet)
        }

        val initialProgress = (currentSettings.chatFontSizeSp - 13f).toInt().coerceIn(0, 11)
        fontSizeSeekBar.progress = initialProgress
        updateFontSizePreview(fontSizeValue, fontSizePreview, currentSettings.chatFontSizeSp)

        fontSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                updateFontSizePreview(fontSizeValue, fontSizePreview, 13f + progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })

        val dialog = dialogBuilder
            .setTitle(getString(R.string.settings_title))
            .setView(dialogView)
            .create()

        cancelButton.setOnClickListener {
            dialog.dismiss()
        }

        saveButton.setOnClickListener {
            val selectedTheme = when (themeGroup.checkedRadioButtonId) {
                R.id.themeMidnight -> AppThemeOption.MIDNIGHT
                R.id.themeForest -> AppThemeOption.FOREST
                R.id.themeViolet -> AppThemeOption.VIOLET
                else -> AppThemeOption.OCEAN
            }
            val selectedFontSize = 13f + fontSizeSeekBar.progress
            settingsStore.save(
                AppSettings(
                    theme = selectedTheme,
                    chatFontSizeSp = selectedFontSize
                )
            )
            dialog.dismiss()
            recreate()
        }

        dialog.show()
    }

    private fun applyTypography(
        statusView: TextView,
        sendButton: Button,
        stopButton: Button
    ) {
        inputEditText.textSize = currentSettings.chatFontSizeSp
        statusView.textSize = (currentSettings.chatFontSizeSp - 2f).coerceAtLeast(12f)
        sendButton.textSize = (currentSettings.chatFontSizeSp - 1f).coerceAtLeast(12f)
        stopButton.textSize = (currentSettings.chatFontSizeSp - 1f).coerceAtLeast(12f)
    }

    private fun updateFontSizePreview(
        fontSizeValue: TextView,
        fontSizePreview: TextView,
        fontSizeSp: Float
    ) {
        fontSizeValue.text = "${fontSizeSp.toInt()} sp"
        fontSizePreview.textSize = fontSizeSp
    }

    private fun loadToolbarLogo(toolbarLogoView: ImageView) {
        runCatching {
            assets.open(TOOLBAR_LOGO_ASSET).use { input ->
                BitmapFactory.decodeStream(input)
            }
        }.getOrNull()?.let { bitmap ->
            toolbarLogoView.setImageBitmap(bitmap)
        }
    }

    private fun scrollChatToBottomIfNeeded() {
        if (!autoScrollDuringGeneration && !autoScrollPendingFinalUpdate) {
            return
        }

        // Keep streamed output anchored to the bottom until the user takes control.
        chatRecyclerView.post {
            if (!autoScrollDuringGeneration && !autoScrollPendingFinalUpdate) {
                return@post
            }

            val lastPosition = chatAdapter.itemCount - 1
            if (lastPosition >= 0) {
                chatRecyclerView.scrollToPosition(lastPosition)
            }

            if (!autoScrollDuringGeneration) {
                autoScrollPendingFinalUpdate = false
            }
        }
    }
}
