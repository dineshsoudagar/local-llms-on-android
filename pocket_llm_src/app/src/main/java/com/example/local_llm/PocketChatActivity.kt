package com.example.local_llm

import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.format.Formatter
import android.view.LayoutInflater
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.snackbar.Snackbar
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

open class PocketChatActivity : AppCompatActivity() {

    companion object {
        private const val TOOLBAR_LOGO_ASSET = "pocket_llm_logo.png"
    }

    private data class ModelDialogViews(
        val dialog: AlertDialog,
        val introView: TextView,
        val listContainer: LinearLayout
    )

    private lateinit var settingsStore: AppSettingsStore
    private lateinit var currentSettings: AppSettings
    private lateinit var modelSelectionStore: ModelSelectionStore
    private lateinit var modelFileResolver: ModelFileResolver
    private lateinit var retainedState: PocketChatViewModel
    private var currentModel: ModelDescriptor? = null
    private var chatController: PersistentChatController? = null
    private var controllerStateJob: Job? = null
    private var modelDownloadStateJob: Job? = null
    private var modelDialogViews: ModelDialogViews? = null
    private var modelDialogForceSelection = false
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var inputEditText: EditText
    private lateinit var toolbarSubtitleView: TextView
    private lateinit var toolbarModelSelector: View
    private lateinit var thinkingToggle: CheckBox
    private lateinit var newChatButton: View
    private lateinit var sendButton: Button
    private lateinit var stopButton: Button
    private lateinit var statusView: TextView
    private var autoScrollDuringGeneration = false
    private var autoScrollPendingFinalUpdate = false
    private var wasGenerating = false
    private var isModelOperationInProgress = false
    private var modelOperationStatusMessage: String? = null
    private var activeDownloadModelId: String? = null
    private var activeDownloadModelName: String? = null
    private var activeDownloadFileName: String? = null
    private var activeDownloadBytes: Long = 0L
    private var activeDownloadTotalBytes: Long? = null
    private val chatAdapterObserver = object : RecyclerView.AdapterDataObserver() {
        override fun onChanged() = scrollChatToBottomIfNeeded()

        override fun onItemRangeInserted(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int, payload: Any?) = scrollChatToBottomIfNeeded()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        settingsStore = AppSettingsStore(this)
        currentSettings = settingsStore.load()
        modelSelectionStore = ModelSelectionStore(this)
        modelFileResolver = ModelFileResolver(this)
        setTheme(currentSettings.theme.styleRes)
        super.onCreate(savedInstanceState)
        retainedState = ViewModelProvider(this)[PocketChatViewModel::class.java]
        setContentView(R.layout.activity_main)

        val toolbar: MaterialToolbar = findViewById(R.id.topToolbar)
        val toolbarLogoView: ImageView = findViewById(R.id.toolbarLogo)
        toolbarModelSelector = findViewById(R.id.modelSelector)
        toolbarSubtitleView = findViewById(R.id.toolbarSubtitle)
        thinkingToggle = findViewById(R.id.thinkingToggle)
        newChatButton = findViewById(R.id.newChatButton)
        inputEditText = findViewById(R.id.userInput)
        sendButton = findViewById(R.id.sendButton)
        stopButton = findViewById(R.id.stopButton)
        statusView = findViewById(R.id.statusView)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)

        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbarSubtitleView.text = getString(R.string.model_picker_empty_subtitle)
        loadToolbarLogo(toolbarLogoView)

        chatAdapter = ChatAdapter(currentSettings.chatFontSizeSp)
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.itemAnimator = null
        chatAdapter.registerAdapterDataObserver(chatAdapterObserver)
        applyTypography(statusView, sendButton, stopButton)
        chatRecyclerView.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrollStateChanged(recyclerView: RecyclerView, newState: Int) {
                super.onScrollStateChanged(recyclerView, newState)
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING && chatController?.state?.value?.isGenerating == true) {
                    autoScrollDuringGeneration = false
                    autoScrollPendingFinalUpdate = false
                }
            }
        })

        toolbarModelSelector.setOnClickListener {
            if (chatController?.state?.value?.isGenerating == true) {
                showTransientMessage(getString(R.string.model_switch_generation_blocked))
                return@setOnClickListener
            }
            showModelSelectionDialog(forceSelection = false)
        }

        newChatButton.setOnClickListener {
            startNewChatFromUi()
        }

        sendButton.setOnClickListener {
            val controller = chatController ?: return@setOnClickListener
            val prompt = inputEditText.text.toString()
            if (prompt.isBlank() || isModelOperationInProgress) {
                return@setOnClickListener
            }

            controller.sendPrompt(prompt)
            inputEditText.text.clear()
        }

        stopButton.setOnClickListener {
            chatController?.cancelGeneration()
        }

        thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
            chatController?.setThinkingEnabled(isChecked)
        }

        val restoredModel = retainedState.modelId?.let(ModelRegistry::findById)
        currentModel = restoredModel ?: modelSelectionStore.loadSelectedModel()

        val retainedController = retainedState.chatController
        if (retainedController != null && currentModel != null) {
            chatController = retainedController
            toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
            thinkingToggle.isChecked = retainedController.isThinkingEnabled()
            observeController(retainedController)
            applyChatState(retainedController.state.value)
        } else {
            val startupModel = currentModel
            if (startupModel != null && modelFileResolver.isModelAvailable(startupModel)) {
                switchToController(startupModel, PersistentChatController(this, startupModel), initialize = true)
            } else {
                if (startupModel != null) {
                    toolbarSubtitleView.text = startupModel.displayName
                    renderNoControllerState(
                        getString(R.string.model_missing_message, startupModel.displayName),
                        preserveTranscript = false
                    )
                } else {
                    renderNoControllerState(
                        getString(R.string.model_required_message),
                        preserveTranscript = false
                    )
                }
                thinkingToggle.isChecked = false
            }
        }

        observeModelDownloadState()
        applyModelDownloadState(ModelDownloadStateStore.state.value)

        if (chatController == null) {
            chatRecyclerView.post {
                showModelSelectionDialog(forceSelection = true)
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.chat_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.menu_sessions -> {
                val controller = requireUsableController() ?: return true
                showPreviousChats(controller)
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
        controllerStateJob?.cancel()
        modelDownloadStateJob?.cancel()
        modelDialogViews?.dialog?.dismiss()
        super.onDestroy()
    }

    private fun requireUsableController(): PersistentChatController? {
        val controller = chatController
        if (controller == null) {
            showTransientMessage(getString(R.string.model_required_message))
            showModelSelectionDialog(forceSelection = true)
            return null
        }

        if (isModelOperationInProgress) {
            showTransientMessage(getString(R.string.model_operation_in_progress))
            return null
        }

        if (!controller.state.value.isReady) {
            showTransientMessage(getString(R.string.model_loading_message))
            return null
        }

        return controller
    }

    private fun observeController(controller: PersistentChatController) {
        controllerStateJob?.cancel()
        controllerStateJob = lifecycleScope.launch {
            controller.state.collect { state ->
                applyChatState(state)
            }
        }
    }

    private fun observeModelDownloadState() {
        modelDownloadStateJob?.cancel()
        modelDownloadStateJob = lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                ModelDownloadStateStore.state.collect { state ->
                    applyModelDownloadState(state)
                }
            }
        }
    }

    private fun applyModelDownloadState(state: ModelDownloadState) {
        when (state) {
            ModelDownloadState.Idle -> {
                if (isModelOperationInProgress || activeDownloadModelId != null) {
                    isModelOperationInProgress = false
                    modelOperationStatusMessage = null
                    resetDownloadState()
                    refreshModelSelectionDialog()
                    chatController?.let { applyChatState(it.state.value) }
                }
            }

            is ModelDownloadState.Running -> {
                isModelOperationInProgress = true
                activeDownloadModelId = state.modelId
                activeDownloadModelName = state.modelName
                activeDownloadFileName = state.fileName
                activeDownloadBytes = state.bytesDownloaded
                activeDownloadTotalBytes = state.totalBytes
                modelOperationStatusMessage = if (state.totalBytes != null && state.totalBytes > 0L) {
                    getString(
                        R.string.model_download_progress_status,
                        state.modelName,
                        formatFileSize(state.bytesDownloaded),
                        formatFileSize(state.totalBytes)
                    )
                } else {
                    getString(
                        R.string.model_download_progress_status_unknown,
                        state.modelName,
                        formatFileSize(state.bytesDownloaded)
                    )
                }

                refreshModelSelectionDialog()
                if (chatController != null) {
                    applyChatState(chatController!!.state.value)
                } else {
                    renderNoControllerState(modelOperationStatusMessage.orEmpty(), preserveTranscript = true)
                }
            }

            is ModelDownloadState.Completed -> {
                val descriptor = ModelRegistry.findById(state.modelId)
                isModelOperationInProgress = false
                modelOperationStatusMessage = null
                resetDownloadState()
                refreshModelSelectionDialog()
                modelDialogViews?.dialog?.dismiss()

                if (descriptor != null && (currentModel?.id != descriptor.id || chatController == null)) {
                    switchToController(
                        descriptor,
                        PersistentChatController(this, descriptor),
                        initialize = true,
                        activeChatSnapshot = chatController?.snapshotActiveChat()
                    )
                } else {
                    chatController?.let { applyChatState(it.state.value) }
                }

                ModelDownloadStateStore.clearTerminalState(state.modelId)
            }

            is ModelDownloadState.Failed -> {
                isModelOperationInProgress = false
                modelOperationStatusMessage = null
                resetDownloadState()
                refreshModelSelectionDialog()
                if (chatController != null) {
                    applyChatState(chatController!!.state.value)
                } else {
                    renderNoControllerState(getString(R.string.model_required_message), preserveTranscript = false)
                }
                showTransientMessage(
                    getString(
                        R.string.model_download_failed_message,
                        state.modelName,
                        state.message
                    )
                )
                ModelDownloadStateStore.clearTerminalState(state.modelId)
            }
        }
    }

    private fun switchToController(
        descriptor: ModelDescriptor,
        controller: PersistentChatController,
        initialize: Boolean,
        activeChatSnapshot: ActiveChatSnapshot? = null
    ) {
        controllerStateJob?.cancel()
        chatController?.close()
        chatController = controller
        currentModel = descriptor
        retainedState.chatController = controller
        retainedState.modelId = descriptor.id
        modelSelectionStore.saveSelectedModel(descriptor.id)
        resetDownloadState()
        isModelOperationInProgress = false
        modelOperationStatusMessage = null
        autoScrollDuringGeneration = false
        autoScrollPendingFinalUpdate = false
        wasGenerating = false
        toolbarSubtitleView.text = descriptor.displayName
        inputEditText.text.clear()
        thinkingToggle.isChecked = false
        observeController(controller)
        applyChatState(controller.state.value)
        if (initialize) {
            controller.initialize(activeChatSnapshot)
        }
    }

    private fun applyChatState(state: ChatUiState) {
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
        toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
        thinkingToggle.visibility = if (state.supportsThinking && !isModelOperationInProgress) View.VISIBLE else View.GONE
        sendButton.visibility = if (state.isGenerating && !isModelOperationInProgress) View.GONE else View.VISIBLE
        stopButton.visibility = if (state.isGenerating && !isModelOperationInProgress) View.VISIBLE else View.GONE
        newChatButton.isEnabled = state.isReady && !state.isGenerating && !isModelOperationInProgress
        sendButton.isEnabled = state.isReady && !state.isGenerating && !isModelOperationInProgress
        stopButton.isEnabled = state.isGenerating && !isModelOperationInProgress

        val effectiveStatus = modelOperationStatusMessage ?: state.statusMessage
        statusView.text = effectiveStatus
        val showInlineStatus = effectiveStatus.isNotBlank() &&
            ((state.transcript.isEmpty() || !state.isReady) || isModelOperationInProgress)
        statusView.visibility = if (showInlineStatus) View.VISIBLE else View.GONE
        applyStatusBackground(effectiveStatus)

        chatAdapter.submitTurns(state.transcript)
        wasGenerating = state.isGenerating
    }

    private fun renderNoControllerState(
        message: String,
        preserveTranscript: Boolean
    ) {
        title = getString(R.string.toolbar_app_title)
        toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
        thinkingToggle.visibility = View.GONE
        newChatButton.isEnabled = false
        sendButton.visibility = View.VISIBLE
        stopButton.visibility = View.GONE
        sendButton.isEnabled = false
        stopButton.isEnabled = false
        statusView.text = message
        statusView.visibility = if (message.isBlank()) View.GONE else View.VISIBLE
        applyStatusBackground(message)
        if (!preserveTranscript) {
            chatAdapter.submitTurns(emptyList())
        }
        autoScrollDuringGeneration = false
        autoScrollPendingFinalUpdate = false
        wasGenerating = false
    }

    private fun showModelSelectionDialog(forceSelection: Boolean) {
        val existingDialog = modelDialogViews?.dialog
        if (existingDialog != null && existingDialog.isShowing) {
            modelDialogForceSelection = forceSelection || modelDialogForceSelection
            refreshModelSelectionDialog()
            return
        }

        modelDialogForceSelection = forceSelection
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_model_selection, null)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()

        val dialogUi = ModelDialogViews(
            dialog = dialog,
            introView = dialogView.findViewById(R.id.modelPickerIntro),
            listContainer = dialogView.findViewById(R.id.modelListContainer)
        )

        dialog.setOnDismissListener {
            modelDialogViews = null
            modelDialogForceSelection = false
        }

        modelDialogViews = dialogUi
        refreshModelSelectionDialog()
        dialog.show()
    }

    private fun refreshModelSelectionDialog() {
        val dialogUi = modelDialogViews ?: return
        val canCancel = !modelDialogForceSelection && !isModelOperationInProgress
        dialogUi.dialog.setCancelable(canCancel)
        dialogUi.dialog.setCanceledOnTouchOutside(canCancel)
        dialogUi.introView.text = if (modelDialogForceSelection) {
            getString(R.string.model_picker_required_intro)
        } else {
            getString(R.string.model_picker_optional_intro)
        }

        dialogUi.listContainer.removeAllViews()
        val inflater = LayoutInflater.from(this)
        ModelRegistry.all.forEach { descriptor ->
            val itemView = inflater.inflate(R.layout.item_model_option, dialogUi.listContainer, false)
            val nameView: TextView = itemView.findViewById(R.id.modelName)
            val metaView: TextView = itemView.findViewById(R.id.modelMeta)
            val recommendationView: TextView = itemView.findViewById(R.id.modelRecommendation)
            val statusTextView: TextView = itemView.findViewById(R.id.modelStatus)
            val actionButton: Button = itemView.findViewById(R.id.modelActionButton)
            val itemProgressBar: ProgressBar = itemView.findViewById(R.id.modelItemProgressBar)
            val itemProgressText: TextView = itemView.findViewById(R.id.modelItemProgressText)

            nameView.text = descriptor.displayName
            val metaText = getString(
                R.string.model_picker_meta_format,
                descriptor.backendLabel,
                descriptor.sizeLabel
            )
            metaView.text = metaText
            val isCurrentModel = currentModel?.id == descriptor.id && chatController != null
            val isDownloadingThisModel = activeDownloadModelId == descriptor.id
            val isDownloaded = modelFileResolver.isModelDownloaded(descriptor)
            val isAvailable = modelFileResolver.isModelAvailable(descriptor)

            val statusText = when {
                isDownloadingThisModel -> getString(R.string.model_status_downloading)
                isCurrentModel -> getString(R.string.model_status_current)
                isDownloaded -> getString(R.string.model_status_downloaded)
                isAvailable -> getString(R.string.model_status_bundled)
                else -> getString(R.string.model_status_not_downloaded)
            }
            statusTextView.text = statusText
            recommendationView.text = getString(
                R.string.model_picker_compact_detail_format,
                descriptor.backendLabel,
                descriptor.sizeLabel,
                descriptor.deviceRecommendation,
                statusText
            )

            actionButton.text = when {
                isDownloadingThisModel -> getString(R.string.cancel_download)
                isCurrentModel -> getString(R.string.model_action_current)
                isAvailable -> getString(R.string.model_action_use)
                else -> getString(R.string.model_action_download)
            }
            actionButton.isEnabled = isDownloadingThisModel || (!isModelOperationInProgress && !isCurrentModel)
            actionButton.setOnClickListener {
                if (isDownloadingThisModel) {
                    cancelModelDownload()
                } else {
                    handleModelSelection(descriptor)
                }
            }

            itemProgressBar.visibility = if (isDownloadingThisModel) View.VISIBLE else View.GONE
            updateProgressBar(itemProgressBar, activeDownloadBytes, activeDownloadTotalBytes)
            itemProgressText.visibility = if (isDownloadingThisModel) View.VISIBLE else View.GONE
            itemProgressText.text = formatDownloadProgressText(activeDownloadBytes, activeDownloadTotalBytes)

            dialogUi.listContainer.addView(itemView)
        }
    }

    private fun handleModelSelection(descriptor: ModelDescriptor) {
        if (isModelOperationInProgress) {
            return
        }

        if (descriptor.id == currentModel?.id && chatController != null) {
            modelDialogViews?.dialog?.dismiss()
            return
        }

        if (modelFileResolver.isModelAvailable(descriptor)) {
            val activeChatSnapshot = chatController?.snapshotActiveChat()
            modelDialogViews?.dialog?.dismiss()
            switchToController(
                descriptor,
                PersistentChatController(this, descriptor),
                initialize = true,
                activeChatSnapshot = activeChatSnapshot
            )
            return
        }

        startModelDownload(descriptor)
    }

    private fun startModelDownload(descriptor: ModelDescriptor) {
        isModelOperationInProgress = true
        activeDownloadModelId = descriptor.id
        activeDownloadModelName = descriptor.displayName
        activeDownloadFileName = null
        activeDownloadBytes = 0L
        activeDownloadTotalBytes = descriptor.approxDownloadBytes
        modelOperationStatusMessage = getString(R.string.model_download_preparing, descriptor.displayName)

        refreshModelSelectionDialog()
        if (chatController != null) {
            applyChatState(chatController!!.state.value)
        } else {
            renderNoControllerState(modelOperationStatusMessage.orEmpty(), preserveTranscript = true)
        }

        ModelDownloadService.start(this, descriptor)
    }

    private fun cancelModelDownload() {
        modelOperationStatusMessage = getString(R.string.model_download_cancelling)
        refreshModelSelectionDialog()
        if (chatController != null) {
            applyChatState(chatController!!.state.value)
        } else {
            renderNoControllerState(modelOperationStatusMessage.orEmpty(), preserveTranscript = true)
        }
        ModelDownloadService.cancel(this)
    }

    private fun resetDownloadState() {
        activeDownloadModelId = null
        activeDownloadModelName = null
        activeDownloadFileName = null
        activeDownloadBytes = 0L
        activeDownloadTotalBytes = null
    }

    private fun startNewChatFromUi() {
        val controller = requireUsableController() ?: return
        if (controller.state.value.isGenerating) {
            return
        }

        inputEditText.text.clear()
        controller.startNewChat()
    }

    private fun showPreviousChats(controller: PersistentChatController) {
        val sessions = controller.listSavedSessions()
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
                controller.loadSession(session.sessionId)
            },
            onDeleteRequested = { session ->
                MaterialAlertDialogBuilder(this)
                    .setTitle(getString(R.string.delete_chat))
                    .setMessage(getString(R.string.delete_chat_confirmation))
                    .setNegativeButton(android.R.string.cancel, null)
                    .setPositiveButton(getString(R.string.delete)) { _, _ ->
                        if (controller.deleteSession(session.sessionId)) {
                            val remainingSessions = controller.listSavedSessions()
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
            val updatedSettings = AppSettings(
                theme = selectedTheme,
                chatFontSizeSp = selectedFontSize
            )
            val themeChanged = updatedSettings.theme != currentSettings.theme
            currentSettings = updatedSettings
            settingsStore.save(updatedSettings)
            dialog.dismiss()
            if (themeChanged) {
                recreate()
            } else {
                chatAdapter.updateFontSize(updatedSettings.chatFontSizeSp)
                applyTypography(
                    findViewById(R.id.statusView),
                    findViewById(R.id.sendButton),
                    findViewById(R.id.stopButton)
                )
            }
        }

        dialog.show()
    }

    private fun applyTypography(
        statusView: TextView,
        sendButton: Button,
        stopButton: Button
    ) {
        inputEditText.textSize = currentSettings.chatFontSizeSp
        statusView.textSize = currentSettings.chatFontSizeSp
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

    private fun updateProgressBar(
        progressBar: ProgressBar,
        downloadedBytes: Long,
        totalBytes: Long?
    ) {
        if (totalBytes == null || totalBytes <= 0L) {
            progressBar.isIndeterminate = true
            return
        }

        progressBar.isIndeterminate = false
        progressBar.max = 1000
        progressBar.progress = ((downloadedBytes * 1000L) / totalBytes)
            .toInt()
            .coerceIn(0, 1000)
    }

    private fun formatDownloadProgressText(downloadedBytes: Long, totalBytes: Long?): String {
        return if (totalBytes != null && totalBytes > 0L) {
            getString(
                R.string.model_download_progress_bytes,
                formatFileSize(downloadedBytes),
                formatFileSize(totalBytes)
            )
        } else {
            getString(
                R.string.download_notification_progress_unknown,
                formatFileSize(downloadedBytes)
            )
        }
    }

    private fun formatFileSize(sizeBytes: Long): String {
        return Formatter.formatFileSize(this, sizeBytes)
    }

    private fun applyStatusBackground(message: String) {
        val backgroundRes = when (message) {
            MODEL_LOADING_STATUS_MESSAGE -> R.drawable.bg_status_loading
            MODEL_READY_STATUS_MESSAGE -> R.drawable.bg_status_ready
            else -> R.drawable.bg_status_chip
        }
        statusView.setBackgroundResource(backgroundRes)
    }

    private fun showTransientMessage(message: String) {
        Snackbar.make(findViewById(android.R.id.content), message, Snackbar.LENGTH_LONG).show()
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
                val remainingScroll = (
                    chatRecyclerView.computeVerticalScrollRange() -
                        chatRecyclerView.computeVerticalScrollExtent() -
                        chatRecyclerView.computeVerticalScrollOffset()
                    ).coerceAtLeast(0)
                if (remainingScroll > 0) {
                    chatRecyclerView.scrollBy(0, remainingScroll)
                }
            }

            if (!autoScrollDuringGeneration) {
                autoScrollPendingFinalUpdate = false
            }
        }
    }
}
