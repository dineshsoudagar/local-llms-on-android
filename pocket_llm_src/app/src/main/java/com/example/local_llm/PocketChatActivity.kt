package com.example.local_llm

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.net.Uri
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.text.method.LinkMovementMethod
import android.text.format.Formatter
import android.text.util.Linkify
import android.util.TypedValue
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatSpinner
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.File

open class PocketChatActivity : AppCompatActivity() {

    private data class ModelDialogViews(
        val dialog: AlertDialog,
        val introView: TextView,
        val listContainer: LinearLayout
    )

    private data class ComposerPrompt(
        val modelText: String,
        val displayText: String
    )

    private enum class PendingImageStatus {
        READING,
        READY,
        FAILED
    }

    private data class PendingImageInput(
        val id: Long,
        val uri: Uri,
        val source: OcrInput.Source,
        val status: PendingImageStatus = PendingImageStatus.READING,
        val recognizedText: String? = null,
        val errorMessage: String? = null,
        val tempFilePath: String? = null
    )

    private lateinit var settingsStore: AppSettingsStore
    private lateinit var currentSettings: AppSettings
    private lateinit var modelInstructionStore: ModelInstructionStore
    private lateinit var modelSelectionStore: ModelSelectionStore
    private lateinit var modelFileResolver: ModelFileResolver
    private lateinit var chatSessionStore: ChatSessionStore
    private lateinit var retainedState: PocketChatViewModel
    private var currentModel: ModelDescriptor? = null
    private var chatController: PersistentChatController? = null
    private var controllerStateJob: Job? = null
    private var modelDownloadStateJob: Job? = null
    private var modelDialogViews: ModelDialogViews? = null
    private var modelDialogForceSelection = false
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var drawerChatsRecyclerView: RecyclerView
    private lateinit var drawerChatsEmptyView: TextView
    private lateinit var drawerSessionsAdapter: DrawerSessionsAdapter
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var inputEditText: EditText
    private lateinit var toolbarSubtitleView: TextView
    private lateinit var toolbarModelSelector: View
    private lateinit var thinkingToggleContainer: View
    private lateinit var thinkingToggle: CheckBox
    private lateinit var newChatButton: View
    private lateinit var sendButton: Button
    private lateinit var stopButton: Button
    private lateinit var micInputButton: MaterialButton
    private lateinit var galleryOcrButton: Button
    private lateinit var cameraOcrButton: Button
    private lateinit var pendingImageInputsScroll: HorizontalScrollView
    private lateinit var pendingImageInputsContainer: LinearLayout
    private lateinit var statusView: TextView
    private lateinit var transientMessageView: TextView
    private var speechInput: SpeechInput? = null
    private var ocrInput: OcrInput? = null
    private var speechBasePromptText: String = ""
    private var speechRecognizedText: String = ""
    private var speechCommittedText: String = ""
    private var speechPartialText: String = ""
    private var pendingSpeechStart = false
    private var pendingCameraOcrStart = false
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraImageCapture: ImageCapture? = null
    private var cameraOcrDialog: AlertDialog? = null
    private var cameraOcrStatusView: TextView? = null
    private val pendingImageInputs = mutableListOf<PendingImageInput>()
    private var nextPendingImageInputId = 1L
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
    private val hideTransientMessageRunnable = Runnable {
        if (::transientMessageView.isInitialized) {
            transientMessageView.visibility = View.GONE
        }
    }
    private val chatAdapterObserver = object : RecyclerView.AdapterDataObserver() {
        override fun onChanged() = scrollChatToBottomIfNeeded()

        override fun onItemRangeInserted(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int) = scrollChatToBottomIfNeeded()

        override fun onItemRangeChanged(positionStart: Int, itemCount: Int, payload: Any?) = scrollChatToBottomIfNeeded()
    }

    private val speechPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        val shouldStart = pendingSpeechStart
        pendingSpeechStart = false
        if (granted && shouldStart) {
            startSpeechInput()
        } else if (!granted) {
            showTransientMessage(getString(R.string.speech_permission_denied))
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        val shouldStart = pendingCameraOcrStart
        pendingCameraOcrStart = false
        if (granted && shouldStart) {
            showCameraOcrDialog()
        } else if (!granted) {
            showTransientMessage(getString(R.string.camera_permission_denied))
        }
    }

    private val galleryImagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let { selectedImage ->
            addPendingImageInput(selectedImage, OcrInput.Source.GALLERY)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        settingsStore = AppSettingsStore(this)
        currentSettings = settingsStore.load()
        modelInstructionStore = ModelInstructionStore(this)
        modelSelectionStore = ModelSelectionStore(this)
        modelFileResolver = ModelFileResolver(this)
        chatSessionStore = ChatSessionStore(this)
        setTheme(currentSettings.accent.styleFor(currentSettings.appearance))
        super.onCreate(savedInstanceState)
        retainedState = ViewModelProvider(this)[PocketChatViewModel::class.java]
        setContentView(R.layout.activity_main)

        val toolbar: MaterialToolbar = findViewById(R.id.topToolbar)
        val toolbarMenuButton: View = findViewById(R.id.toolbarMenuButton)
        drawerLayout = findViewById(R.id.drawerLayout)
        drawerChatsRecyclerView = findViewById(R.id.drawerChatsRecyclerView)
        drawerChatsEmptyView = findViewById(R.id.drawerChatsEmpty)
        toolbarModelSelector = findViewById(R.id.modelSelector)
        toolbarSubtitleView = findViewById(R.id.toolbarSubtitle)
        thinkingToggleContainer = findViewById(R.id.thinkingToggleContainer)
        thinkingToggle = findViewById(R.id.thinkingToggle)
        newChatButton = findViewById(R.id.newChatButton)
        inputEditText = findViewById(R.id.userInput)
        sendButton = findViewById(R.id.sendButton)
        stopButton = findViewById(R.id.stopButton)
        micInputButton = findViewById(R.id.micInputButton)
        galleryOcrButton = findViewById(R.id.galleryOcrButton)
        cameraOcrButton = findViewById(R.id.cameraOcrButton)
        pendingImageInputsScroll = findViewById(R.id.pendingImageInputsScroll)
        pendingImageInputsContainer = findViewById(R.id.pendingImageInputsContainer)
        statusView = findViewById(R.id.statusView)
        transientMessageView = findViewById(R.id.transientMessageView)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        speechInput = createSpeechInput()
        ocrInput = createOcrInput()

        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbarMenuButton.setOnClickListener {
            refreshDrawerSessions()
            drawerLayout.openDrawer(GravityCompat.START)
        }
        toolbarSubtitleView.text = getString(R.string.model_picker_empty_subtitle)
        configureDrawer()

        chatAdapter = ChatAdapter(currentSettings.chatFontSizeSp)
        chatRecyclerView.layoutManager = LinearLayoutManager(this)
        chatRecyclerView.adapter = chatAdapter
        chatRecyclerView.itemAnimator = null
        chatAdapter.registerAdapterDataObserver(chatAdapterObserver)
        applyTypography(statusView, sendButton, stopButton)
        updateMicRecordingState(false)
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

        micInputButton.setOnClickListener {
            handleSpeechInputClick()
        }

        galleryOcrButton.setOnClickListener {
            galleryImagePickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }

        cameraOcrButton.setOnClickListener {
            handleCameraOcrClick()
        }

        sendButton.setOnClickListener {
            val controller = chatController ?: return@setOnClickListener
            val prompt = buildComposerPrompt() ?: return@setOnClickListener
            if (prompt.modelText.isBlank() || isModelOperationInProgress) {
                return@setOnClickListener
            }

            clearSpeechInputState()
            controller.sendPrompt(prompt.modelText, prompt.displayText)
            inputEditText.text.clear()
            clearPendingImageInputs()
            refreshDrawerSessions()
        }

        stopButton.setOnClickListener {
            chatController?.cancelGeneration()
        }

        thinkingToggleContainer.setOnClickListener {
            thinkingToggle.isChecked = !thinkingToggle.isChecked
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
        refreshDrawerSessions()

        if (chatController == null) {
            chatRecyclerView.post {
                showModelSelectionDialog(forceSelection = true)
            }
        }
    }

    override fun onDestroy() {
        if (::chatAdapter.isInitialized) {
            chatAdapter.unregisterAdapterDataObserver(chatAdapterObserver)
        }
        if (::transientMessageView.isInitialized) {
            transientMessageView.removeCallbacks(hideTransientMessageRunnable)
        }
        controllerStateJob?.cancel()
        modelDownloadStateJob?.cancel()
        stopCameraOcr()
        cameraOcrDialog?.dismiss()
        speechInput?.destroy()
        ocrInput?.close()
        clearPendingImageInputs()
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

    private fun createSpeechInput(): SpeechInput {
        return SpeechInput(
            this,
            object : SpeechInput.Listener {
                override fun onSpeechStarted() {
                    runOnUiThread {
                        updateMicRecordingState(true)
                        showTransientMessage(getString(R.string.speech_listening))
                    }
                }

                override fun onSpeechPartial(text: String) {
                    runOnUiThread {
                        handleSpeechRecognizedText(text, final = false)
                    }
                }

                override fun onSpeechFinal(text: String, confidenceScores: FloatArray?) {
                    runOnUiThread {
                        handleSpeechRecognizedText(text, final = true)
                    }
                }

                override fun onSpeechError(message: String) {
                    runOnUiThread {
                        showTransientMessage(message)
                    }
                }

                override fun onSpeechEnded() {
                    runOnUiThread {
                        updateMicRecordingState(false)
                    }
                }
            }
        )
    }

    private fun createOcrInput(): OcrInput {
        return OcrInput(
            this,
            object : OcrInput.Listener {
                override fun onOcrStarted(source: OcrInput.Source, requestId: Long) = Unit

                override fun onOcrTextRecognized(
                    text: String,
                    source: OcrInput.Source,
                    requestId: Long
                ) {
                    runOnUiThread {
                        handlePendingImageOcrText(text, requestId)
                    }
                }

                override fun onOcrFailed(
                    message: String,
                    source: OcrInput.Source,
                    requestId: Long
                ) {
                    runOnUiThread {
                        markPendingImageInputFailed(requestId, message)
                        showTransientMessage(message)
                    }
                }
            }
        )
    }

    private fun handleSpeechInputClick() {
        val speech = speechInput ?: return
        if (speech.isRecording) {
            speech.stop()
            return
        }

        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            pendingSpeechStart = true
            speechPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        startSpeechInput()
    }

    private fun startSpeechInput() {
        speechBasePromptText = inputEditText.text.toString()
        speechRecognizedText = ""
        speechCommittedText = ""
        speechPartialText = ""
        speechInput?.start()
    }

    private fun handleSpeechRecognizedText(text: String, final: Boolean) {
        val recognizedText = PromptPreprocessor.normalize(text)
        if (recognizedText.isBlank()) {
            return
        }

        if (final) {
            speechCommittedText = PromptPreprocessor.mergeTypedAndRecognized(
                speechCommittedText,
                recognizedText
            )
            speechPartialText = ""
        } else {
            speechPartialText = recognizedText
        }

        speechRecognizedText = PromptPreprocessor.mergeTypedAndRecognized(
            speechCommittedText,
            speechPartialText
        )
        setPromptInputText(
            PromptPreprocessor.mergeTypedAndRecognized(
                speechBasePromptText,
                recognizedText
            )
        )
    }

    private fun handleCameraOcrClick() {
        if (!hasPermission(Manifest.permission.CAMERA)) {
            pendingCameraOcrStart = true
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }

        showCameraOcrDialog()
    }

    private fun showCameraOcrDialog() {
        if (cameraOcrDialog?.isShowing == true) {
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_camera_ocr, null)
        val previewView: PreviewView = dialogView.findViewById(R.id.cameraPreviewView)
        val statusText: TextView = dialogView.findViewById(R.id.cameraOcrStatus)
        val cancelButton: Button = dialogView.findViewById(R.id.cameraOcrCancelButton)
        val captureButton: Button = dialogView.findViewById(R.id.cameraOcrCaptureButton)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()

        cameraOcrDialog = dialog
        cameraOcrStatusView = statusText

        cancelButton.setOnClickListener {
            dialog.dismiss()
        }

        captureButton.setOnClickListener {
            captureCameraOcrPhoto(dialog)
        }

        dialog.setOnDismissListener {
            stopCameraOcr()
            cameraOcrDialog = null
            cameraOcrStatusView = null
        }

        showPanelDialog(dialog)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        startCameraPreview(previewView)
    }

    private fun startCameraPreview(previewView: PreviewView) {
        cameraOcrStatusView?.text = getString(R.string.camera_ocr_status_starting)
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener(
            Runnable {
                val provider = runCatching { providerFuture.get() }.getOrElse { error ->
                    cameraOcrStatusView?.text = error.message ?: "Camera could not start."
                    return@Runnable
                }

                if (cameraOcrDialog?.isShowing != true) {
                    provider.unbindAll()
                    return@Runnable
                }

                val preview = Preview.Builder()
                    .build()
                    .also { cameraPreview ->
                        cameraPreview.setSurfaceProvider(previewView.surfaceProvider)
                    }

                val imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .build()

                runCatching {
                    provider.unbindAll()
                    provider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture
                    )
                    cameraProvider = provider
                    cameraImageCapture = imageCapture
                    previewView.post {
                        previewView.display?.rotation?.let { rotation ->
                            cameraImageCapture?.targetRotation = rotation
                        }
                    }
                    cameraOcrStatusView?.text = getString(R.string.camera_ocr_status_ready)
                }.onFailure { error ->
                    cameraOcrStatusView?.text = error.message ?: "Camera OCR could not start."
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun captureCameraOcrPhoto(dialog: AlertDialog) {
        val imageCapture = cameraImageCapture ?: run {
            showTransientMessage(getString(R.string.camera_ocr_status_starting))
            return
        }
        val outputFile = createCameraOcrOutputFile()
        val outputOptions = ImageCapture.OutputFileOptions.Builder(outputFile).build()
        cameraOcrStatusView?.text = getString(R.string.camera_ocr_status_capturing)

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    addPendingImageInput(
                        uri = Uri.fromFile(outputFile),
                        source = OcrInput.Source.CAMERA,
                        tempFilePath = outputFile.absolutePath
                    )
                    dialog.dismiss()
                }

                override fun onError(exception: ImageCaptureException) {
                    runCatching { outputFile.delete() }
                    val message = exception.message ?: "Could not capture photo."
                    cameraOcrStatusView?.text = message
                    showTransientMessage(message)
                }
            }
        )
    }

    private fun createCameraOcrOutputFile(): File {
        val directory = File(cacheDir, "ocr_images").apply { mkdirs() }
        return File.createTempFile("camera_ocr_", ".jpg", directory)
    }

    private fun stopCameraOcr() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        cameraImageCapture = null
    }

    private fun addPendingImageInput(
        uri: Uri,
        source: OcrInput.Source,
        tempFilePath: String? = null
    ) {
        val requestId = nextPendingImageInputId++
        pendingImageInputs += PendingImageInput(
            id = requestId,
            uri = uri,
            source = source,
            tempFilePath = tempFilePath
        )
        renderPendingImageInputs()
        ocrInput?.recognizeImageUri(uri, source, requestId)
            ?: markPendingImageInputFailed(requestId, "OCR is not available.")
    }

    private fun handlePendingImageOcrText(text: String, requestId: Long) {
        val recognizedText = PromptPreprocessor.normalize(text)
        if (recognizedText.isBlank()) {
            markPendingImageInputFailed(requestId, getString(R.string.ocr_no_text))
            showTransientMessage(getString(R.string.ocr_no_text))
            return
        }

        updatePendingImageInput(
            requestId = requestId,
            status = PendingImageStatus.READY,
            recognizedText = recognizedText,
            errorMessage = null
        )
    }

    private fun markPendingImageInputFailed(requestId: Long, message: String) {
        updatePendingImageInput(
            requestId = requestId,
            status = PendingImageStatus.FAILED,
            recognizedText = null,
            errorMessage = message
        )
    }

    private fun updatePendingImageInput(
        requestId: Long,
        status: PendingImageStatus,
        recognizedText: String?,
        errorMessage: String?
    ) {
        val index = pendingImageInputs.indexOfFirst { it.id == requestId }
        if (index == -1) {
            return
        }

        pendingImageInputs[index] = pendingImageInputs[index].copy(
            status = status,
            recognizedText = recognizedText,
            errorMessage = errorMessage
        )
        renderPendingImageInputs()
    }

    private fun removePendingImageInput(requestId: Long) {
        val index = pendingImageInputs.indexOfFirst { it.id == requestId }
        if (index == -1) {
            return
        }

        deletePendingImageTempFile(pendingImageInputs.removeAt(index))
        renderPendingImageInputs()
    }

    private fun clearPendingImageInputs() {
        if (pendingImageInputs.isEmpty()) {
            renderPendingImageInputs()
            return
        }

        pendingImageInputs.forEach(::deletePendingImageTempFile)
        pendingImageInputs.clear()
        renderPendingImageInputs()
    }

    private fun deletePendingImageTempFile(input: PendingImageInput) {
        input.tempFilePath?.let { path ->
            runCatching { File(path).delete() }
        }
    }

    private fun renderPendingImageInputs() {
        if (!::pendingImageInputsContainer.isInitialized || !::pendingImageInputsScroll.isInitialized) {
            return
        }

        pendingImageInputsScroll.visibility = if (pendingImageInputs.isEmpty()) View.GONE else View.VISIBLE
        pendingImageInputsContainer.removeAllViews()
        pendingImageInputs.forEach { input ->
            pendingImageInputsContainer.addView(createPendingImageChip(input))
        }
    }

    private fun createPendingImageChip(input: PendingImageInput): View {
        val chip = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = ContextCompat.getDrawable(this@PocketChatActivity, R.drawable.bg_image_input_chip)
            setPadding(dp(5), dp(4), dp(4), dp(4))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                dp(46)
            ).apply {
                marginEnd = dp(6)
            }
        }

        chip.addView(
            ImageView(this).apply {
                layoutParams = LinearLayout.LayoutParams(dp(34), dp(34))
                scaleType = ImageView.ScaleType.CENTER_CROP
                setBackgroundColor(Color.BLACK)
                setImageURI(input.uri)
            }
        )

        chip.addView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(dp(58), ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    marginStart = dp(6)
                }

                addView(
                    TextView(this@PocketChatActivity).apply {
                        text = if (input.source == OcrInput.Source.CAMERA) {
                            getString(R.string.photo_input_label)
                        } else {
                            getString(R.string.image_input_label)
                        }
                        setTextColor(resolveThemeColor(R.attr.colorAssistantText))
                        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                        setTypeface(typeface, android.graphics.Typeface.BOLD)
                        maxLines = 1
                    }
                )

                addView(
                    TextView(this@PocketChatActivity).apply {
                        text = pendingImageStatusText(input)
                        setTextColor(pendingImageStatusColor(input))
                        setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                        maxLines = 1
                    }
                )
            }
        )

        chip.addView(
            ImageButton(this).apply {
                layoutParams = LinearLayout.LayoutParams(dp(26), dp(26))
                background = null
                contentDescription = getString(R.string.remove_image_input)
                setImageResource(R.drawable.ic_close_18)
                setColorFilter(resolveThemeColor(R.attr.colorStatusText))
                setPadding(dp(4), dp(4), dp(4), dp(4))
                setOnClickListener { removePendingImageInput(input.id) }
            }
        )

        return chip
    }

    private fun pendingImageStatusText(input: PendingImageInput): String {
        return when (input.status) {
            PendingImageStatus.READING -> getString(R.string.image_input_reading)
            PendingImageStatus.READY -> getString(R.string.image_input_ready)
            PendingImageStatus.FAILED -> getString(R.string.image_input_failed)
        }
    }

    private fun pendingImageStatusColor(input: PendingImageInput): Int {
        return when (input.status) {
            PendingImageStatus.FAILED -> ContextCompat.getColor(this, R.color.delete_red)
            else -> resolveThemeColor(R.attr.colorStatusText)
        }
    }

    private fun buildComposerPrompt(): ComposerPrompt? {
        val typedText = PromptPreprocessor.normalize(inputEditText.text.toString())
        val voiceText = currentVoiceTranscriptForPrompt(typedText)
        val typedTextWithoutVoice = if (voiceText != null) {
            PromptPreprocessor.normalize(speechBasePromptText)
        } else {
            typedText
        }
        if (pendingImageInputs.any { it.status == PendingImageStatus.READING }) {
            showTransientMessage(getString(R.string.image_input_processing))
            return null
        }
        if (pendingImageInputs.any { it.status == PendingImageStatus.FAILED }) {
            showTransientMessage(getString(R.string.image_input_remove_failed))
            return null
        }

        val readyImages = pendingImageInputs.filter {
            it.status == PendingImageStatus.READY && !it.recognizedText.isNullOrBlank()
        }
        val modelText = buildSourceAwareModelText(
            typedText = typedTextWithoutVoice,
            voiceText = voiceText,
            images = readyImages
        )
        if (modelText.isBlank()) {
            return null
        }

        return ComposerPrompt(
            modelText = modelText,
            displayText = buildComposerDisplayText(typedText, readyImages.size)
        )
    }

    private fun currentVoiceTranscriptForPrompt(currentPromptText: String): String? {
        val voiceText = PromptPreprocessor.normalize(speechRecognizedText)
        if (voiceText.isBlank()) {
            return null
        }

        val expectedPrompt = PromptPreprocessor.mergeTypedAndRecognized(
            speechBasePromptText,
            voiceText
        )
        return voiceText.takeIf { currentPromptText == expectedPrompt }
    }

    private fun buildSourceAwareModelText(
        typedText: String,
        voiceText: String?,
        images: List<PendingImageInput>
    ): String {
        val hasTypedText = typedText.isNotBlank()
        val hasVoiceText = !voiceText.isNullOrBlank()
        val hasImageText = images.any { !it.recognizedText.isNullOrBlank() }

        if (!hasVoiceText && !hasImageText) {
            return typedText
        }

        return buildString {
            append("The following user input contains labeled sources. Use each section according to its label.")

            if (hasTypedText) {
                append("\n\n[Typed user message]\n")
                append(typedText)
            }

            if (hasVoiceText) {
                append("\n\n[Voice input transcription]\n")
                append("This text was transcribed from the user's speech. Treat it as user-provided context or request text.\n")
                append(voiceText)
            }

            val imageText = buildImageContextText(images)
            if (imageText.isNotBlank()) {
                append("\n\n")
                append(imageText)
            }
        }.trim()
    }

    private fun buildImageContextText(images: List<PendingImageInput>): String {
        if (images.isEmpty()) {
            return ""
        }

        return buildString {
            append("[Image OCR context]\n")
            append("The following text was extracted from attached image input using OCR. Use it as context from the image; it was not typed directly by the user.")
            images.forEachIndexed { index, input ->
                val sourceLabel = if (input.source == OcrInput.Source.CAMERA) {
                    "camera photo"
                } else {
                    "gallery image"
                }
                append("\n\n[Image ")
                append(index + 1)
                append(" - ")
                append(sourceLabel)
                append("]\n")
                append(input.recognizedText.orEmpty().trim())
            }
        }
    }

    private fun buildComposerDisplayText(typedText: String, imageCount: Int): String {
        if (imageCount <= 0) {
            return typedText
        }

        val imageSummary = if (imageCount == 1) {
            getString(R.string.image_input_display_one)
        } else {
            getString(R.string.image_input_display_many, imageCount)
        }

        return when {
            typedText.isBlank() -> imageSummary
            else -> "$typedText\n\n$imageSummary"
        }
    }

    private fun setPromptInputText(text: String) {
        inputEditText.setText(text)
        inputEditText.setSelection(inputEditText.text.length)
    }

    private fun clearSpeechInputState() {
        speechInput?.cancel()
        updateMicRecordingState(false)
        speechBasePromptText = ""
        speechRecognizedText = ""
        speechCommittedText = ""
        speechPartialText = ""
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
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
        refreshDrawerSessions()
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
        thinkingToggleContainer.visibility = if (state.supportsThinking && !isModelOperationInProgress) View.VISIBLE else View.GONE
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
        if (generationFinished) {
            refreshDrawerSessions()
        }
    }

    private fun renderNoControllerState(
        message: String,
        preserveTranscript: Boolean
    ) {
        title = getString(R.string.toolbar_app_title)
        toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
        thinkingToggleContainer.visibility = View.GONE
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
            val deleteButton: ImageButton = itemView.findViewById(R.id.modelDeleteButton)
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
                isDownloadingThisModel -> getString(R.string.stop_download)
                isCurrentModel -> getString(R.string.model_action_current)
                isAvailable -> getString(R.string.model_action_use)
                else -> getString(R.string.model_action_download)
            }
            actionButton.isEnabled = isDownloadingThisModel || !isModelOperationInProgress
            actionButton.setOnClickListener {
                if (isDownloadingThisModel) {
                    cancelModelDownload()
                } else {
                    handleModelSelection(descriptor)
                }
            }

            val canDeleteDownloadedModel = isDownloaded && !isDownloadingThisModel
            deleteButton.visibility = if (canDeleteDownloadedModel) View.VISIBLE else View.GONE
            deleteButton.isEnabled = !isModelOperationInProgress
            deleteButton.setOnClickListener {
                confirmDeleteModel(descriptor)
            }

            itemProgressBar.visibility = if (isDownloadingThisModel) View.VISIBLE else View.GONE
            updateProgressBar(itemProgressBar, activeDownloadBytes, activeDownloadTotalBytes)
            itemProgressText.visibility = if (isDownloadingThisModel) View.VISIBLE else View.GONE
            itemProgressText.text = formatDownloadProgressText(activeDownloadBytes, activeDownloadTotalBytes)

            dialogUi.listContainer.addView(itemView)
        }
    }

    private fun configureDrawer() {
        drawerSessionsAdapter = DrawerSessionsAdapter(
            fontSizeSp = currentSettings.chatFontSizeSp,
            onSessionSelected = { session ->
                drawerLayout.closeDrawer(GravityCompat.START)
                drawerLayout.post {
                    val controller = requireUsableController()
                    if (controller != null) {
                        controller.loadSession(session.sessionId)
                    }
                }
            },
            onDeleteRequested = { session ->
                confirmDeleteSession(session)
            }
        )
        drawerChatsRecyclerView.layoutManager = LinearLayoutManager(this)
        drawerChatsRecyclerView.adapter = drawerSessionsAdapter

        findViewById<View>(R.id.drawerSettingsRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                showSettingsDialog()
            }
        }

        findViewById<View>(R.id.drawerModelSettingsRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                showModelSettingsDialog()
            }
        }

        findViewById<View>(R.id.drawerAboutRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                showAboutDialog()
            }
        }
    }

    private fun refreshDrawerSessions() {
        if (!::drawerSessionsAdapter.isInitialized) {
            return
        }

        val sessions = chatSessionStore.list()
        drawerSessionsAdapter.submitList(sessions)
        drawerChatsEmptyView.visibility = if (sessions.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun confirmDeleteSession(session: ChatSessionSummary) {
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.delete_chat))
            .setMessage(getString(R.string.delete_chat_confirmation))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                val deleted = chatController?.deleteSession(session.sessionId)
                    ?: chatSessionStore.delete(session.sessionId)
                if (deleted) {
                    refreshDrawerSessions()
                    showTransientMessage(getString(R.string.chat_deleted))
                }
            }
            .create()

        dialog.setOnShowListener {
            applyDeleteConfirmationButtonColors(dialog)
        }
        dialog.show()
    }

    private fun confirmDeleteModel(descriptor: ModelDescriptor) {
        if (currentModel?.id == descriptor.id) {
            showTransientMessage(getString(R.string.delete_model_current_blocked))
            return
        }
        if (isModelOperationInProgress) {
            showTransientMessage(getString(R.string.model_operation_in_progress))
            return
        }

        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.delete_model))
            .setMessage(getString(R.string.delete_model_confirmation, descriptor.displayName))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                if (modelFileResolver.deleteDownloadedModel(descriptor)) {
                    refreshModelSelectionDialog()
                    showTransientMessage(getString(R.string.model_deleted_message, descriptor.displayName))
                } else {
                    showTransientMessage(getString(R.string.delete_model_failed, descriptor.displayName))
                }
            }
            .create()

        dialog.setOnShowListener {
            applyDeleteConfirmationButtonColors(dialog)
        }
        dialog.show()
    }

    private fun applyDeleteConfirmationButtonColors(dialog: AlertDialog) {
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE)
            ?.setTextColor(resolveThemeColor(R.attr.colorStatusText))
        dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            ?.setTextColor(ContextCompat.getColor(this, R.color.delete_red))
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
        clearSpeechInputState()
        clearPendingImageInputs()
        controller.startNewChat()
        refreshDrawerSessions()
    }

    private fun showSettingsDialog() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_settings, null)
        val appearanceModeGroup: RadioGroup = dialogView.findViewById(R.id.appearanceModeGroup)
        val accentColorGroup: RadioGroup = dialogView.findViewById(R.id.accentColorGroup)
        val fontSizeValue: TextView = dialogView.findViewById(R.id.fontSizeValue)
        val fontSizePreview: TextView = dialogView.findViewById(R.id.fontSizePreview)
        val fontSizeSeekBar: SeekBar = dialogView.findViewById(R.id.fontSizeSeekBar)
        val cancelButton: Button = dialogView.findViewById(R.id.settingsCancelButton)
        val saveButton: Button = dialogView.findViewById(R.id.settingsSaveButton)

        when (currentSettings.appearance) {
            AppAppearanceMode.LIGHT -> appearanceModeGroup.check(R.id.appearanceLight)
            AppAppearanceMode.DARK -> appearanceModeGroup.check(R.id.appearanceDark)
        }

        when (currentSettings.accent) {
            AppAccentOption.OCEAN -> accentColorGroup.check(R.id.themeOcean)
            AppAccentOption.MIDNIGHT -> accentColorGroup.check(R.id.themeMidnight)
            AppAccentOption.FOREST -> accentColorGroup.check(R.id.themeForest)
            AppAccentOption.VIOLET -> accentColorGroup.check(R.id.themeViolet)
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
            .setView(dialogView)
            .create()

        cancelButton.setOnClickListener {
            dialog.dismiss()
        }

        saveButton.setOnClickListener {
            val selectedAppearance = when (appearanceModeGroup.checkedRadioButtonId) {
                R.id.appearanceLight -> AppAppearanceMode.LIGHT
                else -> AppAppearanceMode.DARK
            }
            val selectedAccent = when (accentColorGroup.checkedRadioButtonId) {
                R.id.themeMidnight -> AppAccentOption.MIDNIGHT
                R.id.themeForest -> AppAccentOption.FOREST
                R.id.themeViolet -> AppAccentOption.VIOLET
                else -> AppAccentOption.OCEAN
            }
            val selectedFontSize = 13f + fontSizeSeekBar.progress
            val updatedSettings = AppSettings(
                accent = selectedAccent,
                appearance = selectedAppearance,
                chatFontSizeSp = selectedFontSize
            )
            val visualThemeChanged = updatedSettings.accent != currentSettings.accent ||
                updatedSettings.appearance != currentSettings.appearance
            currentSettings = updatedSettings
            settingsStore.save(updatedSettings)
            dialog.dismiss()
            if (visualThemeChanged) {
                recreate()
            } else {
                chatAdapter.updateFontSize(updatedSettings.chatFontSizeSp)
                drawerSessionsAdapter.updateFontSize(updatedSettings.chatFontSizeSp)
                applyTypography(
                    findViewById(R.id.statusView),
                    findViewById(R.id.sendButton),
                    findViewById(R.id.stopButton)
                )
            }
        }

        showPanelDialog(dialog)
    }

    private fun showAboutDialog() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_about, null)
        val versionView: TextView = dialogView.findViewById(R.id.aboutVersion)
        val githubLinkView: TextView = dialogView.findViewById(R.id.aboutGithubLink)
        val okButton: Button = dialogView.findViewById(R.id.aboutOkButton)

        versionView.text = getString(R.string.about_version_format, currentVersionName())
        githubLinkView.movementMethod = LinkMovementMethod.getInstance()
        Linkify.addLinks(githubLinkView, Linkify.WEB_URLS)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()

        okButton.setOnClickListener {
            dialog.dismiss()
        }

        showPanelDialog(dialog)
    }

    @Suppress("DEPRECATION")
    private fun currentVersionName(): String {
        return runCatching {
            packageManager.getPackageInfo(packageName, 0).versionName
                ?.takeIf { it.isNotBlank() }
                ?: "unknown"
        }.getOrDefault("unknown")
    }

    private fun showModelSettingsDialog() {
        val descriptor = currentModel
        if (descriptor == null || chatController == null) {
            showTransientMessage(getString(R.string.model_required_message))
            showModelSelectionDialog(forceSelection = true)
            return
        }

        if (chatController?.state?.value?.isGenerating == true) {
            showTransientMessage(getString(R.string.model_settings_generation_blocked))
            return
        }

        if (!modelFileResolver.isModelDownloaded(descriptor)) {
            showTransientMessage(getString(R.string.model_settings_download_required))
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_model_settings, null)
        val modelNameView: TextView = dialogView.findViewById(R.id.modelSettingsModelName)
        val presetSpinner: AppCompatSpinner = dialogView.findViewById(R.id.modelInstructionPresetSpinner)
        val instructionInput: EditText = dialogView.findViewById(R.id.modelInstructionInput)
        val cancelButton: Button = dialogView.findViewById(R.id.modelSettingsCancelButton)
        val saveButton: Button = dialogView.findViewById(R.id.modelSettingsSaveButton)

        modelNameView.text = descriptor.displayName
        val presets = InstructionPreset.entries.toList()
        val presetLabels = presets.map { it.label }
        val presetAdapter = ArrayAdapter(
            this,
            R.layout.item_instruction_preset_spinner,
            presetLabels
        ).apply {
            setDropDownViewResource(R.layout.item_instruction_preset_dropdown)
        }
        presetSpinner.adapter = presetAdapter

        var selectedPreset = modelInstructionStore.loadPreset(descriptor)
        var applyingPresetText = false
        var switchingToCustomFromEdit = false

        val currentInstruction = modelInstructionStore.loadInstruction(descriptor)
        val selectedPresetIndex = presets.indexOf(selectedPreset).coerceAtLeast(0)
        var customInstructionText = if (selectedPreset == InstructionPreset.CUSTOM) {
            currentInstruction
        } else {
            InstructionPreset.CUSTOM.instruction
        }
        var ignoreInitialPresetSelection = true
        presetSpinner.setSelection(selectedPresetIndex, false)
        applyingPresetText = true
        instructionInput.setText(currentInstruction)
        instructionInput.setSelection(currentInstruction.length)
        applyingPresetText = false

        presetSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val preset = presets.getOrNull(position) ?: return
                if (ignoreInitialPresetSelection && position == selectedPresetIndex) {
                    ignoreInitialPresetSelection = false
                    return
                }
                ignoreInitialPresetSelection = false
                selectedPreset = preset
                if (switchingToCustomFromEdit) {
                    return
                }

                applyingPresetText = true
                val presetInstruction = if (preset == InstructionPreset.CUSTOM) {
                    customInstructionText
                } else {
                    preset.instruction
                }
                instructionInput.setText(presetInstruction)
                instructionInput.setSelection(instructionInput.text.length)
                applyingPresetText = false
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }

        instructionInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(
                text: CharSequence?,
                start: Int,
                count: Int,
                after: Int
            ) = Unit

            override fun onTextChanged(
                text: CharSequence?,
                start: Int,
                before: Int,
                count: Int
            ) = Unit

            override fun afterTextChanged(text: Editable?) {
                if (applyingPresetText) {
                    return
                }

                val editedInstruction = text?.toString().orEmpty()
                if (selectedPreset == InstructionPreset.CUSTOM) {
                    customInstructionText = editedInstruction
                    return
                }

                if (editedInstruction.trim() == selectedPreset.instruction) {
                    return
                }

                customInstructionText = editedInstruction
                selectedPreset = InstructionPreset.CUSTOM
                val customPresetIndex = presets.indexOf(InstructionPreset.CUSTOM)
                if (presetSpinner.selectedItemPosition != customPresetIndex) {
                    switchingToCustomFromEdit = true
                    presetSpinner.setSelection(customPresetIndex)
                    switchingToCustomFromEdit = false
                }
            }
        })

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()

        cancelButton.setOnClickListener {
            dialog.dismiss()
        }

        saveButton.setOnClickListener {
            val instruction = instructionInput.text.toString().trim()
            if (instruction.isBlank()) {
                showTransientMessage(getString(R.string.model_instruction_empty))
                return@setOnClickListener
            }

            val presetToSave = if (
                selectedPreset != InstructionPreset.CUSTOM &&
                instruction != selectedPreset.instruction
            ) {
                InstructionPreset.CUSTOM
            } else {
                selectedPreset
            }

            modelInstructionStore.saveInstruction(descriptor, instruction, presetToSave)
            dialog.dismiss()
            showTransientMessage(getString(R.string.model_instruction_saved))
        }

        showPanelDialog(dialog)
    }

    private fun showPanelDialog(dialog: AlertDialog) {
        dialog.show()
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
    }

    private fun resolveThemeColor(attrId: Int): Int {
        val typedValue = TypedValue()
        theme.resolveAttribute(attrId, typedValue, true)
        return typedValue.data
    }

    private fun applyTypography(
        statusView: TextView,
        sendButton: Button,
        stopButton: Button
    ) {
        inputEditText.textSize = currentSettings.chatFontSizeSp
        statusView.textSize = currentSettings.chatFontSizeSp
        transientMessageView.textSize = (currentSettings.chatFontSizeSp - 1f).coerceAtLeast(12f)
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

    private fun updateMicRecordingState(active: Boolean) {
        if (!::micInputButton.isInitialized) {
            return
        }

        val backgroundColor = if (active) {
            resolveThemeColor(R.attr.colorStopFill)
        } else {
            resolveThemeColor(R.attr.colorFrameBackground)
        }
        val strokeColor = if (active) {
            resolveThemeColor(R.attr.colorStopStroke)
        } else {
            resolveThemeColor(R.attr.colorSendStroke)
        }
        val iconColor = if (active) {
            ContextCompat.getColor(this, R.color.white)
        } else {
            resolveThemeColor(R.attr.colorAssistantText)
        }

        micInputButton.isSelected = active
        micInputButton.backgroundTintList = ColorStateList.valueOf(backgroundColor)
        micInputButton.strokeColor = ColorStateList.valueOf(strokeColor)
        micInputButton.iconTint = ColorStateList.valueOf(iconColor)
    }

    private fun showTransientMessage(message: String) {
        if (message.isBlank() || !::transientMessageView.isInitialized || isDestroyed) {
            return
        }

        transientMessageView.removeCallbacks(hideTransientMessageRunnable)
        transientMessageView.text = message
        transientMessageView.visibility = View.VISIBLE
        transientMessageView.bringToFront()
        transientMessageView.postDelayed(hideTransientMessageRunnable, 1300L)
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
