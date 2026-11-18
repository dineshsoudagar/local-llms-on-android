package com.example.local_llm

import android.app.ActivityManager
import kotlinx.coroutines.delay
import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageDecoder
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.media.MediaRecorder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.Editable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ImageSpan
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.text.TextWatcher
import android.text.method.LinkMovementMethod
import android.text.method.HideReturnsTransformationMethod
import android.text.format.Formatter
import android.text.util.Linkify
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.WindowManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import androidx.annotation.DrawableRes
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.PopupMenu
import android.widget.PopupWindow
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.StringRes
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Camera
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.res.ResourcesCompat
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.CircularProgressIndicator
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import kotlin.math.ceil
import kotlin.math.max

open class PocketChatActivity : AppCompatActivity(), ProFeatureLauncher {

    companion object {
        private const val TAG = "PocketChatActivity"
        private const val PLAY_PACKAGE = "io.github.dineshsoudagar.pocketllm"
        private const val IMAGE_INPUT_MAX_DIMENSION = 1024
        private const val IMAGE_INPUT_MIN_NORMALIZED_DIMENSION = 256
        private const val IMAGE_INPUT_EXTREME_ASPECT_RATIO = 4f
        private const val IMAGE_INPUT_JPEG_QUALITY = 90
        private const val CAMERA_CROP_MAX_IMAGE_DIMENSION = 1600
        private const val CAMERA_CROP_JPEG_QUALITY = 95
        private const val STATE_SETTINGS_PAGE_OPEN = "settings_page_open"
    }

    private data class ModelDialogViews(
        val dialog: AlertDialog,
        val introView: TextView,
        val listContainer: LinearLayout
    )

    private data class ComposerPrompt(
        val modelText: String,
        val displayText: String,
        val imageFilePaths: List<String> = emptyList(),
        val nativeAudioInputs: List<NativeAudioInput> = emptyList(),
        val attachmentContext: AttachmentContext? = null,
        val imageTurns: List<ChatTurn> = emptyList(),
        val attachmentTurns: List<ChatTurn> = emptyList()
    )

    private data class ImagePreprocessingResult(
        val text: String,
        val tempFilePath: String?,
        val directImageFilePath: String? = null
    )

    private data class ComposerDraft(
        val typedText: String,
        val typedTextWithoutVoice: String,
        val voiceText: String?
    )

    private enum class PendingImageStatus {
        PENDING,
        READING,
        READY,
        FAILED
    }

    private enum class ImageInputContextSource {
        OCR,
        GEMMA_DIRECT
    }

    private data class PendingImageInput(
        val id: Long,
        val savedImageId: String,
        val savedImagePath: String,
        val uri: Uri,
        val source: OcrInput.Source,
        val contextSource: ImageInputContextSource,
        val status: PendingImageStatus = PendingImageStatus.PENDING,
        val recognizedText: String? = null,
        val errorMessage: String? = null,
        val tempFilePath: String? = null,
        val directImageFilePath: String? = null
    )

    private lateinit var settingsStore: AppSettingsStore
    private lateinit var currentSettings: AppSettings
    private var settingsPageOpen = false
    private lateinit var modelInstructionStore: ModelInstructionStore
    private lateinit var modelRuntimeSettingsStore: ModelRuntimeSettingsStore
    private lateinit var modelSelectionStore: ModelSelectionStore
    private lateinit var modelFileResolver: ModelFileResolver
    private lateinit var modelLoadRecoveryStore: ModelLoadRecoveryStore
    private lateinit var modelLoadCoordinator: ModelLoadCoordinator
    private lateinit var modelPreflightChecker: ModelPreflightChecker
    private lateinit var modelFileValidator: ModelFileValidator
    private lateinit var chatSessionStore: ChatSessionStore
    private lateinit var savedImageStore: SavedImageStore
    private lateinit var attachmentRepository: AttachmentRepository
    private lateinit var attachmentWorkerCoordinator: AttachmentWorkerCoordinator
    private lateinit var whisperModelStore: WhisperModelStore
    private lateinit var retainedState: PocketChatViewModel
    private var currentModel: ModelDescriptor? = null
    private var currentImageInputMode: ImageInputMode = ImageInputMode.OCR
    private var chatController: PersistentChatController? = null
    private var controllerStateJob: Job? = null
    private var modelPreparationJob: Job? = null
    private var modelDownloadStateJob: Job? = null
    private var lastGemmaDirectImageInputAvailable: Boolean? = null
    private var modelDialogViews: ModelDialogViews? = null
    // The Manage models row that shows the running download, so progress ticks can update just that row.
    private var downloadRowProgress: Pair<ProgressBar, TextView>? = null
    private var modelCardTarget: LinearLayout? = null
    private var downloadSheet: BottomSheetDialog? = null
    private var downloadSheetViews: DownloadSheetViews? = null

    private class DownloadSheetViews(val ring: ProgressBar, val title: TextView, val sub: TextView)
    private var downloadRowModelId: String? = null
    private var modelRecoveryDialog: AlertDialog? = null
    private val contextProbeHost = object : ContextProbeFeature.Host {
        override val activity: AppCompatActivity get() = this@PocketChatActivity
        override fun showTransientMessage(message: String) = this@PocketChatActivity.showTransientMessage(message)
        override fun resolveThemeColor(attrId: Int): Int = this@PocketChatActivity.resolveThemeColor(attrId)
        override fun updateKeepScreenOn() = this@PocketChatActivity.updateKeepScreenOn()
        override val isModelPreparing: Boolean get() = modelPreparationJob?.isActive == true
        override val isModelLoading: Boolean get() = chatController?.state?.value?.isLoading == true
        // The chat model's GPU memory must be free, or the test would measure what is left over.
        override fun unloadModelForProbe(message: String): () -> Unit {
            val modelToReload = currentModel
            cancelCurrentInitializationMarker()
            val controller = chatController
            val snapshot = controller?.snapshotActiveChat() ?: retainedState.pendingModelLoadSnapshot
            if (controller != null) {
                controllerStateJob?.cancel()
                SharedChatControllerRegistry.clearIf(controller)
                chatController = null
                retainedState.chatController = null
                controller.close()
            }
            renderNoControllerState(message, preserveTranscript = true)
            return {
                modelToReload?.let { requestModelLoad(it, snapshot) }
            }
        }
    }
    private var modelPickerPopup: BottomSheetDialog? = null
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var drawerChatsRecyclerView: RecyclerView
    private lateinit var drawerChatsEmptyView: TextView
    private lateinit var drawerSessionsAdapter: DrawerSessionsAdapter
    private lateinit var chatRecyclerView: RecyclerView
    private lateinit var inputEditText: EditText
    private lateinit var modelLoadingCard: ModelLoadingCard
    private lateinit var toolbarSubtitleView: TextView
    private lateinit var toolbarModelSelector: View
    private lateinit var selectedModelLabel: TextView
    private lateinit var thinkingToggleContainer: View
    private lateinit var thinkingToggle: CheckBox
    private var currentModelSupportsThinking = false
    private lateinit var newChatButton: View
    private lateinit var sendButton: Button
    private lateinit var imageModePanel: View
    private lateinit var webSearchButton: MaterialButton
    private lateinit var createImageModeButton: MaterialButton
    private lateinit var imageModelChip: MaterialButton
    private lateinit var refinePromptButton: MaterialButton
    private lateinit var imageModeStatus: TextView
    private lateinit var noModelHint: TextView

    private val webSearchHost = object : WebSearchFeature.Host {
        override val activity: AppCompatActivity get() = this@PocketChatActivity
        override val chatController: PersistentChatController? get() = this@PocketChatActivity.chatController
        override fun updateWebSearchButton() = this@PocketChatActivity.updateWebSearchButton()
        override fun requireProOrTrial(feature: ProFeature, action: () -> Unit) = this@PocketChatActivity.requireProOrTrial(feature, action)
        override fun resolveThemeColor(attrId: Int): Int = this@PocketChatActivity.resolveThemeColor(attrId)
    }
    private val chatHost = object : ImageGenFeature.ChatHost {
        override val activity: AppCompatActivity get() = this@PocketChatActivity
        override val chatController: PersistentChatController? get() = this@PocketChatActivity.chatController
        override val inputEditText: EditText get() = this@PocketChatActivity.inputEditText
        override val sendButton: Button get() = this@PocketChatActivity.sendButton
        override val stopButton: Button get() = this@PocketChatActivity.stopButton
        override val attachmentButton: MaterialButton get() = this@PocketChatActivity.attachmentButton
        override val galleryOcrButton: Button get() = this@PocketChatActivity.galleryOcrButton
        override val cameraOcrButton: Button get() = this@PocketChatActivity.cameraOcrButton
        override val createImageModeButton: MaterialButton get() = this@PocketChatActivity.createImageModeButton
        override val imageModelChip: MaterialButton get() = this@PocketChatActivity.imageModelChip
        override val refinePromptButton: MaterialButton get() = this@PocketChatActivity.refinePromptButton
        override val webSearchButton: MaterialButton get() = this@PocketChatActivity.webSearchButton
        override val imageModeStatus: TextView get() = this@PocketChatActivity.imageModeStatus
        override val imageModePanel: View get() = this@PocketChatActivity.imageModePanel
        override val proGate: ProGate get() = this@PocketChatActivity.proGate
        override val modelFileResolver: ModelFileResolver get() = this@PocketChatActivity.modelFileResolver
        override fun requireProOrTrial(feature: ProFeature, action: () -> Unit) = this@PocketChatActivity.requireProOrTrial(feature, action)
        override fun showTransientMessage(message: String) = this@PocketChatActivity.showTransientMessage(message)
        override fun resolveThemeColor(attrId: Int): Int = this@PocketChatActivity.resolveThemeColor(attrId)
        override fun updateWebSearchButton() = this@PocketChatActivity.updateWebSearchButton()
        override fun applyChatState() = this@PocketChatActivity.chatController?.let { this@PocketChatActivity.applyChatState(it.state.value) } ?: this@PocketChatActivity.applyImageModeComposer()
        override fun setPromptInputText(text: String) = this@PocketChatActivity.setPromptInputText(text)
        override fun openImageViewer(path: String) = this@PocketChatActivity.openImageViewer(path)
        override fun showManageModelsDialog() = this@PocketChatActivity.showManageModelsDialog()
        override fun startModelDownload(spec: ModelDescriptor) = this@PocketChatActivity.startModelDownload(spec)
        override fun refreshProBadges() = this@PocketChatActivity.refreshProBadges()
        override fun proLabel(resId: Int, feature: ProFeature): CharSequence = this@PocketChatActivity.proLabel(resId, feature)
        override fun launchImageGen(intent: Intent) = this@PocketChatActivity.imageGenLauncher.launch(intent)
        override fun addDownloadableModelOption(descriptor: ModelDescriptor, onUseModel: () -> Unit) {
            this@PocketChatActivity.modelDialogViews?.let {
                this@PocketChatActivity.addDownloadableModelOption(this@PocketChatActivity.layoutInflater, it.listContainer, descriptor, false, onUseModel)
            }
        }
        override fun dismissModelDialog() {
            this@PocketChatActivity.modelDialogViews?.dialog?.dismiss()
        }
    }
    private var refineJob: Job? = null
    private lateinit var contextUsageIndicator: CircularProgressIndicator
    private lateinit var stopButton: Button
    private lateinit var micInputButton: MaterialButton
    private lateinit var attachmentButton: MaterialButton
    private lateinit var galleryOcrButton: Button
    private lateinit var cameraOcrButton: Button
    private lateinit var pendingImageInputsScroll: HorizontalScrollView
    private lateinit var pendingImageInputsContainer: LinearLayout
    private lateinit var activeAttachmentContainer: LinearLayout
    private lateinit var activeAttachmentText: TextView
    private lateinit var activeAttachmentProgress: ProgressBar
    private lateinit var activeAttachmentRetry: MaterialButton
    private lateinit var activeAttachmentCancel: MaterialButton
    private lateinit var activeAttachmentDetach: MaterialButton
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
    private var imagePreprocessingJob: Job? = null
    private var attachmentImportJob: Job? = null
    private var audioPreparationJob: Job? = null
    private var activeAttachment: AttachmentDescriptor? = null
    private var attachmentImportInProgress = false
    private var confirmedAttachmentOperationKey: String? = null
    private var activeAttachmentWork: AttachmentWorkRecord? = null
    private var activeAttachmentWorkState: androidx.work.WorkInfo.State? = null
    private var preparedGemmaAudio: SegmentedNativeAudio? = null
    private var preparedGemmaAudioAttachmentId: String? = null
    private var dictationRecorder: MediaRecorder? = null
    private var dictationRecordingFile: File? = null
    private var dictationRecordingDialog: AlertDialog? = null
    private var dictationTranscriptionDialog: AlertDialog? = null
    private var pendingDictationRecordingStart = false
    private var dictationTranscriptionJob: Job? = null
    private val dictationTimerHandler = Handler(Looper.getMainLooper())
    private var dictationStartedAtMillis = 0L
    private var dictationTimer: Runnable? = null
    private var isImagePreprocessingForSend = false
    private var autoScrollDuringGeneration = false
    private var autoScrollPendingFinalUpdate = false
    private var wasGenerating = false
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

    private val translatorLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        TranslatorFeature.instance?.resultChatText(result.resultCode, result.data)?.let(::setPromptInputText)
    }

    private fun openTranslator(text: String? = null, imagePath: String? = null, pendingModelId: String? = null) {
        val feature = TranslatorFeature.instance ?: return
        translatorLauncher.launch(
            feature.createIntent(this, text, imagePath).putExtra(SharedContent.EXTRA_PENDING_MODEL, pendingModelId)
        )
    }

    private var translatorLoadStartedAt = 0L

    private val translatorHost = object : TranslatorFeature.ChatHost {
        override fun loadModel(descriptor: ModelDescriptor): String? {
            if (ContextProbeFeature.instance?.isRunning() == true || LanServerFeature.isRunning()) {
                return getString(R.string.lan_server_model_switch_blocked)
            }
            if (chatController?.state?.value?.isGenerating == true ||
                isImagePreprocessingForSend || attachmentImportInProgress ||
                audioPreparationJob?.isActive == true
            ) {
                return getString(R.string.model_switch_generation_blocked)
            }
            translatorLoadStartedAt = System.currentTimeMillis()
            // The snapshot carries the open chat over to the new model.
            requestModelLoad(descriptor, chatController?.snapshotActiveChat())
            return null
        }

        override fun loadFailure(modelId: String): String? =
            modelLoadCoordinator.record
                ?.takeIf {
                    it.modelId == modelId && it.phase == ModelLoadPhase.FAILED &&
                        it.timestampMillis >= translatorLoadStartedAt
                }
                ?.let { it.failureReason ?: getString(R.string.model_load_failed_generic) }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleSharedContent(intent)
    }

    /** Text or an image shared from another app, after the user chose what to do with it. */
    private fun handleSharedContent(intent: Intent) {
        val action = intent.getStringExtra(SharedContent.EXTRA_ACTION) ?: return
        // Handled once; a rotation must not repeat it.
        intent.removeExtra(SharedContent.EXTRA_ACTION)
        if (!LegalConsentStore(this).hasAcceptedCurrentVersion()) {
            showTransientMessage(getString(R.string.share_needs_consent))
            return
        }
        val text = intent.getStringExtra(SharedContent.EXTRA_TEXT)
        val imagePath = intent.getStringExtra(SharedContent.EXTRA_IMAGE_PATH)
        if (action == SharedContent.ACTION_LOAD_MODEL) {
            val descriptor = text?.let(ModelRegistry::findById) ?: return
            val refused = translatorHost.loadModel(descriptor)
            if (refused != null) showTransientMessage(refused)
            openTranslator(pendingModelId = descriptor.id.takeIf { refused == null })
            return
        }
        if (action == SharedContent.ACTION_TRANSLATE) {
            openTranslator(text, imagePath)
            return
        }
        if (chatController?.state?.value?.isGenerating == true) {
            showTransientMessage(getString(R.string.share_chat_busy))
            return
        }
        if (action == SharedContent.ACTION_NEW_CHAT && chatController?.state?.value?.isReady == true) {
            startNewChatFromUi()
        }
        if (text != null) {
            val draft = inputEditText.text.toString()
            setPromptInputText(if (draft.isBlank()) text else draft.trimEnd() + "\n\n" + text)
        }
        if (imagePath != null) importAndAddPendingImage(Uri.fromFile(java.io.File(imagePath)), OcrInput.Source.GALLERY)
    }

    private val imageGenLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        refreshProBadges()
        if (ImageGenFeature.instance?.isResultManageModels(result.resultCode) == true) showManageModelsDialog()
        if (ImageGenFeature.instance?.isResultUseInChat(result.resultCode) == true) {
            ImageGenFeature.instance?.getResultPath(result.data)?.let(::attachSavedImageByPath)
        }
    }

    private val hfBrowserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode != RESULT_OK) return@registerForActivityResult
        ModelRegistry.loadCustomModels(this)
        ImageGenFeature.instance?.loadCustomModels(this)
        result.data?.getStringExtra("added_model_name")?.let {
            showTransientMessage(getString(R.string.hf_added, it))
        }
        showManageModelsDialog()
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

    private val cameraPermissionLauncher: androidx.activity.result.ActivityResultLauncher<String> = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        val shouldStart = pendingCameraOcrStart
        pendingCameraOcrStart = false
        if (granted && shouldStart) {
            showCameraOcrDialog()
        } else if (!granted) {
            CameraPermissionPrompt.show(
                this,
                onTryAgain = {
                    pendingCameraOcrStart = true
                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                },
                onOpenSettings = {},
                onCancel = { showTransientMessage(getString(R.string.camera_permission_denied)) }
            )
        }
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) {
        // The foreground service still runs if the user denies this. The permission only
        // controls whether Android 13+ can show the progress notification in the drawer.
    }

    private val galleryImagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let { selectedImage ->
            importAndAddPendingImage(selectedImage, OcrInput.Source.GALLERY)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        settingsStore = AppSettingsStore(this)
        currentSettings = settingsStore.load()
        settingsPageOpen = savedInstanceState?.getBoolean(STATE_SETTINGS_PAGE_OPEN) == true
        modelInstructionStore = ModelInstructionStore(this)
        modelRuntimeSettingsStore = ModelRuntimeSettingsStore(this)
        modelSelectionStore = ModelSelectionStore(this)
        modelFileResolver = ModelFileResolver(this)
        ModelRegistry.loadCustomModels(this)
        ImageGenFeature.instance?.loadCustomModels(this)
        modelLoadRecoveryStore = ModelLoadRecoveryStore(this)
        modelLoadCoordinator = ModelLoadCoordinator(modelLoadRecoveryStore.load())
        modelPreflightChecker = ModelPreflightChecker(this, modelFileResolver)
        modelFileValidator = ModelFileValidator(modelFileResolver)
        chatSessionStore = ChatSessionStore(this)
        savedImageStore = SavedImageStore(this)
        attachmentRepository = AttachmentRepository(this)
        attachmentWorkerCoordinator = AttachmentWorkerCoordinator(this)
        whisperModelStore = WhisperModelStore(this)
        setTheme(currentSettings.accent.styleFor(currentSettings.appearance, this))
        if (currentSettings.isDark(this)) theme.applyStyle(currentSettings.background.overlayRes, true)
        super.onCreate(savedInstanceState)
        // The manifest theme is light; paint the window in the chosen theme at once so a theme change does not flash.
        window.setBackgroundDrawable(ColorDrawable(resolveThemeColor(com.google.android.material.R.attr.colorSurface)))
        applyEdgeToEdge()
        retainedState = ViewModelProvider(this)[PocketChatViewModel::class.java]
        if (retainedState.chatController == null) {
            val interruptedLoad = modelLoadCoordinator.record
                ?.takeIf { it.phase == ModelLoadPhase.INITIALIZING }
            val interruptedCause = interruptedLoad
                ?.let { ProcessExitInspector.interruptedLoadCause(this, it.timestampMillis) }
                ?: InterruptedLoadCause.UNKNOWN
            if (interruptedLoad != null && interruptedCause == InterruptedLoadCause.NATIVE_CRASH) {
                modelLoadRecoveryStore.markGpuUnsafe(interruptedLoad.modelId)
            }
            modelLoadCoordinator.recoverInterrupted(System.currentTimeMillis(), interruptedCause)
            persistModelLoadRecord()
        }
        setContentView(R.layout.activity_main)

        val toolbar: MaterialToolbar = findViewById(R.id.topToolbar)
        val toolbarMenuButton: View = findViewById(R.id.toolbarMenuButton)
        drawerLayout = findViewById(R.id.drawerLayout)
        drawerChatsRecyclerView = findViewById(R.id.drawerChatsRecyclerView)
        drawerChatsEmptyView = findViewById(R.id.drawerChatsEmpty)
        toolbarModelSelector = findViewById(R.id.modelSelector)
        selectedModelLabel = findViewById(R.id.selectedModelLabel)
        toolbarSubtitleView = findViewById(R.id.toolbarSubtitle)
        findViewById<TextView>(R.id.toolbarTitle).typeface = ResourcesCompat.getFont(this, R.font.space_grotesk_medium)
        applyThemeGradient(findViewById(R.id.toolbarTitle))
        thinkingToggle = findViewById(R.id.thinkingToggle)
        thinkingToggleContainer = thinkingToggle
        thinkingToggle.setChipIcon(R.drawable.ic_tier_bulb)
        thinkingToggle.compoundDrawableTintList = ContextCompat.getColorStateList(this, R.color.think_chip_content)
        newChatButton = findViewById(R.id.newChatButton)
        inputEditText = findViewById(R.id.userInput)
        sendButton = findViewById(R.id.sendButton)
        contextUsageIndicator = findViewById(R.id.contextUsageIndicator)
        contextUsageIndicator.setOnClickListener {
            val state = chatController?.state?.value ?: return@setOnClickListener
            if (state.contextWindowTokens <= 0) return@setOnClickListener
            showContextUsageDialog(state)
        }
        stopButton = findViewById(R.id.stopButton)
        micInputButton = findViewById(R.id.micInputButton)
        attachmentButton = findViewById(R.id.attachmentButton)
        imageModePanel = findViewById(R.id.imageModePanel)
        webSearchButton = findViewById(R.id.webSearchButton)
        // Web search is a Pro feature: the free build has no implementation, so hide the chip.
        webSearchButton.visibility = if (WebSearchFeature.isAvailable) View.VISIBLE else View.GONE
        webSearchButton.setOnClickListener { onWebSearchChipClicked() }
        webSearchButton.setOnLongClickListener { showManageWebSearchDialog(); true }
        createImageModeButton = findViewById(R.id.createImageModeButton)
        imageModelChip = findViewById(R.id.imageModelChip)
        refinePromptButton = findViewById(R.id.refinePromptButton)
        imageModeStatus = findViewById(R.id.imageModeStatus)
        noModelHint = findViewById(R.id.noModelHint)
        val noModelLink = getString(R.string.no_model_hint_link)
        noModelHint.text = SpannableString(getString(R.string.no_model_hint_text) + " " + noModelLink).apply {
            val start = length - noModelLink.length
            setSpan(ForegroundColorSpan(resolveThemeColor(R.attr.colorSendFill)), start, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            setSpan(android.text.style.StyleSpan(android.graphics.Typeface.BOLD), start, length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        noModelHint.setOnClickListener { showManageModelsDialog() }
        galleryOcrButton = findViewById(R.id.galleryOcrButton)
        cameraOcrButton = findViewById(R.id.cameraOcrButton)
        pendingImageInputsScroll = findViewById(R.id.pendingImageInputsScroll)
        pendingImageInputsContainer = findViewById(R.id.pendingImageInputsContainer)
        activeAttachmentContainer = findViewById(R.id.activeAttachmentContainer)
        activeAttachmentText = findViewById(R.id.activeAttachmentText)
        activeAttachmentProgress = findViewById(R.id.activeAttachmentProgress)
        activeAttachmentRetry = findViewById(R.id.activeAttachmentRetry)
        activeAttachmentCancel = findViewById(R.id.activeAttachmentCancel)
        activeAttachmentDetach = findViewById(R.id.activeAttachmentDetach)
        statusView = findViewById(R.id.statusView)
        modelLoadingCard = findViewById(R.id.modelLoadingCard)
        modelLoadingCard.onHidden = { lastChatState?.let { updateChatGreeting(it) } }
        transientMessageView = findViewById(R.id.transientMessageView)
        chatRecyclerView = findViewById(R.id.chatRecyclerView)
        speechInput = createSpeechInput()
        ocrInput = createOcrInput()

        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)
        toolbarMenuButton.setOnClickListener {
            refreshDrawerSessions()
            refreshProBadges()
            drawerLayout.openDrawer(GravityCompat.START)
        }
        toolbarSubtitleView.text = getString(R.string.model_picker_empty_subtitle)
        configureDrawer()

        chatAdapter = ChatAdapter(
            fontSizeSp = currentSettings.chatFontSizeSp,
            onImageTurnSelected = { turn ->
                turn.imagePath?.let(::openImageViewer)
            },
            onImageTurnLongPressed = ::showImageTurnActions,
            onImageTurnReported = ::reportGeneratedImage,
            onAttachmentTurnSelected = ::reactivateAttachmentTurn,
            onReportResponse = { responseText ->
                LegalDialogs.showReportResponse(
                    this,
                    currentModel?.displayName,
                    currentVersionName(),
                    responseText
                )
            },
            onTranslateResponse = if (TranslatorFeature.isAvailable) { text ->
                requireProOrTrial(ProFeature.TRANSLATOR) { openTranslator(text = text) }
            } else null,
            onShareResponse = { text ->
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, text)
                }
                startActivity(Intent.createChooser(send, null))
            },
            onRetryResponse = ::retryLastResponse,
            onMagicAction = { turn, tool ->
                turn.imagePath?.let { openImageGeneration(toolName = tool.name, photoPath = it) }
            }
        )
        chatAdapter.setMagicNote(magicActionsNote())

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
            showModelPickerPopup()
        }

        newChatButton.setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            startNewChatFromUi()
        }
        findViewById<View>(R.id.toolbarNewChat).setOnClickListener { startNewChatFromUi() }
        findViewById<View>(R.id.toolsButton).setOnClickListener(::showToolsMenu)

        micInputButton.setOnClickListener {
            showDictationRecordingDialog()
        }

        attachmentButton.setOnClickListener(::showPlusMenu)
        activeAttachmentDetach.setOnClickListener { detachActiveAttachment() }
        activeAttachmentRetry.setOnClickListener { retryActiveAttachment() }
        activeAttachmentCancel.setOnClickListener { cancelActiveAttachmentWork() }

        galleryOcrButton.setOnClickListener {
            galleryImagePickerLauncher.launch(
                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
            )
        }

        cameraOcrButton.setOnClickListener {
            handleCameraOcrClick()
        }

        sendButton.setOnClickListener {
            handleSendClick()
        }

        stopButton.setOnClickListener {
            if (ImageGenFeature.instance?.isImageModeActive() == true && ImageGenFeature.instance?.isBusy() == true) ImageGenFeature.instance?.cancel() else chatController?.cancelGeneration()
        }
        ImageGenFeature.instance?.setupImageMode(chatHost)

        thinkingToggle.setOnCheckedChangeListener { _, isChecked ->
            chatController?.setThinkingEnabled(isChecked)
            updateModelSelectorLabel()
        }

        val restoredModel = retainedState.modelId?.let(ModelRegistry::findById)
        currentModel = restoredModel ?: modelSelectionStore.loadSelectedModel()
        currentImageInputMode = modelSelectionStore.loadSelectedImageInputMode()
        ensureImageInputModeAllowedForCurrentModel()
        updateImageInputButtonDescriptions()
        updateModelSelectorLabel()

        val retainedController = retainedState.chatController
        if (retainedController != null && currentModel != null) {
            chatController = retainedController
            SharedChatControllerRegistry.register(currentModel!!.id, retainedController)
            toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
            thinkingToggle.isChecked = retainedController.isThinkingEnabled()
            observeController(retainedController)
            applyChatState(retainedController.state.value)
        } else {
            val startupModel = currentModel
            val failedStartupLoad = startupModel?.let { model ->
                modelLoadCoordinator.record?.takeIf {
                    it.modelId == model.id && it.phase == ModelLoadPhase.FAILED
                }
            }
            if (startupModel != null && failedStartupLoad != null) {
                toolbarSubtitleView.text = startupModel.displayName
                renderNoControllerState(
                    failedStartupLoad.failureReason ?: getString(R.string.model_load_failed_generic),
                    preserveTranscript = false
                )
            } else if (startupModel != null && modelFileResolver.isModelAvailable(startupModel)) {
                requestModelLoad(startupModel, activeChatSnapshot = null)
            } else {
                if (startupModel != null) {
                    currentModel = null
                    retainedState.modelId = null
                    modelSelectionStore.clearSelectedModel()
                    updateModelSelectorLabel()
                }
                renderNoControllerState(
                    getString(R.string.model_required_message),
                    preserveTranscript = false
                )
                thinkingToggle.isChecked = false
            }
        }

        observeModelDownloadState()
        applyModelDownloadState(ModelDownloadStateStore.state.value)
        refreshDrawerSessions()
        reconnectActiveAttachmentWorkIfNeeded()


        chatRecyclerView.post {
            LegalDialogs.showConsentIfNeeded(this, LegalConsentStore(this))
        }

        TranslatorFeature.chatHost = translatorHost
        if (savedInstanceState == null) chatRecyclerView.post { handleSharedContent(intent) }
    }

    override fun onResume() {
        super.onResume()
        updateKeepScreenOn()
        // Tries may have been used in the Translator or Image tools screens.
        refreshProBadges()
    }

    /** Keeps the screen on while the context test or the LAN server runs, so the phone doesn't lock mid-way. */
    private fun updateKeepScreenOn() {
        if (ContextProbeFeature.instance?.isRunning() == true || LanServerFeature.isRunning()) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }

    override fun onStop() {
        pendingDictationRecordingStart = false
        discardDictationRecording()
        super.onStop()
    }

    override fun onDestroy() {
        if (TranslatorFeature.chatHost === translatorHost) TranslatorFeature.chatHost = null
        proGateInstance?.close()
        proGateInstance = null
        // The test's dialogs belong to this activity; stop it rather than leave it running unseen.
        ContextProbeFeature.instance?.onDestroy()
        if (::chatAdapter.isInitialized) {
            chatAdapter.unregisterAdapterDataObserver(chatAdapterObserver)
        }
        if (::transientMessageView.isInitialized) {
            transientMessageView.removeCallbacks(hideTransientMessageRunnable)
        }
        controllerStateJob?.cancel()
        modelDownloadStateJob?.cancel()
        imagePreprocessingJob?.cancel()
        attachmentImportJob?.cancel()
        audioPreparationJob?.cancel()
        cleanupPreparedGemmaAudio()
        dictationTranscriptionJob?.cancel()
        dictationTranscriptionDialog?.dismiss()
        discardDictationRecording()
        isImagePreprocessingForSend = false
        stopCameraOcr()
        cameraOcrDialog?.dismiss()
        speechInput?.destroy()
        ocrInput?.close()
        clearPendingImageInputs()
        modelDialogViews?.dialog?.dismiss()
        modelRecoveryDialog?.dismiss()
        modelPickerPopup?.dismiss()
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean(STATE_SETTINGS_PAGE_OPEN, settingsPageOpen)
        super.onSaveInstanceState(outState)
    }

    private fun requireUsableController(): PersistentChatController? {
        val controller = chatController
        if (controller == null) {
            showTransientMessage(getString(R.string.model_required_message))
            showModelPickerPopup()
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
            statusListener = { message -> runOnUiThread { showTransientMessage(message) } }
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
        val titleView: TextView = dialogView.findViewById(R.id.cameraOcrDialogTitle)
        val previewView: PreviewView = dialogView.findViewById(R.id.cameraPreviewView)
        val zoomLabelView: TextView = dialogView.findViewById(R.id.cameraZoomLabel)
        val zoomSeekBar: SeekBar = dialogView.findViewById(R.id.cameraZoomSeekBar)
        val statusText: TextView = dialogView.findViewById(R.id.cameraOcrStatus)
        val cancelButton: Button = dialogView.findViewById(R.id.cameraOcrCancelButton)
        val captureButton: Button = dialogView.findViewById(R.id.cameraOcrCaptureButton)

        titleView.text = if (currentImageInputMode == ImageInputMode.OCR) {
            getString(R.string.camera_ocr_title)
        } else {
            getString(R.string.camera_image_title)
        }

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
        startCameraPreview(previewView, zoomSeekBar, zoomLabelView)
    }

    private fun startCameraPreview(
        previewView: PreviewView,
        zoomSeekBar: SeekBar,
        zoomLabelView: TextView
    ) {
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
                    val boundCamera = provider.bindToLifecycle(
                        this,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageCapture
                    )
                    cameraProvider = provider
                    cameraImageCapture = imageCapture
                    configureCameraZoomControls(boundCamera, zoomSeekBar, zoomLabelView)
                    previewView.post {
                        previewView.display?.rotation?.let { rotation ->
                            cameraImageCapture?.targetRotation = rotation
                        }
                    }
                    cameraOcrStatusView?.text = cameraReadyStatusText()
                }.onFailure { error ->
                    cameraOcrStatusView?.text = error.message ?: "Camera OCR could not start."
                }
            },
            ContextCompat.getMainExecutor(this)
        )
    }

    private fun configureCameraZoomControls(
        boundCamera: Camera,
        zoomSeekBar: SeekBar,
        zoomLabelView: TextView
    ) {
        val zoomState = boundCamera.cameraInfo.zoomState.value
        val minZoomRatio = zoomState?.minZoomRatio ?: 1f
        val maxZoomRatio = zoomState?.maxZoomRatio ?: minZoomRatio
        val hasZoom = maxZoomRatio > minZoomRatio + 0.01f

        zoomSeekBar.max = 100
        zoomSeekBar.progress = 0
        zoomSeekBar.isEnabled = hasZoom
        updateCameraZoomLabel(zoomLabelView, minZoomRatio)

        zoomSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser || !hasZoom) {
                    return
                }

                val linearZoom = (progress / 100f).coerceIn(0f, 1f)
                boundCamera.cameraControl.setLinearZoom(linearZoom)
                updateCameraZoomLabel(
                    zoomLabelView,
                    minZoomRatio + ((maxZoomRatio - minZoomRatio) * linearZoom)
                )
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
        })
    }

    private fun updateCameraZoomLabel(labelView: TextView, zoomRatio: Float) {
        labelView.text = getString(R.string.camera_zoom_label_format, zoomRatio.coerceAtLeast(1f))
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
                    dialog.dismiss()
                    showCameraPhotoReviewDialog(outputFile)
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

    private fun showCameraPhotoReviewDialog(photoFile: File) {
        val reviewBitmap = runCatching { decodeBitmapForCameraCrop(photoFile) }.getOrElse { error ->
            runCatching { photoFile.delete() }
            val message = error.message ?: getString(R.string.image_input_file_error)
            showTransientMessage(message)
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_camera_photo_review, null)
        val previewImage: ImageView = dialogView.findViewById(R.id.cameraPhotoReviewImage)
        val retakeButton: Button = dialogView.findViewById(R.id.cameraPhotoRetakeButton)
        val cropButton: Button = dialogView.findViewById(R.id.cameraPhotoCropButton)
        val useButton: Button = dialogView.findViewById(R.id.cameraPhotoUseButton)
        previewImage.setImageBitmap(reviewBitmap)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()
        var handled = false

        retakeButton.setOnClickListener {
            handled = true
            runCatching { photoFile.delete() }
            dialog.dismiss()
            showCameraOcrDialog()
        }

        cropButton.setOnClickListener {
            handled = true
            dialog.dismiss()
            showCameraCropDialog(photoFile)
        }

        useButton.setOnClickListener {
            handled = true
            attachCameraPhoto(photoFile)
            dialog.dismiss()
        }

        dialog.setOnDismissListener {
            previewImage.setImageDrawable(null)
            reviewBitmap.recycle()
            if (!handled) {
                runCatching { photoFile.delete() }
            }
        }

        showPanelDialog(dialog)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
    }

    private fun showCameraCropDialog(photoFile: File) {
        val sourceBitmap = runCatching { decodeBitmapForCameraCrop(photoFile) }.getOrElse { error ->
            runCatching { photoFile.delete() }
            val message = error.message ?: getString(R.string.image_input_file_error)
            showTransientMessage(message)
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_camera_crop, null)
        val cropImageView: CropImageView = dialogView.findViewById(R.id.cameraCropImageView)
        val statusView: TextView = dialogView.findViewById(R.id.cameraCropStatus)
        val retakeButton: Button = dialogView.findViewById(R.id.cameraCropRetakeButton)
        val backButton: Button = dialogView.findViewById(R.id.cameraCropBackButton)
        val useButton: Button = dialogView.findViewById(R.id.cameraCropUseButton)
        cropImageView.setImageBitmap(sourceBitmap)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()
        var handled = false

        retakeButton.setOnClickListener {
            handled = true
            runCatching { photoFile.delete() }
            dialog.dismiss()
            showCameraOcrDialog()
        }

        backButton.setOnClickListener {
            handled = true
            dialog.dismiss()
            showCameraPhotoReviewDialog(photoFile)
        }

        useButton.setOnClickListener {
            val finalOutputFile = runCatching {
                val croppedBitmap = cropImageView.cropBitmap()
                try {
                    writeCroppedCameraPhoto(photoFile, croppedBitmap)
                } finally {
                    croppedBitmap.recycle()
                }
            }.getOrElse { error ->
                val message = error.message ?: getString(R.string.camera_crop_failed)
                statusView.text = message
                showTransientMessage(message)
                return@setOnClickListener
            }

            handled = true
            attachCameraPhoto(finalOutputFile)
            dialog.dismiss()
        }

        dialog.setOnDismissListener {
            cropImageView.clearImage()
            sourceBitmap.recycle()
            if (!handled) {
                runCatching { photoFile.delete() }
            }
        }

        showPanelDialog(dialog)
        dialog.window?.setLayout(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
    }

    private fun attachCameraPhoto(photoFile: File) {
        importAndAddPendingImage(
            uri = Uri.fromFile(photoFile),
            source = OcrInput.Source.CAMERA,
            tempFilePath = photoFile.absolutePath
        )
    }

    private fun writeCroppedCameraPhoto(originalFile: File, croppedBitmap: Bitmap): File {
        val croppedFile = createCameraOcrOutputFile()
        try {
            FileOutputStream(croppedFile).use { output ->
                val compressed = croppedBitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    CAMERA_CROP_JPEG_QUALITY,
                    output
                )
                if (!compressed) {
                    throw IOException(getString(R.string.camera_crop_failed))
                }
            }

            if (croppedFile.length() <= 0L) {
                throw IOException(getString(R.string.camera_crop_failed))
            }

            runCatching { originalFile.delete() }
            return croppedFile
        } catch (error: Exception) {
            runCatching { croppedFile.delete() }
            throw error
        }
    }

    private fun decodeBitmapForCameraCrop(imageFile: File): Bitmap {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            runCatching {
                val source = ImageDecoder.createSource(imageFile)
                return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    val width = info.size.width
                    val height = info.size.height
                    if (width > 0 && height > 0) {
                        val (targetWidth, targetHeight) = cameraCropTargetSize(width, height)
                        decoder.setTargetSize(targetWidth, targetHeight)
                    }
                }
            }.onFailure { error ->
                Log.w(TAG, "ImageDecoder could not decode camera crop image; falling back to BitmapFactory.", error)
            }
        }

        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        BitmapFactory.decodeFile(imageFile.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IOException(getString(R.string.camera_crop_failed))
        }

        val options = BitmapFactory.Options().apply {
            inSampleSize = calculateCameraCropSampleSize(bounds.outWidth, bounds.outHeight)
        }
        val decodedBitmap = BitmapFactory.decodeFile(imageFile.absolutePath, options)
            ?: throw IOException(getString(R.string.camera_crop_failed))
        return ImageBitmapLoader.applyExifOrientation(decodedBitmap, imageFile)
    }

    private fun cameraCropTargetSize(width: Int, height: Int): Pair<Int, Int> {
        val largestDimension = max(width, height)
        if (largestDimension <= CAMERA_CROP_MAX_IMAGE_DIMENSION) {
            return width.coerceAtLeast(1) to height.coerceAtLeast(1)
        }

        val scale = CAMERA_CROP_MAX_IMAGE_DIMENSION.toFloat() / largestDimension.toFloat()
        return (width * scale).toInt().coerceAtLeast(1) to
            (height * scale).toInt().coerceAtLeast(1)
    }

    private fun calculateCameraCropSampleSize(width: Int, height: Int): Int {
        val largestDimension = max(width, height)
        if (largestDimension <= CAMERA_CROP_MAX_IMAGE_DIMENSION) {
            return 1
        }

        val ratio = ceil(largestDimension.toDouble() / CAMERA_CROP_MAX_IMAGE_DIMENSION.toDouble()).toInt()
        var sampleSize = 1
        while (sampleSize * 2 <= ratio) {
            sampleSize *= 2
        }
        return sampleSize.coerceAtLeast(1)
    }

    private fun createCameraOcrOutputFile(): File {
        val directory = File(cacheDir, "ocr_images").apply { mkdirs() }
        return File.createTempFile("camera_ocr_", ".jpg", directory)
    }

    private fun cameraReadyStatusText(): String {
        return if (currentImageInputMode == ImageInputMode.OCR) {
            getString(R.string.camera_ocr_status_ready)
        } else {
            getString(R.string.camera_image_status_ready)
        }
    }

    private fun currentImageContextSource(): ImageInputContextSource {
        return when (currentImageInputMode) {
            ImageInputMode.OCR -> ImageInputContextSource.OCR
            ImageInputMode.GEMMA_DIRECT -> {
                if (isGemmaDirectImageInputAvailable()) {
                    ImageInputContextSource.GEMMA_DIRECT
                } else {
                    ImageInputContextSource.OCR
                }
            }
        }
    }

    private fun isGemmaDirectImageInputAvailable(): Boolean {
        val descriptor = currentModel ?: return false
        val state = chatController?.state?.value ?: return descriptor.directImageInputAvailable
        return when {
            state.isReady -> state.supportsDirectImageInput
            state.isLoading -> descriptor.directImageInputAvailable
            else -> false
        }
    }

    private fun isGemmaDirectImageInputSupportedBySelectedModel(): Boolean {
        return currentModel?.directImageInputAvailable == true ||
            chatController?.state?.value?.let { it.isReady && it.supportsDirectImageInput } == true
    }

    private fun usesNativeAudioForSelectedModel(): Boolean =
        currentModel?.directAudioInputAvailable == true ||
            chatController?.state?.value?.let { it.isReady && it.supportsNativeAudioInput } == true

    private fun ensureImageInputModeAllowedForCurrentModel() {
        if (currentImageInputMode == ImageInputMode.GEMMA_DIRECT && !isGemmaDirectImageInputAvailable()) {
            selectOcrImageInputMode(resetGemmaDirectInputs = true)
        }
    }

    private fun ensureImageInputModeAllowedForModel(descriptor: ModelDescriptor) {
        if (!descriptor.directImageInputAvailable) {
            selectOcrImageInputMode(resetGemmaDirectInputs = true)
        }
    }

    private fun selectOcrImageInputMode(resetGemmaDirectInputs: Boolean) {
        currentImageInputMode = ImageInputMode.OCR
        modelSelectionStore.saveSelectedImageInputMode(ImageInputMode.OCR)
        if (resetGemmaDirectInputs) {
            resetGemmaDirectPendingInputsToOcr()
        }
        updateImageInputButtonDescriptions()
    }

    private fun resetGemmaDirectPendingInputsToOcr() {
        var changed = false
        for (index in pendingImageInputs.indices) {
            val input = pendingImageInputs[index]
            if (input.contextSource != ImageInputContextSource.GEMMA_DIRECT) {
                continue
            }

            deletePendingImageTempFile(input)
            pendingImageInputs[index] = input.copy(
                contextSource = ImageInputContextSource.OCR,
                status = PendingImageStatus.PENDING,
                recognizedText = null,
                errorMessage = null,
                tempFilePath = null,
                directImageFilePath = null
            )
            changed = true
        }

        if (changed) {
            renderPendingImageInputs()
        }
    }

    private fun stopCameraOcr() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        cameraImageCapture = null
    }

    private fun importAndAddPendingImage(
        uri: Uri,
        source: OcrInput.Source,
        tempFilePath: String? = null
    ) {
        lifecycleScope.launch {
            val savedImage = runCatching {
                withContext(Dispatchers.IO) {
                    savedImageStore.importImage(uri, source, tempFilePath)
                }
            }.getOrElse { error ->
                tempFilePath?.let { path -> runCatching { File(path).delete() } }
                showTransientMessage(error.message ?: getString(R.string.image_input_file_error))
                return@launch
            }

            tempFilePath?.let { path ->
                if (path != savedImage.filePath) {
                    runCatching { File(path).delete() }
                }
            }
            addPendingImageInput(savedImage)
            refreshDrawerSessions()
        }
    }

    private fun addPendingImageInput(savedImage: SavedImageEntry) {
        if (activeAttachment != null || attachmentImportInProgress) {
            showTransientMessage(getString(R.string.attachment_mixed_input_blocked))
            return
        }
        if (isImagePreprocessingForSend) {
            showTransientMessage(getString(R.string.image_input_processing))
            return
        }

        ensureImageInputModeAllowedForCurrentModel()
        val imageContextSource = currentImageContextSource()
        if (imageContextSource == ImageInputContextSource.GEMMA_DIRECT && !isGemmaDirectImageInputAvailable()) {
            showTransientMessage(getString(R.string.gemma_direct_image_model_unavailable))
            return
        }

        val requestId = nextPendingImageInputId++
        pendingImageInputs += PendingImageInput(
            id = requestId,
            savedImageId = savedImage.imageId,
            savedImagePath = savedImage.filePath,
            uri = Uri.fromFile(File(savedImage.filePath)),
            source = savedImage.source,
            contextSource = imageContextSource,
            tempFilePath = null
        )
        renderPendingImageInputs()
    }

    private fun handleSendClick() {
        if (ImageGenFeature.instance?.isImageModeActive() == true) {
            ImageGenFeature.instance?.handleCreateImageSend(chatHost)
            return
        }
        val controller = chatController ?: return
        if (isImagePreprocessingForSend || attachmentImportInProgress || audioPreparationJob?.isActive == true) {
            return
        }

        if (pendingImageInputs.any { it.status == PendingImageStatus.FAILED }) {
            showTransientMessage(getString(R.string.image_input_remove_failed))
            return
        }

        if (imagePreprocessingJob?.isActive == true) {
            return
        }

        if (WebSearchFeature.instance?.isUsable(this) == true && controller.supportsNativeToolCalling() &&
            inputEditText.text.isNotBlank()
        ) {
            if (ProUiFeature.isAvailable) {
                // The try is used only when this answer finishes; see [settleWebSearchTry].
                if (!ProUiFeature.instance!!.hasTrial(this, proGate, ProFeature.WEB_SEARCH)) {
                    WebSearchFeature.instance?.setUsable(this, false)
                    refreshProBadges()
                    return
                }
            } else {
                if (!proGate.isUnlocked()) {
                    proGate.requireUnlocked(this, ProFeature.WEB_SEARCH) {}
                    WebSearchFeature.instance?.setUsable(this, false)
                    refreshProBadges()
                    return
                }
            }
            webSearchTryFrom = controller.state.value.finishedAnswers
            webSearchTryStarted = false
            refreshProBadges()
        }

        val draft = captureComposerDraft()
        val imagesForSend = pendingImageInputs.toList()
        val shouldPreprocessAfterSend = imagesForSend.any { it.status == PendingImageStatus.PENDING }
        if (!shouldPreprocessAfterSend) {
            val prompt = buildComposerPrompt(draft, imagesForSend) ?: return
            if (prompt.modelText.isBlank()) {
                return
            }

            clearSpeechInputState()
            val started = controller.sendPrompt(
                prompt.modelText,
                prompt.displayText,
                prompt.imageFilePaths,
                nativeAudioInputs = prompt.nativeAudioInputs,
                attachmentContext = prompt.attachmentContext,
                leadingTurns = prompt.imageTurns + prompt.attachmentTurns
            )
            if (started) {
                confirmedAttachmentOperationKey = null
                inputEditText.text.clear()
                clearPendingImageInputs(keepTempFilePaths = prompt.imageFilePaths.toSet())
                clearSentAttachmentFromComposer()
                refreshDrawerSessions()
            } else {
                prompt.attachmentContext?.temporaryFiles?.forEach { File(it).delete() }
            }
            return
        }

        val displayText = buildComposerDisplayText(draft.typedText)
        if (displayText.isBlank() && imagesForSend.isEmpty()) {
            return
        }

        isImagePreprocessingForSend = true
        autoScrollDuringGeneration = true
        if (
            !controller.beginPromptPreparation(
                displayText = displayText,
                statusText = getString(R.string.image_input_processing_status),
                leadingTurns = buildImageChatTurns(imagesForSend)
            )
        ) {
            isImagePreprocessingForSend = false
            autoScrollDuringGeneration = false
            applyChatState(controller.state.value)
            return
        }
        clearSpeechInputState()
        inputEditText.text.clear()
        detachPendingImageInputs()
        applyChatState(controller.state.value)

        imagePreprocessingJob = lifecycleScope.launch {
            var promptStarted = false
            var processedImagesForSend = imagesForSend
            try {
                val processedImages = preprocessImagesForSend(imagesForSend)
                if (processedImages == null) {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    return@launch
                }
                processedImagesForSend = processedImages
                isImagePreprocessingForSend = false
                chatController?.let { applyChatState(it.state.value) }

                val prompt = buildComposerPrompt(draft, processedImagesForSend) ?: run {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    restorePendingImageInputs(processedImagesForSend)
                    return@launch
                }
                if (prompt.modelText.isBlank()) {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    restorePendingImageInputs(processedImagesForSend)
                    return@launch
                }

                promptStarted = controller.sendPreparedPrompt(
                    prompt.modelText,
                    prompt.displayText,
                    prompt.imageFilePaths,
                    nativeAudioInputs = prompt.nativeAudioInputs,
                    attachmentContext = prompt.attachmentContext
                )
                if (!promptStarted) {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    restorePendingImageInputs(processedImagesForSend)
                    return@launch
                }
                deletePendingImageTempFiles(
                    imagesForSend + processedImagesForSend,
                    keepTempFilePaths = prompt.imageFilePaths.toSet()
                )
                refreshDrawerSessions()
            } catch (_: CancellationException) {
                if (!promptStarted) {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    restorePendingImageInputs(processedImagesForSend)
                }
                // Activity shutdown or a future explicit cancellation stops preprocessing.
            } catch (error: Exception) {
                if (!promptStarted) {
                    controller.cancelPromptPreparation()
                    restoreComposerDraft(draft)
                    restorePendingImageInputs(processedImagesForSend)
                }
                showTransientMessage(error.message ?: getString(R.string.image_input_processing_failed))
            } finally {
                imagePreprocessingJob = null
                isImagePreprocessingForSend = false
                chatController?.let { applyChatState(it.state.value) }
                if (!promptStarted) {
                    autoScrollDuringGeneration = false
                }
            }
        }
    }

    private suspend fun preprocessImagesForSend(imagesForSend: List<PendingImageInput>): List<PendingImageInput>? {
        val processedImages = imagesForSend.toMutableList()
        if (processedImages.none { it.status == PendingImageStatus.PENDING }) {
            return processedImages
        }

        isImagePreprocessingForSend = true
        chatController?.let { applyChatState(it.state.value) }

        for (index in processedImages.indices) {
            val input = processedImages[index]
            if (input.status != PendingImageStatus.PENDING) {
                continue
            }

            val readingInput = input.copy(
                status = PendingImageStatus.READING,
                recognizedText = null,
                errorMessage = null
            )
            processedImages[index] = readingInput

            val result = when (readingInput.contextSource) {
                ImageInputContextSource.OCR -> runCatching {
                    preprocessImageWithOcr(readingInput)
                }.getOrElse { error ->
                    markImagePreprocessingFailed(processedImages, index, readingInput, error)
                    return null
                }
                ImageInputContextSource.GEMMA_DIRECT -> runCatching {
                    preprocessImageForGemmaDirect(readingInput)
                }.getOrElse { error ->
                    markImagePreprocessingFailed(processedImages, index, readingInput, error)
                    return null
                }
            }

            val isDirectGemmaInput = readingInput.contextSource == ImageInputContextSource.GEMMA_DIRECT
            if (!isDirectGemmaInput && result.text.isBlank()) {
                val message = getString(R.string.ocr_no_text)
                processedImages[index] = readingInput.copy(
                    status = PendingImageStatus.FAILED,
                    recognizedText = null,
                    errorMessage = message,
                    tempFilePath = result.tempFilePath
                )
                restorePendingImageInputs(processedImages)
                showTransientMessage(message)
                return null
            }
            if (isDirectGemmaInput && result.directImageFilePath.isNullOrBlank()) {
                val message = getString(R.string.gemma_direct_image_processing_failed)
                processedImages[index] = readingInput.copy(
                    status = PendingImageStatus.FAILED,
                    recognizedText = null,
                    errorMessage = message,
                    tempFilePath = result.tempFilePath,
                    directImageFilePath = null
                )
                restorePendingImageInputs(processedImages)
                showTransientMessage(message)
                return null
            }

            processedImages[index] = readingInput.copy(
                status = PendingImageStatus.READY,
                recognizedText = result.text,
                errorMessage = null,
                tempFilePath = result.tempFilePath,
                directImageFilePath = result.directImageFilePath
            )
        }

        return processedImages
    }

    private fun markImagePreprocessingFailed(
        processedImages: MutableList<PendingImageInput>,
        index: Int,
        input: PendingImageInput,
        error: Throwable
    ) {
        val message = imagePreprocessingErrorMessage(input, error)
        processedImages[index] = input.copy(
            status = PendingImageStatus.FAILED,
            recognizedText = null,
            errorMessage = message
        )
        restorePendingImageInputs(processedImages)
        showTransientMessage(message)
    }

    private fun imagePreprocessingErrorMessage(input: PendingImageInput, error: Throwable): String {
        return when {
            input.contextSource == ImageInputContextSource.GEMMA_DIRECT -> error.message
                ?: getString(R.string.gemma_direct_image_processing_failed)
            input.contextSource == ImageInputContextSource.OCR -> error.message
                ?: getString(R.string.ocr_no_text)
            else -> error.message ?: getString(R.string.image_input_processing_failed)
        }
    }

    private suspend fun preprocessImageWithOcr(input: PendingImageInput): ImagePreprocessingResult {
        val rawText = ocrInput?.recognizeImageUriText(input.uri)
            ?: throw IllegalStateException("OCR is not available.")
        return ImagePreprocessingResult(
            text = PromptPreprocessor.normalize(rawText),
            tempFilePath = input.tempFilePath
        )
    }

    private suspend fun preprocessImageForGemmaDirect(input: PendingImageInput): ImagePreprocessingResult {
        if (!isGemmaDirectImageInputAvailable()) {
            throw IllegalStateException(getString(R.string.gemma_direct_image_model_unavailable))
        }

        val imageFile = runCatching {
            prepareDirectImageFile(input.uri, input.tempFilePath)
        }.getOrElse { error ->
            throw IOException(getString(R.string.gemma_direct_image_processing_failed), error)
        }
        return ImagePreprocessingResult(
            text = "",
            tempFilePath = imageFile.absolutePath,
            directImageFilePath = imageFile.absolutePath
        )
    }

    private fun removePendingImageInput(requestId: Long) {
        if (isImagePreprocessingForSend) {
            showTransientMessage(getString(R.string.image_input_processing))
            return
        }

        val index = pendingImageInputs.indexOfFirst { it.id == requestId }
        if (index == -1) {
            return
        }

        deletePendingImageTempFile(pendingImageInputs.removeAt(index))
        renderPendingImageInputs()
    }

    private fun clearPendingImageInputs(keepTempFilePaths: Set<String> = emptySet()) {
        if (isImagePreprocessingForSend) {
            return
        }

        if (pendingImageInputs.isEmpty()) {
            renderPendingImageInputs()
            return
        }

        pendingImageInputs.forEach { input ->
            deletePendingImageTempFile(input, keepTempFilePaths)
        }
        pendingImageInputs.clear()
        renderPendingImageInputs()
    }

    private fun detachPendingImageInputs() {
        pendingImageInputs.clear()
        renderPendingImageInputs()
    }

    private fun restorePendingImageInputs(inputs: List<PendingImageInput>) {
        pendingImageInputs.clear()
        pendingImageInputs.addAll(inputs)
        renderPendingImageInputs()
    }

    private fun deletePendingImageTempFiles(
        inputs: List<PendingImageInput>,
        keepTempFilePaths: Set<String> = emptySet()
    ) {
        inputs.forEach { input ->
            deletePendingImageTempFile(input, keepTempFilePaths)
        }
    }

    private fun deletePendingImageTempFile(
        input: PendingImageInput,
        keepTempFilePaths: Set<String> = emptySet()
    ) {
        input.tempFilePath?.let { path ->
            if (path in keepTempFilePaths) {
                return@let
            }
            runCatching { File(path).delete() }
        }
    }

    private fun prepareDirectImageFile(uri: Uri, tempFilePath: String?): File {
        val existingTempFile = tempFilePath
            ?.let(::File)
            ?.takeIf { it.exists() && it.length() > 0L }
        if (existingTempFile != null && isPreparedDirectImageFile(existingTempFile)) {
            return existingTempFile
        }

        return normalizeImageForDirectInput(uri, existingTempFile)
    }

    private fun isPreparedDirectImageFile(file: File): Boolean {
        return file.parentFile?.name == "direct_image_inputs" &&
            file.extension.equals("jpg", ignoreCase = true)
    }

    private fun normalizeImageForDirectInput(uri: Uri, sourceFile: File?): File {
        val decodedBitmap = decodeBitmapForDirectInput(uri, sourceFile)
        val decodedWidth = decodedBitmap.width
        val decodedHeight = decodedBitmap.height
        val normalizedBitmap = normalizeBitmapForDirectInputJpeg(decodedBitmap)
        if (normalizedBitmap !== decodedBitmap) {
            decodedBitmap.recycle()
        }

        val directory = File(cacheDir, "direct_image_inputs").apply { mkdirs() }
        val outputFile = File.createTempFile("direct_image_input_", ".jpg", directory)
        try {
            Log.d(
                TAG,
                "Prepared direct image input: ${decodedWidth}x${decodedHeight} -> " +
                    "${normalizedBitmap.width}x${normalizedBitmap.height}, source=${imageInputSourceLabel(sourceFile)}"
            )
            FileOutputStream(outputFile).use { output ->
                val compressed = normalizedBitmap.compress(
                    Bitmap.CompressFormat.JPEG,
                    IMAGE_INPUT_JPEG_QUALITY,
                    output
                )
                if (!compressed) {
                    throw IOException(getString(R.string.image_input_file_error))
                }
            }

            if (outputFile.length() <= 0L) {
                throw IOException(getString(R.string.image_input_file_error))
            }

            return outputFile
        } catch (error: Exception) {
            runCatching { outputFile.delete() }
            throw error
        } finally {
            normalizedBitmap.recycle()
        }
    }

    private fun decodeBitmapForDirectInput(uri: Uri, sourceFile: File?): Bitmap {
        val orientationSourceFile = resolveImageSourceFile(uri, sourceFile)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            decodeBitmapWithImageDecoder(uri, sourceFile)
                .onSuccess { bitmap ->
                    return ImageBitmapLoader.applyExifOrientation(bitmap, orientationSourceFile)
                }
                .onFailure { error ->
                    Log.w(TAG, "ImageDecoder could not decode direct image input; falling back to BitmapFactory.", error)
                }
        }

        return ImageBitmapLoader.applyExifOrientation(
            decodeBitmapWithBitmapFactory(uri, sourceFile),
            orientationSourceFile
        )
    }

    private fun decodeBitmapWithImageDecoder(uri: Uri, sourceFile: File?): Result<Bitmap> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.P) {
            return Result.failure(UnsupportedOperationException("ImageDecoder requires Android 9 or newer."))
        }

        return runCatching {
            val source = if (sourceFile != null && sourceFile.exists()) {
                ImageDecoder.createSource(sourceFile)
            } else {
                ImageDecoder.createSource(contentResolver, uri)
            }

            ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                val width = info.size.width
                val height = info.size.height
                if (width > 0 && height > 0) {
                    val (targetWidth, targetHeight) = directImageTargetSize(width, height)
                    decoder.setTargetSize(targetWidth, targetHeight)
                }
            }
        }
    }

    private fun decodeBitmapWithBitmapFactory(uri: Uri, sourceFile: File?): Bitmap {
        val bounds = BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        openDirectImageInput(uri, sourceFile).use { input ->
            BitmapFactory.decodeStream(input, null, bounds)
        }

        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IOException(getString(R.string.image_input_file_error))
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = calculateDirectImageSampleSize(bounds.outWidth, bounds.outHeight)
        }
        return openDirectImageInput(uri, sourceFile).use { input ->
            BitmapFactory.decodeStream(input, null, decodeOptions)
        } ?: throw IOException(getString(R.string.image_input_file_error))
    }

    private fun openDirectImageInput(uri: Uri, sourceFile: File?): java.io.InputStream {
        if (sourceFile != null && sourceFile.exists()) {
            return sourceFile.inputStream()
        }

        return contentResolver.openInputStream(uri)
            ?: throw IOException(getString(R.string.image_input_file_error))
    }

    private fun resolveImageSourceFile(uri: Uri, sourceFile: File?): File? {
        if (sourceFile != null && sourceFile.exists()) {
            return sourceFile
        }

        if (uri.scheme.equals("file", ignoreCase = true)) {
            return uri.path?.let(::File)?.takeIf { it.exists() }
        }

        return null
    }

    private fun calculateDirectImageSampleSize(width: Int, height: Int): Int {
        val largestDimension = max(width, height)
        if (largestDimension <= IMAGE_INPUT_MAX_DIMENSION) {
            return 1
        }

        val ratio = ceil(largestDimension.toDouble() / IMAGE_INPUT_MAX_DIMENSION.toDouble()).toInt()
        var sampleSize = 1
        while (sampleSize * 2 <= ratio) {
            sampleSize *= 2
        }
        return sampleSize.coerceAtLeast(1)
    }

    private fun directImageTargetSize(width: Int, height: Int): Pair<Int, Int> {
        val largestDimension = max(width, height)
        if (largestDimension <= IMAGE_INPUT_MAX_DIMENSION) {
            return width.coerceAtLeast(1) to height.coerceAtLeast(1)
        }

        val scale = IMAGE_INPUT_MAX_DIMENSION.toFloat() / largestDimension.toFloat()
        return (width * scale).toInt().coerceAtLeast(1) to
            (height * scale).toInt().coerceAtLeast(1)
    }

    private fun scaleBitmapForDirectInput(bitmap: Bitmap): Bitmap {
        val largestDimension = max(bitmap.width, bitmap.height)
        if (largestDimension <= IMAGE_INPUT_MAX_DIMENSION) {
            return bitmap
        }

        val scale = IMAGE_INPUT_MAX_DIMENSION.toFloat() / largestDimension.toFloat()
        val targetWidth = (bitmap.width * scale).toInt().coerceAtLeast(1)
        val targetHeight = (bitmap.height * scale).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
    }

    private fun normalizeBitmapForDirectInputJpeg(bitmap: Bitmap): Bitmap {
        val scaledBitmap = scaleBitmapForDirectInput(bitmap)
        val outputBitmap = if (shouldPadBitmapForDirectInput(scaledBitmap)) {
            val canvasSize = max(scaledBitmap.width, scaledBitmap.height)
                .coerceAtLeast(IMAGE_INPUT_MIN_NORMALIZED_DIMENSION)
                .coerceAtMost(IMAGE_INPUT_MAX_DIMENSION)
            Bitmap.createBitmap(canvasSize, canvasSize, Bitmap.Config.ARGB_8888).also { canvasBitmap ->
                val canvas = Canvas(canvasBitmap)
                canvas.drawColor(Color.WHITE)
                canvas.drawBitmap(
                    scaledBitmap,
                    ((canvasSize - scaledBitmap.width) / 2f).coerceAtLeast(0f),
                    ((canvasSize - scaledBitmap.height) / 2f).coerceAtLeast(0f),
                    null
                )
            }
        } else if (scaledBitmap.hasAlpha()) {
            Bitmap.createBitmap(scaledBitmap.width, scaledBitmap.height, Bitmap.Config.ARGB_8888).also { canvasBitmap ->
                val canvas = Canvas(canvasBitmap)
                canvas.drawColor(Color.WHITE)
                canvas.drawBitmap(scaledBitmap, 0f, 0f, null)
            }
        } else {
            scaledBitmap
        }

        if (outputBitmap !== scaledBitmap && scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }
        return outputBitmap
    }

    private fun shouldPadBitmapForDirectInput(bitmap: Bitmap): Boolean {
        val smallerDimension = minOf(bitmap.width, bitmap.height).coerceAtLeast(1)
        val largerDimension = max(bitmap.width, bitmap.height).coerceAtLeast(1)
        return smallerDimension < IMAGE_INPUT_MIN_NORMALIZED_DIMENSION &&
            largerDimension.toFloat() / smallerDimension.toFloat() > IMAGE_INPUT_EXTREME_ASPECT_RATIO
    }

    private fun imageInputSourceLabel(sourceFile: File?): String {
        return sourceFile?.let { file ->
            "${file.name} (${file.length()} bytes)"
        } ?: "content uri"
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
                val previewBitmap = ImageBitmapLoader.decodeSampledBitmap(
                    imageFile = File(input.savedImagePath),
                    requestedWidth = dp(34),
                    requestedHeight = dp(34)
                )
                if (previewBitmap != null) {
                    setImageBitmap(previewBitmap)
                } else {
                    setImageResource(R.drawable.ic_image_24)
                }
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

        if (input.status == PendingImageStatus.FAILED && !input.errorMessage.isNullOrBlank()) {
            chip.isClickable = true
            chip.isFocusable = true
            chip.setOnClickListener { showImageInputError(input) }
        }

        return chip
    }

    private fun showImageInputError(input: PendingImageInput) {
        val message = input.errorMessage ?: return
        MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.image_input_error_title))
            .setMessage(message)
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun pendingImageStatusText(input: PendingImageInput): String {
        return when (input.status) {
            PendingImageStatus.PENDING -> getString(R.string.image_input_pending)
            PendingImageStatus.READING -> when (input.contextSource) {
                ImageInputContextSource.GEMMA_DIRECT -> getString(R.string.image_input_preparing)
                ImageInputContextSource.OCR -> getString(R.string.image_input_reading)
            }
            PendingImageStatus.READY -> getString(R.string.image_input_ready)
            PendingImageStatus.FAILED -> when (input.contextSource) {
                ImageInputContextSource.OCR -> getString(R.string.image_input_failed)
                ImageInputContextSource.GEMMA_DIRECT -> getString(R.string.image_input_failed_generic)
            }
        }
    }

    private fun pendingImageStatusColor(input: PendingImageInput): Int {
        return when (input.status) {
            PendingImageStatus.FAILED -> ContextCompat.getColor(this, R.color.delete_red)
            else -> resolveThemeColor(R.attr.colorStatusText)
        }
    }

    private fun captureComposerDraft(): ComposerDraft {
        val typedText = PromptPreprocessor.normalize(inputEditText.text.toString())
        val voiceText = currentVoiceTranscriptForPrompt(typedText)
        val typedTextWithoutVoice = if (voiceText != null) {
            PromptPreprocessor.normalize(speechBasePromptText)
        } else {
            typedText
        }

        return ComposerDraft(
            typedText = typedText,
            typedTextWithoutVoice = typedTextWithoutVoice,
            voiceText = voiceText
        )
    }

    private fun restoreComposerDraft(draft: ComposerDraft) {
        if (draft.typedText.isNotBlank()) {
            setPromptInputText(draft.typedText)
        }
    }

    private fun buildComposerPrompt(
        draft: ComposerDraft = captureComposerDraft(),
        images: List<PendingImageInput> = pendingImageInputs
    ): ComposerPrompt? {
        if (attachmentImportInProgress) {
            showTransientMessage(getString(R.string.attachment_importing))
            return null
        }
        val descriptor = activeAttachment
        if (descriptor != null && images.isNotEmpty()) {
            showTransientMessage(getString(R.string.attachment_mixed_input_blocked))
            return null
        }
        if (descriptor?.status == AttachmentStatus.FAILED) {
            showTransientMessage(descriptor.errorMessage ?: "Attachment preparation failed.")
            return null
        }
        if (images.any { it.status == PendingImageStatus.PENDING || it.status == PendingImageStatus.READING }) {
            showTransientMessage(getString(R.string.image_input_processing))
            return null
        }
        if (images.any { it.status == PendingImageStatus.FAILED }) {
            showTransientMessage(getString(R.string.image_input_remove_failed))
            return null
        }

        val readyTextImages = images.filter {
            it.status == PendingImageStatus.READY &&
                it.contextSource != ImageInputContextSource.GEMMA_DIRECT &&
                !it.recognizedText.isNullOrBlank()
        }
        val directGemmaImagePaths = images
            .filter {
                it.status == PendingImageStatus.READY &&
                    it.contextSource == ImageInputContextSource.GEMMA_DIRECT &&
                    !it.directImageFilePath.isNullOrBlank()
            }
            .mapNotNull { it.directImageFilePath }
        val modelText = buildSourceAwareModelText(
            typedText = draft.typedTextWithoutVoice,
            voiceText = draft.voiceText,
            images = readyTextImages,
            hasDirectGemmaImages = directGemmaImagePaths.isNotEmpty()
        ).ifBlank {
            if (directGemmaImagePaths.isNotEmpty()) {
                getString(R.string.gemma_direct_default_prompt)
            } else {
                ""
            }
        }
        if (descriptor != null && modelText.isBlank()) {
            showTransientMessage(getString(R.string.attachment_question_required))
            return null
        }
        if (modelText.isBlank()) {
            return null
        }

        val attachmentContext = descriptor?.let { buildActiveAttachmentContext(it, modelText) }
            ?: if (descriptor != null) return null else null

        return ComposerPrompt(
            modelText = modelText,
            displayText = buildComposerDisplayText(draft.typedText),
            imageFilePaths = directGemmaImagePaths,
            nativeAudioInputs = attachmentContext?.nativeAudioInputs.orEmpty(),
            attachmentContext = attachmentContext,
            imageTurns = buildImageChatTurns(images),
            attachmentTurns = descriptor?.let(::buildAttachmentChatTurn)?.let(::listOf).orEmpty()
        )
    }

    private fun buildActiveAttachmentContext(
        descriptor: AttachmentDescriptor,
        userPrompt: String
    ): AttachmentContext? {
        val controller = chatController ?: return null
        return when (descriptor.kind) {
            AttachmentKind.TEXT, AttachmentKind.PDF -> {
                val chunks = attachmentRepository.loadChunks(descriptor)
                if (chunks.isEmpty()) {
                    showTransientMessage("No extracted attachment text is available.")
                    null
                } else {
                    buildPlannedAttachmentContext(controller, descriptor, chunks, userPrompt)
                }
            }

            AttachmentKind.AUDIO -> {
                if (usesNativeAudioForSelectedModel()) {
                    if (!controller.state.value.supportsNativeAudioInput) {
                        showTransientMessage(getString(R.string.gemma_audio_unavailable))
                        return null
                    }
                    val segmenter = GemmaAudioSegmenter(this, currentModel!!.nativeAudioSegmentMillis)
                    val plan = runCatching { segmenter.plan(descriptor) }.getOrElse { error ->
                        showTransientMessage(error.message ?: "Audio duration is unavailable.")
                        return null
                    }
                    val confirmationKey = "${descriptor.id}:${userPrompt.hashCode()}:${currentModel?.id}:native-audio"
                    if (
                        plan.segmentCount > 1 &&
                        confirmedAttachmentOperationKey != confirmationKey
                    ) {
                        MaterialAlertDialogBuilder(this)
                            .setTitle("Process the full audio?")
                            .setMessage(
                                "The selected model will analyze ${plan.segmentCount} overlapping native-audio segments " +
                                    "in ${plan.inferencePasses} inference passes and then combine them. " +
                                    "It may take several minutes."
                            )
                            .setPositiveButton("Continue") { _, _ ->
                                confirmedAttachmentOperationKey = confirmationKey
                                handleSendClick()
                            }
                            .setNegativeButton(android.R.string.cancel, null)
                            .show()
                        return null
                    }
                    val segmented = if (plan.segmentCount == 1) {
                        SegmentedNativeAudio(
                            inputs = listOf(NativeAudioInput(descriptor.sourcePath, 0L, descriptor.durationMillis)),
                            temporaryFiles = emptyList()
                        )
                    } else {
                        preparedGemmaAudio?.takeIf { preparedGemmaAudioAttachmentId == descriptor.id }
                            ?: run {
                                prepareGemmaAudioAfterConfirmation(descriptor)
                                return null
                            }
                    }
                    preparedGemmaAudio = null
                    preparedGemmaAudioAttachmentId = null
                    AttachmentContext(
                        attachmentId = descriptor.id,
                        displayName = descriptor.displayName,
                        route = AttachmentProcessingRoute.GEMMA_NATIVE_AUDIO,
                        sourceRefs = segmented.inputs.map { input ->
                            AttachmentSourceRef(startMillis = input.startMillis, endMillis = input.endMillis)
                        },
                        nativeAudioInputs = segmented.inputs,
                        task = AttachmentTaskRouter.route(userPrompt),
                        temporaryFiles = segmented.temporaryFiles
                    )
                } else {
                    val transcriptChunks = attachmentRepository.loadChunks(descriptor)
                    if (transcriptChunks.isEmpty()) {
                        if (whisperModelStore.isAvailable()) {
                            if (currentModel is GemmaLiteRtSpec) {
                                showTransientMessage(getString(R.string.gemma_audio_transcribing))
                            }
                            prepareWhisperTranscript(descriptor)
                        } else {
                            offerWhisperPreparation(descriptor)
                            showTransientMessage(
                                getString(
                                    if (currentModel is GemmaLiteRtSpec) {
                                        R.string.gemma_audio_transcription_download_required
                                    } else {
                                        R.string.non_gemma_audio_download_required
                                    }
                                )
                            )
                        }
                        null
                    } else {
                        buildPlannedAttachmentContext(controller, descriptor, transcriptChunks, userPrompt)
                    }
                }
            }
        }
    }

    private fun buildPlannedAttachmentContext(
        controller: PersistentChatController,
        descriptor: AttachmentDescriptor,
        chunks: List<AttachmentChunk>,
        userPrompt: String
    ): AttachmentContext? {
        val context = runCatching { controller.buildAttachmentContext(descriptor, chunks, userPrompt) }
            .getOrElse { error ->
                showTransientMessage(error.message ?: "The attachment exceeds this model's prompt limits.")
                return null
            }
        if (!context.requiresFinalSynthesis) return context
        val confirmationKey = "${descriptor.id}:${userPrompt.hashCode()}:${context.task}"
        if (confirmedAttachmentOperationKey == confirmationKey) return context
        MaterialAlertDialogBuilder(this)
            .setTitle("Process the full attachment?")
            .setMessage(
                "This ${context.task.name.lowercase()} needs ${context.promptBatches.size} on-device passes " +
                    "plus final combination. It may take several minutes."
            )
            .setPositiveButton("Continue") { _, _ ->
                confirmedAttachmentOperationKey = confirmationKey
                handleSendClick()
            }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
        return null
    }

    private fun buildAttachmentChatTurn(descriptor: AttachmentDescriptor): ChatTurn {
        return ChatTurn(
            role = ChatRole.USER,
            text = "",
            contentType = ChatTurnContentType.ATTACHMENT,
            attachmentId = descriptor.id,
            attachmentName = attachmentDisplayName(descriptor),
            attachmentKind = descriptor.kind,
            attachmentProcessingRoute = descriptor.processingRoute
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
        images: List<PendingImageInput>,
        hasDirectGemmaImages: Boolean
    ): String {
        val hasTypedText = typedText.isNotBlank()
        val hasVoiceText = !voiceText.isNullOrBlank()
        val hasImageText = images.any { !it.recognizedText.isNullOrBlank() }

        if (!hasVoiceText && !hasImageText && !hasDirectGemmaImages) {
            return typedText
        }
        if (!hasTypedText && !hasVoiceText && !hasImageText && hasDirectGemmaImages) {
            return ""
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

            if (hasDirectGemmaImages) {
                append("\n\n[Attached image input]\n")
                append("The image file is attached directly to this message. Use the visual content itself when answering.")
            }
        }.trim()
    }

    private fun buildImageContextText(images: List<PendingImageInput>): String {
        if (images.isEmpty()) {
            return ""
        }

        return buildString {
            append("[Image context]\n")
            append("The following information was extracted from attached image input with OCR. Use it as context from the image; it was not typed directly by the user.")
            images.forEachIndexed { index, input ->
                val sourceLabel = if (input.source == OcrInput.Source.CAMERA) {
                    "camera photo"
                } else {
                    "gallery image"
                }
                val contextLabel = when (input.contextSource) {
                    ImageInputContextSource.OCR -> "OCR text"
                    ImageInputContextSource.GEMMA_DIRECT -> "Native image"
                }
                append("\n\n[Image ")
                append(index + 1)
                append(" - ")
                append(sourceLabel)
                append(" - ")
                append(contextLabel)
                append("]\n")
                append(input.recognizedText.orEmpty().trim())
            }
        }
    }

    private fun buildComposerDisplayText(typedText: String): String {
        return typedText
    }

    private fun buildImageChatTurns(images: List<PendingImageInput>): List<ChatTurn> {
        return images.map { input ->
            ChatTurn(
                role = ChatRole.USER,
                text = "",
                contentType = ChatTurnContentType.IMAGE,
                imagePath = input.savedImagePath
            )
        }
    }

    private fun setPromptInputText(text: String) {
        inputEditText.setText(text)
        inputEditText.setSelection(inputEditText.text.length)
    }

    /** The "+" in the message bar: Camera, Photo, Document, Audio file, then the four image tools. */
    private fun showPlusMenu(anchor: View) {
        val content = layoutInflater.inflate(R.layout.popup_plus_menu, null) as LinearLayout
        val popup = PopupWindow(content, ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, true).apply {
            setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
            isOutsideTouchable = true
            elevation = dp(12).toFloat()
        }
        val locked = !proGate.isUnlocked()
        fun heading(@StringRes title: Int) {
            content.addView(TextView(this).apply {
                setText(title)
                setTextColor(resolveThemeColor(R.attr.colorStatusText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                setPadding(dp(14), dp(10), dp(14), dp(2))
            })
        }
        fun row(@DrawableRes icon: Int, @StringRes label: Int, pro: Boolean, action: () -> Unit) {
            val view = layoutInflater.inflate(R.layout.item_plus_menu_row, content, false)
            view.findViewById<ImageView>(R.id.plusRowIcon).apply {
                setImageResource(icon)
                imageTintList = ColorStateList.valueOf(resolveThemeColor(if (pro) R.attr.colorSendFill else R.attr.colorStatusText))
            }
            view.findViewById<TextView>(R.id.plusRowLabel).setText(label)
            view.findViewById<View>(R.id.plusRowPro).visibility = if (pro && locked) View.VISIBLE else View.GONE
            view.setOnClickListener {
                popup.dismiss()
                action()
            }
            content.addView(view)
        }
        fun attachment(action: () -> Unit) {
            if (pendingImageInputs.isNotEmpty()) showTransientMessage(getString(R.string.attachment_mixed_input_blocked))
            else action()
        }
        heading(R.string.plus_menu_add)
        row(R.drawable.ic_camera_24, R.string.plus_menu_camera, false) { handleCameraOcrClick() }
        row(R.drawable.ic_image_24, R.string.plus_menu_photo, false) {
            galleryImagePickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        row(R.drawable.ic_doc_20, R.string.plus_menu_document, false) {
            attachment {
                documentPickerLauncher.launch(
                    arrayOf("application/pdf", "text/*", "application/json", "application/xml", "application/yaml")
                )
            }
        }
        row(R.drawable.ic_attach_file_24, R.string.plus_menu_audio, false) {
            attachment { audioPickerLauncher.launch(arrayOf("audio/*", "video/mp4")) }
        }
        if (ImageGenFeature.isAvailable) {
            content.addView(View(this).apply {
                setBackgroundColor(resolveThemeColor(R.attr.colorPocketOutline))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)).apply {
                    setMargins(dp(14), dp(4), dp(14), dp(4))
                }
            })
            row(R.drawable.ic_sparkle_20, R.string.plus_menu_create, true) { openImageGeneration(toolName = ImageTool.CREATE.name) }
        }

        val button = anchor as MaterialButton
        button.setIconResource(R.drawable.ic_close_18)
        button.contentDescription = getString(R.string.plus_menu_close)
        popup.setOnDismissListener {
            button.setIconResource(R.drawable.ic_plus_24)
            button.contentDescription = getString(R.string.plus_menu_open)
        }
        content.measure(View.MeasureSpec.UNSPECIFIED, View.MeasureSpec.UNSPECIFIED)
        popup.showAsDropDown(anchor, 0, -(anchor.height + content.measuredHeight + dp(8)))
    }

    private fun showDictationRecordingDialog() {
        if (dictationRecorder != null || dictationTranscriptionJob?.isActive == true) return
        val controller = requireUsableController() ?: return
        if (controller.state.value.isGenerating) return
        // A model with working audio input transcribes the recording itself, so no Whisper download is needed.
        if (!canDictateWithNativeAudio() && !whisperModelStore.isAvailable()) {
            val content = layoutInflater.inflate(R.layout.dialog_dictation_download, null)
            val message: TextView = content.findViewById(R.id.dictationDownloadMessage)
            val downloadButton: MaterialButton = content.findViewById(R.id.downloadDictationButton)
            val notNowButton: MaterialButton = content.findViewById(R.id.notNowDictationButton)
            message.text = getString(R.string.dictation_download_message, WhisperModelStore.DOWNLOAD_SIZE_LABEL)
            val dialog = MaterialAlertDialogBuilder(this)
                .setTitle(R.string.dictation_download_title)
                .setView(content)
                .create()
            dialog.setOnShowListener {
                downloadButton.setOnClickListener {
                    dialog.dismiss()
                    lifecycleScope.launch {
                        runCatching { whisperModelStore.download() }
                            .onSuccess { startDictationRecording() }
                            .onFailure { showTransientMessage(it.message ?: getString(R.string.dictation_download_failed)) }
                    }
                }
                notNowButton.setOnClickListener { dialog.dismiss() }
            }
            dialog.show()
            return
        }
        if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
            pendingDictationRecordingStart = true
            dictationRecordingPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            return
        }
        startDictationRecording()
    }

    @Suppress("DEPRECATION")
    private fun startDictationRecording() {
        if (dictationRecorder != null || dictationTranscriptionJob?.isActive == true) return
        val directory = File(cacheDir, "temporary_dictation").apply { mkdirs() }
        val recording = File.createTempFile("dictation_", ".m4a", directory)
        val recorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) MediaRecorder(this) else MediaRecorder()
        try {
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC)
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            recorder.setAudioSamplingRate(44_100)
            recorder.setAudioEncodingBitRate(128_000)
            recorder.setMaxDuration(AttachmentLimits.MAX_AUDIO_DURATION_MILLIS.toInt())
            recorder.setOutputFile(recording.absolutePath)
            recorder.prepare()
            recorder.start()
            dictationRecorder = recorder
            dictationRecordingFile = recording
            updateMicRecordingState(true)
        } catch (error: Throwable) {
            runCatching { recorder.release() }
            recording.delete()
            showTransientMessage(error.message ?: getString(R.string.record_audio_coming_soon))
            return
        }

        val content = layoutInflater.inflate(R.layout.dialog_audio_recording, null)
        val timer: TextView = content.findViewById(R.id.audioRecordingTimer)
        val useButton: MaterialButton = content.findViewById(R.id.useAudioRecordingButton)
        val discardButton: MaterialButton = content.findViewById(R.id.discardAudioRecordingButton)
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle(R.string.dictation_record_title)
            .setView(content)
            .create()
        dictationRecordingDialog = dialog
        startDictationTimer(timer)
        dialog.setOnShowListener {
            useButton.setOnClickListener {
                finishDictationRecording(transcribe = true)
                dialog.dismiss()
            }
            discardButton.setOnClickListener {
                finishDictationRecording(transcribe = false)
                dialog.dismiss()
            }
        }
        dialog.setOnCancelListener { finishDictationRecording(transcribe = false) }
        dialog.setOnDismissListener {
            if (dictationRecordingDialog === dialog) dictationRecordingDialog = null
        }
        dialog.show()
    }

    private fun finishDictationRecording(transcribe: Boolean) {
        val recorder = dictationRecorder
        val recording = dictationRecordingFile
        dictationRecorder = null
        dictationRecordingFile = null
        stopDictationTimer()
        updateMicRecordingState(false)
        val stopped = recorder?.let { runCatching { it.stop() }.isSuccess } ?: false
        runCatching { recorder?.release() }
        if (transcribe && stopped && recording != null && recording.length() > 0L) {
            transcribeDictation(recording)
        } else {
            recording?.delete()
        }
    }

    private fun canDictateWithNativeAudio(): Boolean {
        return DictationFeature.isAvailable &&
            chatController?.state?.value?.let { it.isReady && it.supportsNativeAudioInput } == true &&
            currentModel?.nativeAudioSegmentMillis != null
    }

    private fun transcribeDictation(recording: File) {
        if (dictationTranscriptionJob?.isActive == true) return
        showTransientMessage(getString(R.string.dictation_transcribing))
        showDictationTranscriptionDialog()
        dictationTranscriptionJob = lifecycleScope.launch {
            val normalized = File(cacheDir, "temporary_dictation/${recording.nameWithoutExtension}.wav")
            try {
                val useNativeAudio = canDictateWithNativeAudio()
                val transcriptionWork: suspend () -> String = {
                    val audio = withContext(Dispatchers.IO) { AudioNormalizer().normalize(recording, normalized) }
                    val descriptor = AttachmentDescriptor(
                        sessionId = "temporary",
                        displayName = recording.name,
                        mimeType = "audio/wav",
                        kind = AttachmentKind.AUDIO,
                        status = AttachmentStatus.READY,
                        processingRoute = if (useNativeAudio) {
                            AttachmentProcessingRoute.GEMMA_NATIVE_AUDIO
                        } else {
                            AttachmentProcessingRoute.SHERPA_WHISPER
                        },
                        sourcePath = audio.file.absolutePath,
                        sizeBytes = audio.file.length(),
                        durationMillis = audio.durationMillis
                    )
                    if (useNativeAudio) {
                        val controller = chatController ?: error("The model is not ready.")
                        val feature = DictationFeature.instance ?: error("Model dictation is not available.")
                        PromptPreprocessor.normalize(
                            feature.transcribe(
                                applicationContext,
                                controller,
                                descriptor,
                                checkNotNull(currentModel?.nativeAudioSegmentMillis)
                            )
                        )
                    } else {
                        WhisperAttachmentTranscriber(this@PocketChatActivity)
                            .transcribe(descriptor)
                            .joinToString(" ") { it.text }
                            .let(PromptPreprocessor::normalize)
                    }
                }
                val transcript = if (useNativeAudio) {
                    transcriptionWork()
                } else {
                    chatController?.withRuntimeLease { transcriptionWork() } ?: transcriptionWork()
                }
                if (transcript.isBlank()) {
                    showTransientMessage(getString(R.string.dictation_no_speech))
                } else {
                    setPromptInputText(
                        PromptPreprocessor.mergeTypedAndRecognized(inputEditText.text.toString(), transcript)
                    )
                }
            } catch (_: CancellationException) {
                // Lifecycle cancellation owns cleanup below.
            } catch (error: Throwable) {
                showTransientMessage(error.message ?: getString(R.string.dictation_transcription_failed))
            } finally {
                recording.delete()
                normalized.delete()
                dictationTranscriptionJob = null
                dictationTranscriptionDialog?.dismiss()
                dictationTranscriptionDialog = null
            }
        }
    }

    private fun showDictationTranscriptionDialog() {
        if (dictationTranscriptionDialog?.isShowing == true) return
        dictationTranscriptionDialog = MaterialAlertDialogBuilder(this)
            .setView(R.layout.dialog_dictation_transcribing)
            .setCancelable(false)
            .create()
            .also(AlertDialog::show)
    }

    private fun discardDictationRecording() {
        finishDictationRecording(transcribe = false)
        dictationRecordingDialog?.dismiss()
        dictationRecordingDialog = null
    }

    private fun startDictationTimer(timer: TextView) {
        stopDictationTimer()
        dictationStartedAtMillis = SystemClock.elapsedRealtime()
        val ticker = object : Runnable {
            override fun run() {
                val seconds = (SystemClock.elapsedRealtime() - dictationStartedAtMillis) / 1_000L
                timer.text = formatDictationDuration(seconds)
                dictationTimerHandler.postDelayed(this, 250L)
            }
        }
        dictationTimer = ticker
        ticker.run()
    }

    private fun stopDictationTimer() {
        dictationTimer?.let(dictationTimerHandler::removeCallbacks)
        dictationTimer = null
    }

    private fun formatDictationDuration(seconds: Long): String = if (seconds >= 3_600L) {
        String.format(Locale.getDefault(), "%02d:%02d:%02d", seconds / 3_600L, (seconds % 3_600L) / 60L, seconds % 60L)
    } else {
        String.format(Locale.getDefault(), "%02d:%02d", seconds / 60L, seconds % 60L)
    }

    private fun importAttachment(uri: Uri, requestedKind: AttachmentKind) {
        if (pendingImageInputs.isNotEmpty()) {
            showTransientMessage(getString(R.string.attachment_mixed_input_blocked))
            return
        }
        val controller = requireUsableController() ?: return
        if (controller.state.value.isGenerating || attachmentImportJob?.isActive == true) return
        runCatching {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        attachmentImportInProgress = true
        activeAttachment = null
        renderActiveAttachment()
        val sessionId = controller.ensureSessionId()
        val record = runCatching {
            attachmentWorkerCoordinator.enqueueImport(
                sessionId = sessionId,
                uri = uri,
                kind = requestedKind,
                gemmaAudio = usesNativeAudioForSelectedModel()
            )
        }.getOrElse { error ->
            attachmentImportInProgress = false
            deleteRecordedAttachmentInput(uri)
            showTransientMessage(error.message ?: "Attachment import could not be started.")
            renderActiveAttachment()
            return
        }
        observeAttachmentWork(record)
    }

    private fun deleteRecordedAttachmentInput(uri: Uri) {
        if (uri.scheme != "file") return
        val file = uri.path?.let(::File) ?: return
        val recordingsDirectory = File(cacheDir, "attachment_recordings")
        val safe = runCatching {
            file.canonicalPath.startsWith(recordingsDirectory.canonicalPath + File.separator)
        }.getOrDefault(false)
        if (safe) file.delete()
    }

    private fun offerWhisperPreparation(descriptor: AttachmentDescriptor) {
        if (whisperModelStore.isAvailable()) {
            prepareWhisperTranscript(descriptor)
            return
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Download multilingual transcription model?")
            .setMessage(
                "This model uses sherpa-onnx Whisper tiny int8 (${WhisperModelStore.DOWNLOAD_SIZE_LABEL}) for transcription. " +
                    "It downloads only after your confirmation and stays on this device."
            )
            .setPositiveButton("Download") { _, _ -> downloadWhisperAndPrepare(descriptor) }
            .setNegativeButton(android.R.string.cancel, null)
            .show()
    }

    private fun downloadWhisperAndPrepare(descriptor: AttachmentDescriptor) {
        if (audioPreparationJob?.isActive == true) return
        attachmentImportInProgress = true
        renderActiveAttachment()
        audioPreparationJob = lifecycleScope.launch {
            try {
                whisperModelStore.download { downloaded, total ->
                    runOnUiThread {
                        activeAttachmentText.text = getString(
                            R.string.whisper_download_progress,
                            (downloaded * 100L / total.coerceAtLeast(1L)).coerceIn(0L, 100L)
                        )
                    }
                }
                attachmentImportInProgress = false
                audioPreparationJob = null
                prepareWhisperTranscript(descriptor)
            } catch (_: CancellationException) {
                // Cancel/detach owns the resulting attachment UI state.
            } catch (error: Throwable) {
                attachmentImportInProgress = false
                audioPreparationJob = null
                showTransientMessage(error.message ?: "Could not download the transcription model.")
                renderActiveAttachment()
            }
        }
    }

    private fun prepareWhisperTranscript(descriptor: AttachmentDescriptor) {
        if (activeAttachmentWork != null) return
        val processing = descriptor.copy(
            status = AttachmentStatus.PROCESSING,
            updatedAtMillis = System.currentTimeMillis(),
            errorMessage = null
        )
        activeAttachment = processing
        attachmentRepository.saveDescriptor(processing)
        renderActiveAttachment()
        val record = runCatching {
            attachmentWorkerCoordinator.enqueueWhisper(processing.sessionId, processing.id)
        }.getOrElse { error ->
            activeAttachment = processing.copy(
                status = AttachmentStatus.FAILED,
                errorMessage = error.message ?: "Audio transcription could not be started.",
                retryOperation = AttachmentWorkOperation.WHISPER_TRANSCRIPTION
            ).also(attachmentRepository::saveDescriptor)
            renderActiveAttachment()
            return
        }
        observeAttachmentWork(record)
    }

    private fun detachActiveAttachment() {
        activeAttachmentWork?.let(::cancelAndForgetWork)
        attachmentImportJob?.cancel()
        attachmentImportJob = null
        audioPreparationJob?.cancel()
        audioPreparationJob = null
        attachmentImportInProgress = false
        activeAttachment = null
        confirmedAttachmentOperationKey = null
        chatController?.setActiveAttachment(null)
        renderActiveAttachment()
    }

    private val dictationRecordingPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        val shouldStart = pendingDictationRecordingStart
        pendingDictationRecordingStart = false
        if (granted && shouldStart) {
            startDictationRecording()
        } else if (!granted) {
            showTransientMessage(getString(R.string.speech_permission_denied))
        }
    }

    private fun clearSentAttachmentFromComposer() {
        if (activeAttachment?.kind != AttachmentKind.AUDIO) return
        activeAttachment = null
        chatController?.setActiveAttachment(null)
        renderActiveAttachment()
    }

    private fun cancelActiveAttachmentWork() {
        val record = activeAttachmentWork
        record?.let {
            attachmentWorkerCoordinator.cancel(java.util.UUID.fromString(it.workId))
            attachmentRepository.deleteWorkRecord(it)
        }
        activeAttachmentWork = null
        activeAttachmentWorkState = androidx.work.WorkInfo.State.CANCELLED
        attachmentImportJob?.cancel()
        attachmentImportJob = null
        audioPreparationJob?.cancel()
        audioPreparationJob = null
        attachmentImportInProgress = false
        activeAttachment = activeAttachment?.copy(
            status = AttachmentStatus.FAILED,
            updatedAtMillis = System.currentTimeMillis(),
            errorMessage = "Processing cancelled. Retry or detach the attachment.",
            retryOperation = record?.operation ?: activeAttachment?.retryOperation
        )?.also(attachmentRepository::saveDescriptor)
        renderActiveAttachment()
    }

    private fun reactivateAttachmentTurn(turn: ChatTurn) {
        val controller = chatController ?: return
        if (controller.state.value.isGenerating || attachmentImportInProgress) return
        val attachmentId = turn.attachmentId ?: return
        val sessionId = controller.snapshotActiveChat().sessionId ?: return
        val descriptor = attachmentRepository.loadDescriptor(sessionId, attachmentId)
        if (descriptor == null) {
            showTransientMessage("That attachment is no longer available.")
            return
        }
        if (pendingImageInputs.isNotEmpty()) clearPendingImageInputs()
        activeAttachment = descriptor
        controller.setActiveAttachment(descriptor.id)
        renderActiveAttachment()
    }

    private fun restoreActiveAttachmentIfNeeded(attachmentId: String?) {
        if (attachmentId == null) {
            if (!attachmentImportInProgress) activeAttachment = null
            renderActiveAttachment()
            return
        }
        if (activeAttachment?.id == attachmentId) return
        val snapshot = chatController?.snapshotActiveChat() ?: return
        val sessionId = snapshot.sessionId ?: return
        activeAttachment = attachmentRepository.loadDescriptor(sessionId, attachmentId)
        renderActiveAttachment()
    }

    private fun reconnectActiveAttachmentWorkIfNeeded() {
        if (attachmentImportJob?.isActive == true || activeAttachmentWork != null) return
        val controller = chatController ?: return
        val sessionId = controller.snapshotActiveChat().sessionId
        val record = AttachmentWorkPolicy.selectForRestoration(
            sessionId,
            attachmentRepository.listWorkRecords()
        ) ?: return
        controller.restoreAttachmentSession(record.sessionId)
        attachmentRepository.loadDescriptor(record.sessionId, record.attachmentId)?.let {
            activeAttachment = it
        }
        observeAttachmentWork(record)
        activeAttachment?.let { controller.setActiveAttachment(it.id) }
    }

    private fun observeAttachmentWork(record: AttachmentWorkRecord) {
        if (activeAttachmentWork?.workId == record.workId && attachmentImportJob?.isActive == true) return
        attachmentImportJob?.cancel()
        activeAttachmentWork = record
        activeAttachmentWorkState = androidx.work.WorkInfo.State.ENQUEUED
        attachmentImportInProgress = record.operation == AttachmentWorkOperation.IMPORT
        renderActiveAttachment()
        attachmentImportJob = lifecycleScope.launch {
            try {
                val awaitTerminal: suspend () -> androidx.work.WorkInfo? = {
                    val flow = attachmentWorkerCoordinator
                        .workInfoFlow(java.util.UUID.fromString(record.workId))
                        .filterNotNull()
                    val firstInfo = withTimeoutOrNull(2_000L) { flow.first() }
                    if (firstInfo == null) null else {
                        activeAttachmentWorkState = firstInfo.state
                        attachmentRepository.loadDescriptor(record.sessionId, record.attachmentId)?.let {
                            activeAttachment = it
                        }
                        renderActiveAttachment()
                        if (firstInfo.state.isFinished) firstInfo else flow.first { info ->
                            activeAttachmentWorkState = info.state
                            attachmentRepository.loadDescriptor(record.sessionId, record.attachmentId)?.let {
                                activeAttachment = it
                            }
                            renderActiveAttachment()
                            info.state.isFinished
                        }
                    }
                }
                val workInfo = if (record.operation == AttachmentWorkOperation.WHISPER_TRANSCRIPTION) {
                    chatController?.withRuntimeLease { awaitTerminal() } ?: awaitTerminal()
                } else {
                    awaitTerminal()
                }
                if (workInfo == null) {
                    attachmentRepository.deleteWorkRecord(record)
                    activeAttachmentWork = null
                    activeAttachmentWorkState = null
                    attachmentImportInProgress = false
                    showTransientMessage("The saved attachment job no longer exists.")
                } else {
                    finishAttachmentWork(record, workInfo)
                }
            } catch (_: CancellationException) {
                // WorkManager owns the operation; a recreated Activity reconnects from the durable record.
            } finally {
                if (activeAttachmentWork?.workId == record.workId && activeAttachmentWorkState?.isFinished == true) {
                    activeAttachmentWork = null
                    attachmentImportInProgress = false
                }
                attachmentImportJob = null
                renderActiveAttachment()
            }
        }
    }

    private fun finishAttachmentWork(record: AttachmentWorkRecord, workInfo: androidx.work.WorkInfo) {
        var descriptor = attachmentRepository.loadDescriptor(record.sessionId, record.attachmentId)
        val succeeded = workInfo.state == androidx.work.WorkInfo.State.SUCCEEDED &&
            descriptor?.status == AttachmentStatus.READY
        val errorMessage = workInfo.outputData.getString(AttachmentPreparationWorker.KEY_ERROR)
            ?: descriptor?.errorMessage
            ?: if (workInfo.state == androidx.work.WorkInfo.State.SUCCEEDED) {
                "The attachment result could not be loaded."
            } else null
            ?: when (workInfo.state) {
                androidx.work.WorkInfo.State.CANCELLED -> "Processing cancelled. Retry or detach the attachment."
                else -> "Attachment processing failed."
            }
        if (!succeeded && descriptor != null) {
            descriptor = AttachmentWorkPolicy.failedDescriptor(
                descriptor,
                record.operation,
                errorMessage
            ).also(attachmentRepository::saveDescriptor)
        }
        attachmentRepository.deleteWorkRecord(record)
        activeAttachmentWorkState = workInfo.state
        attachmentImportInProgress = false
        activeAttachment = descriptor
        descriptor?.let {
            chatController?.restoreAttachmentSession(record.sessionId)
            chatController?.setActiveAttachment(it.id)
        }
        if (!succeeded) {
            showTransientMessage(errorMessage)
        } else if (
            descriptor?.kind == AttachmentKind.AUDIO &&
            descriptor.status == AttachmentStatus.READY &&
            !usesNativeAudioForSelectedModel() &&
            record.operation == AttachmentWorkOperation.IMPORT
        ) {
            offerWhisperPreparation(descriptor)
        }
    }

    private fun retryActiveAttachment() {
        val descriptor = activeAttachment ?: return
        if (activeAttachmentWork != null) return
        val record = runCatching {
            when (descriptor.retryOperation) {
                AttachmentWorkOperation.IMPORT -> attachmentWorkerCoordinator.enqueueImportRetry(descriptor)
                AttachmentWorkOperation.WHISPER_TRANSCRIPTION ->
                    attachmentWorkerCoordinator.enqueueWhisper(descriptor.sessionId, descriptor.id)
                null -> return
            }
        }.getOrElse { error ->
            activeAttachment = descriptor.copy(
                status = AttachmentStatus.FAILED,
                updatedAtMillis = System.currentTimeMillis(),
                errorMessage = error.message ?: "Attachment retry could not be started."
            ).also(attachmentRepository::saveDescriptor)
            renderActiveAttachment()
            return
        }
        observeAttachmentWork(record)
    }

    private fun cancelAndForgetWork(record: AttachmentWorkRecord) {
        attachmentWorkerCoordinator.cancel(java.util.UUID.fromString(record.workId))
        attachmentRepository.deleteWorkRecord(record)
        activeAttachmentWork = null
        activeAttachmentWorkState = null
    }

    private fun prepareGemmaAudioAfterConfirmation(descriptor: AttachmentDescriptor) {
        val segmentMillis = currentModel?.nativeAudioSegmentMillis ?: return
        if (audioPreparationJob?.isActive == true) return
        audioPreparationJob = lifecycleScope.launch {
            try {
                val segmented = withContext(Dispatchers.IO) {
                    GemmaAudioSegmenter(applicationContext, segmentMillis).segment(descriptor)
                }
                preparedGemmaAudio = segmented
                preparedGemmaAudioAttachmentId = descriptor.id
            } catch (_: CancellationException) {
                cleanupPreparedGemmaAudio()
                return@launch
            } catch (error: Throwable) {
                cleanupPreparedGemmaAudio()
                showTransientMessage(error.message ?: "Audio segmentation failed.")
                return@launch
            } finally {
                audioPreparationJob = null
            }
            handleSendClick()
        }
    }

    private fun cleanupPreparedGemmaAudio() {
        preparedGemmaAudio?.temporaryFiles?.forEach { File(it).delete() }
        preparedGemmaAudio = null
        preparedGemmaAudioAttachmentId = null
    }

    private fun renderActiveAttachment() {
        if (!::activeAttachmentContainer.isInitialized) return
        val descriptor = activeAttachment
        val compactReadyAudio = descriptor?.kind == AttachmentKind.AUDIO &&
            descriptor.status == AttachmentStatus.READY &&
            !attachmentImportInProgress
        val containerParams = activeAttachmentContainer.layoutParams
        containerParams.width = if (compactReadyAudio) {
            ViewGroup.LayoutParams.WRAP_CONTENT
        } else {
            ViewGroup.LayoutParams.MATCH_PARENT
        }
        activeAttachmentContainer.layoutParams = containerParams
        val textParams = activeAttachmentText.layoutParams as LinearLayout.LayoutParams
        textParams.width = if (compactReadyAudio) {
            ViewGroup.LayoutParams.WRAP_CONTENT
        } else {
            0
        }
        textParams.weight = if (compactReadyAudio) 0f else 1f
        activeAttachmentText.layoutParams = textParams
        activeAttachmentContainer.setBackgroundResource(
            if (compactReadyAudio) R.drawable.bg_active_audio_attachment else R.drawable.bg_image_input_chip
        )
        activeAttachmentText.setTextColor(
            if (compactReadyAudio) resolveThemeColor(R.attr.colorOnAccent) else resolveThemeColor(R.attr.colorInputText)
        )
        activeAttachmentContainer.visibility = if (attachmentImportInProgress || descriptor != null) {
            View.VISIBLE
        } else {
            View.GONE
        }
        activeAttachmentProgress.visibility = if (
            attachmentImportInProgress || descriptor?.status == AttachmentStatus.PROCESSING || descriptor?.status == AttachmentStatus.IMPORTING
        ) View.VISIBLE else View.GONE
        activeAttachmentRetry.visibility = if (descriptor?.status == AttachmentStatus.FAILED) View.VISIBLE else View.GONE
        activeAttachmentCancel.visibility = if (
            attachmentImportInProgress || descriptor?.status == AttachmentStatus.PROCESSING || descriptor?.status == AttachmentStatus.IMPORTING
        ) View.VISIBLE else View.GONE
        activeAttachmentText.text = when {
            attachmentImportInProgress && activeAttachmentWorkState in listOf(
                androidx.work.WorkInfo.State.ENQUEUED,
                androidx.work.WorkInfo.State.BLOCKED
            ) -> getString(R.string.attachment_queued)
            attachmentImportInProgress -> getString(R.string.attachment_importing)
            descriptor == null -> ""
            descriptor.status == AttachmentStatus.FAILED -> getString(
                R.string.attachment_failed,
                descriptor.errorMessage.orEmpty()
            )
            descriptor.status == AttachmentStatus.PROCESSING ->
                getString(R.string.attachment_using, attachmentDisplayName(descriptor)) + " · " +
                    if (activeAttachmentWorkState == androidx.work.WorkInfo.State.ENQUEUED) "Queued…" else "Transcribing…"
            else -> getString(R.string.attachment_using, attachmentDisplayName(descriptor)) + " · " +
                getString(R.string.attachment_ready)
        }
    }

    private fun attachmentDisplayName(descriptor: AttachmentDescriptor): String = when (descriptor.kind) {
        AttachmentKind.AUDIO -> getString(R.string.audio_attached)
        else -> descriptor.displayName
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
        modelPreparationJob?.cancel()
        controllerStateJob = lifecycleScope.launch {
            controller.state.collect { state ->
                reconcileModelLoadState(state)
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
                    updateNoModelHint()
                }
            }
        }
    }

    private fun applyModelDownloadState(state: ModelDownloadState) {
        when (state) {
            ModelDownloadState.Idle -> {
                if (activeDownloadModelId != null) {
                    modelOperationStatusMessage = null
                    resetDownloadState()
                    refreshModelSelectionDialog()
                    if (chatController != null) {
                        applyChatState(chatController!!.state.value)
                    } else {
                        renderNoControllerState(getString(R.string.model_required_message), preserveTranscript = true)
                    }
                }
            }

            is ModelDownloadState.Running -> {
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

                // A progress tick only needs the running row to change. Rebuilding the whole list here
                // inflated every row and checked every model file on the main thread, once a second,
                // while the download was saturating the disk.
                val row = downloadRowProgress
                if (row != null && downloadRowModelId == state.modelId && row.first.isAttachedToWindow) {
                    updateProgressBar(row.first, activeDownloadBytes, activeDownloadTotalBytes)
                    row.second.text = formatDownloadProgressText(activeDownloadBytes, activeDownloadTotalBytes)
                    updateDownloadSheet()
                } else {
                    refreshModelSelectionDialog()
                }
                if (chatController == null) {
                    renderNoControllerState(modelOperationStatusMessage.orEmpty(), preserveTranscript = true)
                }
            }

            is ModelDownloadState.Completed -> {
                modelOperationStatusMessage = null
                resetDownloadState()
                refreshModelSelectionDialog()
                if (chatController != null) {
                    applyChatState(chatController!!.state.value)
                } else {
                    renderNoControllerState(getString(R.string.model_required_message), preserveTranscript = true)
                }
                showTransientMessage(getString(R.string.download_notification_complete, state.modelName))

                ModelDownloadStateStore.clearTerminalState(state.modelId)
            }

            is ModelDownloadState.Failed -> {
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
        activeChatSnapshot: ActiveChatSnapshot? = null,
        attempt: ModelLoadRecord? = null
    ) {
        controllerStateJob?.cancel()
        if (currentModel?.id != descriptor.id && activeChatSnapshot == null) {
            inputEditText.text.clear()
            clearSpeechInputState()
            clearPendingImageInputs()
            activeAttachment = null
            renderActiveAttachment()
        }
        chatController?.close()
        chatController = controller
        SharedChatControllerRegistry.register(descriptor.id, controller)
        currentModel = descriptor
        ensureImageInputModeAllowedForCurrentModel()
        retainedState.chatController = controller
        retainedState.modelId = descriptor.id
        modelSelectionStore.saveSelectedModel(descriptor.id)
        modelOperationStatusMessage = null
        autoScrollDuringGeneration = false
        autoScrollPendingFinalUpdate = false
        wasGenerating = false
        toolbarSubtitleView.text = descriptor.displayName
        updateModelSelectorLabel()
        lastGemmaDirectImageInputAvailable = null
        updateImageInputButtonDescriptions()
        refreshDrawerSessions()
        thinkingToggle.isChecked = settingsStore.loadThinkByDefault()
        observeController(controller)
        applyChatState(controller.state.value)
        if (attempt != null) {
            if (controller.state.value.isReady) {
                handleModelInitializationResult(descriptor, attempt, Result.success(Unit))
            } else {
                controller.initialize(activeChatSnapshot) { result ->
                    handleModelInitializationResult(descriptor, attempt, result)
                }
            }
        }
    }

    private val documentPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { importAttachment(it, AttachmentKind.TEXT) } }

    private val audioPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { importAttachment(it, AttachmentKind.AUDIO) } }

    private var proGateInstance: ProGate? = null
    private val proGate: ProGate
        get() = proGateInstance ?: createProGate(this).also { proGateInstance = it }

    private val customModelPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let(::importCustomModel) }

    private fun showCustomModelWarning() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val warningView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_custom_model_warning, null)
        val dialog = dialogBuilder
            .setView(warningView)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.custom_model_choose_file) { _, _ ->
                customModelPickerLauncher.launch(arrayOf("*/*"))
            }
            .create()
        dialog.show()
    }

    private val llamaHost = object : LlamaFeature.Host {
        override val activity: AppCompatActivity get() = this@PocketChatActivity
        override fun showModelImportFailure(error: Throwable) = this@PocketChatActivity.showModelImportFailure(error)
        override fun onModelImported(id: String, isChat: Boolean, displayName: String) {
            ModelRegistry.loadCustomModels(this@PocketChatActivity)
            ImageGenFeature.instance?.loadCustomModels(this@PocketChatActivity)
            refreshModelSelectionDialog()
            if (isChat) {
                ModelRegistry.findById(id)?.let(::handleModelSelection)
            } else {
                showTransientMessage(getString(R.string.local_import_image_done, displayName))
            }
        }
    }

    /** Routes a picked file: .litertlm to the LiteRT import, .gguf (chat or image) and .safetensors (image) to the local importer. */
    private fun importCustomModel(uri: Uri) {
        val fileName = runCatching {
            contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
                ?.substringAfterLast('/')
                ?: throw IllegalArgumentException("Could not read the selected file name.")
        }.getOrElse {
            showModelImportFailure(it)
            return
        }
        when {
            fileName.endsWith(".litertlm", ignoreCase = true) -> importLiteRtModel(uri)
            fileName.endsWith(".gguf", ignoreCase = true) || fileName.endsWith(".safetensors", ignoreCase = true) -> {
                if (LlamaFeature.isAvailable) {
                    LlamaFeature.instance?.handleLocalImport(llamaHost, uri, fileName)
                } else {
                    showModelImportFailure(IllegalArgumentException(getString(R.string.local_import_unsupported)))
                }
            }
            else -> showModelImportFailure(IllegalArgumentException(getString(R.string.local_import_unsupported)))
        }
    }

    private fun showModelImportFailure(error: Throwable) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.custom_model_add)
            .setMessage(getString(R.string.custom_model_import_failed, error.message ?: "Unknown error"))
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun importLiteRtModel(uri: Uri) {
        val progress = MaterialAlertDialogBuilder(this)
            .setTitle(R.string.custom_model_add)
            .setMessage(R.string.custom_model_importing)
            .setCancelable(false)
            .create()
        progress.show()
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { CustomLiteRtModelStore(this@PocketChatActivity).import(uri) }
            }
            progress.dismiss()
            result.fold(
                onSuccess = { descriptor ->
                    ModelRegistry.loadCustomModels(this@PocketChatActivity)
                    refreshModelSelectionDialog()
                    handleModelSelection(descriptor)
                },
                onFailure = { error ->
                    MaterialAlertDialogBuilder(this@PocketChatActivity)
                        .setTitle(R.string.custom_model_add)
                        .setMessage(getString(R.string.custom_model_import_failed, error.message ?: "Unknown error"))
                        .setPositiveButton(android.R.string.ok, null)
                        .show()
                }
            )
        }
    }

    private fun requestModelLoad(
        descriptor: ModelDescriptor,
        activeChatSnapshot: ActiveChatSnapshot?
    ) {
        modelPreparationJob?.cancel()
        modelPreparationJob = lifecycleScope.launch {
            val initialPreflight = withContext(Dispatchers.IO) {
                modelPreflightChecker.checkLoad(descriptor, descriptor.approxDownloadBytes)
            }
            if (initialPreflight.kind == ModelPreflightKind.BLOCKED) {
                recordPreflightFailure(descriptor, initialPreflight.message.orEmpty())
                return@launch
            }

            val validation = withContext(Dispatchers.IO) { modelFileValidator.validate(descriptor) }
            if (!validation.valid) {
                recordPreflightFailure(
                    descriptor,
                    validation.message ?: getString(R.string.model_file_invalid_generic)
                )
                return@launch
            }

            val preflight = withContext(Dispatchers.IO) {
                modelPreflightChecker.checkLoad(descriptor, validation.totalBytes)
            }
            if (preflight.kind == ModelPreflightKind.BLOCKED) {
                recordPreflightFailure(descriptor, preflight.message.orEmpty())
                return@launch
            }
            startPreparedModelLoad(
                descriptor,
                activeChatSnapshot,
                BackendInitializationPolicy(allowCpuFallback = preflight.allowCpuFallback)
            )
        }
    }

    private fun startPreparedModelLoad(
        descriptor: ModelDescriptor,
        activeChatSnapshot: ActiveChatSnapshot?,
        policy: BackendInitializationPolicy
    ) {
        cancelCurrentInitializationMarker()
        val attempt = modelLoadCoordinator.begin(descriptor.id, System.currentTimeMillis())
        persistModelLoadRecord()
        retainedState.pendingModelLoadSnapshot = activeChatSnapshot
        val controller = SharedChatControllerRegistry.getOrCreate(this, descriptor, policy)
        switchToController(descriptor, controller, activeChatSnapshot, attempt)
    }

    private fun handleModelInitializationResult(
        descriptor: ModelDescriptor,
        attempt: ModelLoadRecord,
        result: Result<Unit>
    ) {
        if (isFinishing || isDestroyed) return
        result.fold(
            onSuccess = {
                if (modelLoadCoordinator.succeed(descriptor.id, attempt.attemptId)) {
                    persistModelLoadRecord()
                }
                retainedState.pendingModelLoadSnapshot = null
                chatController?.contextDecision?.let { showContextCapNoticeIfNeeded(descriptor, it) }
            },
            onFailure = { error ->
                val reason = usefulModelLoadError(descriptor, error)
                if (modelLoadCoordinator.fail(
                        descriptor.id,
                        attempt.attemptId,
                        reason,
                        System.currentTimeMillis()
                    )
                ) {
                    persistModelLoadRecord()
                    renderNoControllerState(reason, preserveTranscript = true)
                    showModelRecoveryDialog()
                }
            }
        )
    }

    private fun reconcileModelLoadState(state: ChatUiState) {
        val descriptor = currentModel ?: return
        val record = modelLoadCoordinator.record ?: return
        if (record.modelId != descriptor.id || record.phase != ModelLoadPhase.INITIALIZING) return
        when {
            state.isReady -> {
                if (modelLoadCoordinator.succeed(descriptor.id, record.attemptId)) {
                    persistModelLoadRecord()
                }
                retainedState.pendingModelLoadSnapshot = null
            }
            !state.isLoading && state.statusMessage.startsWith("Error:") -> {
                val reason = state.statusMessage.removePrefix("Error:").trim()
                    .ifBlank { getString(R.string.model_load_failed_generic) }
                if (modelLoadCoordinator.fail(descriptor.id, record.attemptId, reason, System.currentTimeMillis())) {
                    persistModelLoadRecord()
                    showModelRecoveryDialog()
                }
            }
        }
    }

    private fun recordPreflightFailure(descriptor: ModelDescriptor, reason: String) {
        cancelCurrentInitializationMarker()
        val snapshot = chatController?.snapshotActiveChat()
            ?: retainedState.pendingModelLoadSnapshot
        controllerStateJob?.cancel()
        chatController?.close()
        chatController = null
        retainedState.chatController = null
        retainedState.pendingModelLoadSnapshot = snapshot
        currentModel = descriptor
        retainedState.modelId = descriptor.id
        modelSelectionStore.saveSelectedModel(descriptor.id)
        val attempt = modelLoadCoordinator.begin(descriptor.id, System.currentTimeMillis())
        modelLoadCoordinator.fail(descriptor.id, attempt.attemptId, reason, System.currentTimeMillis())
        persistModelLoadRecord()
        toolbarSubtitleView.text = descriptor.displayName
        renderNoControllerState(reason, preserveTranscript = true)
        showModelRecoveryDialog()
    }

    private fun cancelCurrentInitializationMarker() {
        val record = modelLoadCoordinator.record ?: return
        if (record.phase != ModelLoadPhase.INITIALIZING) return
        chatController?.cancelInitialization()
        if (modelLoadCoordinator.cancel(record.modelId, record.attemptId)) {
            persistModelLoadRecord()
        }
    }

    private fun showContextCapNoticeIfNeeded(descriptor: ModelDescriptor, decision: ContextDecision) {
        if (!decision.isCapped || isFinishing || isDestroyed) return
        if (!ContextMemoryGuard(this).markCapNoticeShown(descriptor.id, decision)) return
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.model_context_capped_title)
            .setMessage(
                getString(
                    if (decision.cappedByCrashHistory) {
                        R.string.model_context_capped_after_crash
                    } else {
                        R.string.model_context_capped_device
                    },
                    decision.effectiveTokens,
                    decision.requestedTokens
                )
            )
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun persistModelLoadRecord() {
        modelLoadRecoveryStore.save(modelLoadCoordinator.record)
    }

    private fun usefulModelLoadError(descriptor: ModelDescriptor, error: Throwable): String {
        val detail = error.message?.trim().orEmpty()
        return if (detail.isBlank()) {
            getString(R.string.model_load_failed_message, descriptor.displayName)
        } else {
            getString(R.string.model_load_failed_detail, descriptor.displayName, detail)
        }
    }

    private fun applyChatState(state: ChatUiState) {
        updateNoModelHint()
        if (state.isReady) reconnectActiveAttachmentWorkIfNeeded()
        restoreActiveAttachmentIfNeeded(state.activeAttachmentId)
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
        updateModelSelectorLabel()
        if (currentModelSupportsThinking != state.supportsThinking) {
            currentModelSupportsThinking = state.supportsThinking
            updateModelSelectorLabel()
        }
        thinkingToggleContainer.visibility = if (state.supportsThinking) View.VISIBLE else View.GONE
        updateWebSearchButton()
        updateChatGreeting(state)
        val gemmaDirectAvailable = isGemmaDirectImageInputAvailable()
        if (!gemmaDirectAvailable && currentImageInputMode == ImageInputMode.GEMMA_DIRECT) {
            selectOcrImageInputMode(resetGemmaDirectInputs = true)
            updateModelSelectorLabel()
        }
        if (lastGemmaDirectImageInputAvailable != gemmaDirectAvailable) {
            lastGemmaDirectImageInputAvailable = gemmaDirectAvailable
            updateImageInputButtonDescriptions()
            refreshModelSelectionDialog()
        }
        val sendWasHidden = sendButton.visibility != View.VISIBLE
        sendButton.visibility = if (state.isGenerating) View.GONE else View.VISIBLE
        if (sendWasHidden && sendButton.visibility == View.VISIBLE) morphIn(sendButton)
        stopButton.visibility = if (state.isGenerating) View.VISIBLE else View.GONE
        findViewById<AnswerLineView>(R.id.answerLine)?.setActive(state.isGenerating)
        setStopGlow(state.isGenerating)
        settleWebSearchTry(state)
        newChatButton.isEnabled = state.isReady && !state.isGenerating && !isImagePreprocessingForSend
        findViewById<View>(R.id.toolbarNewChat).apply {
            isEnabled = newChatButton.isEnabled
            alpha = if (isEnabled) 1f else 0.4f
        }
        updateChatToolChips()
        sendButton.isEnabled = state.isReady && !state.isGenerating && !isImagePreprocessingForSend
        attachmentButton.isEnabled = state.isReady && !state.isGenerating &&
            !attachmentImportInProgress && audioPreparationJob?.isActive != true
        stopButton.isEnabled = state.isGenerating
        applyImageModeComposer()
        if (state.contextWindowTokens > 0) {
            contextUsageIndicator.visibility = View.VISIBLE
            contextUsageIndicator.setProgressCompat(contextUsagePercent(state), true)
        } else {
            contextUsageIndicator.visibility = View.GONE
        }

        findViewById<View>(R.id.modelStateDot).backgroundTintList = ColorStateList.valueOf(
            when {
                state.isReady -> getColor(R.color.status_running)
                state.isLoading -> getColor(R.color.state_warn)
                else -> getColor(R.color.pocket_faint)
            }
        )
        val effectiveStatus = if (state.isReady) {
            state.statusMessage
        } else {
            modelOperationStatusMessage ?: state.statusMessage
        }
        statusView.text = effectiveStatus
        val greetingShows = state.isReady && !state.isGenerating && state.transcript.isEmpty()
        val showInlineStatus = effectiveStatus.isNotBlank() && !greetingShows &&
            (state.transcript.isEmpty() || !state.isReady)
        val showLoadingCard = state.isLoading && !state.isReady && state.transcript.isEmpty() &&
            effectiveStatus == MODEL_LOADING_STATUS_MESSAGE
        if (showLoadingCard) {
            currentModel?.displayName?.let { modelLoadingCard.setModelName(it) }
            modelLoadingCard.show(currentModel?.id)
        } else {
            modelLoadingCard.hide(success = state.isReady)
        }
        inputEditText.alpha = if (state.isLoading && !state.isReady) 0.5f else 1f
        statusView.visibility = if (showInlineStatus && !showLoadingCard) View.VISIBLE else View.GONE
        applyStatusBackground(effectiveStatus)

        chatAdapter.submitTurns(state.transcript)
        wasGenerating = state.isGenerating
        if (generationFinished) {
            refreshDrawerSessions()
        }
    }

    private fun showContextUsageDialog(state: ChatUiState) {
        val view = layoutInflater.inflate(R.layout.dialog_context_usage, null)
        val window = state.contextWindowTokens
        val used = state.contextUsedTokens.coerceIn(0, window)
        view.findViewById<TextView>(R.id.contextUsageUsed).text = getString(R.string.context_usage_tokens, used)
        view.findViewById<TextView>(R.id.contextUsageTotal).text = getString(R.string.context_usage_total, window)
        view.findViewById<TextView>(R.id.contextUsagePercent).text =
            getString(R.string.context_usage_percent, contextUsagePercent(state))
        view.findViewById<TextView>(R.id.contextUsageNote).setText(
            if (currentModel is OnnxQwenSpec) R.string.context_usage_note_trimmed else R.string.context_usage_note
        )

        val breakdown = state.contextBreakdown
        val parts = listOf(
            Triple(R.string.context_usage_instruction, breakdown.instructionTokens, resolveThemeColor(R.attr.colorSendFill)),
            Triple(R.string.context_usage_summary, breakdown.summaryTokens, ContextCompat.getColor(this, R.color.context_summary)),
            Triple(R.string.context_usage_documents, breakdown.documentTokens, ContextCompat.getColor(this, R.color.context_documents)),
            Triple(R.string.context_usage_conversation, breakdown.conversationTokens, ContextCompat.getColor(this, R.color.context_conversation))
        )
        val bar = view.findViewById<LinearLayout>(R.id.contextUsageBar)
        bar.clipToOutline = true
        parts.filter { it.second > 0 }.forEach { (_, tokens, color) ->
            bar.addView(View(this).apply {
                setBackgroundColor(color)
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, tokens.toFloat())
            })
        }
        val free = window - used
        if (free > 0) {
            bar.addView(View(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, free.toFloat())
            })
        }

        val rows = view.findViewById<LinearLayout>(R.id.contextUsageRows)
        parts.filter { it.second > 0 || it.first == R.string.context_usage_conversation }
            .forEach { (label, tokens, color) -> rows.addView(contextUsageRow(label, tokens, color)) }
        rows.addView(contextUsageRow(R.string.context_usage_free, free, resolveThemeColor(com.google.android.material.R.attr.colorOutlineVariant), outlined = true))

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.context_usage_description)
            .setView(view)
            .setPositiveButton(android.R.string.ok, null)
            .show()
    }

    private fun contextUsageRow(label: Int, tokens: Int, color: Int, outlined: Boolean = false): View {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(6), 0, dp(6))
            addView(View(this@PocketChatActivity).apply {
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(color)
                    if (outlined) setStroke(dp(1), resolveThemeColor(com.google.android.material.R.attr.colorOutline))
                }
                layoutParams = LinearLayout.LayoutParams(dp(10), dp(10))
            })
            addView(TextView(this@PocketChatActivity).apply {
                text = getString(label)
                setTextColor(resolveThemeColor(R.attr.colorAssistantText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginStart = dp(12)
                }
            })
            addView(TextView(this@PocketChatActivity).apply {
                text = getString(R.string.context_usage_tokens, tokens)
                setTextColor(resolveThemeColor(R.attr.colorStatusText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                fontFeatureSettings = "tnum"
            })
        }
    }

    private fun contextUsagePercent(state: ChatUiState): Int =
        (state.contextUsedTokens.toLong() * 100 / state.contextWindowTokens.coerceAtLeast(1)).toInt().coerceIn(0, 100)

    private fun renderNoControllerState(
        message: String,
        preserveTranscript: Boolean
    ) {
        title = getString(R.string.toolbar_app_title)
        toolbarSubtitleView.text = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
        updateModelSelectorLabel()
        thinkingToggleContainer.visibility = View.GONE
        currentModelSupportsThinking = false
        newChatButton.isEnabled = false
        findViewById<View>(R.id.toolbarNewChat).apply { isEnabled = false; alpha = 0.4f }
        updateChatToolChips()
        sendButton.visibility = View.VISIBLE
        stopButton.visibility = View.GONE
        sendButton.isEnabled = false
        stopButton.isEnabled = false
        applyImageModeComposer()
        contextUsageIndicator.visibility = View.GONE
        // With the no-model hint in the middle of the screen, the same request is not repeated here.
        val shown = if (message == getString(R.string.model_required_message) && resources.getBoolean(R.bool.show_no_model_hint)) "" else message
        statusView.text = shown
        statusView.visibility = if (shown.isBlank()) View.GONE else View.VISIBLE
        applyStatusBackground(shown)
        if (!preserveTranscript) {
            chatAdapter.submitTurns(emptyList())
        }
        autoScrollDuringGeneration = false
        autoScrollPendingFinalUpdate = false
        wasGenerating = false
    }

    private fun updateNoModelHint() {
        if (!::noModelHint.isInitialized) return
        val state = chatController?.state?.value
        val show = resources.getBoolean(R.bool.show_no_model_hint) &&
            state?.isReady != true && state?.isLoading != true &&
            modelPreparationJob?.isActive != true && activeDownloadModelId == null
        noModelHint.visibility = if (show) View.VISIBLE else View.GONE
    }

    private fun updateModelSelectorLabel() {
        updateNoModelHint()
        if (!::selectedModelLabel.isInitialized) return
        val modelName = currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
        // Thinking has its own chip in the input pill, so the dropdown shows only the model name.
        selectedModelLabel.text = modelName
        toolbarModelSelector.contentDescription = getString(
            R.string.model_popup_selected_format,
            selectedModelLabel.text,
            if (currentImageInputMode == ImageInputMode.GEMMA_DIRECT) {
                getString(R.string.gemma_direct_image_model_name)
            } else {
                getString(R.string.ocr_image_model_name)
            }
        )
    }

    /**
     * Choose a model: a half-height bottom sheet whose list scrolls. Groups: Chat, Image input, Image creation,
     * Web search, Not downloaded (opens Manage models) and More (Import from Hugging Face).
     */
    private fun showModelPickerPopup() {
        modelPickerPopup?.dismiss()
        val content = layoutInflater.inflate(R.layout.popup_model_picker, null)
        val options: LinearLayout = content.findViewById(R.id.modelPickerOptions)
        val scroll: View = content.findViewById(R.id.modelPickerScroll)
        val popup = BottomSheetDialog(this)
        popup.setContentView(content)
        modelPickerPopup = popup
        popup.setOnDismissListener {
            if (modelPickerPopup === popup) modelPickerPopup = null
        }
        val accentColor = resolveThemeColor(com.google.android.material.R.attr.colorPrimary)
        val onSelectedColor = resolveThemeColor(com.google.android.material.R.attr.colorOnSecondaryContainer)

        fun addHeading(title: String) {
            options.addView(TextView(this).apply {
                text = title.uppercase(java.util.Locale.getDefault())
                setTextColor(accentColor)
                typeface = ResourcesCompat.getFont(this@PocketChatActivity, R.font.space_grotesk_medium)
                letterSpacing = 0.06f
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                setPadding(dp(12), dp(12), dp(12), dp(6))
            })
        }

        fun addRow(
            title: String,
            detail: String? = null,
            initials: String? = null,
            icon: Int? = null,
            selected: Boolean = false,
            enabled: Boolean = true,
            actionText: String? = null,
            chevron: Boolean = false,
            onClick: () -> Unit
        ) {
            val row = layoutInflater.inflate(R.layout.item_model_sheet_row, options, false)
            val titleView: TextView = row.findViewById(R.id.modelRowTitle)
            val detailView: TextView = row.findViewById(R.id.modelRowDetail)
            val initialsView: TextView = row.findViewById(R.id.modelRowInitials)
            val iconView: ImageView = row.findViewById(R.id.modelRowIcon)
            val actionView: TextView = row.findViewById(R.id.modelRowAction)
            val trailing: ImageView = row.findViewById(R.id.modelRowTrailing)
            titleView.text = title
            if (detail != null) {
                detailView.text = detail
                detailView.visibility = View.VISIBLE
            }
            if (icon != null) {
                iconView.setImageResource(icon)
                iconView.visibility = View.VISIBLE
            } else {
                initialsView.text = initials ?: title.take(1)
            }
            if (actionText != null) {
                actionView.text = actionText
                actionView.visibility = View.VISIBLE
            }
            when {
                selected -> {
                    trailing.setImageResource(R.drawable.ic_check_20)
                    trailing.imageTintList = ColorStateList.valueOf(onSelectedColor)
                    trailing.visibility = View.VISIBLE
                    row.background = ContextCompat.getDrawable(this, R.drawable.bg_model_row_selected)
                    titleView.setTextColor(onSelectedColor)
                    detailView.setTextColor(onSelectedColor)
                    detailView.alpha = 0.8f
                }
                chevron -> {
                    trailing.setImageResource(R.drawable.ic_chevron_right_16)
                    trailing.imageTintList = ColorStateList.valueOf(resolveThemeColor(R.attr.colorStatusText))
                    trailing.visibility = View.VISIBLE
                }
            }
            row.isEnabled = enabled
            row.alpha = if (enabled) 1f else 0.6f
            row.contentDescription = if (selected) "$title, ${getString(R.string.model_status_current)}" else title
            if (enabled) row.setOnClickListener {
                popup.dismiss()
                onClick()
            }
            options.addView(row)
        }

        fun initialsOf(descriptor: ModelDescriptor): String {
            val name = descriptor.shortName()
            val digit = name.firstOrNull { it.isDigit() }?.toString() ?: name.getOrNull(1)?.toString().orEmpty()
            return name.take(1).uppercase(java.util.Locale.getDefault()) + digit
        }

        // Image input pills stay pinned above the list.
        val pills: LinearLayout = content.findViewById(R.id.modelPickerPills)
        fun addPill(label: String, hint: String, selected: Boolean, enabled: Boolean, onPick: () -> Unit) {
            pills.addView(TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, dp(54), 1f)
                    .apply { if (pills.childCount > 0) marginStart = dp(8) }
                gravity = Gravity.CENTER
                setPadding(dp(8), 0, dp(8), 0)
                val onFill = resolveThemeColor(com.google.android.material.R.attr.colorOnPrimary)
                val main = if (selected) onFill else resolveThemeColor(R.attr.colorAssistantText)
                text = android.text.SpannableStringBuilder().apply {
                    append(label)
                    setSpan(android.text.style.StyleSpan(Typeface.BOLD), 0, length, 0)
                    setSpan(android.text.style.ForegroundColorSpan(main), 0, length, 0)
                    val start = length + 1
                    append('\n').append(hint)
                    setSpan(android.text.style.RelativeSizeSpan(0.8f), start, length, 0)
                    setSpan(android.text.style.ForegroundColorSpan(if (selected) onFill else resolveThemeColor(R.attr.colorStatusText)), start, length, 0)
                }
                textSize = 14f
                background = GradientDrawable().apply {
                    cornerRadius = dp(27).toFloat()
                    if (selected) setColor(accentColor) else setStroke(dp(1), resolveThemeColor(R.attr.colorPocketOutline))
                }
                alpha = if (enabled) 1f else 0.45f
                isEnabled = enabled
                if (enabled) setOnClickListener {
                    popup.dismiss()
                    onPick()
                }
            })
        }
        addPill(getString(R.string.model_sheet_pill_ocr), getString(R.string.model_sheet_pill_ocr_hint), currentImageInputMode == ImageInputMode.OCR, true) {
            handleOcrImageModelSelection()
        }
        val nativeSupported = isGemmaDirectImageInputSupportedBySelectedModel()
        addPill(
            getString(R.string.model_sheet_pill_native),
            getString(R.string.model_sheet_pill_native_hint),
            nativeSupported && currentImageInputMode == ImageInputMode.GEMMA_DIRECT,
            nativeSupported && isGemmaDirectImageInputAvailable()
        ) { handleGemmaDirectImageModelSelection() }

        // Downloaded chat models first, then the curated list to download, all in one scrolling list.
        val (availableModels, notOnPhone) = ModelRegistry.all.partition(modelFileResolver::isModelAvailable)
        val recommendedTier = ModelTier.recommendedFor(this)
        val toDownload = notOnPhone.sortedBy { if (ModelTier.of(it) == recommendedTier) 0 else 1 }
        fun addDownloadRow(descriptor: ModelDescriptor) {
            val detail = listOfNotNull(
                ModelTier.of(descriptor)?.let { getString(it.labelRes) },
                descriptor.sizeLabel,
                if (descriptor is GgufSpec) getString(R.string.pro_chip) else null
            ).joinToString(" · ")
            addRow(
                descriptor.displayName,
                detail,
                initials = initialsOf(descriptor),
                actionText = getString(R.string.model_action_download)
            ) { startModelDownload(descriptor) }
        }
        if (availableModels.isEmpty()) {
            addHeading(getString(R.string.model_sheet_empty_recommended))
            val suggested = ModelRegistry.gemma4E2B
            addDownloadRow(suggested)
            val rest = toDownload.filter { it.id != suggested.id }
            if (rest.isNotEmpty()) {
                addHeading(getString(R.string.model_sheet_more_models))
                rest.forEach(::addDownloadRow)
            }
        } else {
            addHeading(getString(R.string.model_section_on_phone))
            availableModels.forEach { descriptor ->
                val loaded = descriptor.id == currentModel?.id && chatController != null
                val tier = ModelTier.of(descriptor)?.let { getString(it.labelRes) }
                val detail = listOfNotNull(
                    if (loaded) getString(R.string.model_sheet_loaded) else null,
                    tier,
                    descriptor.sizeLabel
                ).joinToString(" · ")
                addRow(descriptor.displayName, detail, initials = initialsOf(descriptor), selected = loaded) {
                    handleModelSelection(descriptor)
                }
            }
            if (toDownload.isNotEmpty()) {
                addHeading(getString(R.string.model_section_download))
                toDownload.forEach(::addDownloadRow)
            }
        }

        content.findViewById<View>(R.id.modelPickerImport).setOnClickListener {
            popup.dismiss()
            showCustomModelWarning()
        }
        content.findViewById<View>(R.id.modelPickerHf).apply {
            if (LlamaFeature.isAvailable) {
                setOnClickListener {
                    popup.dismiss()
                    LlamaFeature.instance?.launchHfBrowser(this@PocketChatActivity, hfBrowserLauncher)
                }
            } else {
                visibility = View.GONE
            }
        }

        content.findViewById<View>(R.id.modelPickerManageLink).setOnClickListener {
            popup.dismiss()
            showManageModelsDialog()
        }

        // Up to three quarters of the screen; the list scrolls inside, and a short list keeps the sheet small.
        val halfScreen = (resources.displayMetrics.heightPixels * 0.75f).toInt()
        options.measure(
            View.MeasureSpec.makeMeasureSpec(resources.displayMetrics.widthPixels, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED)
        )
        scroll.layoutParams = scroll.layoutParams.apply {
            height = minOf(options.measuredHeight, halfScreen - dp(200))
        }
        popup.setOnShowListener {
            popup.behavior.skipCollapsed = true
            popup.behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
        }
        popup.show()
    }

    private fun showManageModelsDialog() {
        val existingDialog = modelDialogViews?.dialog
        if (existingDialog != null && existingDialog.isShowing) {
            refreshModelSelectionDialog()
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_model_selection, null)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()
        dialog.setOnShowListener {
            dialog.window?.apply {
                setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                val pageColor = resolveThemeColor(R.attr.colorAppBackground)
                setBackgroundDrawable(ColorDrawable(pageColor))
                setDimAmount(0f)
                statusBarColor = pageColor
                navigationBarColor = pageColor
            }
        }
        dialogView.findViewById<View>(R.id.modelPickerBack).setOnClickListener { dialog.dismiss() }
        dialogView.findViewById<View>(R.id.modelPickerHelp).setOnClickListener {
            MaterialAlertDialogBuilder(this)
                .setTitle(R.string.model_picker_title)
                .setMessage(R.string.manage_models_intro)
                .setPositiveButton(android.R.string.ok, null)
                .show()
        }

        val dialogUi = ModelDialogViews(
            dialog = dialog,
            introView = dialogView.findViewById(R.id.modelPickerIntro),
            listContainer = dialogView.findViewById(R.id.modelListContainer)
        )

        dialogView.findViewById<View>(R.id.importLocalModelButton).setOnClickListener {
            showCustomModelWarning()
        }
        val addFromHuggingFaceButton = dialogView.findViewById<View>(R.id.addFromHuggingFaceButton)
        if (LlamaFeature.isAvailable) {
            addFromHuggingFaceButton.setOnClickListener {
                dialog.dismiss()
                LlamaFeature.instance?.launchHfBrowser(this, hfBrowserLauncher)
            }
        } else {
            addFromHuggingFaceButton.visibility = View.GONE
        }

        dialog.setOnDismissListener {
            modelDialogViews = null
        }

        modelDialogViews = dialogUi
        refreshModelSelectionDialog()
        dialog.show()
    }

    private fun showModelRecoveryDialog() {
        val record = modelLoadCoordinator.record?.takeIf { it.phase == ModelLoadPhase.FAILED } ?: return
        val descriptor = ModelRegistry.findById(record.modelId) ?: return
        if (modelRecoveryDialog?.isShowing == true) return
        val choices = arrayOf(
            getString(R.string.model_recovery_choose_another),
            getString(R.string.model_recovery_delete),
            getString(R.string.model_recovery_retry),
            getString(R.string.model_recovery_continue)
        )
        modelRecoveryDialog = MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.model_recovery_title))
            .setMessage(
                getString(
                    when {
                        record.failureReason?.contains("Some ops are not accelerated", ignoreCase = true) == true ->
                            R.string.model_recovery_accelerator_message
                        else -> when (ModelLoadFailureKind.of(record.failureReason)) {
                            ModelLoadFailureKind.UNSUPPORTED_ARCHITECTURE -> R.string.model_recovery_unsupported_message
                            ModelLoadFailureKind.OUT_OF_MEMORY -> R.string.model_recovery_memory_message
                            ModelLoadFailureKind.DAMAGED_FILE -> R.string.model_recovery_damaged_message
                            ModelLoadFailureKind.OTHER -> R.string.model_recovery_message
                        }
                    },
                    descriptor.displayName,
                    record.failureReason ?: getString(R.string.model_load_failed_generic)
                )
            )
            .setItems(choices) { dialog, which ->
                when (which) {
                    0 -> {
                        dialog.dismiss()
                        showModelPickerPopup()
                    }
                    1 -> {
                        dialog.dismiss()
                        confirmDeleteModel(descriptor)
                    }
                    2 -> {
                        dialog.dismiss()
                        modelLoadCoordinator.clearForModel(descriptor.id)
                        persistModelLoadRecord()
                        requestModelLoad(
                            descriptor,
                            activeChatSnapshot = chatController?.snapshotActiveChat()
                                ?: retainedState.pendingModelLoadSnapshot,
                        )
                    }
                    3 -> {
                        dialog.dismiss()
                        detachFailedController(descriptor)
                    }
                }
            }
            .setOnDismissListener { modelRecoveryDialog = null }
            .create()
        modelRecoveryDialog?.show()
    }

    private fun detachFailedController(descriptor: ModelDescriptor) {
        controllerStateJob?.cancel()
        chatController?.close()
        chatController = null
        retainedState.chatController = null
        retainedState.modelId = descriptor.id
        currentModel = descriptor
        renderNoControllerState(getString(R.string.model_recovery_continued), preserveTranscript = true)
    }

    private fun refreshModelSelectionDialog() {
        val dialogUi = modelDialogViews ?: return
        dialogUi.dialog.setCancelable(true)
        dialogUi.dialog.setCanceledOnTouchOutside(true)
        dialogUi.introView.text = getString(R.string.manage_models_intro)

        dialogUi.listContainer.removeAllViews()
        downloadRowProgress = null
        downloadRowModelId = null
        val inflater = LayoutInflater.from(this)
        val (onPhone, notOnPhone) = ModelRegistry.all.partition(modelFileResolver::isModelAvailable)
        // The recommended tier comes first, as in the mock-up; the rest keep the registry order.
        val recommendedTier = ModelTier.recommendedFor(this)
        val toDownload = notOnPhone.sortedBy { if (ModelTier.of(it) == recommendedTier) 0 else 1 }
        if (onPhone.isNotEmpty()) {
            addModelSectionHeader(dialogUi.listContainer, getString(R.string.model_section_on_phone))
            modelCardTarget = newModelCard(dialogUi.listContainer)
            onPhone.forEach { addTextModelOption(inflater, dialogUi.listContainer, it) }
        }
        if (toDownload.isNotEmpty()) {
            addModelSectionHeader(dialogUi.listContainer, getString(R.string.model_section_download))
            modelCardTarget = newModelCard(dialogUi.listContainer)
            toDownload.forEach { addTextModelOption(inflater, dialogUi.listContainer, it) }
        }
        if (ImageGenFeature.instance?.isCpuSupported() == true) {
            addModelSectionHeader(dialogUi.listContainer, getString(R.string.image_gen_manage_heading))
            modelCardTarget = newModelCard(dialogUi.listContainer)
            ImageGenFeature.instance?.addDownloadableModelOptions(chatHost)
        }
        modelCardTarget = null
    }

    // Image mode: the composer creates pictures with the image model instead of sending a chat message.

    /** Runs after every chat state update so image mode keeps control of the Send and Stop buttons. */
    private fun applyImageModeComposer() {
        ImageGenFeature.instance?.applyImageModeComposer(chatHost)
    }

    private fun showImageTurnActions(turn: ChatTurn) {
        val path = turn.imagePath ?: return
        MaterialAlertDialogBuilder(this)
            .setItems(
                arrayOf(
                    getString(R.string.image_turn_attach),
                    getString(R.string.image_turn_open),
                    getString(R.string.image_turn_report)
                )
            ) { _, which ->
                when (which) {
                    0 -> attachSavedImageByPath(path)
                    1 -> openImageViewer(path)
                    else -> reportGeneratedImage(turn)
                }
            }
            .show()
    }

    private fun reportGeneratedImage(turn: ChatTurn) {
        ImageGenFeature.instance?.reportGeneratedImage(chatHost, turn)
    }

    private fun attachSavedImageByPath(path: String) {
        val entry = savedImageStore.list()
            .firstOrNull { it.filePath == path }
            ?.let { savedImageStore.load(it.imageId) }
        if (entry == null) {
            showTransientMessage(getString(R.string.saved_image_missing))
            return
        }
        if (ImageGenFeature.instance?.isImageModeActive() == true) ImageGenFeature.instance?.setImageMode(chatHost, false)
        addPendingImageInput(entry)
    }

    private fun openModelCard(url: String) {
        runCatching { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
            .onFailure { showTransientMessage(url) }
    }

    private fun openImageGeneration(specId: String? = null, toolName: String? = null, photoPath: String? = null) {
        ImageGenFeature.instance?.openImageGeneration(chatHost, specId, toolName, photoPath)
    }

    private fun magicActionsNote(): ChatAdapter.MagicNote {
        if (ProUiFeature.isAvailable) {
            return ProUiFeature.instance!!.magicActionsNote(this, proGate)
        }
        return ChatAdapter.MagicNote(getString(R.string.magic_note_pro), showPro = false)
    }

    private fun addModelSectionHeader(container: LinearLayout, title: String) {
        container.addView(
            TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply {
                    topMargin = if (container.childCount == 0) 0 else dp(18)
                    bottomMargin = dp(8)
                    marginStart = dp(2)
                }
                text = title
                setTextColor(resolveThemeColor(R.attr.colorStatusText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                letterSpacing = 0.02f
            }
        )
    }

    /** One rounded card per section; its rows are separated by hairlines. */
    private fun newModelCard(container: LinearLayout): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundResource(R.drawable.bg_card_line)
            setPadding(dp(12), dp(4), dp(12), dp(4))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            container.addView(this)
        }

    private fun addTextModelOption(
        inflater: LayoutInflater,
        container: LinearLayout,
        descriptor: ModelDescriptor
    ) {
        addDownloadableModelOption(
            inflater = inflater,
            container = container,
            descriptor = descriptor,
            isCurrentModel = currentModel?.id == descriptor.id && chatController != null,
            onUseModel = { handleModelSelection(descriptor) }
        )
    }

    private fun addDownloadableModelOption(
        inflater: LayoutInflater,
        container: LinearLayout,
        descriptor: ModelDescriptor,
        isCurrentModel: Boolean,
        onUseModel: () -> Unit
    ) {
        val parent = modelCardTarget ?: container
        val itemView = inflater.inflate(R.layout.item_model_option, parent, false)
        itemView.findViewById<View>(R.id.modelRowDivider).visibility =
            if (parent.childCount == 0) View.GONE else View.VISIBLE

        val tier = ModelTier.of(descriptor)
        val isDownloadingThisModel = activeDownloadModelId == descriptor.id
        val isDownloaded = modelFileResolver.isModelDownloaded(descriptor)
        val isAvailable = modelFileResolver.isModelAvailable(descriptor)
        val accent = resolveThemeColor(R.attr.colorSendFill)
        val text = resolveThemeColor(R.attr.colorAssistantText)

        // Icon, tier label and name
        val iconBox: View = itemView.findViewById(R.id.modelIconBox)
        val icon: ImageView = itemView.findViewById(R.id.modelIcon)
        icon.setImageResource(
            when (tier) {
                ModelTier.FAST -> R.drawable.ic_tier_bolt
                ModelTier.BALANCED -> R.drawable.ic_tier_balanced
                ModelTier.BEST -> R.drawable.ic_tier_diamond
                ModelTier.REASONING -> R.drawable.ic_tier_bulb
                null -> R.drawable.ic_sparkle_20
            }
        )
        iconBox.setBackgroundResource(if (isCurrentModel) R.drawable.bg_model_icon_acc else R.drawable.bg_model_icon)
        icon.imageTintList = ColorStateList.valueOf(if (isCurrentModel) accent else text)

        val recommended = tier != null && tier == ModelTier.recommendedFor(this) && !isAvailable
        val labelParts = listOfNotNull(
            tier?.let { getString(it.labelRes).uppercase() },
            if (isCurrentModel) getString(R.string.model_section_in_use) else null,
            if (recommended) getString(R.string.model_recommended_short) else null
        )
        itemView.findViewById<TextView>(R.id.modelTierLabel).apply {
            visibility = if (labelParts.isEmpty()) View.GONE else View.VISIBLE
            this.text = labelParts.joinToString(" · ")
        }
        itemView.findViewById<TextView>(R.id.modelName).text =
            if (tier != null) descriptor.shortName() else descriptor.displayName

        // Right side: size when in use, Use when on the phone, Get with the size otherwise, a ring while downloading
        val actionButton: MaterialButton = itemView.findViewById(R.id.modelActionButton)
        val metaView: TextView = itemView.findViewById(R.id.modelMeta)
        val ring: ProgressBar = itemView.findViewById(R.id.modelItemProgressBar)
        val progressText: TextView = itemView.findViewById(R.id.modelItemProgressText)
        itemView.findViewById<TextView>(R.id.modelStatus).text = when {
            isDownloadingThisModel -> getString(R.string.model_status_downloading)
            isCurrentModel -> getString(R.string.model_status_current)
            isDownloaded -> getString(R.string.model_status_downloaded)
            isAvailable -> getString(R.string.model_status_bundled)
            else -> getString(R.string.model_status_not_downloaded)
        }
        actionButton.visibility = View.GONE
        metaView.visibility = View.GONE
        ring.visibility = View.GONE
        progressText.visibility = View.GONE
        when {
            isDownloadingThisModel -> {
                ring.visibility = View.VISIBLE
                progressText.visibility = View.VISIBLE
                updateProgressBar(ring, activeDownloadBytes, activeDownloadTotalBytes)
                progressText.text = formatDownloadProgressText(activeDownloadBytes, activeDownloadTotalBytes)
                downloadRowProgress = ring to progressText
                downloadRowModelId = descriptor.id
            }
            isCurrentModel -> {
                metaView.visibility = View.VISIBLE
                metaView.text = descriptor.sizeLabel
            }
            isAvailable -> {
                actionButton.visibility = View.VISIBLE
                actionButton.text = getString(R.string.model_action_use)
                actionButton.icon = null
                styleModelOption(actionButton, ModelOptionAction.USE)
                actionButton.setOnClickListener { onUseModel() }
            }
            else -> {
                actionButton.visibility = View.VISIBLE
                actionButton.text = descriptor.sizeLabel
                actionButton.setIconResource(R.drawable.ic_download_16)
                actionButton.isEnabled = activeDownloadModelId == null
                styleModelOption(actionButton, ModelOptionAction.DOWNLOAD)
                actionButton.setOnClickListener {
                    onUseModel()
                    if (activeDownloadModelId == descriptor.id) showDownloadSheet(descriptor)
                }
            }
        }

        // Details open under the row on a tap; a running download opens its sheet instead.
        val detailsBox: View = itemView.findViewById(R.id.modelDetailsBox)
        val modelLicense = descriptor.license
        itemView.findViewById<TextView>(R.id.modelDetailsText).text =
            tier?.let { getString(it.blurbRes) } ?: descriptor.deviceRecommendation
        itemView.findViewById<TextView>(R.id.modelDetailRunsOn).text = descriptor.backendLabel
        itemView.findViewById<TextView>(R.id.modelLicense).apply {
            if (modelLicense == null) {
                itemView.findViewById<View>(R.id.modelLicenseLabel).visibility = View.GONE
                visibility = View.GONE
            } else {
                this.text = getString(R.string.model_licence_value, modelLicense.summary().removePrefix("License: "))
                modelLicense.url?.let { url -> setOnClickListener { openModelCard(url) } }
            }
        }
        itemView.findViewById<TextView>(R.id.modelDeleteText).apply {
            visibility = if (isDownloaded && !isDownloadingThisModel) View.VISIBLE else View.GONE
            setOnClickListener { confirmDeleteModel(descriptor) }
        }
        // The recommended model opens its details by default, as in the mock-up.
        if (recommended) detailsBox.visibility = View.VISIBLE
        itemView.findViewById<View>(R.id.modelRowMain).setOnClickListener {
            if (isDownloadingThisModel) {
                showDownloadSheet(descriptor)
            } else {
                detailsBox.visibility = if (detailsBox.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            }
        }

        parent.addView(itemView)
    }

    /** The mock-up's download sheet: a ring with time left, the details, and Cancel. */
    private fun showDownloadSheet(descriptor: ModelDescriptor) {
        downloadSheet?.dismiss()
        val sheet = BottomSheetDialog(this)
        val view = layoutInflater.inflate(R.layout.sheet_model_download, null)
        val tier = ModelTier.of(descriptor)
        view.findViewById<ImageView>(R.id.sheetIcon).setImageResource(
            when (tier) {
                ModelTier.FAST -> R.drawable.ic_tier_bolt
                ModelTier.BALANCED -> R.drawable.ic_tier_balanced
                ModelTier.BEST -> R.drawable.ic_tier_diamond
                ModelTier.REASONING -> R.drawable.ic_tier_bulb
                null -> R.drawable.ic_sparkle_20
            }
        )
        view.findViewById<TextView>(R.id.sheetTier).apply {
            visibility = if (tier == null) View.GONE else View.VISIBLE
            text = tier?.let { getString(it.labelRes).uppercase() }
        }
        view.findViewById<TextView>(R.id.sheetName).text = if (tier != null) descriptor.shortName() else descriptor.displayName
        view.findViewById<TextView>(R.id.sheetSize).text = descriptor.sizeLabel
        view.findViewById<TextView>(R.id.sheetRunsOn).text = descriptor.backendLabel
        val license = descriptor.license
        view.findViewById<TextView>(R.id.sheetLicence).text = license?.summary()?.removePrefix("License: ").orEmpty()
        view.findViewById<View>(R.id.sheetCardRow).apply {
            visibility = if (license == null) View.GONE else View.VISIBLE
            license?.url?.let { url -> setOnClickListener { openModelCard(url) } }
        }
        val stop = View.OnClickListener {
            cancelModelDownload()
            sheet.dismiss()
        }
        view.findViewById<View>(R.id.sheetCancel).setOnClickListener(stop)
        view.findViewById<View>(R.id.sheetRingBox).setOnClickListener(stop)
        downloadSheetViews = DownloadSheetViews(
            ring = view.findViewById(R.id.sheetRing),
            title = view.findViewById(R.id.sheetTitle),
            sub = view.findViewById(R.id.sheetSub)
        )
        sheet.setContentView(view)
        sheet.setOnShowListener {
            (view.parent as? View)?.setBackgroundColor(Color.TRANSPARENT)
            sheet.behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
        }
        sheet.setOnDismissListener {
            if (downloadSheet === sheet) {
                downloadSheet = null
                downloadSheetViews = null
            }
        }
        downloadSheet = sheet
        updateDownloadSheet()
        sheet.show()
    }

    private fun updateDownloadSheet() {
        val views = downloadSheetViews ?: return
        val total = activeDownloadTotalBytes
        updateProgressBar(views.ring, activeDownloadBytes, total)
        if (total != null && total > 0L) {
            val percent = (activeDownloadBytes * 100L / total).toInt().coerceIn(0, 100)
            views.title.text = getString(R.string.model_sheet_downloading, percent)
            val left = downloadTimeLeftText(activeDownloadBytes, total)
                ?.replaceFirstChar { it.uppercase() }
                ?: formatDownloadProgressText(activeDownloadBytes, total)
            views.sub.text = getString(R.string.model_sheet_sub, left)
        } else {
            views.title.text = getString(R.string.model_sheet_preparing)
            views.sub.text = ""
        }
    }

    private enum class ModelOptionAction { CURRENT, USE, DOWNLOAD, STOP }

    private fun styleModelOption(button: MaterialButton, action: ModelOptionAction) {
        val soft = resolveThemeColor(R.attr.colorUserBubble)
        val text = resolveThemeColor(R.attr.colorAssistantText)
        val outline = resolveThemeColor(R.attr.colorPocketOutline)
        val (fill, stroke) = when (action) {
            ModelOptionAction.DOWNLOAD -> soft to Color.TRANSPARENT
            else -> Color.TRANSPARENT to outline
        }
        button.backgroundTintList = ColorStateList.valueOf(fill)
        button.strokeColor = ColorStateList.valueOf(stroke)
        button.setTextColor(text)
        button.iconTint = ColorStateList.valueOf(text)
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
            onMenuRequested = { session, anchor ->
                showSessionMenu(session, anchor)
            }
        )
        drawerChatsRecyclerView.layoutManager = LinearLayoutManager(this)
        drawerChatsRecyclerView.adapter = drawerSessionsAdapter
        findViewById<EditText>(R.id.drawerSearchChats).addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
            override fun afterTextChanged(s: android.text.Editable?) = refreshDrawerSessions()
        })

        val settingsPage = findViewById<View>(R.id.settingsPage)
        val backCallback = object : androidx.activity.OnBackPressedCallback(false) {
            override fun handleOnBackPressed() {
                settingsPage.visibility = View.GONE
                settingsPageOpen = false
                isEnabled = false
            }
        }
        onBackPressedDispatcher.addCallback(this, backCallback)
        fun openSettingsPage() {
            refreshSettingsPage()
            settingsPage.visibility = View.VISIBLE
            settingsPageOpen = true
            backCallback.isEnabled = true
        }
        findViewById<View>(R.id.drawerOpenSettingsRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            openSettingsPage()
        }
        // The Play build has its own Pro row; the open-source build links to the Play version.
        findViewById<View>(R.id.drawerGetPlayRow).apply {
            visibility = if (resources.getBoolean(R.bool.show_rate_on_play)) View.GONE else View.VISIBLE
            setOnClickListener {
                drawerLayout.closeDrawer(GravityCompat.START)
                openPlayStorePage()
            }
        }
        findViewById<View>(R.id.settingsPageBack).setOnClickListener { backCallback.handleOnBackPressed() }
        bindSettingsPage()
        if (settingsPageOpen) openSettingsPage()

        findViewById<View>(R.id.drawerProStatus).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                proGate.showProPage(this) { refreshProBadges() }
            }
        }

        drawerLayout.addDrawerListener(object : DrawerLayout.SimpleDrawerListener() {
            override fun onDrawerOpened(drawerView: View) {
                refreshProBadges()
                if (proGate.isUnlocked()) ProUiFeature.instance?.playShine(findViewById<View>(R.id.drawerTitlePro))
            }
        })

        findViewById<View>(R.id.drawerLanServerRow).visibility =
            if (LanServerFeature.isAvailable) View.VISIBLE else View.GONE
        findViewById<View>(R.id.drawerLanServerRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                showLanServerDialog()
            }
        }

        findViewById<View>(R.id.drawerImagesRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                showImageLibraryDialog()
            }
        }

        findViewById<View>(R.id.drawerImageToolsRow).visibility =
            if (ImageGenFeature.isAvailable) View.VISIBLE else View.GONE
        findViewById<View>(R.id.drawerImageToolsRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                openImageGeneration()
            }
        }

        findViewById<View>(R.id.drawerWebSearchRow).visibility =
            if (WebSearchFeature.isAvailable) View.VISIBLE else View.GONE
        findViewById<View>(R.id.drawerWebSearchRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                requireProOrTrial(ProFeature.WEB_SEARCH) { showManageWebSearchDialog() }
            }
        }

        findViewById<View>(R.id.drawerTranslatorRow).visibility =
            if (TranslatorFeature.isAvailable) View.VISIBLE else View.GONE
        findViewById<View>(R.id.drawerTranslatorRow).setOnClickListener {
            drawerLayout.closeDrawer(GravityCompat.START)
            drawerLayout.post {
                openTranslator()
            }
        }

    }

    /** Menu A for free users (PRO tags, free tries, lock when used up), Menu C for owners (gold icons, live status). */
    private fun refreshProBadges() {
        if (ProUiFeature.isAvailable) {
            ProUiFeature.instance!!.refreshProBadges(this, proGate)
        } else {
            findViewById<View>(R.id.drawerProStatus).visibility = View.GONE
            findViewById<View>(R.id.drawerTitlePro).visibility = View.GONE
            listOf(
                R.id.drawerImageToolsProChip, R.id.drawerImageToolsLock,
                R.id.drawerWebSearchProChip, R.id.drawerWebSearchLock,
                R.id.drawerTranslatorProChip, R.id.drawerTranslatorLock
            ).forEach { findViewById<View>(it).visibility = View.GONE }

            val lan = LanServerFeature.instance?.drawerState(this)
            val lanStatus = when (lan) {
                LanServerFeature.DrawerState.RUNNING -> getString(R.string.drawer_lan_running)
                LanServerFeature.DrawerState.STARTING -> getString(R.string.drawer_lan_starting)
                else -> null
            }
            val webStatus = if (WebSearchFeature.instance?.isUsable(this) == true) getString(R.string.drawer_web_search_on) else null

            fun updateRow(statusId: Int, status: String?, live: Boolean) {
                val statusView = findViewById<TextView>(statusId)
                statusView.text = status
                statusView.visibility = if (status == null) View.GONE else View.VISIBLE
                statusView.setTextColor(if (live) getColor(R.color.status_running) else resolveThemeColor(R.attr.colorStatusText))
            }
            updateRow(R.id.drawerLanServerStatus, lanStatus, live = true)
            updateRow(R.id.drawerImageToolsStatus, null, live = false)
            updateRow(R.id.drawerWebSearchStatus, webStatus, live = false)
        }
        updateWebSearchButton()
        updateChatToolChips()
        if (ImageGenFeature.instance?.isImageModeActive() != true) {
            createImageModeButton.text = proLabel(R.string.image_tools, ProFeature.IMAGE_GENERATION)
        }
        if (::chatAdapter.isInitialized) chatAdapter.setMagicNote(magicActionsNote())
    }

    /** Button text with a small "3 free" or "PRO" tag while Pro is locked. */
    private fun proLabel(@StringRes label: Int, feature: ProFeature): CharSequence {
        if (ProUiFeature.isAvailable) {
            return ProUiFeature.instance!!.proLabel(this, proGate, label, feature)
        }
        return getString(label)
    }

    /** Runs [action] when Pro is unlocked or a free try of [feature] is left, otherwise shows the offer. */
    private fun requireProOrTrial(feature: ProFeature, action: () -> Unit) {
        if (ProUiFeature.isAvailable) {
            ProUiFeature.instance!!.requireProOrTrial(this, proGate, feature) {
                refreshProBadges()
                action()
            }
        } else {
            proGate.requireUnlocked(this, feature) {
                refreshProBadges()
                action()
            }
        }
    }

    /** Links from the Welcome to Pro page. */
    override fun openProFeature(feature: ProFeature) {
        when (feature) {
            ProFeature.IMAGE_GENERATION -> openImageGeneration()
            ProFeature.WEB_SEARCH -> showManageWebSearchDialog()
            ProFeature.TRANSLATOR -> openTranslator()
            ProFeature.PRO_MENU -> Unit
        }
    }

    private val lanHost = object : LanServerFeature.Host {
        override val activity: AppCompatActivity get() = this@PocketChatActivity
        override val currentModel: ModelDescriptor? get() = this@PocketChatActivity.currentModel
        override val chatController: PersistentChatController? get() = this@PocketChatActivity.chatController
        override val proGate: ProGate get() = this@PocketChatActivity.proGate
        override fun showTransientMessage(message: String) = this@PocketChatActivity.showTransientMessage(message)
        override fun resolveThemeColor(attrId: Int): Int = this@PocketChatActivity.resolveThemeColor(attrId)
        override fun updateKeepScreenOn() = this@PocketChatActivity.updateKeepScreenOn()
        override val installedModels: List<ModelDescriptor>
            get() = ModelRegistry.all.filter(modelFileResolver::isModelAvailable)
        override fun selectModel(descriptor: ModelDescriptor) = handleModelSelection(descriptor)
        override fun openManageModels() = showManageModelsDialog()
    }

    private fun showLanServerDialog() {
        LanServerFeature.instance?.showDialog(lanHost)
    }

    private fun refreshDrawerSessions() {
        if (!::drawerSessionsAdapter.isInitialized) {
            return
        }

        val query = findViewById<EditText>(R.id.drawerSearchChats).text.toString().trim()
        val all = chatSessionStore.list()
        val sessions = if (query.isEmpty()) all else all.filter { it.title.contains(query, ignoreCase = true) }
        drawerSessionsAdapter.submitList(sessions)
        drawerChatsEmptyView.setText(if (query.isEmpty()) R.string.no_saved_chats else R.string.drawer_search_no_match)
        drawerChatsEmptyView.visibility = if (sessions.isEmpty()) View.VISIBLE else View.GONE
    }

    private var pendingExportSessionId: String? = null

    private val exportTextLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("text/plain")) { uri ->
        uri?.let { writeSessionExport(it, asJson = false) }
    }
    private val exportJsonLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri ->
        uri?.let { writeSessionExport(it, asJson = true) }
    }

    /** The row menu: export as text or JSON, or delete (which still asks first). */
    private fun showSessionMenu(session: ChatSessionSummary, anchor: View) {
        val menu = android.widget.PopupMenu(this, anchor)
        menu.menu.add(0, 1, 0, R.string.chat_export_text)
        menu.menu.add(0, 2, 1, R.string.chat_export_json)
        menu.menu.add(0, 3, 2, R.string.delete)
        menu.setOnMenuItemClickListener { item ->
            val name = session.title.replace(Regex("[^\\p{L}\\p{N} _-]"), "").trim().ifBlank { "chat" }.take(40)
            when (item.itemId) {
                1 -> { pendingExportSessionId = session.sessionId; exportTextLauncher.launch("$name.txt") }
                2 -> { pendingExportSessionId = session.sessionId; exportJsonLauncher.launch("$name.json") }
                else -> confirmDeleteSession(session)
            }
            true
        }
        menu.show()
    }

    private fun writeSessionExport(uri: Uri, asJson: Boolean) {
        val sessionId = pendingExportSessionId ?: return
        pendingExportSessionId = null
        lifecycleScope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching {
                    val content = (if (asJson) chatSessionStore.exportJson(sessionId) else chatSessionStore.exportText(sessionId))
                        ?: error("missing session")
                    contentResolver.openOutputStream(uri, "wt")?.use { it.write(content.toByteArray(Charsets.UTF_8)) } ?: error("no stream")
                }.isSuccess
            }
            showTransientMessage(getString(if (ok) R.string.chat_export_done else R.string.chat_export_failed))
        }
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

    private fun showImageLibraryDialog() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_image_library, null)
        val subtitleView: TextView = dialogView.findViewById(R.id.imageLibrarySubtitle)
        val emptyView: TextView = dialogView.findViewById(R.id.imageLibraryEmpty)
        val recyclerView: RecyclerView = dialogView.findViewById(R.id.imageLibraryRecyclerView)
        val closeButton: Button = dialogView.findViewById(R.id.imageLibraryCloseButton)
        val deleteButton: Button = dialogView.findViewById(R.id.imageLibraryDeleteButton)
        val images = savedImageStore.list()
        val spanCount = ((resources.displayMetrics.widthPixels / resources.displayMetrics.density) / 120f)
            .toInt()
            .coerceAtLeast(2)

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()
        val gridAdapter = SavedImageGridAdapter(
            fontSizeSp = currentSettings.chatFontSizeSp,
            onOpenImage = { image ->
                dialog.dismiss()
                openImageViewer(image.filePath)
            },
            onSelectionChanged = { selectedCount ->
                subtitleView.text = if (selectedCount > 0) {
                    getString(R.string.image_library_selection_count, selectedCount)
                } else {
                    getString(R.string.image_library_hint)
                }
                deleteButton.isEnabled = selectedCount > 0
                deleteButton.text = if (selectedCount > 0) {
                    getString(R.string.delete_selected_images_count, selectedCount)
                } else {
                    getString(R.string.delete_selected_images)
                }
            }
        )

        recyclerView.layoutManager = GridLayoutManager(this, spanCount)
        recyclerView.adapter = gridAdapter
        gridAdapter.submitImages(images)
        emptyView.visibility = if (images.isEmpty()) View.VISIBLE else View.GONE
        recyclerView.visibility = if (images.isEmpty()) View.GONE else View.VISIBLE

        closeButton.setOnClickListener {
            dialog.dismiss()
        }
        deleteButton.setOnClickListener {
            val selectedIds = gridAdapter.selectedIds()
            if (selectedIds.isEmpty()) {
                return@setOnClickListener
            }
            confirmDeleteSavedImages(selectedIds, dialog)
        }

        showPanelDialog(dialog)
    }

    private fun confirmDeleteSavedImages(
        imageIds: Set<String>,
        parentDialog: AlertDialog? = null
    ) {
        if (isImagePreprocessingForSend && pendingImageInputs.any { it.savedImageId in imageIds }) {
            showTransientMessage(getString(R.string.image_input_processing))
            return
        }

        val confirmationMessage = if (imageIds.size == 1) {
            getString(R.string.delete_image_confirmation)
        } else {
            getString(R.string.delete_images_confirmation, imageIds.size)
        }
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.delete_image))
            .setMessage(confirmationMessage)
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                parentDialog?.dismiss()
                deleteSavedImages(imageIds)
            }
            .create()

        dialog.setOnShowListener {
            applyDeleteConfirmationButtonColors(dialog)
        }
        dialog.show()
    }

    private fun deleteSavedImages(imageIds: Set<String>) {
        imageIds.forEach(::removePendingInputsForSavedImage)
        val deletedCount = imageIds.count { imageId -> savedImageStore.delete(imageId) }
        if (deletedCount <= 0) {
            return
        }

        chatController?.let { applyChatState(it.state.value) }
        chatAdapter.notifyDataSetChanged()
        showTransientMessage(
            if (deletedCount == 1) {
                getString(R.string.image_deleted)
            } else {
                getString(R.string.images_deleted, deletedCount)
            }
        )
    }

    private fun removePendingInputsForSavedImage(imageId: String) {
        val removedInputs = pendingImageInputs
            .filter { it.savedImageId == imageId }
            .toList()
        if (removedInputs.isEmpty()) {
            return
        }

        removedInputs.forEach { input ->
            deletePendingImageTempFile(input)
        }
        pendingImageInputs.removeAll { it.savedImageId == imageId }
        renderPendingImageInputs()
    }

    private fun openImageViewer(imagePath: String) {
        val imageFile = File(imagePath)
        if (!imageFile.exists()) {
            showTransientMessage(getString(R.string.saved_image_missing))
            return
        }

        startActivity(
            Intent(this, ImageViewerActivity::class.java)
                .putExtra(ImageViewerActivity.EXTRA_IMAGE_PATH, imagePath)
        )
    }

    private fun confirmDeleteModel(descriptor: ModelDescriptor) {
        val isFailedModel = modelLoadCoordinator.record?.let {
            it.modelId == descriptor.id && it.phase == ModelLoadPhase.FAILED
        } == true
        if (currentModel?.id == descriptor.id && !isFailedModel) {
            showTransientMessage(getString(R.string.delete_model_current_blocked))
            return
        }
        if (activeDownloadModelId == descriptor.id) {
            showTransientMessage(getString(R.string.model_download_delete_blocked))
            return
        }

        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle(getString(R.string.delete_model))
            .setMessage(getString(R.string.delete_model_confirmation, descriptor.displayName))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                if (modelFileResolver.deleteDownloadedModel(descriptor)) {
                    // Custom and Hugging Face entries live in their model folder, so deleting removes them.
                    ModelRegistry.loadCustomModels(this)
                    ImageGenFeature.instance?.loadCustomModels(this)
                    if (isFailedModel) {
                        controllerStateJob?.cancel()
                        chatController?.close()
                        chatController = null
                        retainedState.chatController = null
                        retainedState.modelId = null
                        currentModel = null
                        modelSelectionStore.clearSelectedModel()
                        modelLoadCoordinator.clearForModel(descriptor.id)
                        persistModelLoadRecord()
                        renderNoControllerState(getString(R.string.model_required_message), preserveTranscript = true)
                    }
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
        if (descriptor.id == currentModel?.id && chatController != null) {
            modelDialogViews?.dialog?.dismiss()
            return
        }

        val isAvailable = modelFileResolver.isModelAvailable(descriptor)
        if (isAvailable) {
            if (LanServerFeature.isRunning()) {
                showTransientMessage(getString(R.string.lan_server_model_switch_blocked))
                return
            }
            if (chatController?.state?.value?.isGenerating == true ||
                isImagePreprocessingForSend || attachmentImportInProgress ||
                audioPreparationJob?.isActive == true
            ) {
                showTransientMessage(getString(R.string.model_switch_generation_blocked))
                return
            }
            ensureImageInputModeAllowedForModel(descriptor)
            modelDialogViews?.dialog?.dismiss()
            requestModelLoad(descriptor, activeChatSnapshot = null)
            return
        }

        if (activeDownloadModelId != null) {
            showTransientMessage(getString(R.string.model_download_already_running))
            return
        }

        startModelDownload(descriptor)
    }

    private fun handleOcrImageModelSelection() {
        if (chatController?.state?.value?.isGenerating == true || isImagePreprocessingForSend) {
            showTransientMessage(getString(R.string.model_switch_generation_blocked))
            return
        }
        selectOcrImageInputMode(resetGemmaDirectInputs = true)
        updateModelSelectorLabel()
        refreshModelSelectionDialog()
        dismissModelDialogAfterImageSelectionIfAllowed()
        showTransientMessage(getString(R.string.ocr_image_model_selected))
    }

    private fun handleGemmaDirectImageModelSelection() {
        if (chatController?.state?.value?.isGenerating == true || isImagePreprocessingForSend) {
            showTransientMessage(getString(R.string.model_switch_generation_blocked))
            return
        }
        if (!isGemmaDirectImageInputAvailable()) {
            showTransientMessage(getString(R.string.gemma_direct_image_model_unavailable))
            return
        }

        currentImageInputMode = ImageInputMode.GEMMA_DIRECT
        modelSelectionStore.saveSelectedImageInputMode(ImageInputMode.GEMMA_DIRECT)
        updateImageInputButtonDescriptions()
        updateModelSelectorLabel()
        refreshModelSelectionDialog()
        dismissModelDialogAfterImageSelectionIfAllowed()
        showTransientMessage(getString(R.string.gemma_direct_image_model_selected))
    }

    private fun dismissModelDialogAfterImageSelectionIfAllowed() {
        modelDialogViews?.dialog?.dismiss()
    }

    /** Asks the same question again: sends the last typed prompt once more as a new turn. */
    private fun retryLastResponse() {
        val controller = chatController ?: return
        val last = controller.state.value.transcript.lastOrNull { it.isUser && !it.isImage && !it.isAttachment } ?: return
        if (controller.state.value.isGenerating) return
        controller.sendPrompt(last.text, last.displayText ?: last.text)
    }

    private fun updateImageInputButtonDescriptions() {
        if (!::galleryOcrButton.isInitialized || !::cameraOcrButton.isInitialized) {
            return
        }

        when (currentImageInputMode) {
            ImageInputMode.OCR -> {
                galleryOcrButton.contentDescription = getString(R.string.gallery_ocr_input)
                cameraOcrButton.contentDescription = getString(R.string.camera_ocr_input)
            }
            ImageInputMode.GEMMA_DIRECT -> {
                galleryOcrButton.contentDescription = getString(R.string.gallery_gemma_direct_input)
                cameraOcrButton.contentDescription = getString(R.string.camera_gemma_direct_input)
            }
        }
    }

    private fun startModelDownload(descriptor: ModelDescriptor) {
        if (activeDownloadModelId != null) {
            showTransientMessage(getString(R.string.model_download_already_running))
            return
        }

        val preflight = modelPreflightChecker.checkDownload(descriptor)
        if (preflight.kind == ModelPreflightKind.BLOCKED) {
            showTransientMessage(preflight.message ?: getString(R.string.model_download_storage_blocked))
            return
        }

        activeDownloadModelId = descriptor.id
        activeDownloadModelName = descriptor.displayName
        activeDownloadFileName = null
        activeDownloadBytes = 0L
        downloadSpeedStartMs = 0L
        activeDownloadTotalBytes = descriptor.approxDownloadBytes
        modelOperationStatusMessage = getString(R.string.model_download_preparing, descriptor.displayName)

        refreshModelSelectionDialog()
        if (chatController != null) {
            applyChatState(chatController!!.state.value)
        } else {
            renderNoControllerState(modelOperationStatusMessage.orEmpty(), preserveTranscript = true)
        }

        requestDownloadNotificationPermissionIfNeeded()
        runCatching {
            ModelDownloadService.start(this, descriptor)
        }.onFailure { error ->
            Log.w(TAG, "Failed to start model download service.", error)
            modelOperationStatusMessage = null
            resetDownloadState()
            refreshModelSelectionDialog()
            if (chatController != null) {
                applyChatState(chatController!!.state.value)
            } else {
                renderNoControllerState(getString(R.string.model_required_message), preserveTranscript = true)
            }
            showTransientMessage(
                getString(
                    R.string.model_download_failed_message,
                    descriptor.displayName,
                    error.message ?: getString(R.string.model_download_failed_generic)
                )
            )
        }
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

    private fun requestDownloadNotificationPermissionIfNeeded() {
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            !hasPermission(Manifest.permission.POST_NOTIFICATIONS)
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    private fun resetDownloadState() {
        activeDownloadModelId = null
        activeDownloadModelName = null
        activeDownloadFileName = null
        activeDownloadBytes = 0L
        activeDownloadTotalBytes = null
        downloadSpeedStartMs = 0L
        downloadSheet?.dismiss()
    }

    private fun startNewChatFromUi() {
        if (isImagePreprocessingForSend || attachmentImportInProgress || audioPreparationJob?.isActive == true) {
            showTransientMessage(getString(R.string.image_input_processing))
            return
        }

        val controller = requireUsableController() ?: return
        if (controller.state.value.isGenerating) {
            return
        }

        inputEditText.text.clear()
        clearSpeechInputState()
        clearPendingImageInputs()
        activeAttachment = null
        renderActiveAttachment()
        controller.startNewChat()
        refreshDrawerSessions()
    }

    /** A start icon at the mock-up's chip size (20dp), tinted like the chip text. */
    private fun TextView.setChipIcon(res: Int) {
        val icon = ContextCompat.getDrawable(this@PocketChatActivity, res)!!.mutate()
        icon.setBounds(0, 0, dp(20), dp(20))
        setCompoundDrawablesRelative(icon, null, null, null)
        compoundDrawableTintList = ColorStateList.valueOf(resolveThemeColor(R.attr.colorAssistantText))
    }

    /**
     * Icon-only chip for free users: "3 free" in gold while tries are left, a small gold lock once they are used.
     * Nothing for Pro users.
     */
    private fun TextView.setProBadge(feature: ProFeature) {
        val left = if (proGate.isUnlocked()) Int.MAX_VALUE
            else ProUiFeature.instance?.triesLeft(this@PocketChatActivity, proGate, feature) ?: 0
        when {
            left == Int.MAX_VALUE -> {
                text = ""
                compoundDrawablePadding = 0
            }
            left > 0 -> {
                text = getString(R.string.chat_chip_tries_left, left)
                textSize = 11f
                // Bright gold on dark themes, deeper gold on light ones for contrast.
                setTextColor(getColor(if (androidx.core.graphics.ColorUtils.calculateLuminance(resolveThemeColor(R.attr.colorAssistantText)) > 0.5) R.color.pro_gold else R.color.pro_gold_dark))
                compoundDrawablePadding = dp(4)
            }
            else -> {
                val lock = ContextCompat.getDrawable(this@PocketChatActivity, R.drawable.ic_lock_gold_12)!!.mutate()
                lock.setBounds(0, 0, dp(12), dp(12))
                text = SpannableString("￼").apply {
                    setSpan(ImageSpan(lock, ImageSpan.ALIGN_CENTER), 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }
                compoundDrawablePadding = dp(4)
            }
        }
    }

    private fun updateChatToolChips() {
        val button = findViewById<TextView>(R.id.toolsButton)
        button.visibility = if (ImageGenFeature.isAvailable || TranslatorFeature.isAvailable) View.VISIBLE else View.GONE
        val label = getString(R.string.chat_tools)
        if (proGate.isUnlocked()) {
            button.text = label
            return
        }
        val pro = ProUiFeature.instance
        val left = maxOf(
            if (ImageGenFeature.isAvailable) pro?.triesLeft(this, proGate, ProFeature.IMAGE_GENERATION) ?: 0 else 0,
            if (TranslatorFeature.isAvailable) pro?.triesLeft(this, proGate, ProFeature.TRANSLATOR) ?: 0 else 0
        )
        val gold = getColor(if (androidx.core.graphics.ColorUtils.calculateLuminance(resolveThemeColor(R.attr.colorAssistantText)) > 0.5) R.color.pro_gold else R.color.pro_gold_dark)
        val text = SpannableStringBuilder(label).append("  ")
        if (left > 0) {
            val start = text.length
            text.append(getString(R.string.chat_chip_tries_left, left))
            text.setSpan(ForegroundColorSpan(gold), start, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            text.setSpan(RelativeSizeSpan(0.85f), start, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        } else {
            val lock = ContextCompat.getDrawable(this, R.drawable.ic_lock_gold_12)!!.mutate()
            lock.setBounds(0, 0, dp(12), dp(12))
            val start = text.length
            text.append("￼")
            text.setSpan(ImageSpan(lock, ImageSpan.ALIGN_CENTER), start, text.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        button.text = text
    }

    /** "Open tools": a bottom sheet of tiles (Create, Enhance, Remove, Translator, LAN server), as in the mock-up. */
    private fun showToolsMenu(anchor: View) {
        val sheet = BottomSheetDialog(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(4), dp(20), dp(24))
        }
        root.addView(TextView(this).apply {
            setText(R.string.tools_sheet_title)
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleLarge)
            setTextColor(resolveThemeColor(R.attr.colorAssistantText))
            setPadding(dp(4), dp(8), 0, dp(12))
        })
        val tiles = mutableListOf<View>()
        fun tile(@DrawableRes icon: Int, @StringRes label: Int, @StringRes hint: Int, feature: ProFeature?, action: () -> Unit) {
            val view = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = ContextCompat.getDrawable(this@PocketChatActivity, R.drawable.bg_card_line)
                setPadding(dp(12), dp(12), dp(12), dp(12))
                isClickable = true
                isFocusable = true
                foreground = ContextCompat.getDrawable(this@PocketChatActivity, android.R.drawable.list_selector_background)
                minimumHeight = dp(120)
            }
            val top = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
            top.addView(ImageView(this).apply {
                setImageResource(icon)
                imageTintList = ColorStateList.valueOf(resolveThemeColor(R.attr.colorSendFill))
                layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            })
            top.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))
            if (feature != null && !proGate.isUnlocked()) {
                val left = ProUiFeature.instance?.triesLeft(this, proGate, feature) ?: 0
                top.addView(TextView(this).apply {
                    background = ContextCompat.getDrawable(this@PocketChatActivity, R.drawable.bg_pro_chip)
                    setPadding(dp(6), dp(1), dp(6), dp(1))
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 10f)
                    setTypeface(typeface, Typeface.BOLD)
                    setTextColor(ContextCompat.getColor(this@PocketChatActivity, R.color.pro_chip_on))
                    text = if (left > 0) getString(R.string.chat_chip_tries_left, left) else getString(R.string.pro_chip)
                })
            }
            view.addView(top)
            view.addView(TextView(this).apply {
                setText(label)
                setTextColor(resolveThemeColor(R.attr.colorAssistantText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
                setTypeface(typeface, Typeface.BOLD)
                setPadding(0, dp(10), 0, 0)
            })
            view.addView(TextView(this).apply {
                setText(hint)
                setTextColor(resolveThemeColor(R.attr.colorStatusText))
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            })
            view.setOnClickListener {
                sheet.dismiss()
                action()
            }
            tiles.add(view)
        }
        if (ImageGenFeature.isAvailable) {
            tile(R.drawable.ic_sparkle_20, R.string.plus_menu_create, R.string.tools_hint_create, ProFeature.IMAGE_GENERATION) {
                openImageGeneration(toolName = ImageTool.CREATE.name)
            }
            tile(R.drawable.ic_tune_20, R.string.image_tool_enhance, R.string.tools_hint_enhance, ProFeature.IMAGE_GENERATION) {
                openImageGeneration(toolName = ImageTool.ENHANCE.name)
            }
            tile(R.drawable.ic_erase_20, R.string.image_tool_remove, R.string.tools_hint_remove, ProFeature.IMAGE_GENERATION) {
                openImageGeneration(toolName = ImageTool.REMOVE_OBJECT.name)
            }
        }
        if (TranslatorFeature.isAvailable) {
            tile(R.drawable.ic_translate_24, R.string.translator_menu, R.string.tools_hint_translator, ProFeature.TRANSLATOR) { openTranslator() }
        }
        if (LanServerFeature.isAvailable) {
            tile(R.drawable.ic_network_20, R.string.lan_server_menu, R.string.tools_hint_lan, null) { showLanServerDialog() }
        }
        tiles.chunked(3).forEach { chunk ->
            val line = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            repeat(3) { i ->
                val cell = chunk.getOrNull(i) ?: View(this)
                line.addView(cell, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    setMargins(dp(4), dp(4), dp(4), dp(4))
                })
            }
            root.addView(line)
        }
        sheet.setContentView(root)
        sheet.show()
    }

    private var stopGlowAnimator: android.animation.ValueAnimator? = null

    /** Finished-answer count when a web search answer was sent; null when no web search try is pending. */
    private var webSearchTryFrom: Int? = null
    private var webSearchTryStarted = false

    /** Uses the web search try once the answer it was sent with finishes; a stopped or failed answer keeps it. */
    private fun settleWebSearchTry(state: ChatUiState) {
        val from = webSearchTryFrom ?: return
        if (state.isGenerating) {
            webSearchTryStarted = true
            return
        }
        if (!webSearchTryStarted) return
        webSearchTryFrom = null
        webSearchTryStarted = false
        if (state.finishedAnswers > from) {
            ProUiFeature.instance?.useTrial(this, proGate, ProFeature.WEB_SEARCH)
            refreshProBadges()
        }
    }

    /** A weak accent glow on the stop button while the model answers; the only glow besides the orb. */
    private fun setStopGlow(on: Boolean) {
        stopGlowAnimator?.cancel()
        stopGlowAnimator = null
        stopButton.elevation = 0f
        if (!on) return
        stopButton.outlineSpotShadowColor = resolveThemeColor(R.attr.colorSendFill)
        stopButton.outlineAmbientShadowColor = resolveThemeColor(R.attr.colorSendFill)
        stopGlowAnimator = android.animation.ValueAnimator.ofFloat(2f, 7f).apply {
            duration = 1100L
            repeatCount = android.animation.ValueAnimator.INFINITE
            repeatMode = android.animation.ValueAnimator.REVERSE
            addUpdateListener { stopButton.elevation = dp(1) * (it.animatedValue as Float) }
            start()
        }
    }

    /** Send and stop swap with a short scale and fade instead of popping in. */
    private fun morphIn(view: View) {
        view.animate().cancel()
        view.alpha = 0f
        view.scaleX = 0.6f
        view.scaleY = 0.6f
        view.animate().alpha(1f).scaleX(1f).scaleY(1f).setDuration(160L).start()
    }

    /** Empty chat with a model ready: one greeting line. */
    private var lastChatState: ChatUiState? = null

    private fun updateChatGreeting(state: ChatUiState) {
        lastChatState = state
        val greeting = findViewById<View>(R.id.chatGreeting) ?: return
        // Only after the loading card (and its progress bar) has completely faded out.
        val loadingGone = !::modelLoadingCard.isInitialized || modelLoadingCard.visibility != View.VISIBLE
        val show = state.isReady && !state.isGenerating && state.transcript.isEmpty() && loadingGone
        greeting.visibility = if (show) View.VISIBLE else View.GONE
        if (!show) return
        val hour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        val hello = when {
            hour < 12 -> R.string.chat_greeting_morning
            hour < 18 -> R.string.chat_greeting_afternoon
            else -> R.string.chat_greeting_evening
        }
        findViewById<TextView>(R.id.chatGreetingHello).apply {
            text = getString(hello)
            applyThemeGradient(this)
        }
    }

    /** Left-to-right gradient across the measured text, from the theme's three gradient colours. */
    private fun applyThemeGradient(view: TextView) {
        val width = view.paint.measureText(view.text.toString()).coerceAtLeast(1f)
        view.paint.shader = android.graphics.LinearGradient(
            0f, 0f, width, 0f,
            intArrayOf(
                resolveThemeColor(R.attr.pocketGradient1),
                resolveThemeColor(R.attr.pocketGradient2),
                resolveThemeColor(R.attr.pocketGradient3)
            ),
            floatArrayOf(0f, 0.55f, 1f),
            android.graphics.Shader.TileMode.CLAMP
        )
        view.invalidate()
    }

    private fun updateWebSearchButton() {
        if (!::webSearchButton.isInitialized) return
        val supported = chatController?.supportsNativeToolCalling() == true
        val on = supported && WebSearchFeature.instance?.isUsable(this) == true
        // M3 filter chip, like Image and Translate: outlined when off, secondaryContainer when on.
        val onColor = resolveThemeColor(R.attr.colorToolActive)
        val text = resolveThemeColor(R.attr.colorAssistantText)
        webSearchButton.alpha = if (supported) 1f else 0.45f
        webSearchButton.backgroundTintList = ColorStateList.valueOf(Color.TRANSPARENT)
        webSearchButton.strokeWidth = 0
        webSearchButton.setTextColor(if (on) onColor else text)
        webSearchButton.iconTint = ColorStateList.valueOf(if (on) onColor else text)
        webSearchButton.setProBadge(ProFeature.WEB_SEARCH)
        // "Web" label next to the icon, followed by the free-tries or lock badge for free users.
        val badge = webSearchButton.text
        val label = SpannableStringBuilder(getString(R.string.chat_chip_web))
        label.setSpan(ForegroundColorSpan(if (on) onColor else text), 0, label.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        if (badge.isNotEmpty()) {
            val start = label.length + 2
            label.append("  ").append(badge)
            label.setSpan(RelativeSizeSpan(0.85f), start, label.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        webSearchButton.textSize = 13f
        webSearchButton.compoundDrawablePadding = 0
        webSearchButton.text = label
        webSearchButton.contentDescription = getString(R.string.chat_chip_web)
    }

    private fun onWebSearchChipClicked() {
        WebSearchFeature.instance?.onWebSearchChipClicked(webSearchHost)
    }

    private fun showManageWebSearchDialog() {
        WebSearchFeature.instance?.showManageDialog(webSearchHost)
    }
    private fun showAboutDialog() {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_about, null)
        val versionView: TextView = dialogView.findViewById(R.id.aboutVersion)
        val privacyLinkView: View = dialogView.findViewById(R.id.aboutPrivacyLink)
        val termsLinkView: View = dialogView.findViewById(R.id.aboutTermsLink)
        val okButton: Button = dialogView.findViewById(R.id.aboutOkButton)
        privacyLinkView.setOnClickListener {
            LegalDialogs.showDocument(this, LegalDocument.PRIVACY_POLICY)
        }
        termsLinkView.setOnClickListener {
            LegalDialogs.showDocument(this, LegalDocument.TERMS_OF_USE)
        }
        dialogView.findViewById<View>(R.id.aboutModelLicensesLink).setOnClickListener {
            showModelLicensesDialog()
        }

        versionView.text = getString(R.string.about_version_by, currentVersionName())
        dialogView.findViewById<View>(R.id.aboutProBadge).visibility =
            if (ProUiFeature.isAvailable && proGate.isUnlocked()) View.VISIBLE else View.GONE
        val showRate = resources.getBoolean(R.bool.show_rate_on_play)
        dialogView.findViewById<View>(R.id.aboutRateRow).visibility = if (showRate) View.VISIBLE else View.GONE
        dialogView.findViewById<View>(R.id.aboutRateDivider).visibility = if (showRate) View.VISIBLE else View.GONE
        dialogView.findViewById<View>(R.id.aboutRateRow).setOnClickListener { openPlayStorePage() }
        dialogView.findViewById<View>(R.id.aboutFeedbackRow).setOnClickListener {
            val mail = Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:" + getString(R.string.about_feedback_email)))
                .putExtra(Intent.EXTRA_SUBJECT, getString(R.string.about_feedback_subject) + " " + currentVersionName())
            runCatching { startActivity(mail) }.onFailure { showTransientMessage(getString(R.string.about_feedback_email)) }
        }

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()

        okButton.setOnClickListener {
            dialog.dismiss()
        }

        showPanelDialog(dialog)
    }

    /** Opens the app's Google Play page: the Play Store app if present, else the browser. */
    private fun openPlayStorePage() {
        val market = Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=$PLAY_PACKAGE"))
        runCatching { startActivity(market) }.onFailure {
            runCatching {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=$PLAY_PACKAGE")))
            }
        }
    }

    private fun showModelLicensesDialog() {
        val builder = MaterialAlertDialogBuilder(this)
            .setTitle(R.string.about_model_licenses)
            .setMessage(R.string.model_licenses_text)
            .setPositiveButton(android.R.string.ok, null)
        val plamoLicense = plamoLicenseResId()
        if (resources.getBoolean(R.bool.show_pro_model_licenses) && plamoLicense != 0) {
            builder.setNeutralButton(R.string.plamo_license_button) { _, _ -> showPlamoLicenseDialog(plamoLicense) }
        }
        val dialog = builder.show()
        dialog.findViewById<TextView>(android.R.id.message)?.let {
            Linkify.addLinks(it, Linkify.WEB_URLS)
            it.movementMethod = LinkMovementMethod.getInstance()
        }
    }

    /** The PLaMo license file ships only in the Play build (src/play/res/raw), so it is looked up by name. */
    @android.annotation.SuppressLint("DiscouragedApi")
    private fun plamoLicenseResId(): Int =
        resources.getIdentifier("plamo_community_license", "raw", packageName)

    /** The PLaMo Community License text, shipped with the app as its Article 4 asks. */
    private fun showPlamoLicenseDialog(resId: Int) {
        val text = resources.openRawResource(resId).bufferedReader().use { it.readText() }
        val body = TextView(this).apply {
            this.text = text
            setTextIsSelectable(true)
            val pad = (24 * resources.displayMetrics.density).toInt()
            setPadding(pad, 0, pad, 0)
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.plamo_license_title)
            .setView(android.widget.ScrollView(this).apply { addView(body) })
            .setPositiveButton(android.R.string.ok, null)
            .show()
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
        val selectedModel = currentModel?.takeIf(modelFileResolver::isModelAvailable)
        if (selectedModel != null) {
            showModelSettingsDialog(selectedModel)
            return
        }

        val availableModels = ModelRegistry.all.filter(modelFileResolver::isModelAvailable)
        if (availableModels.isEmpty()) {
            MaterialAlertDialogBuilder(this)
                .setTitle(R.string.model_settings_title)
                .setMessage(R.string.model_settings_no_models)
                .setNegativeButton(android.R.string.cancel, null)
                .setPositiveButton(R.string.manage_models) { _, _ -> showManageModelsDialog() }
                .show()
            return
        }

        if (availableModels.size == 1) {
            showModelSettingsDialog(availableModels.single())
            return
        }

        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.model_settings_choose_model)
            .setItems(availableModels.map { it.displayName }.toTypedArray()) { _, index ->
                showModelSettingsDialog(availableModels[index])
            }
            .show()
    }

    private fun showModelSettingsDialog(descriptor: ModelDescriptor) {
        val editingActiveModel = currentModel?.id == descriptor.id && chatController != null

        if (editingActiveModel && chatController?.state?.value?.isGenerating == true) {
            showTransientMessage(getString(R.string.model_settings_generation_blocked))
            return
        }

        if (editingActiveModel && LanServerFeature.isRunning()) {
            showTransientMessage(getString(R.string.model_settings_lan_blocked))
            return
        }

        if (!modelFileResolver.isModelAvailable(descriptor)) {
            showTransientMessage(getString(R.string.model_settings_download_required))
            return
        }

        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_model_settings, null)
        val modelNameView: TextView = dialogView.findViewById(R.id.modelSettingsModelName)
        val contextLengthInput: EditText = dialogView.findViewById(R.id.modelContextLengthInput)
        val contextLengthHelp: TextView = dialogView.findViewById(R.id.modelContextLengthHelp)
        val presetChipGroup: ChipGroup = dialogView.findViewById(R.id.modelInstructionPresetChips)
        val instructionInput: EditText = dialogView.findViewById(R.id.modelInstructionInput)
        val cancelButton: Button = dialogView.findViewById(R.id.modelSettingsCancelButton)
        val saveButton: Button = dialogView.findViewById(R.id.modelSettingsSaveButton)

        modelNameView.text = descriptor.displayName
        val runtimeSettings = modelRuntimeSettingsStore.load(descriptor)
        val minContextLength = ModelRuntimeSettingsLimits.minFor(descriptor)
        val maxContextLength = ModelRuntimeSettingsLimits.maxFor(descriptor)
        contextLengthInput.setText(runtimeSettings.contextLengthTokens.toString())
        contextLengthInput.setSelection(contextLengthInput.text.length)
        val contextMemoryGuard = ContextMemoryGuard(this)
        val contextDecision = contextMemoryGuard.decide(descriptor, runtimeSettings.contextLengthTokens)
        contextLengthHelp.text = getString(
            R.string.model_context_length_help,
            minContextLength,
            maxContextLength
        )
        val deviceContextLimit = minOf(
            contextDecision.deviceLimitTokens,
            contextDecision.learnedLimitTokens ?: Int.MAX_VALUE,
            maxContextLength
        )
        // Saving a new value clears the crash-learned limit, so warn against the estimate alone.
        val estimatedDeviceLimit = minOf(contextDecision.deviceLimitTokens, maxContextLength)
        if (descriptor !is OnnxQwenSpec) {
            val limitText = if (contextDecision.deviceLimitMeasured) {
                R.string.model_context_length_measured_limit
            } else {
                R.string.model_context_length_device_limit
            }
            contextLengthHelp.append(" " + getString(limitText, deviceContextLimit))
            val loadedDecision = chatController?.contextDecision?.takeIf { editingActiveModel }
            if (loadedDecision != null && loadedDecision.isCapped) {
                contextLengthHelp.append(
                    " " + getString(R.string.model_context_length_loaded_with, loadedDecision.effectiveTokens)
                )
            }
        }
        val presets = InstructionPreset.entries.toList()
        val presetChipInflater = LayoutInflater.from(dialogView.context)
        val presetChipIds = presets.map { preset ->
            val chip = presetChipInflater.inflate(R.layout.item_preset_chip, presetChipGroup, false) as Chip
            chip.id = View.generateViewId()
            chip.text = preset.label
            presetChipGroup.addView(chip)
            chip.id
        }

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
        presetChipGroup.check(presetChipIds[selectedPresetIndex])
        applyingPresetText = true
        instructionInput.setText(currentInstruction)
        instructionInput.setSelection(0)
        applyingPresetText = false

        presetChipGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            val position = presetChipIds.indexOf(checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener)
            val preset = presets.getOrNull(position) ?: return@setOnCheckedStateChangeListener
            if (preset == selectedPreset) {
                return@setOnCheckedStateChangeListener
            }
            selectedPreset = preset
            if (switchingToCustomFromEdit) {
                return@setOnCheckedStateChangeListener
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
                val customChipId = presetChipIds[customPresetIndex]
                if (presetChipGroup.checkedChipId != customChipId) {
                    switchingToCustomFromEdit = true
                    presetChipGroup.check(customChipId)
                    switchingToCustomFromEdit = false
                }
            }
        })

        val dialog = dialogBuilder
            .setView(dialogView)
            .create()
        dialog.setOnShowListener {
            dialog.window?.apply {
                setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                val pageColor = resolveThemeColor(R.attr.colorAppBackground)
                setBackgroundDrawable(ColorDrawable(pageColor))
                statusBarColor = pageColor
                navigationBarColor = pageColor
                setSoftInputMode(android.view.WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
            }
        }
        dialogView.findViewById<View>(R.id.modelSettingsBack).setOnClickListener { dialog.dismiss() }

        // Chat memory in plain words; the exact token count stays in Advanced.
        val advancedToggle: TextView = dialogView.findViewById(R.id.modelAdvancedToggle)
        val advancedSection: View = dialogView.findViewById(R.id.modelAdvancedSection)
        fun renderAdvancedToggle() {
            val open = advancedSection.visibility == View.VISIBLE
            val tokens = contextLengthInput.text.toString().trim().toIntOrNull()
            val summary = if (tokens != null) getString(R.string.model_advanced_tokens, tokens) else ""
            advancedToggle.text = getString(R.string.model_advanced_label) + " " + summary.trim() + if (open) " ‹" else " ›"
        }
        advancedToggle.setOnClickListener {
            advancedSection.visibility = if (advancedSection.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            renderAdvancedToggle()
        }
        renderAdvancedToggle()
        val memoryGroup: MaterialButtonToggleGroup = dialogView.findViewById(R.id.modelMemoryGroup)
        val memoryShortValue = 2048.coerceIn(minContextLength, maxContextLength)
        val memoryStandardValue = minOf(8192, deviceContextLimit).coerceIn(minContextLength, maxContextLength)
        val memoryLongValue = deviceContextLimit.coerceIn(minContextLength, maxContextLength)
        val memoryButtons = listOf(
            R.id.modelMemoryShort to memoryShortValue,
            R.id.modelMemoryStandard to memoryStandardValue,
            R.id.modelMemoryLong to memoryLongValue
        )
        var syncingMemory = false
        fun syncMemoryGroup() {
            syncingMemory = true
            val typed = contextLengthInput.text.toString().trim().toIntOrNull()
            val match = memoryButtons.lastOrNull { it.second == typed }
            if (match != null) memoryGroup.check(match.first) else memoryGroup.clearChecked()
            syncingMemory = false
        }
        memoryGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (syncingMemory || !isChecked) return@addOnButtonCheckedListener
            memoryButtons.firstOrNull { it.first == checkedId }?.let {
                contextLengthInput.setText(it.second.toString())
            }
        }
        contextLengthInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(t: CharSequence?, a: Int, b: Int, c: Int) = Unit
            override fun onTextChanged(t: CharSequence?, a: Int, b: Int, c: Int) = Unit
            override fun afterTextChanged(t: Editable?) { if (!syncingMemory) syncMemoryGroup(); renderAdvancedToggle() }
        })
        syncMemoryGroup()
        val editInstruction: TextView = dialogView.findViewById(R.id.modelInstructionEdit)
        editInstruction.setOnClickListener {
            instructionInput.maxLines = 12
            instructionInput.requestFocus()
            instructionInput.setSelection(instructionInput.text.length)
            editInstruction.visibility = View.GONE
        }

        fun saveModelSettings(
            instruction: String,
            presetToSave: InstructionPreset,
            requestedContextLength: Int
        ) {
            modelInstructionStore.saveInstruction(descriptor, instruction, presetToSave)
            val updatedRuntimeSettings = modelRuntimeSettingsStore.save(
                descriptor,
                ModelRuntimeSettings(requestedContextLength)
            )
            val runtimeSettingsChanged =
                updatedRuntimeSettings.contextLengthTokens != runtimeSettings.contextLengthTokens
            // A new explicit choice replaces the limit learned from an earlier crash.
            if (runtimeSettingsChanged) contextMemoryGuard.clearLearnedLimit(descriptor.id)
            dialog.dismiss()
            if (runtimeSettingsChanged) {
                if (editingActiveModel) {
                    showTransientMessage(getString(R.string.model_context_length_saved))
                    reloadCurrentModelAfterRuntimeSettings()
                } else {
                    showTransientMessage(getString(R.string.model_settings_saved))
                }
            } else {
                showTransientMessage(getString(R.string.model_instruction_saved))
            }
        }

        val contextTestButton: Button = dialogView.findViewById(R.id.modelContextTestButton)
        contextTestButton.visibility = if (ContextProbeFeature.isAvailable && descriptor !is OnnxQwenSpec && descriptor !is GgufSpec) View.VISIBLE else View.GONE
        contextTestButton.setOnClickListener {
            if (LanServerFeature.isRunning()) {
                showTransientMessage(getString(R.string.model_settings_lan_blocked))
                return@setOnClickListener
            }
            if (chatController?.state?.value?.isGenerating == true) {
                showTransientMessage(getString(R.string.model_settings_generation_blocked))
                return@setOnClickListener
            }
            ContextProbeFeature.instance?.onContextTestButtonClicked(contextProbeHost, descriptor) {
                dialog.dismiss()
            }
        }

        cancelButton.setOnClickListener {
            dialog.dismiss()
        }

        saveButton.setOnClickListener {
            val instruction = instructionInput.text.toString().trim()
            if (instruction.isBlank()) {
                showTransientMessage(getString(R.string.model_instruction_empty))
                return@setOnClickListener
            }

            val requestedContextLength = contextLengthInput.text.toString().trim().toIntOrNull()
            if (
                requestedContextLength == null ||
                requestedContextLength !in minContextLength..maxContextLength
            ) {
                showTransientMessage(
                    getString(
                        R.string.model_context_length_invalid,
                        minContextLength,
                        maxContextLength
                    )
                )
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

            if (ModelRuntimeSettingsLimits.requiresMemoryWarning(requestedContextLength)) {
                val warningDialog = MaterialAlertDialogBuilder(this)
                    .setTitle(R.string.model_context_length_warning_title)
                    .setMessage(
                        if (descriptor !is OnnxQwenSpec && requestedContextLength > estimatedDeviceLimit) {
                            getString(
                                R.string.model_context_length_over_device_limit_message,
                                estimatedDeviceLimit,
                                requestedContextLength
                            )
                        } else {
                            getString(
                                R.string.model_context_length_warning_message,
                                ModelRuntimeSettingsLimits.HIGH_CONTEXT_WARNING_THRESHOLD,
                                requestedContextLength
                            )
                        }
                    )
                    .setNegativeButton(android.R.string.cancel, null)
                    .setPositiveButton(R.string.model_context_length_warning_continue) { _, _ ->
                        saveModelSettings(instruction, presetToSave, requestedContextLength)
                    }
                    .create()
                warningDialog.setOnShowListener {
                    warningDialog.findViewById<TextView>(androidx.appcompat.R.id.alertTitle)
                        ?.setTextColor(ContextCompat.getColor(this, R.color.delete_red))
                    warningDialog.getButton(AlertDialog.BUTTON_POSITIVE)
                        ?.setTextColor(resolveThemeColor(R.attr.colorAssistantText))
                }
                warningDialog.show()
            } else {
                saveModelSettings(instruction, presetToSave, requestedContextLength)
            }
        }

        showPanelDialog(dialog)
    }

    private fun reloadCurrentModelAfterRuntimeSettings() {
        val descriptor = currentModel ?: return
        val controller = chatController ?: return
        val activeChatSnapshot = controller.snapshotActiveChat()
        controllerStateJob?.cancel()
        SharedChatControllerRegistry.clearIf(controller)
        chatController = null
        retainedState.chatController = null
        controller.close()
        renderNoControllerState(getString(R.string.model_loading_message), preserveTranscript = true)
        requestModelLoad(descriptor, activeChatSnapshot)
    }

    /** Settings pages open as a full page in the app's own background instead of a floating card. */
    private fun showFullScreenDialog(dialog: AlertDialog) {
        dialog.setOnShowListener {
            dialog.window?.apply {
                setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
                val pageColor = resolveThemeColor(R.attr.colorAppBackground)
                setBackgroundDrawable(ColorDrawable(pageColor))
                statusBarColor = pageColor
                navigationBarColor = pageColor
            }
        }
        dialog.show()
    }

    private fun showPanelDialog(dialog: AlertDialog) {
        dialog.show()
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
    }

    /** Fills the single Settings page: rows open the detail screens, Appearance is live. */
    private fun bindSettingsPage() {
        findViewById<View>(R.id.settingsPersonalityRow).setOnClickListener { showModelSettingsDialog() }
        findViewById<View>(R.id.settingsContextRow).setOnClickListener { showModelSettingsDialog() }
        findViewById<View>(R.id.settingsAboutRow).setOnClickListener { showAboutDialog() }
        findViewById<View>(R.id.settingsWebSearchHeader).visibility = if (WebSearchFeature.isAvailable) View.VISIBLE else View.GONE
        findViewById<View>(R.id.settingsWebSearchRow).apply {
            visibility = if (WebSearchFeature.isAvailable) View.VISIBLE else View.GONE
            setOnClickListener { requireProOrTrial(ProFeature.WEB_SEARCH) { showManageWebSearchDialog() } }
        }
        findViewById<MaterialSwitch>(R.id.settingsThinkDefault).setOnCheckedChangeListener { _, checked ->
            settingsStore.saveThinkByDefault(checked)
        }

        // Appearance mode.
        val modeGroup: RadioGroup = findViewById(R.id.appearanceModeGroup)
        modeGroup.check(
            when (currentSettings.appearance) {
                AppAppearanceMode.SYSTEM -> R.id.appearanceSystem
                AppAppearanceMode.LIGHT -> R.id.appearanceLight
                AppAppearanceMode.DARK -> R.id.appearanceDark
            }
        )
        modeGroup.setOnCheckedChangeListener { _, checkedId ->
            val mode = when (checkedId) {
                R.id.appearanceSystem -> AppAppearanceMode.SYSTEM
                R.id.appearanceLight -> AppAppearanceMode.LIGHT
                else -> AppAppearanceMode.DARK
            }
            if (mode != currentSettings.appearance) applyVisualSettings(currentSettings.copy(appearance = mode))
        }

        // Accent swatches.
        val accentRow: LinearLayout = findViewById(R.id.accentSwatchRow)
        val ringColor = resolveThemeColor(R.attr.colorAssistantText)
        AppAccentOption.entries.forEachIndexed { index, accent ->
            val selected = accent == currentSettings.accent
            accentRow.addView(View(this).apply {
                layoutParams = LinearLayout.LayoutParams(dp(36), dp(36)).apply { if (index > 0) marginStart = dp(10) }
                background = GradientDrawable().apply {
                    shape = GradientDrawable.OVAL
                    setColor(ContextCompat.getColor(this@PocketChatActivity, accent.swatchColorRes))
                    if (selected) setStroke(dp(3), ringColor)
                }
                contentDescription = getString(accent.labelResId)
                setOnClickListener {
                    if (accent != currentSettings.accent) applyVisualSettings(currentSettings.copy(accent = accent))
                }
            })
        }

        // Background chips.
        val backgroundRow: LinearLayout = findViewById(R.id.backgroundRow)
        val activeColor = resolveThemeColor(androidx.appcompat.R.attr.colorPrimary)
        val lineColor = resolveThemeColor(R.attr.colorPocketOutline)
        AppBackgroundOption.entries.forEachIndexed { index, option ->
            val selected = option == currentSettings.background
            backgroundRow.addView(TextView(this).apply {
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(36))
                    .apply { if (index > 0) marginStart = dp(8) }
                gravity = Gravity.CENTER
                setPadding(dp(14), 0, dp(14), 0)
                text = getString(option.labelResId)
                textSize = 13f
                setTextColor(ringColor)
                background = GradientDrawable().apply {
                    cornerRadius = dp(18).toFloat()
                    setStroke(if (selected) dp(2) else dp(1), if (selected) activeColor else lineColor)
                }
                setOnClickListener {
                    if (option != currentSettings.background) applyVisualSettings(currentSettings.copy(background = option))
                }
            })
        }

        // Text size.
        val fontValue: TextView = findViewById(R.id.fontSizeValue)
        val fontPreview: TextView = findViewById(R.id.fontSizePreview)
        val fontSeek: SeekBar = findViewById(R.id.fontSizeSeekBar)
        fontSeek.progress = (currentSettings.chatFontSizeSp - 13f).toInt().coerceIn(0, 11)
        updateFontSizePreview(fontValue, fontPreview, currentSettings.chatFontSizeSp)
        fontSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                updateFontSizePreview(fontValue, fontPreview, 13f + progress)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                currentSettings = currentSettings.copy(chatFontSizeSp = 13f + fontSeek.progress)
                settingsStore.save(currentSettings)
                chatAdapter.updateFontSize(currentSettings.chatFontSizeSp)
                drawerSessionsAdapter.updateFontSize(currentSettings.chatFontSizeSp)
                applyTypography(findViewById(R.id.statusView), findViewById(R.id.sendButton), findViewById(R.id.stopButton))
            }
        })

        // Speech language.
        val speechSpinner: Spinner = findViewById(R.id.speechLanguageSpinner)
        val speechStore = SpeechLanguageStore(this)
        val speechCodes = listOf(SpeechLanguages.SETTING_PHONE, SpeechLanguages.SETTING_AUTO) +
            SpeechLanguages.languages.map(SpeechLanguageOption::code)
        val speechLabels = listOf(
            getString(R.string.speech_language_phone, Locale.getDefault().displayLanguage),
            getString(R.string.speech_language_auto)
        ) + SpeechLanguages.languages.map(SpeechLanguageOption::label)
        val spinnerTextColor = ringColor
        speechSpinner.adapter = object : ArrayAdapter<String>(this, android.R.layout.simple_spinner_dropdown_item, speechLabels) {
            override fun getView(position: Int, convertView: View?, parent: ViewGroup): View =
                (super.getView(position, convertView, parent) as TextView).apply { setTextColor(spinnerTextColor) }

            override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View =
                (super.getDropDownView(position, convertView, parent) as TextView).apply { setTextColor(spinnerTextColor) }
        }
        speechSpinner.setSelection(speechCodes.indexOf(speechStore.load()).coerceAtLeast(0))
        val speechHint: View = findViewById(R.id.speechLanguageHint)
        speechHint.visibility = if (speechCodes.getOrNull(speechSpinner.selectedItemPosition) == SpeechLanguages.SETTING_AUTO) View.VISIBLE else View.GONE
        speechSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                val code = speechCodes.getOrNull(position) ?: return
                speechHint.visibility = if (code == SpeechLanguages.SETTING_AUTO) View.VISIBLE else View.GONE
                if (code != speechStore.load()) speechStore.save(code)
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
        }
    }

    /** Values that change while the app runs: think default, personality and context summaries. */
    private fun refreshSettingsPage() {
        findViewById<MaterialSwitch>(R.id.settingsThinkDefault).isChecked = settingsStore.loadThinkByDefault()
        val state = chatController?.state?.value
        val contextTokens = state?.contextWindowTokens ?: 0
        // Small windows (the ONNX models use 512) would round down to "0K".
        findViewById<TextView>(R.id.settingsContextValue).text = when {
            contextTokens >= 1024 -> getString(R.string.settings_context_value, contextTokens / 1024)
            contextTokens > 0 -> getString(R.string.settings_context_value_small, contextTokens)
            else -> getString(R.string.settings_context_none)
        }
        findViewById<TextView>(R.id.settingsPersonalityValue).text =
            currentModel?.displayName ?: getString(R.string.model_picker_empty_subtitle)
    }

    /** Saves a theme change and rebuilds the screen; the Settings page opens again afterwards. */
    private fun applyVisualSettings(updated: AppSettings) {
        currentSettings = updated
        settingsStore.save(updated)
        recreateWithoutFlash()
    }

    /** Recreates for a theme change with no window transition, so the screen does not flash. */
    private fun recreateWithoutFlash() {
        if (Build.VERSION.SDK_INT >= 34) {
            overrideActivityTransition(OVERRIDE_TRANSITION_OPEN, 0, 0)
            overrideActivityTransition(OVERRIDE_TRANSITION_CLOSE, 0, 0)
        } else {
            @Suppress("DEPRECATION")
            overridePendingTransition(0, 0)
        }
        recreate()
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
        fontSizeValue.setText(
            when {
                fontSizeSp <= 14f -> R.string.font_size_small
                fontSizeSp <= 17f -> R.string.font_size_default
                else -> R.string.font_size_large
            }
        )
        fontSizePreview.textSize = fontSizeSp
        (fontSizePreview.parent as? View)
            ?.findViewById<TextView>(R.id.fontSizePreviewReply)
            ?.textSize = fontSizeSp
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

    private var downloadSpeedStartMs = 0L
    private var downloadSpeedStartBytes = 0L

    /** Rough time left from the average speed since the first reading; null until it is meaningful. */
    private fun downloadTimeLeftText(downloadedBytes: Long, totalBytes: Long): String? {
        val now = SystemClock.elapsedRealtime()
        if (downloadSpeedStartMs == 0L || downloadedBytes < downloadSpeedStartBytes) {
            downloadSpeedStartMs = now
            downloadSpeedStartBytes = downloadedBytes
            return null
        }
        val elapsedMs = now - downloadSpeedStartMs
        val movedBytes = downloadedBytes - downloadSpeedStartBytes
        if (elapsedMs < 5_000L || movedBytes <= 0L) return null
        val secondsLeft = (totalBytes - downloadedBytes).coerceAtLeast(0L) * elapsedMs / movedBytes / 1000L
        return if (secondsLeft < 60L) {
            getString(R.string.model_download_time_left_short)
        } else {
            getString(R.string.model_download_time_left, ((secondsLeft + 59L) / 60L).toInt())
        }
    }

    private fun formatDownloadProgressText(downloadedBytes: Long, totalBytes: Long?): String {
        return if (totalBytes != null && totalBytes > 0L) {
            val percent = (downloadedBytes * 100L / totalBytes).toInt().coerceIn(0, 100)
            val timeLeft = downloadTimeLeftText(downloadedBytes, totalBytes)
                ?: getString(
                    R.string.model_download_progress_bytes,
                    formatFileSize(downloadedBytes),
                    formatFileSize(totalBytes)
                )
            getString(R.string.model_download_progress_percent, percent, timeLeft)
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
        // Plain italic text rather than a chip, so the status never reads as a button.
        val textColor = when (message) {
            MODEL_LOADING_STATUS_MESSAGE -> ContextCompat.getColor(this, R.color.status_loading_background)
            MODEL_READY_STATUS_MESSAGE -> ContextCompat.getColor(this, R.color.status_ready_background)
            else -> resolveThemeColor(R.attr.colorStatusText)
        }
        statusView.background = null
        statusView.setTextColor(textColor)
    }

    private fun updateMicRecordingState(active: Boolean) {
        if (!::micInputButton.isInitialized) {
            return
        }

        val backgroundColor = if (active) {
            resolveThemeColor(R.attr.colorStopFill)
        } else {
            Color.TRANSPARENT
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
