package com.example.local_llm

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.text.method.LinkMovementMethod
import android.util.TypedValue
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.latex.JLatexMathPlugin
import io.noties.markwon.ext.tables.TableAwareMovementMethod
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.inlineparser.MarkwonInlineParserPlugin
import java.io.File
import kotlin.math.roundToInt

class ChatAdapter(
    private var fontSizeSp: Float = 16f,
    private val onImageTurnSelected: (ChatTurn) -> Unit = {},
    private val onImageTurnLongPressed: (ChatTurn) -> Unit = {},
    private val onImageTurnReported: (ChatTurn) -> Unit = {},
    private val onAttachmentTurnSelected: (ChatTurn) -> Unit = {},
    private val onReportResponse: (String) -> Unit = {},
    /** Null when the build has no Translator; the reply's translate button is then hidden. */
    private val onTranslateResponse: ((String) -> Unit)? = null,
    private val onShareResponse: (String) -> Unit = {},
    private val onRetryResponse: () -> Unit = {},
    private val onMagicAction: (ChatTurn, ImageTool) -> Unit = { _, _ -> }
) : ListAdapter<ChatTurn, ChatAdapter.MessageViewHolder>(DiffCallback) {

    private val expandedThoughtIds = mutableSetOf<String>()
    private val markwonByTextSize = mutableMapOf<Int, Markwon>()

    /** Line under the gold photo actions: free tries left, or "runs offline" for Pro owners. */
    data class MagicNote(val text: String, val showPro: Boolean)

    private var magicNote: MagicNote? = null
    private var lastUserImageId: String? = null

    companion object {
        private const val USER_BUBBLE_LEFT_SPACE_FRACTION = 0.15f
        private const val BUBBLE_HORIZONTAL_PADDING_DP = 16
        private const val BUBBLE_VERTICAL_PADDING_DP = 12
        private const val COPY_BUTTON_RESERVED_END_PADDING_DP = 36
        private const val ACTION_COLUMN_MIN_HEIGHT_DP = 60
        private const val THOUGHT_COLLAPSED_ICON = "\u25B8"
        private const val THOUGHT_EXPANDED_ICON = "\u25BE"
        private const val IMAGE_BUBBLE_MAX_HEIGHT_DP = 320
        private const val IMAGE_BUBBLE_MAX_WIDTH_FRACTION = 0.66f
    }

    enum class TextRenderMode {
        PLAIN,
        MARKDOWN
    }

    data class TextRenderKey(
        val turnId: String,
        val mode: TextRenderMode,
        val textSizeKey: Int,
        val text: String
    )

    /** Ids of messages that arrived after the chat was shown; they rise in once. */
    private val enterAnimationIds = mutableSetOf<String>()

    fun submitTurns(turns: List<ChatTurn>) {
        if (currentList.isNotEmpty()) {
            val known = currentList.mapTo(mutableSetOf<String>()) { it.id }
            turns.filter { it.id !in known }.forEach { enterAnimationIds.add(it.id) }
        }
        expandedThoughtIds.retainAll(turns.mapTo(mutableSetOf<String>()) { it.id })
        lastUserImageId = turns.lastOrNull { it.isUser && it.isImage }?.id
        submitList(turns.toList())
    }

    fun setMagicNote(note: MagicNote) {
        if (note == magicNote) return
        magicNote = note
        val index = currentList.indexOfFirst { it.id == lastUserImageId }
        if (index >= 0) notifyItemChanged(index)
    }

    fun updateFontSize(fontSizeSp: Float) {
        this.fontSizeSp = fontSizeSp
        notifyItemRangeChanged(0, itemCount)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)
        return MessageViewHolder(view)
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val turn = getItem(position)
        if (enterAnimationIds.remove(turn.id) && holder.itemView.context.animationsEnabled()) {
            val rise = 12 * holder.itemView.resources.displayMetrics.density
            holder.itemView.alpha = 0f
            holder.itemView.translationY = rise
            holder.itemView.animate().alpha(1f).translationY(0f).setDuration(220)
                .setInterpolator(android.view.animation.PathInterpolator(0.05f, 0.7f, 0.1f, 1f)).start()
        } else {
            holder.itemView.animate().cancel()
            holder.itemView.alpha = 1f
            holder.itemView.translationY = 0f
        }
        holder.magicActions.visibility = View.GONE
        val markwon = markwonFor(holder.textView.context)
        val bubbleBackground = if (turn.isUser) R.drawable.bg_bubble_user else R.drawable.bg_bubble_bot
        val textColorAttr = if (turn.isUser) R.attr.colorUserText else R.attr.colorAssistantText
        val messageLayoutParams = holder.bubbleFrame.layoutParams as LinearLayout.LayoutParams
        val textLayoutParams = holder.textView.layoutParams as FrameLayout.LayoutParams
        val statusLayoutParams = holder.preResponseStatusContainer.layoutParams as LinearLayout.LayoutParams
        val thoughtLayoutParams = holder.thoughtContainer.layoutParams as LinearLayout.LayoutParams

        holder.textView.setBackgroundResource(bubbleBackground)
        holder.messageImage.setBackgroundResource(bubbleBackground)
        holder.textView.setTextColor(resolveThemeColor(holder.textView, textColorAttr))
        val tokensPerSecond = turn.tokensPerSecond
            ?.takeIf { !turn.isUser && !turn.isStreaming && !turn.isImage && !turn.isAttachment }
        holder.speedView.visibility = if (tokensPerSecond == null) View.GONE else View.VISIBLE
        holder.speedView.text = tokensPerSecond?.let {
            holder.speedView.context.getString(R.string.tokens_per_second_format, it)
        }
        holder.textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)
        holder.preResponseStatusView.setTextColor(resolveThemeColor(holder.preResponseStatusView, R.attr.colorStatusText))
        holder.preResponseStatusView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 2f).coerceAtLeast(12f))
        holder.thoughtHeaderView.setTextColor(resolveThemeColor(holder.thoughtHeaderView, R.attr.colorStatusText))
        holder.thoughtHeaderView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 2f).coerceAtLeast(12f))
        holder.thoughtView.setTextColor(resolveThemeColor(holder.thoughtView, R.attr.colorStatusText))
        holder.thoughtView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 1f).coerceAtLeast(12f))
        holder.container.gravity = if (turn.isUser) Gravity.END else Gravity.START
        messageLayoutParams.width = if (turn.isUser) ViewGroup.LayoutParams.WRAP_CONTENT else ViewGroup.LayoutParams.MATCH_PARENT
        messageLayoutParams.marginStart = calculateUserBubbleStartMargin(holder, turn.isUser)
        textLayoutParams.width = if (turn.isUser) ViewGroup.LayoutParams.WRAP_CONTENT else ViewGroup.LayoutParams.MATCH_PARENT
        holder.textView.maxWidth = calculateUserBubbleMaxWidth(holder, turn.isUser)
        statusLayoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT
        thoughtLayoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT
        holder.bubbleFrame.layoutParams = messageLayoutParams
        holder.textView.layoutParams = textLayoutParams
        holder.preResponseStatusContainer.layoutParams = statusLayoutParams
        holder.thoughtContainer.layoutParams = thoughtLayoutParams

        val preResponseStatusText = turn.preResponseStatusText
            ?.takeIf { !turn.isUser && it.isNotBlank() }
        holder.preResponseStatusContainer.visibility = if (preResponseStatusText == null) View.GONE else View.VISIBLE
        holder.preResponseStatusView.text = preResponseStatusText.orEmpty()

        val thoughtText = buildThoughtText(turn)
        val hasThought = !thoughtText.isNullOrBlank()
        val isLiveThought = hasThought && turn.thinkingDurationMillis == null && turn.text.isBlank()
        val isThoughtExpanded = hasThought && (isLiveThought || expandedThoughtIds.contains(turn.id))
        holder.thoughtContainer.visibility = if (hasThought) View.VISIBLE else View.GONE
        holder.thoughtHeaderView.text = buildThoughtHeader(turn, isThoughtExpanded)
        holder.thoughtHeaderView.setOnClickListener {
            if (!hasThought) {
                return@setOnClickListener
            }

            if (isThoughtExpanded) {
                expandedThoughtIds.remove(turn.id)
            } else {
                expandedThoughtIds.add(turn.id)
            }
            val adapterPosition = holder.bindingAdapterPosition
            if (adapterPosition != RecyclerView.NO_POSITION) {
                notifyItemChanged(adapterPosition)
            }
        }
        holder.thoughtView.visibility = if (isThoughtExpanded) View.VISIBLE else View.GONE
        if (isThoughtExpanded) {
            setThoughtMarkdown(
                holder,
                turn.id,
                normalizeEquationDelimiters(thoughtText.orEmpty()),
                markwon
            )
        } else {
            setThoughtPlainText(holder, turn.id, "")
        }

        if (turn.isImage) {
            bindImageTurn(holder, turn, bubbleBackground)
            return
        }

        val bubbleText = buildBubbleText(turn)
        val hasBubbleText = bubbleText.isNotBlank()
        val canCopyResponse = !turn.isUser && hasBubbleText && !turn.isStreaming
        holder.messageImage.visibility = View.GONE
        holder.messageImage.setImageDrawable(null)
        holder.bubbleFrame.setOnClickListener(
            if (turn.isAttachment) View.OnClickListener { onAttachmentTurnSelected(turn) } else null
        )
        holder.bubbleFrame.setOnLongClickListener(null)
        holder.bubbleFrame.visibility = if (hasBubbleText) View.VISIBLE else View.GONE
        holder.textView.visibility = if (hasBubbleText) View.VISIBLE else View.GONE
        holder.textView.setPaddingRelative(
            dp(holder.textView, BUBBLE_HORIZONTAL_PADDING_DP),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP),
            dp(holder.textView, BUBBLE_HORIZONTAL_PADDING_DP),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP)
        )
        if (holder.textView.visibility == View.VISIBLE && !turn.isUser && turn.renderAsMarkdown) {
            setMessageMarkdown(holder, turn.id, normalizeEquationDelimiters(bubbleText), markwon)
        } else {
            setMessagePlainText(holder, turn.id, bubbleText)
        }

        // Text replies: a row of small icons under the bubble. The in-bubble buttons are only for pictures.
        holder.copyButton.visibility = View.GONE
        holder.reportButton.visibility = View.GONE
        holder.textView.minHeight = 0
        holder.replyActions.visibility = if (canCopyResponse) View.VISIBLE else View.GONE
        if (canCopyResponse) {
            val isLast = holder.bindingAdapterPosition == itemCount - 1
            holder.replyRetry.visibility = if (isLast) View.VISIBLE else View.GONE
            holder.replyCopy.setOnClickListener { view -> copyResponseToClipboard(view.context, bubbleText) }
            val translate = onTranslateResponse
            holder.replyTranslate.visibility = if (translate != null) View.VISIBLE else View.GONE
            holder.replyTranslate.setOnClickListener(translate?.let { t -> View.OnClickListener { t(bubbleText) } })
            holder.replyShare.setOnClickListener { onShareResponse(bubbleText) }
            holder.replyRetry.setOnClickListener { onRetryResponse() }
            holder.replyReport.setOnClickListener { onReportResponse(bubbleText) }
        } else {
            listOf(holder.replyCopy, holder.replyTranslate, holder.replyShare, holder.replyRetry, holder.replyReport)
                .forEach { it.setOnClickListener(null) }
        }
    }

    /** Gold Upscale, Cut out and Restyle buttons under the latest photo the user sent. */
    private fun bindMagicActions(holder: MessageViewHolder, turn: ChatTurn) {
        val note = magicNote
        if (!turn.isUser || turn.id != lastUserImageId || note == null) return
        if (ImageGenFeature.isAvailable) {
            holder.magicActions.visibility = View.VISIBLE
            holder.magicProChip.visibility = if (note.showPro) View.VISIBLE else View.GONE
            holder.magicNote.text = note.text
            holder.magicCutout.setOnClickListener { onMagicAction(turn, ImageTool.REMOVE_BACKGROUND) }
            holder.magicRestyle.visibility = View.GONE
        } else {
            holder.magicActions.visibility = View.GONE
        }
    }

    private fun bindImageTurn(
        holder: MessageViewHolder,
        turn: ChatTurn,
        bubbleBackground: Int
    ) {
        holder.replyActions.visibility = View.GONE
        holder.copyButton.visibility = View.GONE
        holder.copyButton.setOnClickListener(null)
        // Generated pictures (assistant side) can be reported like text answers.
        holder.reportButton.visibility = if (turn.isUser) View.GONE else View.VISIBLE
        holder.reportButton.setOnClickListener(if (turn.isUser) null else View.OnClickListener { onImageTurnReported(turn) })
        holder.bubbleFrame.visibility = View.VISIBLE
        holder.messageImage.visibility = View.VISIBLE
        holder.textView.visibility = View.GONE
        holder.messageTextRenderKey = null
        holder.messageImage.setBackgroundResource(bubbleBackground)
        holder.messageImage.maxWidth = calculateImageBubbleMaxWidth(holder)
        holder.messageImage.maxHeight = dp(holder.messageImage, IMAGE_BUBBLE_MAX_HEIGHT_DP)
        holder.textView.setPaddingRelative(
            dp(holder.textView, BUBBLE_HORIZONTAL_PADDING_DP),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP),
            dp(holder.textView, BUBBLE_HORIZONTAL_PADDING_DP),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP)
        )

        val imageFile = turn.imagePath?.let(::File)
        if (imageFile == null || !imageFile.exists()) {
            holder.messageImage.visibility = View.GONE
            holder.textView.visibility = View.VISIBLE
            holder.textView.setBackgroundResource(bubbleBackground)
            setMessagePlainText(
                holder,
                turn.id,
                holder.itemView.context.getString(R.string.saved_image_missing)
            )
            holder.bubbleFrame.setOnClickListener(null)
            return
        }

        val previewBitmap = ImageBitmapLoader.decodeSampledBitmap(
            imageFile = imageFile,
            requestedWidth = holder.messageImage.maxWidth.coerceAtLeast(1),
            requestedHeight = holder.messageImage.maxHeight.coerceAtLeast(1)
        )
        if (previewBitmap == null) {
            holder.messageImage.visibility = View.GONE
            holder.textView.visibility = View.VISIBLE
            holder.textView.setBackgroundResource(bubbleBackground)
            setMessagePlainText(
                holder,
                turn.id,
                holder.itemView.context.getString(R.string.saved_image_missing)
            )
            holder.bubbleFrame.setOnClickListener(null)
            return
        }

        holder.messageImage.setImageBitmap(previewBitmap)
        bindMagicActions(holder, turn)
        holder.bubbleFrame.setOnClickListener {
            onImageTurnSelected(turn)
        }
        holder.bubbleFrame.setOnLongClickListener {
            onImageTurnLongPressed(turn)
            true
        }
    }

    private fun setMessageMarkdown(
        holder: MessageViewHolder,
        turnId: String,
        markdown: String,
        markwon: Markwon
    ) {
        val key = TextRenderKey(turnId, TextRenderMode.MARKDOWN, textSizeKey(fontSizeSp), markdown)
        if (holder.messageTextRenderKey != key) {
            markwon.setMarkdown(holder.textView, markdown)
            holder.messageTextRenderKey = key
        }
        holder.textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)
    }

    private fun setMessagePlainText(
        holder: MessageViewHolder,
        turnId: String,
        text: String
    ) {
        val key = TextRenderKey(turnId, TextRenderMode.PLAIN, textSizeKey(fontSizeSp), text)
        if (holder.messageTextRenderKey != key) {
            holder.textView.text = text
            holder.messageTextRenderKey = key
        }
    }

    private fun setThoughtMarkdown(
        holder: MessageViewHolder,
        turnId: String,
        markdown: String,
        markwon: Markwon
    ) {
        val thoughtTextSizeSp = thoughtTextSizeSp()
        val key = TextRenderKey(turnId, TextRenderMode.MARKDOWN, textSizeKey(thoughtTextSizeSp), markdown)
        if (holder.thoughtTextRenderKey != key) {
            markwon.setMarkdown(holder.thoughtView, markdown)
            holder.thoughtTextRenderKey = key
        }
        holder.thoughtView.setTextSize(TypedValue.COMPLEX_UNIT_SP, thoughtTextSizeSp)
    }

    private fun setThoughtPlainText(
        holder: MessageViewHolder,
        turnId: String,
        text: String
    ) {
        val thoughtTextSizeSp = thoughtTextSizeSp()
        val key = TextRenderKey(turnId, TextRenderMode.PLAIN, textSizeKey(thoughtTextSizeSp), text)
        if (holder.thoughtTextRenderKey != key) {
            holder.thoughtView.text = text
            holder.thoughtTextRenderKey = key
        }
    }

    private fun markwonFor(context: Context): Markwon {
        val textSizeKey = textSizeKey(fontSizeSp)
        return markwonByTextSize.getOrPut(textSizeKey) {
            val textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_SP,
                fontSizeSp,
                context.resources.displayMetrics
            )
            Markwon.builder(context)
                .usePlugin(TablePlugin.create(context))
                .usePlugin(MarkwonInlineParserPlugin.create())
                .usePlugin(
                    JLatexMathPlugin.create(textSizePx) { builder ->
                        builder.inlinesEnabled(true)
                        builder.errorHandler { _, _ -> null }
                    }
                )
                .build()
        }
    }

    private fun textSizeKey(sizeSp: Float): Int {
        return (sizeSp * 10f).roundToInt()
    }

    private fun thoughtTextSizeSp(): Float {
        return (fontSizeSp - 1f).coerceAtLeast(12f)
    }

    private fun normalizeEquationDelimiters(markdown: String): String {
        return markdown
            .replace(Regex("""\\\[(.+?)\\\]""", RegexOption.DOT_MATCHES_ALL)) { match ->
                "\n$$\n${match.groupValues[1].trim()}\n$$\n"
            }
            .replace(Regex("""\\\((.+?)\\\)""")) { match ->
                "$$${match.groupValues[1].trim()}$$"
            }
    }

    private fun buildThoughtText(turn: ChatTurn): String? {
        if (turn.isUser) {
            return null
        }

        return turn.thinkingText
            ?.let(::normalizeThoughtText)
            ?.takeIf { it.isNotBlank() }
    }

    private fun buildThoughtHeader(turn: ChatTurn, isExpanded: Boolean): String {
        val state = if (isExpanded) THOUGHT_EXPANDED_ICON else THOUGHT_COLLAPSED_ICON
        val durationMillis = turn.thinkingDurationMillis
        if (durationMillis == null) {
            return "$state Thinking..."
        }

        val seconds = ((durationMillis + 999L) / 1000L).coerceAtLeast(1L)
        val unit = if (seconds == 1L) "second" else "seconds"
        return "$state Thought for $seconds $unit"
    }

    private fun buildBubbleText(turn: ChatTurn): String {
        if (turn.isAttachment) {
            return "📎 ${turn.attachmentName ?: "Attachment"}"
        }
        if (turn.isImagePrompt) {
            return "🎨 ${turn.text}"
        }
        if (turn.isUser) {
            return turn.transcriptText
        }

        return buildString {
            if (turn.text.isNotBlank()) {
                append(turn.text.trimEnd())
            }

            if (turn.stopped && turn.text.isNotBlank()) {
                if (isNotEmpty()) {
                    append("\n\n")
                }
                append("_Generation stopped_")
            }
        }
    }

    private fun normalizeThoughtText(rawThought: String): String {
        return rawThought
            .replace("\r\n", "\n")
            .lineSequence()
            .filterNot { line -> line.trim().equals("Thinking Process:", ignoreCase = true) }
            .joinToString("\n")
            .trim()
    }

    private fun calculateUserBubbleStartMargin(holder: MessageViewHolder, isUser: Boolean): Int {
        if (!isUser) {
            return 0
        }

        return (availableMessageWidth(holder) * USER_BUBBLE_LEFT_SPACE_FRACTION).toInt()
    }

    private fun calculateUserBubbleMaxWidth(holder: MessageViewHolder, isUser: Boolean): Int {
        if (!isUser) {
            return Int.MAX_VALUE
        }

        return (availableMessageWidth(holder) * (1f - USER_BUBBLE_LEFT_SPACE_FRACTION))
            .toInt()
            .coerceAtLeast(1)
    }

    private fun calculateImageBubbleMaxWidth(holder: MessageViewHolder): Int {
        return (availableMessageWidth(holder) * IMAGE_BUBBLE_MAX_WIDTH_FRACTION)
            .toInt()
            .coerceAtLeast(dp(holder.itemView, 140))
    }

    private fun availableMessageWidth(holder: MessageViewHolder): Int {
        val containerWidth = holder.container.width.takeIf { it > 0 }
            ?: holder.itemView.resources.displayMetrics.widthPixels

        return (containerWidth - holder.container.paddingStart - holder.container.paddingEnd)
            .coerceAtLeast(1)
    }

    private fun dp(view: View, value: Int): Int {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            view.resources.displayMetrics
        ).toInt()
    }

    private fun copyResponseToClipboard(context: Context, responseText: String) {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText(context.getString(R.string.copy_response), responseText)
        clipboard.setPrimaryClip(clip)
        Toast.makeText(context, R.string.response_copied, Toast.LENGTH_SHORT).show()
    }

    class MessageViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val container: LinearLayout = view.findViewById(R.id.messageContainer)
        val preResponseStatusContainer: LinearLayout = view.findViewById(R.id.preResponseStatusContainer)
        val preResponseStatusView: TextView = view.findViewById(R.id.preResponseStatusView)
        val thoughtContainer: LinearLayout = view.findViewById(R.id.thoughtContainer)
        val thoughtHeaderView: TextView = view.findViewById(R.id.thoughtHeaderView)
        val thoughtView: TextView = view.findViewById(R.id.thoughtView)
        val bubbleFrame: FrameLayout = view.findViewById(R.id.messageBubbleFrame)
        val messageImage: ImageView = view.findViewById(R.id.messageImage)
        val textView: TextView = view.findViewById(R.id.messageText)
        val copyButton: ImageButton = view.findViewById(R.id.copyResponseButton)
        val reportButton: ImageButton = view.findViewById(R.id.reportResponseButton)
        val replyActions: View = view.findViewById(R.id.replyActions)
        val replyCopy: ImageButton = view.findViewById(R.id.replyCopy)
        val replyTranslate: ImageButton = view.findViewById(R.id.replyTranslate)
        val replyShare: ImageButton = view.findViewById(R.id.replyShare)
        val replyRetry: ImageButton = view.findViewById(R.id.replyRetry)
        val replyReport: ImageButton = view.findViewById(R.id.replyReport)
        val speedView: TextView = view.findViewById(R.id.messageSpeed)
        val magicActions: View = view.findViewById(R.id.magicActions)
        val magicCutout: View = view.findViewById(R.id.magicCutout)
        val magicRestyle: View = view.findViewById(R.id.magicRestyle)
        val magicProChip: View = view.findViewById(R.id.magicProChip)
        val magicNote: TextView = view.findViewById(R.id.magicNote)
        var messageTextRenderKey: TextRenderKey? = null
        var thoughtTextRenderKey: TextRenderKey? = null

        init {
            textView.movementMethod = runCatching {
                TableAwareMovementMethod.create()
            }.getOrElse {
                LinkMovementMethod.getInstance()
            }
            thoughtView.movementMethod = runCatching {
                TableAwareMovementMethod.create()
            }.getOrElse {
                LinkMovementMethod.getInstance()
            }
        }
    }

    private fun resolveThemeColor(view: View, attrId: Int): Int {
        val typedValue = TypedValue()
        view.context.theme.resolveAttribute(attrId, typedValue, true)
        return typedValue.data
    }

    private object DiffCallback : DiffUtil.ItemCallback<ChatTurn>() {
        override fun areItemsTheSame(oldItem: ChatTurn, newItem: ChatTurn): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ChatTurn, newItem: ChatTurn): Boolean {
            return oldItem == newItem
        }
    }
}
