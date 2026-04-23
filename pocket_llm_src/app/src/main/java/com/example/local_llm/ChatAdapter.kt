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
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.tables.TableAwareMovementMethod
import io.noties.markwon.ext.tables.TablePlugin

class ChatAdapter(
    private var fontSizeSp: Float = 16f
) : ListAdapter<ChatTurn, ChatAdapter.MessageViewHolder>(DiffCallback) {

    private val expandedThoughtIds = mutableSetOf<String>()

    companion object {
        private const val USER_BUBBLE_LEFT_SPACE_FRACTION = 0.15f
        private const val BUBBLE_HORIZONTAL_PADDING_DP = 16
        private const val BUBBLE_VERTICAL_PADDING_DP = 12
        private const val COPY_BUTTON_RESERVED_END_PADDING_DP = 36
        private const val THOUGHT_COLLAPSED_ICON = "\u25B8"
        private const val THOUGHT_EXPANDED_ICON = "\u25BE"
    }

    fun submitTurns(turns: List<ChatTurn>) {
        expandedThoughtIds.retainAll(turns.mapTo(mutableSetOf<String>()) { it.id })
        submitList(turns.toList())
    }

    fun updateFontSize(fontSizeSp: Float) {
        this.fontSizeSp = fontSizeSp
        notifyItemRangeChanged(0, itemCount)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)
        return MessageViewHolder(
            view,
            Markwon.builder(parent.context)
                .usePlugin(TablePlugin.create(parent.context))
                .build()
        )
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val turn = getItem(position)
        val bubbleBackground = if (turn.isUser) R.drawable.bg_bubble_user else R.drawable.bg_bubble_bot
        val textColorAttr = if (turn.isUser) R.attr.colorUserText else R.attr.colorAssistantText
        val messageLayoutParams = holder.bubbleFrame.layoutParams as LinearLayout.LayoutParams
        val textLayoutParams = holder.textView.layoutParams as FrameLayout.LayoutParams
        val statusLayoutParams = holder.preResponseStatusContainer.layoutParams as LinearLayout.LayoutParams
        val thoughtLayoutParams = holder.thoughtContainer.layoutParams as LinearLayout.LayoutParams

        holder.textView.setBackgroundResource(bubbleBackground)
        holder.textView.setTextColor(resolveThemeColor(holder.textView, textColorAttr))
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
            holder.markwon.setMarkdown(holder.thoughtView, thoughtText.orEmpty())
            holder.thoughtView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 1f).coerceAtLeast(12f))
        } else {
            holder.thoughtView.text = ""
        }

        val bubbleText = buildBubbleText(turn)
        val hasBubbleText = bubbleText.isNotBlank()
        val canCopyResponse = !turn.isUser && hasBubbleText && !turn.isStreaming
        holder.bubbleFrame.visibility = if (hasBubbleText) View.VISIBLE else View.GONE
        holder.textView.visibility = if (hasBubbleText) View.VISIBLE else View.GONE
        holder.textView.setPaddingRelative(
            dp(holder.textView, BUBBLE_HORIZONTAL_PADDING_DP),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP),
            dp(
                holder.textView,
                if (canCopyResponse) COPY_BUTTON_RESERVED_END_PADDING_DP else BUBBLE_HORIZONTAL_PADDING_DP
            ),
            dp(holder.textView, BUBBLE_VERTICAL_PADDING_DP)
        )
        if (holder.textView.visibility == View.VISIBLE && !turn.isUser && turn.renderAsMarkdown) {
            holder.markwon.setMarkdown(holder.textView, bubbleText)
            holder.textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)
        } else {
            holder.textView.text = bubbleText
        }

        holder.copyButton.visibility = if (canCopyResponse) View.VISIBLE else View.GONE
        if (canCopyResponse) {
            holder.copyButton.setOnClickListener { view ->
                copyResponseToClipboard(view.context, bubbleText)
            }
        } else {
            holder.copyButton.setOnClickListener(null)
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
        if (turn.isUser) {
            return turn.displayText ?: turn.text
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

    class MessageViewHolder(view: View, val markwon: Markwon) : RecyclerView.ViewHolder(view) {
        val container: LinearLayout = view.findViewById(R.id.messageContainer)
        val preResponseStatusContainer: LinearLayout = view.findViewById(R.id.preResponseStatusContainer)
        val preResponseStatusView: TextView = view.findViewById(R.id.preResponseStatusView)
        val thoughtContainer: LinearLayout = view.findViewById(R.id.thoughtContainer)
        val thoughtHeaderView: TextView = view.findViewById(R.id.thoughtHeaderView)
        val thoughtView: TextView = view.findViewById(R.id.thoughtView)
        val bubbleFrame: FrameLayout = view.findViewById(R.id.messageBubbleFrame)
        val textView: TextView = view.findViewById(R.id.messageText)
        val copyButton: ImageButton = view.findViewById(R.id.copyResponseButton)

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
