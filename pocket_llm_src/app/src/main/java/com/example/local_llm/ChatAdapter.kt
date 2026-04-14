package com.example.local_llm

import android.text.method.LinkMovementMethod
import android.util.TypedValue
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon
import io.noties.markwon.ext.tables.TableAwareMovementMethod
import io.noties.markwon.ext.tables.TablePlugin

class ChatAdapter(
    private var fontSizeSp: Float = 16f
) : ListAdapter<ChatTurn, ChatAdapter.MessageViewHolder>(DiffCallback) {

    companion object {
        private val STEP_PREFIX_REGEX = Regex("(?m)^\\s*(?:Step\\s*)?\\d+[.)]\\s*")
        private val BOLD_STEP_HEADING_REGEX = Regex("^\\s*\\*\\*[^*\\n]{1,80}\\*\\*:\\s*")
    }

    fun submitTurns(turns: List<ChatTurn>) {
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
        val messageLayoutParams = holder.textView.layoutParams as LinearLayout.LayoutParams
        val thoughtLayoutParams = holder.thoughtView.layoutParams as LinearLayout.LayoutParams

        holder.textView.setBackgroundResource(bubbleBackground)
        holder.textView.setTextColor(resolveThemeColor(holder.textView, textColorAttr))
        holder.textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)
        holder.thoughtView.setTextColor(resolveThemeColor(holder.thoughtView, R.attr.colorStatusText))
        holder.thoughtView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 1f).coerceAtLeast(12f))
        holder.container.gravity = if (turn.isUser) Gravity.END else Gravity.START
        messageLayoutParams.width = if (turn.isUser) ViewGroup.LayoutParams.WRAP_CONTENT else ViewGroup.LayoutParams.MATCH_PARENT
        thoughtLayoutParams.width = ViewGroup.LayoutParams.MATCH_PARENT
        holder.textView.layoutParams = messageLayoutParams
        holder.thoughtView.layoutParams = thoughtLayoutParams

        val thoughtText = buildThoughtText(turn)
        holder.thoughtView.visibility = if (thoughtText.isNullOrBlank()) View.GONE else View.VISIBLE
        holder.thoughtView.text = thoughtText

        val bubbleText = buildBubbleText(turn)
        holder.textView.visibility = if (bubbleText.isBlank()) View.GONE else View.VISIBLE
        if (holder.textView.visibility == View.VISIBLE && !turn.isUser && turn.renderAsMarkdown) {
            holder.markwon.setMarkdown(holder.textView, bubbleText)
            holder.textView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp)
        } else {
            holder.textView.text = bubbleText
        }
    }

    private fun buildThoughtText(turn: ChatTurn): String? {
        if (turn.isUser || turn.text.isNotBlank()) {
            return null
        }

        return turn.thinkingText
            ?.let(::extractCurrentThoughtStep)
            ?.takeIf { it.isNotBlank() }
            ?.let { stepText ->
                buildString {
                    append("Thinking...")
                    if (stepText.isNotBlank()) {
                        append("\n")
                        append(stepText)
                    }
                }
            }
    }

    private fun buildBubbleText(turn: ChatTurn): String {
        if (turn.isUser) {
            return turn.text
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

    private fun extractCurrentThoughtStep(rawThought: String): String {
        val normalizedThought = rawThought
            .replace("\r\n", "\n")
            .lineSequence()
            .filterNot { line -> line.trim().equals("Thinking Process:", ignoreCase = true) }
            .joinToString("\n")
            .trim()

        if (normalizedThought.isBlank()) {
            return ""
        }

        val currentStep = STEP_PREFIX_REGEX.findAll(normalizedThought)
            .lastOrNull()
            ?.let { match -> normalizedThought.substring(match.range.first) }
            ?: normalizedThought

        return currentStep
            .replaceFirst(STEP_PREFIX_REGEX, "")
            .replaceFirst(BOLD_STEP_HEADING_REGEX, "")
            .replace("**", "")
            .trim()
    }

    class MessageViewHolder(view: View, val markwon: Markwon) : RecyclerView.ViewHolder(view) {
        val container: LinearLayout = view.findViewById(R.id.messageContainer)
        val thoughtView: TextView = view.findViewById(R.id.thoughtView)
        val textView: TextView = view.findViewById(R.id.messageText)

        init {
            textView.movementMethod = runCatching {
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
