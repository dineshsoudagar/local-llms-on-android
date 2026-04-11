package com.example.local_llm

import android.text.method.LinkMovementMethod
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon

class ChatAdapter : ListAdapter<ChatTurn, ChatAdapter.MessageViewHolder>(DiffCallback) {

    fun submitTurns(turns: List<ChatTurn>) {
        submitList(turns.toList())
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)
        return MessageViewHolder(view, Markwon.create(parent.context))
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val turn = getItem(position)
        val bubbleBackground = if (turn.isUser) R.drawable.bg_bubble_user else R.drawable.bg_bubble_bot
        val textColor = if (turn.isUser) R.color.user_text else R.color.assistant_text

        holder.textView.setBackgroundResource(bubbleBackground)
        holder.textView.setTextColor(ContextCompat.getColor(holder.textView.context, textColor))
        holder.container.gravity = if (turn.isUser) Gravity.END else Gravity.START

        val displayText = buildDisplayText(turn)
        if (!turn.isUser && turn.renderAsMarkdown) {
            holder.markwon.setMarkdown(holder.textView, displayText)
        } else {
            holder.textView.text = displayText
        }
    }

    private fun buildDisplayText(turn: ChatTurn): String {
        if (turn.isUser) {
            return turn.text
        }

        return buildString {
            if (!turn.thinkingText.isNullOrBlank()) {
                append("**Thinking**\n")
                append(turn.thinkingText.trimEnd())
            }

            if (turn.text.isNotBlank()) {
                if (isNotEmpty()) {
                    append("\n\n")
                }
                append(turn.text.trimEnd())
            }

            if (turn.stopped) {
                if (isNotEmpty()) {
                    append("\n\n")
                }
                append("_Generation stopped_")
            }

            if (isEmpty()) {
                append("Thinking...")
            }
        }
    }

    class MessageViewHolder(view: View, val markwon: Markwon) : RecyclerView.ViewHolder(view) {
        val container: LinearLayout = view.findViewById(R.id.messageContainer)
        val textView: TextView = view.findViewById(R.id.messageText)

        init {
            textView.movementMethod = LinkMovementMethod.getInstance()
        }
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
