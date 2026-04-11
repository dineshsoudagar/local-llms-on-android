package com.example.local_llm

import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

class ChatAdapter : ListAdapter<ChatTurn, ChatAdapter.MessageViewHolder>(DiffCallback) {

    fun submitTurns(turns: List<ChatTurn>) {
        submitList(turns.toList())
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MessageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)
        return MessageViewHolder(view)
    }

    override fun onBindViewHolder(holder: MessageViewHolder, position: Int) {
        val turn = getItem(position)
        holder.textView.text = buildDisplayText(turn)
        holder.textView.setBackgroundResource(
            if (turn.isUser) R.drawable.bg_bubble_user else R.drawable.bg_bubble_bot
        )
        holder.container.gravity = if (turn.isUser) Gravity.END else Gravity.START
    }

    private fun buildDisplayText(turn: ChatTurn): String {
        if (turn.isUser) {
            return turn.text
        }

        return buildString {
            if (!turn.thinkingText.isNullOrBlank()) {
                append("[Thinking]\n")
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
                    append("\n")
                }
                append("[Generation stopped]")
            }

            if (isEmpty()) {
                append("Thinking...")
            }
        }
    }

    class MessageViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val container: LinearLayout = view.findViewById(R.id.messageContainer)
        val textView: TextView = view.findViewById(R.id.messageText)
    }

    private object DiffCallback : DiffUtil.ItemCallback<ChatTurn>() {
        override fun areItemsTheSame(oldItem: ChatTurn, newItem: ChatTurn): Boolean {
            return oldItem.role == newItem.role && oldItem.text == newItem.text
        }

        override fun areContentsTheSame(oldItem: ChatTurn, newItem: ChatTurn): Boolean {
            return oldItem == newItem
        }
    }
}
