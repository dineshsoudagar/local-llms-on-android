package com.example.local_llm

import android.text.format.DateUtils
import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

class SavedSessionsAdapter(
    private val fontSizeSp: Float,
    private val onSessionSelected: (ChatSessionSummary) -> Unit
) : ListAdapter<ChatSessionSummary, SavedSessionsAdapter.SavedSessionViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SavedSessionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_saved_chat, parent, false)
        return SavedSessionViewHolder(view)
    }

    override fun onBindViewHolder(holder: SavedSessionViewHolder, position: Int) {
        val session = getItem(position)
        val updated = DateUtils.getRelativeTimeSpanString(
            session.updatedAtMillis,
            System.currentTimeMillis(),
            DateUtils.MINUTE_IN_MILLIS
        )

        holder.titleView.text = session.title
        holder.metaView.text = "${session.modelDisplayName} - $updated"
        holder.previewView.text = session.preview.ifBlank {
            holder.itemView.context.getString(R.string.saved_chat_fallback_preview)
        }
        holder.titleView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp.coerceAtLeast(15f))
        holder.metaView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 2f).coerceAtLeast(12f))
        holder.previewView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 1f).coerceAtLeast(13f))
        holder.itemView.setOnClickListener { onSessionSelected(session) }
    }

    class SavedSessionViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleView: TextView = view.findViewById(R.id.sessionTitle)
        val metaView: TextView = view.findViewById(R.id.sessionMeta)
        val previewView: TextView = view.findViewById(R.id.sessionPreview)
    }

    private object DiffCallback : DiffUtil.ItemCallback<ChatSessionSummary>() {
        override fun areItemsTheSame(oldItem: ChatSessionSummary, newItem: ChatSessionSummary): Boolean {
            return oldItem.sessionId == newItem.sessionId
        }

        override fun areContentsTheSame(oldItem: ChatSessionSummary, newItem: ChatSessionSummary): Boolean {
            return oldItem == newItem
        }
    }
}
