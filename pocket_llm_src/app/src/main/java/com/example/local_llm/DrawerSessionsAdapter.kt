package com.example.local_llm

import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

class DrawerSessionsAdapter(
    private var fontSizeSp: Float,
    private val onSessionSelected: (ChatSessionSummary) -> Unit,
    private val onDeleteRequested: (ChatSessionSummary) -> Unit
) : ListAdapter<ChatSessionSummary, DrawerSessionsAdapter.DrawerSessionViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): DrawerSessionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_drawer_chat, parent, false)
        return DrawerSessionViewHolder(view)
    }

    override fun onBindViewHolder(holder: DrawerSessionViewHolder, position: Int) {
        val session = getItem(position)
        holder.titleView.text = session.title
        holder.titleView.setTextSize(TypedValue.COMPLEX_UNIT_SP, fontSizeSp.coerceAtLeast(14f))
        holder.itemView.setOnClickListener { onSessionSelected(session) }
        holder.deleteButton.setOnClickListener { onDeleteRequested(session) }
    }

    fun updateFontSize(updatedFontSizeSp: Float) {
        fontSizeSp = updatedFontSizeSp
        notifyItemRangeChanged(0, itemCount)
    }

    class DrawerSessionViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleView: TextView = view.findViewById(R.id.drawerSessionTitle)
        val deleteButton: ImageButton = view.findViewById(R.id.drawerSessionDeleteButton)
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
