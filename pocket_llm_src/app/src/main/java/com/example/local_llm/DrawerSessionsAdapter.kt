package com.example.local_llm

import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.text.format.DateUtils
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView

class DrawerSessionsAdapter(
    private var fontSizeSp: Float,
    private val onSessionSelected: (ChatSessionSummary) -> Unit,
    private val onMenuRequested: (ChatSessionSummary, View) -> Unit
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
        holder.itemView.setOnLongClickListener {
            onMenuRequested(session, holder.menuView)
            true
        }
        holder.menuView.setOnClickListener { onMenuRequested(session, holder.menuView) }
        holder.dividerView.visibility = if (position == 0) View.GONE else View.VISIBLE
        holder.timeView.text = formatTime(holder.itemView, session.updatedAtMillis)
    }

    /** Time today, weekday within the last week, date otherwise. */
    private fun formatTime(view: View, millis: Long): String {
        val now = System.currentTimeMillis()
        val flags = when {
            DateUtils.isToday(millis) -> DateUtils.FORMAT_SHOW_TIME
            now - millis < 6 * DateUtils.DAY_IN_MILLIS -> DateUtils.FORMAT_SHOW_WEEKDAY or DateUtils.FORMAT_ABBREV_WEEKDAY
            else -> DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_ABBREV_MONTH or DateUtils.FORMAT_NO_YEAR
        }
        return DateUtils.formatDateTime(view.context, millis, flags)
    }

    fun updateFontSize(updatedFontSizeSp: Float) {
        fontSizeSp = updatedFontSizeSp
        notifyItemRangeChanged(0, itemCount)
    }

    class DrawerSessionViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val titleView: TextView = view.findViewById(R.id.drawerSessionTitle)
        val timeView: TextView = view.findViewById(R.id.drawerSessionTime)
        val dividerView: View = view.findViewById(R.id.drawerSessionDivider)
        val menuView: View = view.findViewById(R.id.drawerSessionMenu)
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
