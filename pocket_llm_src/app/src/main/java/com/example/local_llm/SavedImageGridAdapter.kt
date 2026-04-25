package com.example.local_llm

import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class SavedImageGridAdapter(
    private val fontSizeSp: Float,
    private val onOpenImage: (SavedImageSummary) -> Unit,
    private val onSelectionChanged: (Int) -> Unit
) : RecyclerView.Adapter<SavedImageGridAdapter.SavedImageGridViewHolder>() {

    private val images = mutableListOf<SavedImageSummary>()
    private val selectedImageIds = linkedSetOf<String>()

    fun submitImages(newImages: List<SavedImageSummary>) {
        images.clear()
        images.addAll(newImages)
        selectedImageIds.retainAll(newImages.mapTo(linkedSetOf()) { it.imageId })
        notifyDataSetChanged()
        onSelectionChanged(selectedImageIds.size)
    }

    fun selectedIds(): Set<String> = selectedImageIds.toSet()

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SavedImageGridViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_saved_image_grid, parent, false)
        return SavedImageGridViewHolder(view)
    }

    override fun onBindViewHolder(holder: SavedImageGridViewHolder, position: Int) {
        val image = images[position]
        val isSelected = image.imageId in selectedImageIds
        holder.titleView.text = image.displayName
        holder.titleView.setTextSize(TypedValue.COMPLEX_UNIT_SP, (fontSizeSp - 2f).coerceAtLeast(12f))
        val thumbnail = ImageBitmapLoader.decodeSampledBitmap(
            imageFile = File(image.filePath),
            requestedWidth = dp(holder.thumbnailView, 120),
            requestedHeight = dp(holder.thumbnailView, 120)
        )
        if (thumbnail != null) {
            holder.thumbnailView.setImageBitmap(thumbnail)
        } else {
            holder.thumbnailView.setImageResource(R.drawable.ic_image_24)
        }

        holder.selectedOverlay.visibility = if (isSelected) View.VISIBLE else View.GONE
        holder.selectedBadge.visibility = if (isSelected) View.VISIBLE else View.GONE
        holder.itemView.setOnClickListener {
            if (selectedImageIds.isEmpty()) {
                onOpenImage(image)
            } else {
                toggleSelection(holder.bindingAdapterPosition)
            }
        }
        holder.itemView.setOnLongClickListener {
            toggleSelection(holder.bindingAdapterPosition)
            true
        }
    }

    override fun getItemCount(): Int = images.size

    private fun toggleSelection(position: Int) {
        if (position !in images.indices) {
            return
        }

        val imageId = images[position].imageId
        if (imageId in selectedImageIds) {
            selectedImageIds.remove(imageId)
        } else {
            selectedImageIds.add(imageId)
        }
        notifyItemChanged(position)
        onSelectionChanged(selectedImageIds.size)
    }

    private fun dp(view: View, value: Int): Int {
        return TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP,
            value.toFloat(),
            view.resources.displayMetrics
        ).toInt()
    }

    class SavedImageGridViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val thumbnailView: ImageView = view.findViewById(R.id.imageGridThumbnail)
        val selectedOverlay: View = view.findViewById(R.id.imageGridSelectedOverlay)
        val selectedBadge: TextView = view.findViewById(R.id.imageGridSelectedBadge)
        val titleView: TextView = view.findViewById(R.id.imageGridTitle)
    }
}
