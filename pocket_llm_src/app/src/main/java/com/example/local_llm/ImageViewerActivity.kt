package com.example.local_llm

import android.os.Bundle
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.appbar.MaterialToolbar
import java.io.File
import kotlin.math.max

class ImageViewerActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_PATH = "image_path"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_image_viewer)

        val toolbar: MaterialToolbar = findViewById(R.id.imageViewerToolbar)
        val imageView: ImageView = findViewById(R.id.imageViewerImage)
        toolbar.setNavigationOnClickListener { onBackPressedDispatcher.onBackPressed() }
        toolbar.title = ""

        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        val imageFile = imagePath?.let(::File)
        if (imageFile == null || !imageFile.exists()) {
            Toast.makeText(this, R.string.saved_image_missing, Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        val displayMetrics = resources.displayMetrics
        val previewBitmap = ImageBitmapLoader.decodeSampledBitmap(
            imageFile = imageFile,
            requestedWidth = max(displayMetrics.widthPixels, 1),
            requestedHeight = max(displayMetrics.heightPixels, 1)
        )
        if (previewBitmap == null) {
            Toast.makeText(this, R.string.saved_image_missing, Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        imageView.setImageBitmap(previewBitmap)
    }
}
