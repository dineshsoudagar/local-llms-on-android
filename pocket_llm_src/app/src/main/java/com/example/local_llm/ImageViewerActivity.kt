package com.example.local_llm

import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max

class ImageViewerActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_IMAGE_PATH = "image_path"
    }

    private var proGateInstance: ProGate? = null
    private val proGate: ProGate
        get() = proGateInstance ?: createProGate(this).also { proGateInstance = it }
    private lateinit var imageView: ImageView
    private lateinit var statusView: TextView
    private lateinit var toolButtons: List<MaterialButton>
    private var currentFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applyEdgeToEdge()
        setContentView(R.layout.activity_image_viewer)

        val toolbar: MaterialToolbar = findViewById(R.id.imageViewerToolbar)
        imageView = findViewById(R.id.imageViewerImage)
        statusView = findViewById(R.id.imageViewerStatus)
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
        currentFile = imageFile

        val removeBackgroundButton: MaterialButton = findViewById(R.id.imageViewerRemoveBackground)
        toolButtons = listOf(removeBackgroundButton)
        removeBackgroundButton.visibility = if (ImageGenFeature.isAvailable) View.VISIBLE else View.GONE
        removeBackgroundButton.setOnClickListener {
            val action = {
                // The free try is used only when the cut-out is saved.
                runTool(onSaved = {
                    if (ProUiFeature.isAvailable) ProUiFeature.instance!!.useTrial(this, proGate, ProFeature.IMAGE_GENERATION)
                }) { file ->
                    ImageGenFeature.instance!!.removeBackground(this, file) { step ->
                        runOnUiThread { statusView.text = step }
                    }
                }
            }
            if (ProUiFeature.isAvailable) {
                if (ProUiFeature.instance!!.hasTrial(this, proGate, ProFeature.IMAGE_GENERATION)) {
                    action()
                }
            } else {
                proGate.requireUnlocked(this, ProFeature.IMAGE_GENERATION, action)
            }
        }
    }

    override fun onDestroy() {
        proGateInstance?.close()
        super.onDestroy()
    }

    /** Runs a Pro image tool on the shown image, then shows the new image (saved to the library). */
    /** Runs [tool] on the shown image; [onSaved] runs only when it produced a new image. */
    private fun runTool(onSaved: () -> Unit = {}, tool: suspend (File) -> SavedImageEntry) {
        val source = currentFile ?: return
        if (ImageGenFeature.instance?.isBusy() == true) {
            Toast.makeText(this, R.string.image_tool_busy, Toast.LENGTH_SHORT).show()
            return
        }
        toolButtons.forEach { it.isEnabled = false }
        statusView.visibility = View.VISIBLE
        statusView.setText(R.string.image_tool_working)
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) { runCatching { tool(source) } }
            toolButtons.forEach { it.isEnabled = true }
            result.onSuccess { entry ->
                onSaved()
                val file = File(entry.filePath)
                currentFile = file
                val metrics = resources.displayMetrics
                imageView.setImageBitmap(
                    ImageBitmapLoader.decodeSampledBitmap(
                        file,
                        max(metrics.widthPixels, 1),
                        max(metrics.heightPixels, 1)
                    )
                )
                statusView.setText(R.string.image_tool_saved)
            }.onFailure {
                statusView.text = getString(R.string.image_gen_failed, it.message ?: "")
            }
        }
    }
}
