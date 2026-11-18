package com.example.local_llm

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.Shader
import android.graphics.SweepGradient
import android.provider.Settings
import android.util.AttributeSet
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.animation.LinearInterpolator
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.res.ResourcesCompat
import androidx.core.graphics.ColorUtils
import com.google.android.material.progressindicator.LinearProgressIndicator
import kotlin.math.exp
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** Reads the system "animator duration scale"; 0 means the user asked for no motion. */
fun Context.animationsEnabled(): Boolean = try {
    Settings.Global.getFloat(contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) > 0f
} catch (_: Exception) {
    true
}

fun Context.themeColor(attr: Int, fallback: Int = 0xFFA98BFF.toInt()): Int {
    val tv = TypedValue()
    return if (theme.resolveAttribute(attr, tv, true)) tv.data else fallback
}

/** A glossy AI orb: colour clouds swirl inside a glass sphere with a turning rim light and a soft halo. */
class OrbView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val accent = context.themeColor(R.attr.colorSendFill)
    private val accent2 = context.themeColor(R.attr.colorAccentGradientEnd)
    private val halo = Paint(Paint.ANTI_ALIAS_FLAG)
    private val base = Paint(Paint.ANTI_ALIAS_FLAG)
    private val cloud = Paint(Paint.ANTI_ALIAS_FLAG).apply { xfermode = PorterDuffXfermode(PorterDuff.Mode.SCREEN) }
    private val rim = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val shine = Paint(Paint.ANTI_ALIAS_FLAG)
    private val clip = Path()
    private var phase = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 8000
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            phase = it.animatedValue as Float
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        setLayerType(LAYER_TYPE_HARDWARE, null)
        if (context.animationsEnabled()) animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height / 2f
        val r = min(width, height) / 2f
        val tau = 2f * Math.PI.toFloat()
        val breathe = 0.5f + 0.5f * sin(phase * tau * 4f)
        val black = 0xFF000000.toInt()
        // Halo.
        val haloR = r * (0.86f + 0.07f * breathe)
        halo.shader = RadialGradient(
            cx, cy, haloR,
            intArrayOf(ColorUtils.setAlphaComponent(accent2, 120), ColorUtils.setAlphaComponent(accent, 50), ColorUtils.setAlphaComponent(accent, 0)),
            floatArrayOf(0.5f, 0.72f, 1f), Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, haloR, halo)
        // Sphere with moving colour clouds.
        val s = r * (0.52f + 0.015f * breathe)
        clip.reset()
        clip.addCircle(cx, cy, s, Path.Direction.CW)
        canvas.save()
        canvas.clipPath(clip)
        base.shader = RadialGradient(
            cx, cy + s * 0.2f, s * 1.1f,
            intArrayOf(ColorUtils.blendARGB(accent2, black, 0.55f), ColorUtils.blendARGB(accent, black, 0.88f)),
            null, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, s, base)
        cloud(canvas, cx + s * 0.5f * cos(phase * tau), cy + s * 0.4f * sin(phase * tau * 2f), s * 1.0f, accent, 235)
        cloud(canvas, cx + s * 0.55f * cos(-phase * tau + 2f), cy + s * 0.5f * sin(-phase * tau + 2f), s * 0.95f, accent2, 225)
        cloud(canvas, cx + s * 0.3f * sin(phase * tau * 2f), cy + s * 0.4f * cos(phase * tau), s * 0.55f, ColorUtils.blendARGB(accent, 0xFFFFFFFF.toInt(), 0.6f), 190)
        // Depth: darker towards the lower rim.
        shine.shader = RadialGradient(
            cx, cy - s * 0.25f, s * 1.25f,
            intArrayOf(0x00000000, ColorUtils.setAlphaComponent(black, 150)),
            floatArrayOf(0.6f, 1f), Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx, cy, s, shine)
        canvas.restore()
        // Rim light turning around the sphere.
        rim.strokeWidth = r * 0.035f
        rim.shader = SweepGradient(
            cx, cy,
            intArrayOf(ColorUtils.setAlphaComponent(accent, 0), ColorUtils.setAlphaComponent(0xFFFFFFFF.toInt(), 230), ColorUtils.setAlphaComponent(accent2, 0)),
            floatArrayOf(0f, 0.25f, 0.5f)
        )
        canvas.save()
        canvas.rotate(phase * 360f * 2f, cx, cy)
        canvas.drawCircle(cx, cy, s, rim)
        canvas.restore()
        // Specular highlight.
        shine.shader = RadialGradient(
            cx - s * 0.38f, cy - s * 0.45f, s * 0.4f,
            intArrayOf(ColorUtils.setAlphaComponent(0xFFFFFFFF.toInt(), 150), 0x00FFFFFF),
            null, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(cx - s * 0.38f, cy - s * 0.45f, s * 0.4f, shine)
    }

    private fun cloud(canvas: Canvas, x: Float, y: Float, radius: Float, color: Int, alpha: Int) {
        cloud.shader = RadialGradient(
            x, y, radius,
            intArrayOf(ColorUtils.setAlphaComponent(color, alpha), ColorUtils.setAlphaComponent(color, 0)),
            null, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(x, y, radius, cloud)
    }
}

/**
 * Shown in the empty chat while a model loads: orb, rotating friendly lines and a progress bar that
 * eases towards 90% (the real load reports no percentage, so it never claims to be finished early).
 */
class ModelLoadingCard @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : LinearLayout(context, attrs) {
    private val title = TextView(context).apply {
        textSize = 18f
        gravity = Gravity.CENTER
        setTextColor(context.themeColor(R.attr.colorAssistantText))
        typeface = ResourcesCompat.getFont(context, R.font.inter_semibold)
    }
    private val line = TextView(context).apply {
        textSize = 14f
        gravity = Gravity.CENTER
        setTextColor(context.themeColor(R.attr.colorStatusText))
    }
    private val progress = LinearProgressIndicator(context).apply {
        max = 1000
        trackCornerRadius = (4 * resources.displayMetrics.density).toInt()
        setIndicatorColor(context.themeColor(R.attr.colorSendFill))
        trackColor = context.themeColor(R.attr.colorPocketOutline)
    }
    private val lines = intArrayOf(
        R.string.loading_line_1, R.string.loading_line_2, R.string.loading_line_3, R.string.loading_line_4
    )
    private var startedAt = 0L
    private var modelKey: String? = null
    private var expectedMillis = DEFAULT_EXPECTED_MILLIS
    private var finishing = false
    private var firstLoad = true
    private val prefs = context.getSharedPreferences("model_load_times", Context.MODE_PRIVATE)
    private val ticker = object : Runnable {
        override fun run() {
            val t = System.currentTimeMillis() - startedAt
            progress.setProgressCompat((1000 * progressFraction(t)).toInt(), true)
            val text = context.getString(lines[min(lines.size - 1, (3 * progressFraction(t) / 0.9f).toInt())])
            if (line.text.toString() != text) swapLine(text)
            postDelayed(this, 250)
        }
    }

    /**
     * Fraction of the bar for [t] ms into the load. Reaches about 90% at the expected time (learned from
     * earlier loads of this model) and then creeps towards 97% so a slow load never looks frozen.
     */
    private fun progressFraction(t: Long): Float {
        val e = expectedMillis.toFloat()
        val x = t / e
        return if (x < 1f) 0.9f * (1f - (1f - x) * (1f - x)) else 0.9f + 0.07f * (1f - exp(-(x - 1f)))
    }

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER
        val d = resources.displayMetrics.density
        setPadding((32 * d).toInt(), 0, (32 * d).toInt(), 0)
        addView(NodeOrbView(context), LayoutParams((96 * d).toInt(), (96 * d).toInt()))
        addView(title, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = (16 * d).toInt() })
        addView(line, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = (6 * d).toInt() })
        addView(progress, LayoutParams((220 * d).toInt(), LayoutParams.WRAP_CONTENT).apply { topMargin = (22 * d).toInt() })
        visibility = GONE
    }

    fun setModelName(name: String) {
        title.text = context.getString(R.string.loading_title, name)
    }

    /** Starts the card for [modelId]; the expected duration comes from this model's earlier loads. */
    fun show(modelId: String?) {
        if (visibility == VISIBLE && !finishing) return
        finishing = false
        removeCallbacks(finishRunnable)
        modelKey = modelId
        val known = modelId?.let { prefs.getLong(it, 0L) }?.takeIf { it > 0L }
        firstLoad = known == null
        expectedMillis = known ?: DEFAULT_EXPECTED_MILLIS
        startedAt = System.currentTimeMillis()
        progress.setProgressCompat(0, false)
        line.text = context.getString(lines[0])
        animate().cancel()
        alpha = 0f
        visibility = VISIBLE
        animate().alpha(1f).setDuration(if (context.animationsEnabled()) 300 else 0).start()
        removeCallbacks(ticker)
        post(ticker)
    }

    /**
     * Called when loading ended. On success the real duration is remembered and the bar runs to 100%. The
     * very first load of a model (duration unknown) stays up for at least [MIN_SHOWN_MILLIS] in total.
     */
    fun hide(success: Boolean) {
        if (visibility != VISIBLE || finishing) return
        finishing = true
        removeCallbacks(ticker)
        val elapsed = System.currentTimeMillis() - startedAt
        if (success) modelKey?.let { key ->
            val old = prefs.getLong(key, 0L)
            prefs.edit().putLong(key, if (old > 0L) (old + elapsed) / 2 else elapsed).apply()
            progress.setProgressCompat(1000, true)
            line.text = context.getString(R.string.loading_line_done)
        }
        // Only a first load (time unknown) is held for the minimum; later loads end when the model is ready.
        val minimum = if (firstLoad) max(0L, MIN_SHOWN_MILLIS - elapsed) else 0L
        val wait = if (success) minimum + 450L else 0L
        postDelayed(finishRunnable, if (context.animationsEnabled()) wait else 0L)
    }

    private val finishRunnable = Runnable {
        animate().alpha(0f).setDuration(if (context.animationsEnabled()) 250 else 0)
            .withEndAction { visibility = GONE; finishing = false; onHidden?.invoke() }.start()
    }

    /** Called once the card has fully faded out, so the greeting can appear after it and not during it. */
    var onHidden: (() -> Unit)? = null

    private fun swapLine(text: String) {
        if (!context.animationsEnabled()) {
            line.text = text
            return
        }
        line.animate().alpha(0f).translationY(-6f).setDuration(150).withEndAction {
            line.text = text
            line.translationY = 8f
            line.animate().alpha(1f).translationY(0f).setDuration(200).start()
        }.start()
    }

    override fun onDetachedFromWindow() {
        removeCallbacks(ticker)
        removeCallbacks(finishRunnable)
        super.onDetachedFromWindow()
    }

    private companion object {
        const val DEFAULT_EXPECTED_MILLIS = 15_000L
        const val MIN_SHOWN_MILLIS = 5_000L
    }
}
