package com.example.local_llm

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RadialGradient
import android.graphics.Shader
import android.util.AttributeSet
import android.util.TypedValue
import android.view.View
import android.view.animation.LinearInterpolator
import androidx.core.graphics.ColorUtils
import kotlin.math.min
import kotlin.math.sin

/**
 * The Pocket LLM brand mark: ten nodes joined by curved lines, no bubble and no background.
 * The same network is the header logo, the app icon, the thinking orb and the model-loading orb.
 */
internal object NodeNetwork {
    /** Node centres in a 432 unit box (the old logo art). Left column, middle column, right column. */
    private val raw = arrayOf(
        floatArrayOf(131f, 137f), floatArrayOf(131f, 212f), floatArrayOf(131f, 287f),
        floatArrayOf(216f, 102f), floatArrayOf(219f, 177f), floatArrayOf(218f, 252f), floatArrayOf(217f, 327f),
        floatArrayOf(300f, 137f), floatArrayOf(300f, 212f), floatArrayOf(300f, 287f)
    )
    val edges = arrayOf(
        intArrayOf(0, 3), intArrayOf(0, 4), intArrayOf(1, 3), intArrayOf(1, 5), intArrayOf(2, 4), intArrayOf(2, 5),
        intArrayOf(2, 6), intArrayOf(3, 7), intArrayOf(4, 8), intArrayOf(5, 9), intArrayOf(6, 9)
    )

    /** Node centres as fractions of the drawing box, centred on 0.5 (the network is 0.75 tall, 0.38 wide). */
    val nodes: Array<FloatArray> = Array(raw.size) { i ->
        floatArrayOf(0.5f + (raw[i][0] - 215.5f) / 300f, 0.5f + (raw[i][1] - 214.5f) / 300f)
    }

    /** Horizontal position 0..1 of each node, used to run a pulse from left to right. */
    fun columnOf(i: Int): Float = (raw[i][0] - 131f) / (300f - 131f)

    /** Column 0..1 of the middle of an edge, used to light the lines the pulse runs along. */
    fun edgeColumn(e: IntArray): Float = (columnOf(e[0]) + columnOf(e[1])) / 2f

    const val NODE_RADIUS = 0.07f
    const val EDGE_WIDTH = 0.024f
}

/** Draws the network at a position and size. Colour comes from the theme: gradient or solid accent. */
internal class NodeNetworkPainter(private val context: Context) {
    private val gradientOn = run {
        val tv = TypedValue()
        !context.theme.resolveAttribute(R.attr.pocketLogoGradient, tv, true) || tv.data != 0
    }
    val accent = context.themeColor(R.attr.pocketGradient1)
    val accent2 = context.themeColor(R.attr.pocketGradient3)
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG)
    private val path = Path()

    /** [pulse] gives each node a 0..1 brightness boost; null draws the plain logo. */
    fun draw(
        canvas: Canvas, left: Float, top: Float, size: Float,
        pulse: ((Int) -> Float)? = null, edgePulse: ((IntArray) -> Float)? = null
    ) {
        val shader: Shader? = if (gradientOn) {
            LinearGradient(left, top, left + size, top + size, accent, accent2, Shader.TileMode.CLAMP)
        } else null
        val solid = context.themeColor(androidx.appcompat.R.attr.colorPrimary)
        line.shader = shader
        dot.shader = shader
        line.color = solid
        dot.color = solid
        line.strokeWidth = size * NodeNetwork.EDGE_WIDTH
        for (e in NodeNetwork.edges) {
            line.alpha = if (edgePulse == null) 170 else (110 + 145 * edgePulse(e)).toInt().coerceIn(0, 255)
            line.strokeWidth = size * NodeNetwork.EDGE_WIDTH * (1f + 0.6f * (edgePulse?.invoke(e) ?: 0f))
            val a = NodeNetwork.nodes[e[0]]
            val b = NodeNetwork.nodes[e[1]]
            val ax = left + a[0] * size
            val ay = top + a[1] * size
            val bx = left + b[0] * size
            val by = top + b[1] * size
            // Slight bow so the lines read as organic curves, like the original artwork.
            val mx = (ax + bx) / 2f
            val my = (ay + by) / 2f
            val bow = (if ((e[0] + e[1]) % 2 == 0) 1f else -1f) * size * 0.025f
            path.reset()
            path.moveTo(ax, ay)
            path.quadTo(mx, my + bow, bx, by)
            canvas.drawPath(path, line)
        }
        for (i in NodeNetwork.nodes.indices) {
            val n = NodeNetwork.nodes[i]
            val boost = pulse?.invoke(i) ?: 0f
            dot.alpha = if (pulse == null) 255 else (185 + 70 * boost).toInt().coerceIn(0, 255)
            canvas.drawCircle(left + n[0] * size, top + n[1] * size, size * NodeNetwork.NODE_RADIUS * (1f + 0.35f * boost), dot)
        }
    }
}

/** Static brand mark for the header, drawer and About dialog. */
class NodeLogoView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val painter = NodeNetworkPainter(context)

    override fun onDraw(canvas: Canvas) {
        val size = min(width, height).toFloat()
        painter.draw(canvas, (width - size) / 2f, (height - size) / 2f, size)
    }
}

/**
 * The thinking and loading orb: the network with a soft glow behind it and a pulse of light running from the
 * left column to the right. With animations off (system setting) it stays still.
 */
class NodeOrbView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val painter = NodeNetworkPainter(context)
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG)
    private var phase = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 2400
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener {
            phase = it.animatedValue as Float
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        if (context.animationsEnabled()) animator.start()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        val size = min(width, height).toFloat()
        val left = (width - size) / 2f
        val top = (height - size) / 2f
        val animated = animator.isRunning
        val breathe = if (animated) 0.5f + 0.5f * sin(phase * 2f * Math.PI.toFloat()) else 0.5f
        val haloR = size * (0.46f + 0.04f * breathe)
        glow.shader = RadialGradient(
            left + size / 2f, top + size / 2f, haloR,
            intArrayOf(ColorUtils.setAlphaComponent(painter.accent, 70), ColorUtils.setAlphaComponent(painter.accent2, 0)),
            null, Shader.TileMode.CLAMP
        )
        canvas.drawCircle(left + size / 2f, top + size / 2f, haloR, glow)
        // A light front sweeps left to right through the nodes and along the lines, then starts again off-screen
        // at the left. Every term is periodic in phase and the front is dark at both ends of the cycle, so the
        // loop has no jump.
        val front = -0.5f + 1.9f * phase
        fun bump(col: Float): Float {
            val t = (col - front) / 0.38f
            val v = (1f - t * t).coerceAtLeast(0f)
            return v * v
        }
        painter.draw(
            canvas, left, top, size,
            pulse = { i -> if (!animated) 0.5f else bump(NodeNetwork.columnOf(i)) },
            edgePulse = if (animated) { e -> bump(NodeNetwork.edgeColumn(e)) } else null
        )
    }
}
