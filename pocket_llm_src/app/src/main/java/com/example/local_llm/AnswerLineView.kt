package com.example.local_llm

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SweepGradient
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import androidx.core.graphics.ColorUtils

/**
 * A thin accent line that travels around the border of the composer pill while the model answers.
 * Place it over the pill with the same size and corner radius; it is hidden otherwise.
 */
class AnswerLineView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {
    private val density = resources.displayMetrics.density
    private val strokeWidth = 2f * density
    private val cornerRadius = 28f * density
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = this@AnswerLineView.strokeWidth
        strokeCap = Paint.Cap.ROUND
    }
    private val rect = RectF()
    private val matrix = Matrix()
    private var sweep: SweepGradient? = null
    private var sweepColor = 0
    private var phase = 0f
    private val animator = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 2200L
        repeatCount = ValueAnimator.INFINITE
        interpolator = LinearInterpolator()
        addUpdateListener { phase = it.animatedValue as Float; invalidate() }
    }

    // No size of its own: it only draws over the pill, which gives the size. Without this the view asks for the
    // whole screen height inside a wrap_content parent and pushes the composer to the top.
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        setMeasuredDimension(resolveSize(0, widthMeasureSpec), resolveSize(0, heightMeasureSpec))
    }

    /** Runs while [active]; nothing is drawn when it is off. */
    fun setActive(active: Boolean) {
        visibility = if (active) VISIBLE else GONE
        if (active) { if (!animator.isStarted) animator.start() } else animator.cancel()
    }

    override fun onDetachedFromWindow() {
        animator.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        val accent = context.themeColor(R.attr.colorSendFill)
        val cx = width / 2f
        val cy = height / 2f
        if (sweep == null || sweepColor != accent) {
            // A comet: transparent tail fading in to a bright head, then a hard cut.
            sweep = SweepGradient(
                cx, cy,
                intArrayOf(
                    ColorUtils.setAlphaComponent(accent, 0),
                    accent,
                    ColorUtils.setAlphaComponent(accent, 0),
                    ColorUtils.setAlphaComponent(accent, 0)
                ),
                floatArrayOf(0f, 0.3f, 0.301f, 1f)
            )
            sweepColor = accent
        }
        matrix.setRotate(phase * 360f, cx, cy)
        sweep!!.setLocalMatrix(matrix)
        paint.shader = sweep
        val inset = strokeWidth / 2f
        rect.set(inset, inset, width - inset, height - inset)
        val radius = minOf(cornerRadius, rect.height() / 2f)
        canvas.drawRoundRect(rect, radius, radius, paint)
    }
}
