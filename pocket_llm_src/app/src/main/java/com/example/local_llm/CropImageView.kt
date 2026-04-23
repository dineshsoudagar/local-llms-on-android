package com.example.local_llm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class CropImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private enum class DragHandle {
        NONE,
        MOVE,
        LEFT,
        TOP,
        RIGHT,
        BOTTOM,
        TOP_LEFT,
        TOP_RIGHT,
        BOTTOM_LEFT,
        BOTTOM_RIGHT
    }

    private val density = resources.displayMetrics.density
    private val imageRect = RectF()
    private val cropRect = RectF()
    private val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val overlayPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(150, 0, 0, 0)
        style = Paint.Style.FILL
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f * density
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(175, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 1f * density
    }
    private val handlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private val handleTouchRadius = 28f * density
    private val handleDrawRadius = 5f * density
    private val minimumCropSizePx = 80f * density
    private var bitmap: Bitmap? = null
    private var activeHandle = DragHandle.NONE
    private var lastTouchX = 0f
    private var lastTouchY = 0f

    fun setImageBitmap(bitmap: Bitmap) {
        this.bitmap = bitmap
        updateImageRect()
        invalidate()
    }

    fun clearImage() {
        bitmap = null
        imageRect.setEmpty()
        cropRect.setEmpty()
        invalidate()
    }

    fun cropBitmap(): Bitmap {
        val source = bitmap ?: throw IllegalStateException("No bitmap set for crop.")
        if (imageRect.width() <= 0f || imageRect.height() <= 0f || cropRect.width() <= 0f || cropRect.height() <= 0f) {
            return Bitmap.createBitmap(source, 0, 0, source.width, source.height)
        }

        val scaleX = source.width.toFloat() / imageRect.width()
        val scaleY = source.height.toFloat() / imageRect.height()
        val left = ((cropRect.left - imageRect.left) * scaleX)
            .roundToInt()
            .coerceIn(0, source.width - 1)
        val top = ((cropRect.top - imageRect.top) * scaleY)
            .roundToInt()
            .coerceIn(0, source.height - 1)
        val right = ((cropRect.right - imageRect.left) * scaleX)
            .roundToInt()
            .coerceIn(left + 1, source.width)
        val bottom = ((cropRect.bottom - imageRect.top) * scaleY)
            .roundToInt()
            .coerceIn(top + 1, source.height)

        return Bitmap.createBitmap(source, left, top, right - left, bottom - top)
    }

    override fun onSizeChanged(width: Int, height: Int, oldWidth: Int, oldHeight: Int) {
        super.onSizeChanged(width, height, oldWidth, oldHeight)
        updateImageRect()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.BLACK)
        val source = bitmap ?: return
        if (imageRect.width() <= 0f || imageRect.height() <= 0f) {
            return
        }

        canvas.drawBitmap(source, null, imageRect, bitmapPaint)
        drawCropOverlay(canvas)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (bitmap == null || cropRect.width() <= 0f || cropRect.height() <= 0f) {
            return false
        }

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                activeHandle = findDragHandle(event.x, event.y)
                if (activeHandle == DragHandle.NONE) {
                    return false
                }
                parent?.requestDisallowInterceptTouchEvent(true)
                lastTouchX = event.x
                lastTouchY = event.y
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = event.x - lastTouchX
                val dy = event.y - lastTouchY
                updateCropRect(activeHandle, dx, dy)
                lastTouchX = event.x
                lastTouchY = event.y
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP,
            MotionEvent.ACTION_CANCEL -> {
                activeHandle = DragHandle.NONE
                parent?.requestDisallowInterceptTouchEvent(false)
                return true
            }
        }

        return super.onTouchEvent(event)
    }

    private fun updateImageRect() {
        val source = bitmap ?: return
        val availableWidth = width - paddingLeft - paddingRight
        val availableHeight = height - paddingTop - paddingBottom
        if (availableWidth <= 0 || availableHeight <= 0 || source.width <= 0 || source.height <= 0) {
            return
        }

        val scale = min(
            availableWidth.toFloat() / source.width.toFloat(),
            availableHeight.toFloat() / source.height.toFloat()
        )
        val drawnWidth = source.width * scale
        val drawnHeight = source.height * scale
        val left = paddingLeft + ((availableWidth - drawnWidth) / 2f)
        val top = paddingTop + ((availableHeight - drawnHeight) / 2f)
        imageRect.set(left, top, left + drawnWidth, top + drawnHeight)
        resetCropRect()
    }

    private fun resetCropRect() {
        if (imageRect.width() <= 0f || imageRect.height() <= 0f) {
            cropRect.setEmpty()
            return
        }

        val horizontalInset = imageRect.width() * 0.08f
        val verticalInset = imageRect.height() * 0.08f
        cropRect.set(
            imageRect.left + horizontalInset,
            imageRect.top + verticalInset,
            imageRect.right - horizontalInset,
            imageRect.bottom - verticalInset
        )
        clampCropRect()
    }

    private fun drawCropOverlay(canvas: Canvas) {
        canvas.drawRect(imageRect.left, imageRect.top, imageRect.right, cropRect.top, overlayPaint)
        canvas.drawRect(imageRect.left, cropRect.bottom, imageRect.right, imageRect.bottom, overlayPaint)
        canvas.drawRect(imageRect.left, cropRect.top, cropRect.left, cropRect.bottom, overlayPaint)
        canvas.drawRect(cropRect.right, cropRect.top, imageRect.right, cropRect.bottom, overlayPaint)

        val thirdWidth = cropRect.width() / 3f
        val thirdHeight = cropRect.height() / 3f
        canvas.drawLine(cropRect.left + thirdWidth, cropRect.top, cropRect.left + thirdWidth, cropRect.bottom, gridPaint)
        canvas.drawLine(cropRect.left + (thirdWidth * 2f), cropRect.top, cropRect.left + (thirdWidth * 2f), cropRect.bottom, gridPaint)
        canvas.drawLine(cropRect.left, cropRect.top + thirdHeight, cropRect.right, cropRect.top + thirdHeight, gridPaint)
        canvas.drawLine(cropRect.left, cropRect.top + (thirdHeight * 2f), cropRect.right, cropRect.top + (thirdHeight * 2f), gridPaint)

        canvas.drawRect(cropRect, borderPaint)
        drawHandle(canvas, cropRect.left, cropRect.top)
        drawHandle(canvas, cropRect.centerX(), cropRect.top)
        drawHandle(canvas, cropRect.right, cropRect.top)
        drawHandle(canvas, cropRect.left, cropRect.centerY())
        drawHandle(canvas, cropRect.right, cropRect.centerY())
        drawHandle(canvas, cropRect.left, cropRect.bottom)
        drawHandle(canvas, cropRect.centerX(), cropRect.bottom)
        drawHandle(canvas, cropRect.right, cropRect.bottom)
    }

    private fun drawHandle(canvas: Canvas, x: Float, y: Float) {
        canvas.drawCircle(x, y, handleDrawRadius, handlePaint)
    }

    private fun findDragHandle(x: Float, y: Float): DragHandle {
        if (isNear(x, y, cropRect.left, cropRect.top)) return DragHandle.TOP_LEFT
        if (isNear(x, y, cropRect.right, cropRect.top)) return DragHandle.TOP_RIGHT
        if (isNear(x, y, cropRect.left, cropRect.bottom)) return DragHandle.BOTTOM_LEFT
        if (isNear(x, y, cropRect.right, cropRect.bottom)) return DragHandle.BOTTOM_RIGHT

        if (abs(x - cropRect.left) <= handleTouchRadius && y >= cropRect.top && y <= cropRect.bottom) {
            return DragHandle.LEFT
        }
        if (abs(x - cropRect.right) <= handleTouchRadius && y >= cropRect.top && y <= cropRect.bottom) {
            return DragHandle.RIGHT
        }
        if (abs(y - cropRect.top) <= handleTouchRadius && x >= cropRect.left && x <= cropRect.right) {
            return DragHandle.TOP
        }
        if (abs(y - cropRect.bottom) <= handleTouchRadius && x >= cropRect.left && x <= cropRect.right) {
            return DragHandle.BOTTOM
        }
        if (cropRect.contains(x, y)) {
            return DragHandle.MOVE
        }
        return DragHandle.NONE
    }

    private fun isNear(x: Float, y: Float, targetX: Float, targetY: Float): Boolean {
        return hypot((x - targetX).toDouble(), (y - targetY).toDouble()) <= handleTouchRadius.toDouble()
    }

    private fun updateCropRect(handle: DragHandle, dx: Float, dy: Float) {
        if (handle == DragHandle.NONE || imageRect.width() <= 0f || imageRect.height() <= 0f) {
            return
        }

        val next = RectF(cropRect)
        when (handle) {
            DragHandle.MOVE -> {
                next.offset(dx, dy)
                clampMovedRect(next)
            }
            DragHandle.LEFT -> setLeft(next, next.left + dx)
            DragHandle.TOP -> setTop(next, next.top + dy)
            DragHandle.RIGHT -> setRight(next, next.right + dx)
            DragHandle.BOTTOM -> setBottom(next, next.bottom + dy)
            DragHandle.TOP_LEFT -> {
                setLeft(next, next.left + dx)
                setTop(next, next.top + dy)
            }
            DragHandle.TOP_RIGHT -> {
                setRight(next, next.right + dx)
                setTop(next, next.top + dy)
            }
            DragHandle.BOTTOM_LEFT -> {
                setLeft(next, next.left + dx)
                setBottom(next, next.bottom + dy)
            }
            DragHandle.BOTTOM_RIGHT -> {
                setRight(next, next.right + dx)
                setBottom(next, next.bottom + dy)
            }
            DragHandle.NONE -> Unit
        }
        cropRect.set(next)
        clampCropRect()
    }

    private fun setLeft(rect: RectF, value: Float) {
        val maxLeft = max(imageRect.left, rect.right - effectiveMinimumCropSize())
        rect.left = value.coerceIn(imageRect.left, maxLeft)
    }

    private fun setTop(rect: RectF, value: Float) {
        val maxTop = max(imageRect.top, rect.bottom - effectiveMinimumCropSize())
        rect.top = value.coerceIn(imageRect.top, maxTop)
    }

    private fun setRight(rect: RectF, value: Float) {
        val minRight = min(imageRect.right, rect.left + effectiveMinimumCropSize())
        rect.right = value.coerceIn(minRight, imageRect.right)
    }

    private fun setBottom(rect: RectF, value: Float) {
        val minBottom = min(imageRect.bottom, rect.top + effectiveMinimumCropSize())
        rect.bottom = value.coerceIn(minBottom, imageRect.bottom)
    }

    private fun clampMovedRect(rect: RectF) {
        if (rect.left < imageRect.left) {
            rect.offset(imageRect.left - rect.left, 0f)
        }
        if (rect.right > imageRect.right) {
            rect.offset(imageRect.right - rect.right, 0f)
        }
        if (rect.top < imageRect.top) {
            rect.offset(0f, imageRect.top - rect.top)
        }
        if (rect.bottom > imageRect.bottom) {
            rect.offset(0f, imageRect.bottom - rect.bottom)
        }
    }

    private fun clampCropRect() {
        cropRect.intersect(imageRect)
        val minSize = effectiveMinimumCropSize()
        if (cropRect.width() < minSize) {
            val centerX = cropRect.centerX()
            cropRect.left = (centerX - (minSize / 2f)).coerceAtLeast(imageRect.left)
            cropRect.right = (cropRect.left + minSize).coerceAtMost(imageRect.right)
            cropRect.left = (cropRect.right - minSize).coerceAtLeast(imageRect.left)
        }
        if (cropRect.height() < minSize) {
            val centerY = cropRect.centerY()
            cropRect.top = (centerY - (minSize / 2f)).coerceAtLeast(imageRect.top)
            cropRect.bottom = (cropRect.top + minSize).coerceAtMost(imageRect.bottom)
            cropRect.top = (cropRect.bottom - minSize).coerceAtLeast(imageRect.top)
        }
    }

    private fun effectiveMinimumCropSize(): Float {
        val imageLimit = min(imageRect.width(), imageRect.height()).coerceAtLeast(1f)
        return min(minimumCropSizePx, imageLimit)
    }
}
