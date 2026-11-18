package com.example.local_llm

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat

/**
 * Shown after the camera permission was denied. While Android still shows its own prompt the
 * dialog offers "Try again"; once the system stops asking (denied twice or "don't ask again")
 * it sends the user to the app's settings page instead.
 */
object CameraPermissionPrompt {

    fun isPermanentlyDenied(activity: Activity): Boolean =
        !ActivityCompat.shouldShowRequestPermissionRationale(activity, Manifest.permission.CAMERA)

    fun show(
        activity: Activity,
        onTryAgain: () -> Unit,
        onOpenSettings: () -> Unit,
        onCancel: () -> Unit
    ) {
        val permanent = isPermanentlyDenied(activity)
        AlertDialog.Builder(activity)
            .setTitle(R.string.camera_permission_title)
            .setMessage(if (permanent) R.string.camera_permission_settings_message else R.string.camera_permission_retry_message)
            .setPositiveButton(if (permanent) R.string.camera_permission_open_settings else R.string.camera_permission_try_again) { _, _ ->
                if (permanent) {
                    openAppSettings(activity)
                    onOpenSettings()
                } else {
                    onTryAgain()
                }
            }
            .setNegativeButton(R.string.camera_permission_not_now) { _, _ -> onCancel() }
            .setOnCancelListener { onCancel() }
            .show()
    }

    private fun openAppSettings(activity: Activity) {
        activity.startActivity(
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", activity.packageName, null))
        )
    }
}
