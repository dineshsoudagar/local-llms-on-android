package com.example.local_llm

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.text.Editable
import android.text.TextWatcher
import android.text.method.HideReturnsTransformationMethod
import android.text.method.PasswordTransformationMethod
import android.view.LayoutInflater
import android.view.View
import android.widget.EditText
import android.widget.ImageButton
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/** Pro-only LAN server. Found by [LanServerFeature.instance] through reflection. */
object LanServerFeatureImpl : LanServerFeature {
    override fun isRunning(): Boolean = LanServerService.isRunning()

    override fun drawerState(context: Context): LanServerFeature.DrawerState =
        when (LanServerStateStore.read(context).status) {
            LanServerStatus.RUNNING -> LanServerFeature.DrawerState.RUNNING
            LanServerStatus.STARTING -> LanServerFeature.DrawerState.STARTING
            else -> LanServerFeature.DrawerState.OFF
        }

    override fun showDialog(host: LanServerFeature.Host) = with(host) { with(activity) {
        val dialogBuilder = MaterialAlertDialogBuilder(this)
        val dialogView = LayoutInflater.from(dialogBuilder.context)
            .inflate(R.layout.dialog_lan_server, null)
        // Keep Start/Stop visible: the options scroll inside about 58% of the screen height.
        val optionsScroll: android.widget.ScrollView = dialogView.findViewById(R.id.lanServerScroll)
        optionsScroll.post {
            val maxHeight = (resources.displayMetrics.heightPixels * 0.58f).toInt()
            val content = optionsScroll.getChildAt(0)
            if (content != null && content.height > maxHeight) {
                optionsScroll.layoutParams = optionsScroll.layoutParams.apply { height = maxHeight }
            }
        }
        val statusView: TextView = dialogView.findViewById(R.id.lanServerStatus)
        val endpointView: TextView = dialogView.findViewById(R.id.lanServerEndpoint)
        val apiEndpointView: TextView = dialogView.findViewById(R.id.lanServerApiEndpoint)
        val statusDot: View = dialogView.findViewById(R.id.lanServerStatusDot)
        val statusDetailView: TextView = dialogView.findViewById(R.id.lanServerStatusDetail)
        val copyApiUrlLink: View = dialogView.findViewById(R.id.lanServerCopyApiUrlLink)
        val passwordInput: EditText = dialogView.findViewById(R.id.lanServerPasswordInput)
        val passwordVisibilityButton: ImageButton = dialogView.findViewById(R.id.lanServerPasswordVisibilityButton)
        val savePasswordButton: MaterialButton = dialogView.findViewById(R.id.lanServerSavePasswordButton)
        val copyUrlLink: View = dialogView.findViewById(R.id.lanServerCopyUrlLink)
        val startButton: MaterialButton = dialogView.findViewById(R.id.lanServerStartButton)
        val closeButton: MaterialButton = dialogView.findViewById(R.id.lanServerCloseButton)
        val passwordWarning: TextView = dialogView.findViewById(R.id.lanServerPasswordWarning)
        val modelButton: MaterialButton = dialogView.findViewById(R.id.lanServerModelButton)
        val startReasonView: TextView = dialogView.findViewById(R.id.lanServerStartReason)
        closeButton.stateListAnimator = null
        var savedPassword = LanServerStateStore.getPassword(this)
        var passwordVisible = false
        passwordInput.setText(savedPassword.orEmpty())
        passwordInput.hint = if (LanServerStateStore.hasPassword(this)) {
            getString(R.string.lan_server_password_change_hint)
        } else {
            getString(R.string.lan_server_password_hint)
        }
        val dialog = dialogBuilder
            .setView(ScrollView(dialogBuilder.context).apply {
                isFillViewport = true
                overScrollMode = View.OVER_SCROLL_IF_CONTENT_SCROLLS
                addView(dialogView)
            })
            .create()

        fun renderState() {
            val storedState = LanServerStateStore.read(this)
            if (storedState.isActive && !LanServerService.isRunning()) {
                LanServerStateStore.markStopped(this)
            }
            val state = LanServerStateStore.read(this)
            updateKeepScreenOn()
            statusView.text = when (state.status) {
                LanServerStatus.STOPPED -> getString(R.string.lan_server_status_stopped)
                LanServerStatus.STARTING -> getString(R.string.lan_server_status_starting)
                LanServerStatus.RUNNING -> getString(R.string.lan_server_status_running)
                LanServerStatus.FAILED -> getString(R.string.lan_server_status_failed) +
                    ": " + state.errorMessage.orEmpty()
            }
            statusDetailView.text = when (state.status) {
                LanServerStatus.RUNNING -> getString(R.string.lan_server_detail_running)
                LanServerStatus.STARTING -> getString(R.string.lan_server_starting)
                else -> getString(R.string.lan_server_detail_stopped)
            }
            statusDot.backgroundTintList = ColorStateList.valueOf(
                when (state.status) {
                    LanServerStatus.RUNNING -> ContextCompat.getColor(this, R.color.status_ready_background)
                    LanServerStatus.STARTING -> ContextCompat.getColor(this, R.color.status_loading_background)
                    LanServerStatus.FAILED -> ContextCompat.getColor(this, R.color.stop_button_fill)
                    LanServerStatus.STOPPED -> resolveThemeColor(R.attr.colorStatusText)
                }
            )
            endpointView.text = state.endpoint ?: "—"
            apiEndpointView.text = state.endpoint?.let(::lanApiBaseUrl) ?: "—"
            val model = currentModel
            val chatState = chatController?.state?.value
            val modelReady = model != null && chatState?.isReady == true
            val modelLoading = model != null && !modelReady && chatState?.isLoading == true
            modelButton.text = when {
                model == null -> getString(R.string.lan_server_model_none)
                modelReady -> getString(R.string.lan_server_model_ready, model.displayName)
                modelLoading -> getString(R.string.lan_server_model_loading, model.displayName)
                else -> getString(R.string.lan_server_model_not_loaded, model.displayName)
            }
            modelButton.isEnabled = !state.isActive
            modelButton.alpha = if (modelButton.isEnabled) 1f else 0.4f
            val reason = when {
                state.isActive -> null
                modelLoading -> getString(R.string.lan_server_reason_loading)
                !modelReady && installedModels.isEmpty() -> getString(R.string.lan_server_reason_download)
                !modelReady -> getString(R.string.lan_server_reason_model)
                LanServerStateStore.hasPassword(this) &&
                    passwordInput.text.toString() != savedPassword.orEmpty() ->
                    getString(R.string.lan_server_password_save_first)
                else -> null
            }
            startReasonView.text = reason
            startReasonView.visibility = if (reason == null) View.GONE else View.VISIBLE
            closeButton.isEnabled = state.isActive
            closeButton.alpha = if (state.isActive) 1f else 0.4f
            startButton.isEnabled = !state.isActive && modelReady &&
                LanServerStateStore.hasPassword(this) &&
                passwordInput.text.toString() == savedPassword.orEmpty()
            val stopRed = ContextCompat.getColor(this, R.color.stop_button_fill)
            closeButton.backgroundTintList = ColorStateList.valueOf(
                if (state.isActive) stopRed else ContextCompat.getColor(this, R.color.stop_soft_fill)
            )
            closeButton.setTextColor(if (state.isActive) Color.WHITE else stopRed)
            startButton.alpha = if (startButton.isEnabled) 1f else 0.4f
            passwordWarning.visibility = if (!state.isActive && !LanServerStateStore.hasPassword(this)) {
                View.VISIBLE
            } else {
                View.GONE
            }
            val webUiAvailable = state.status == LanServerStatus.RUNNING && !state.endpoint.isNullOrBlank()
            copyUrlLink.isEnabled = webUiAvailable
            copyApiUrlLink.isEnabled = webUiAvailable
            copyUrlLink.alpha = if (webUiAvailable) 1f else 0.4f
            copyApiUrlLink.alpha = if (webUiAvailable) 1f else 0.4f
            passwordInput.isEnabled = !state.isActive
            savePasswordButton.visibility = if (!state.isActive &&
                passwordInput.text.toString() != savedPassword.orEmpty()
            ) View.VISIBLE else View.GONE
        }

        passwordVisibilityButton.setOnClickListener {
            passwordVisible = !passwordVisible
            passwordInput.transformationMethod = if (passwordVisible) {
                HideReturnsTransformationMethod.getInstance()
            } else {
                PasswordTransformationMethod.getInstance()
            }
            passwordInput.setSelection(passwordInput.text.length)
            passwordVisibilityButton.setImageResource(
                if (passwordVisible) R.drawable.ic_visibility_off else R.drawable.ic_visibility
            )
            passwordVisibilityButton.contentDescription = getString(
                if (passwordVisible) R.string.lan_server_hide_password else R.string.lan_server_show_password
            )
        }
        passwordInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = renderState()
            override fun afterTextChanged(s: Editable?) = Unit
        })
        savePasswordButton.setOnClickListener {
            val password = passwordInput.text.toString()
            if (password.length < LanServerStateStore.MIN_PASSWORD_LENGTH) {
                showTransientMessage(getString(
                    R.string.lan_server_password_too_short,
                    LanServerStateStore.MIN_PASSWORD_LENGTH
                ))
                return@setOnClickListener
            }
            if (runCatching { LanServerStateStore.setPassword(this, password) }.isFailure) {
                showTransientMessage(getString(R.string.lan_server_password_save_failed))
                return@setOnClickListener
            }
            savedPassword = password
            renderState()
        }

        val refreshRunnable = object : Runnable {
            override fun run() {
                if (!dialog.isShowing) return
                renderState()
                dialogView.postDelayed(this, 500L)
            }
        }

        modelButton.setOnClickListener {
            val models = installedModels
            val popup = PopupMenu(this, modelButton)
            models.forEachIndexed { index, descriptor ->
                popup.menu.add(0, index, index, descriptor.displayName)
            }
            popup.menu.add(0, models.size, models.size, getString(R.string.lan_server_manage_models))
            popup.setOnMenuItemClickListener { item ->
                if (item.itemId == models.size) openManageModels() else selectModel(models[item.itemId])
                renderState()
                true
            }
            popup.show()
        }

        startButton.setOnClickListener {
            val model = currentModel
            val controller = chatController
            if (model == null || controller?.state?.value?.isReady != true) {
                showTransientMessage(getString(R.string.model_required_message))
                return@setOnClickListener
            }
            if (controller.state.value.isGenerating) {
                showTransientMessage(getString(R.string.lan_server_generation_blocked))
                return@setOnClickListener
            }
            if (passwordInput.text.toString() != savedPassword.orEmpty()) {
                showTransientMessage(getString(R.string.lan_server_password_save_first))
                return@setOnClickListener
            }
            if (!LanServerStateStore.hasPassword(this)) {
                showTransientMessage(getString(R.string.lan_server_password_required))
                return@setOnClickListener
            }
            SharedChatControllerRegistry.register(model.id, controller)
            LanServerService.start(this, model.id)
            renderState()
        }
        copyUrlLink.setOnClickListener {
            val endpoint = LanServerStateStore.read(this).endpoint
            if (endpoint.isNullOrBlank()) {
                showTransientMessage(getString(R.string.lan_server_not_running))
                return@setOnClickListener
            }
            getSystemService(ClipboardManager::class.java)?.setPrimaryClip(
                ClipData.newPlainText(getString(R.string.lan_server_url_label), endpoint)
            )
            showTransientMessage(getString(R.string.lan_server_url_copied))
        }
        copyApiUrlLink.setOnClickListener {
            val apiEndpoint = LanServerStateStore.read(this).endpoint?.let(::lanApiBaseUrl)
            if (apiEndpoint.isNullOrBlank()) {
                showTransientMessage(getString(R.string.lan_server_not_running))
                return@setOnClickListener
            }
            getSystemService(ClipboardManager::class.java)?.setPrimaryClip(
                ClipData.newPlainText(getString(R.string.lan_server_api_url_label), apiEndpoint)
            )
            showTransientMessage(getString(R.string.lan_server_api_url_copied))
        }
        closeButton.setOnClickListener {
            if (LanServerStateStore.read(this).isActive) {
                LanServerService.stop(this)
                window.decorView.postDelayed({ updateKeepScreenOn() }, 1500)
            }
            dialog.dismiss()
        }
        dialog.setOnShowListener {
            renderState()
            dialogView.post(refreshRunnable)
        }
        dialog.setOnDismissListener {
            dialogView.removeCallbacks(refreshRunnable)
        }
        dialog.show()
        renderState()
    } }
}

private fun lanApiBaseUrl(webUiUrl: String): String {
    val normalized = webUiUrl.trimEnd('/')
    return if (normalized.endsWith("/ui")) {
        normalized.removeSuffix("/ui") + "/v1"
    } else {
        "$normalized/v1"
    }
}
