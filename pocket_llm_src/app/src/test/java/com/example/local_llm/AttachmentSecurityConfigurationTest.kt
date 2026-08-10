package com.example.local_llm

import android.content.pm.ServiceInfo
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.w3c.dom.Element

class AttachmentSecurityConfigurationTest {
    @Test
    fun legacyBackupExcludesPersistentAttachmentPayloads() {
        val root = parseMainSource("res/xml/backup_rules.xml").documentElement

        assertEquals("full-backup-content", root.tagName)
        assertTrue(root.hasFileExclusion("chat_attachments/"))
    }

    @Test
    fun android12BackupAndTransferExcludePersistentAttachmentPayloads() {
        val root = parseMainSource("res/xml/data_extraction_rules.xml").documentElement

        val cloudBackup = root.getElementsByTagName("cloud-backup").item(0) as Element
        val deviceTransfer = root.getElementsByTagName("device-transfer").item(0) as Element
        assertTrue(cloudBackup.hasFileExclusion("chat_attachments/"))
        assertTrue(deviceTransfer.hasFileExclusion("chat_attachments/"))
    }

    @Test
    fun workManagerForegroundServiceAndWorkerBothUseDataSync() {
        val manifest = parseMainSource("AndroidManifest.xml")
        val services = manifest.getElementsByTagName("service")
        val workManagerService = (0 until services.length)
            .map { services.item(it) as Element }
            .single { it.androidAttribute("name") == WORK_MANAGER_FOREGROUND_SERVICE }

        assertEquals("dataSync", workManagerService.androidAttribute("foregroundServiceType"))
        assertEquals(
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
            ATTACHMENT_FOREGROUND_SERVICE_TYPE
        )

        val permissions = manifest.getElementsByTagName("uses-permission")
        val permissionNames = (0 until permissions.length)
            .map { (permissions.item(it) as Element).androidAttribute("name") }
            .toSet()
        assertTrue("android.permission.FOREGROUND_SERVICE" in permissionNames)
        assertTrue("android.permission.FOREGROUND_SERVICE_DATA_SYNC" in permissionNames)
    }

    private fun parseMainSource(relativePath: String) = DocumentBuilderFactory.newInstance().apply {
        isNamespaceAware = true
    }.newDocumentBuilder().parse(mainSourceFile(relativePath))

    private fun mainSourceFile(relativePath: String): File {
        val moduleRelative = File("src/main", relativePath)
        return if (moduleRelative.exists()) moduleRelative else File("app/src/main", relativePath)
    }

    private fun Element.hasFileExclusion(path: String): Boolean {
        val exclusions = getElementsByTagName("exclude")
        return (0 until exclusions.length)
            .map { exclusions.item(it) as Element }
            .any { it.getAttribute("domain") == "file" && it.getAttribute("path") == path }
    }

    private fun Element.androidAttribute(name: String): String =
        getAttributeNS(ANDROID_NAMESPACE, name)

    private companion object {
        const val ANDROID_NAMESPACE = "http://schemas.android.com/apk/res/android"
        const val WORK_MANAGER_FOREGROUND_SERVICE =
            "androidx.work.impl.foreground.SystemForegroundService"
    }
}
