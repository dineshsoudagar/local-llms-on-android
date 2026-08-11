package com.example.local_llm

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WhisperModelInstallerTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun validStagedInstallationIsVerifiedAndInstalled() {
        val payloads = payloads("encoder-v1", "decoder-v1", "tokens-v1")
        val installer = installer("revision-1", payloads)
        val staging = installer.createStagingDirectory().also { writePayloads(it, payloads) }

        installer.verifyAndPromote(staging)

        assertTrue(installer.isInstalled())
        assertFalse(staging.exists())
    }

    @Test
    fun hashMismatchIsRejected() {
        val payloads = payloads("good", "decoder", "tokens")
        val installer = installer("revision-1", payloads)
        val staging = installer.createStagingDirectory().also { writePayloads(it, payloads) }
        File(staging, "encoder.onnx").writeBytes("baad".toByteArray())

        val error = runCatching { installer.verifyAndPromote(staging) }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("SHA-256"))
        assertFalse(installer.isInstalled())
    }

    @Test
    fun truncatedFileIsRejected() {
        val payloads = payloads("encoder", "decoder", "tokens")
        val installer = installer("revision-1", payloads)
        val staging = installer.createStagingDirectory().also { writePayloads(it, payloads) }
        File(staging, "decoder.onnx").writeBytes("dec".toByteArray())

        val error = runCatching { installer.verifyAndPromote(staging) }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(error!!.message.orEmpty().contains("expected"))
        assertFalse(installer.isInstalled())
    }

    @Test
    fun interruptedDownloadIsRemovedBeforeTheNextDownload() {
        val payloads = payloads("encoder", "decoder", "tokens")
        val root = temporaryFolder.newFolder("interrupted-root")
        val first = installer("revision-1", payloads, root = root)
        val staging = first.createStagingDirectory()
        File(staging, "encoder.onnx").writeBytes(payloads.getValue("encoder.onnx"))

        val restarted = installer("revision-1", payloads, root = root)
        val nextStaging = restarted.createStagingDirectory()

        assertFalse(staging.exists())
        assertFalse(restarted.isInstalled())
        restarted.discardStagingDirectory(nextStaging)
    }

    @Test
    fun existingGoodInstallationIsPreservedWhenReplacementPromotionFails() {
        val oldPayloads = payloads("encoder-v1", "decoder-v1", "tokens-v1")
        val oldInstaller = installer("revision-1", oldPayloads)
        oldInstaller.createStagingDirectory().also { staging ->
            writePayloads(staging, oldPayloads)
            oldInstaller.verifyAndPromote(staging)
        }
        val replacementPayloads = payloads("encoder-v2", "decoder-v2", "tokens-v2")
        val failingInstaller = installer(
            revision = "revision-2",
            payloads = replacementPayloads,
            fileOperations = FailStagingPromotionOperations("whisper-test")
        )
        val staging = failingInstaller.createStagingDirectory().also {
            writePayloads(it, replacementPayloads)
        }

        val error = runCatching { failingInstaller.verifyAndPromote(staging) }.exceptionOrNull()

        assertNotNull(error)
        assertTrue(oldInstaller.isInstalled())
        assertArrayEquals(
            oldPayloads.getValue("encoder.onnx"),
            File(oldInstaller.installedDirectory, "encoder.onnx").readBytes()
        )
    }

    @Test
    fun staleTemporaryDirectoryIsCleanedSafely() {
        val root = temporaryFolder.newFolder("stale-root")
        val stale = File(root, ".whisper-test.staging-abandoned").apply {
            mkdir()
            resolve("partial").writeText("partial")
        }

        val installer = installer("revision-1", payloads("encoder", "decoder", "tokens"), root = root)
        val fresh = installer.createStagingDirectory()

        assertFalse(stale.exists())
        installer.discardStagingDirectory(fresh)
    }

    @Test
    fun partiallyInstalledDestinationIsRejected() {
        val root = temporaryFolder.newFolder("partial-root")
        File(root, "whisper-test").apply {
            mkdir()
            resolve("encoder.onnx").writeText("encoder")
        }

        val installer = installer(
            "revision-1",
            payloads("encoder", "decoder", "tokens"),
            root = root
        )

        assertFalse(installer.isInstalled())
    }

    @Test
    fun completeReplacementIsPromotedAndOldBackupIsRemoved() {
        val oldPayloads = payloads("encoder-v1", "decoder-v1", "tokens-v1")
        val oldInstaller = installer("revision-1", oldPayloads)
        oldInstaller.createStagingDirectory().also { staging ->
            writePayloads(staging, oldPayloads)
            oldInstaller.verifyAndPromote(staging)
        }
        val replacementPayloads = payloads("encoder-v2", "decoder-v2", "tokens-v2")
        val replacementInstaller = installer("revision-2", replacementPayloads)
        replacementInstaller.createStagingDirectory().also { staging ->
            writePayloads(staging, replacementPayloads)
            replacementInstaller.verifyAndPromote(staging)
        }

        assertTrue(replacementInstaller.isInstalled())
        assertArrayEquals(
            replacementPayloads.getValue("decoder.onnx"),
            File(replacementInstaller.installedDirectory, "decoder.onnx").readBytes()
        )
        assertFalse(File(replacementInstaller.installedDirectory.parentFile, ".whisper-test.backup").exists())
    }

    private fun installer(
        revision: String,
        payloads: Map<String, ByteArray>,
        root: File = temporaryFolder.newFolder(),
        fileOperations: WhisperInstallerFileOperations = DefaultWhisperInstallerFileOperations
    ) = WhisperModelInstaller(
        rootDirectory = root,
        modelDirectoryName = "whisper-test",
        revision = revision,
        artifacts = payloads.map { (fileName, bytes) ->
            WhisperModelArtifact(fileName, bytes.size.toLong(), sha256(bytes))
        },
        fileOperations = fileOperations
    )

    private fun payloads(encoder: String, decoder: String, tokens: String) = linkedMapOf(
        "encoder.onnx" to encoder.toByteArray(),
        "decoder.onnx" to decoder.toByteArray(),
        "tokens.txt" to tokens.toByteArray()
    )

    private fun writePayloads(directory: File, payloads: Map<String, ByteArray>) {
        payloads.forEach { (fileName, bytes) -> File(directory, fileName).writeBytes(bytes) }
    }

    private fun sha256(bytes: ByteArray): String = MessageDigest
        .getInstance("SHA-256")
        .digest(bytes)
        .joinToString(separator = "") { byte -> "%02x".format(byte) }

    private class FailStagingPromotionOperations(
        private val modelDirectoryName: String
    ) : WhisperInstallerFileOperations {
        override fun rename(source: File, destination: File): Boolean {
            if (source.name.startsWith(".$modelDirectoryName.staging-") &&
                destination.name == modelDirectoryName
            ) return false
            return source.renameTo(destination)
        }

        override fun deleteRecursively(target: File): Boolean = target.deleteRecursively()
    }
}
