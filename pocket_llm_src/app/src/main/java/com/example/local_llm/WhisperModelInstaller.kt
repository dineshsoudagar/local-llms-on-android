package com.example.local_llm

import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID

internal data class WhisperModelArtifact(
    val fileName: String,
    val expectedBytes: Long,
    val sha256: String
)

internal interface WhisperInstallerFileOperations {
    fun rename(source: File, destination: File): Boolean
    fun deleteRecursively(target: File): Boolean
}

internal object DefaultWhisperInstallerFileOperations : WhisperInstallerFileOperations {
    override fun rename(source: File, destination: File): Boolean = source.renameTo(destination)
    override fun deleteRecursively(target: File): Boolean = target.deleteRecursively()
}

internal class WhisperModelInstaller(
    private val rootDirectory: File,
    private val modelDirectoryName: String,
    private val revision: String,
    val artifacts: List<WhisperModelArtifact>,
    private val fileOperations: WhisperInstallerFileOperations = DefaultWhisperInstallerFileOperations
) {
    val installedDirectory = File(rootDirectory, modelDirectoryName)
    private val backupDirectory = File(rootDirectory, ".$modelDirectoryName.backup")
    private val stagingPrefix = ".$modelDirectoryName.staging-"
    private val manifestFileName = ".verified-install-v1"
    private var verifiedInstalledSignature: String? = null
    private val expectedManifest = buildString {
        appendLine("whisper-model-install-v1")
        appendLine("revision=$revision")
        artifacts.sortedBy(WhisperModelArtifact::fileName).forEach { artifact ->
            appendLine("${artifact.fileName}|${artifact.expectedBytes}|${artifact.sha256.lowercase()}")
        }
    }.toByteArray(Charsets.UTF_8)

    init {
        require(modelDirectoryName.isNotBlank() && '/' !in modelDirectoryName && '\\' !in modelDirectoryName)
        require(artifacts.isNotEmpty())
        require(artifacts.map(WhisperModelArtifact::fileName).toSet().size == artifacts.size)
        artifacts.forEach { artifact ->
            require(artifact.fileName.isNotBlank() && '/' !in artifact.fileName && '\\' !in artifact.fileName)
            require(artifact.expectedBytes > 0L)
            require(artifact.sha256.matches(Regex("[0-9a-fA-F]{64}")))
        }
        check(rootDirectory.mkdirs() || rootDirectory.isDirectory) {
            "Could not create the Whisper model directory."
        }
    }

    fun isInstalled(): Boolean = synchronized(INSTALLATION_FILE_LOCK) {
        recoverInterruptedPromotion()
        isCompleteInstallation(installedDirectory)
    }

    fun createStagingDirectory(): File = synchronized(INSTALLATION_FILE_LOCK) {
        recoverInterruptedPromotion()
        cleanupStaleStagingDirectories()
        val staging = File(rootDirectory, stagingPrefix + UUID.randomUUID())
        check(staging.mkdir()) { "Could not create a temporary Whisper model directory." }
        staging
    }

    fun verifyAndPromote(stagingDirectory: File) {
        synchronized(INSTALLATION_FILE_LOCK) {
            requireSafeStagingDirectory(stagingDirectory)
            verifyStagedArtifacts(stagingDirectory)
            writeVerifiedManifest(stagingDirectory)
            promote(stagingDirectory)
        }
    }

    fun discardStagingDirectory(stagingDirectory: File) {
        synchronized(INSTALLATION_FILE_LOCK) {
            requireSafeStagingDirectory(stagingDirectory)
            fileOperations.deleteRecursively(stagingDirectory)
        }
    }

    private fun cleanupStaleStagingDirectories() {
        rootDirectory.listFiles()
            .orEmpty()
            .filter { it.isDirectory && it.name.startsWith(stagingPrefix) }
            .forEach { stale ->
                check(fileOperations.deleteRecursively(stale)) {
                    "Could not remove stale Whisper staging directory ${stale.name}."
                }
            }
    }

    private fun verifyStagedArtifacts(stagingDirectory: File) {
        val stagedNames = stagingDirectory.listFiles().orEmpty().map(File::getName).toSet()
        val expectedNames = artifacts.map(WhisperModelArtifact::fileName).toSet()
        require(stagedNames == expectedNames) { "The staged Whisper model set is incomplete or unexpected." }

        artifacts.forEach { artifact ->
            val file = File(stagingDirectory, artifact.fileName)
            require(file.isFile && file.name == artifact.fileName) {
                "Missing Whisper model file ${artifact.fileName}."
            }
            require(file.length() == artifact.expectedBytes) {
                "Whisper model file ${artifact.fileName} has size ${file.length()}, expected ${artifact.expectedBytes}."
            }
            val actualHash = sha256(file)
            require(actualHash.equals(artifact.sha256, ignoreCase = true)) {
                "Whisper model file ${artifact.fileName} failed SHA-256 verification."
            }
        }
    }

    private fun writeVerifiedManifest(stagingDirectory: File) {
        FileOutputStream(File(stagingDirectory, manifestFileName)).use { output ->
            output.write(expectedManifest)
            output.fd.sync()
        }
    }

    private fun promote(stagingDirectory: File) {
        verifiedInstalledSignature = null
        recoverInterruptedPromotion()
        check(!backupDirectory.exists()) { "A stale Whisper backup could not be recovered." }

        val hadExistingInstallation = installedDirectory.exists()
        if (hadExistingInstallation) {
            check(fileOperations.rename(installedDirectory, backupDirectory)) {
                "Could not preserve the existing Whisper model installation."
            }
        }

        if (!fileOperations.rename(stagingDirectory, installedDirectory)) {
            if (hadExistingInstallation && !installedDirectory.exists() && backupDirectory.exists()) {
                check(fileOperations.rename(backupDirectory, installedDirectory)) {
                    "Whisper promotion failed and the previous installation could not be restored."
                }
            }
            throw IllegalStateException("Could not atomically promote the verified Whisper model set.")
        }

        if (!isCompleteInstallation(installedDirectory)) {
            if (backupDirectory.exists()) {
                fileOperations.deleteRecursively(installedDirectory)
                check(fileOperations.rename(backupDirectory, installedDirectory)) {
                    "The promoted Whisper model was invalid and the previous installation could not be restored."
                }
            }
            throw IllegalStateException("The promoted Whisper model set failed installation validation.")
        }

        if (backupDirectory.exists()) {
            check(fileOperations.deleteRecursively(backupDirectory)) {
                "The verified Whisper model was installed, but its old backup could not be removed."
            }
        }
    }

    private fun recoverInterruptedPromotion() {
        if (!backupDirectory.exists()) return
        verifiedInstalledSignature = null

        val installedIsComplete = isCompleteInstallation(installedDirectory)
        val backupIsComplete = isCompleteInstallation(backupDirectory)
        when {
            installedIsComplete -> check(fileOperations.deleteRecursively(backupDirectory)) {
                "Could not remove a stale Whisper model backup."
            }

            backupIsComplete -> {
                if (installedDirectory.exists()) {
                    check(fileOperations.deleteRecursively(installedDirectory)) {
                        "Could not remove an incomplete Whisper model installation."
                    }
                }
                check(fileOperations.rename(backupDirectory, installedDirectory)) {
                    "Could not recover the previous Whisper model installation."
                }
            }

            else -> check(fileOperations.deleteRecursively(backupDirectory)) {
                "Could not remove an incomplete Whisper model backup."
            }
        }
    }

    private fun isCompleteInstallation(directory: File): Boolean {
        if (!directory.isDirectory) return false
        val expectedNames = artifacts.map(WhisperModelArtifact::fileName).toSet() + manifestFileName
        val actualFiles = directory.listFiles() ?: return false
        if (actualFiles.map(File::getName).toSet() != expectedNames) return false
        if (artifacts.any { artifact ->
                val file = File(directory, artifact.fileName)
                !file.isFile || file.length() != artifact.expectedBytes
            }
        ) return false
        val manifest = File(directory, manifestFileName)
        if (!manifest.isFile || !manifest.readBytes().contentEquals(expectedManifest)) return false
        val signature = buildString {
            artifacts.sortedBy(WhisperModelArtifact::fileName).forEach { artifact ->
                val file = File(directory, artifact.fileName)
                append("${artifact.fileName}:${file.length()}:${file.lastModified()}|")
            }
            append("$manifestFileName:${manifest.length()}:${manifest.lastModified()}")
        }
        val isInstalledDirectory = directory.canonicalFile == installedDirectory.canonicalFile
        if (isInstalledDirectory && signature == verifiedInstalledSignature) return true
        if (artifacts.any { artifact ->
                !sha256(File(directory, artifact.fileName)).equals(artifact.sha256, ignoreCase = true)
            }
        ) return false
        if (isInstalledDirectory) verifiedInstalledSignature = signature
        return true
    }

    private fun requireSafeStagingDirectory(stagingDirectory: File) {
        require(stagingDirectory.name.startsWith(stagingPrefix)) { "Unexpected Whisper staging directory." }
        require(stagingDirectory.canonicalFile.parentFile == rootDirectory.canonicalFile) {
            "Whisper staging directory escaped its app-owned model root."
        }
    }

    private fun sha256(file: File): String = FileInputStream(file).use { input ->
        val digest = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            digest.update(buffer, 0, read)
        }
        digest.digest().joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    companion object {
        private val INSTALLATION_FILE_LOCK = Any()
    }
}
