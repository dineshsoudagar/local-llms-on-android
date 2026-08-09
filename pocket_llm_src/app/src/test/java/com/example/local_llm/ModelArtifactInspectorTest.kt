package com.example.local_llm

import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.RandomAccessFile

class ModelArtifactInspectorTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun acceptsModelWithMatchingDescriptorSize() {
        val file = temporaryFolder.newFile("model.litertlm")
        RandomAccessFile(file, "rw").use { it.setLength(1024L * 1024L) }
        val artifact = ModelDownloadFile(file.name, "unused", expectedBytes = file.length())

        assertNull(ModelArtifactInspector.findProblem(file, artifact))
    }

    @Test
    fun rejectsDescriptorOrManifestSizeMismatch() {
        val file = temporaryFolder.newFile("model.onnx")
        RandomAccessFile(file, "rw").use { it.setLength(1024L * 1024L) }

        assertNotNull(
            ModelArtifactInspector.findProblem(
                file,
                ModelDownloadFile(file.name, "unused", expectedBytes = file.length() + 1L)
            )
        )
        assertNotNull(
            ModelArtifactInspector.findProblem(
                file,
                ModelDownloadFile(file.name, "unused"),
                manifestLength = file.length() + 1L
            )
        )
    }

    @Test
    fun rejectsMissingTinyAndErrorPageModels() {
        val missing = temporaryFolder.root.resolve("missing.onnx")
        assertNotNull(ModelArtifactInspector.findProblem(missing, ModelDownloadFile(missing.name, "unused")))

        val tiny = temporaryFolder.newFile("tiny.onnx").apply { writeText("tiny") }
        assertNotNull(ModelArtifactInspector.findProblem(tiny, ModelDownloadFile(tiny.name, "unused")))

        val html = temporaryFolder.newFile("error.onnx")
        RandomAccessFile(html, "rw").use {
            it.write("<!doctype html><html>Access denied</html>".toByteArray())
            it.setLength(1024L * 1024L)
        }
        assertNotNull(ModelArtifactInspector.findProblem(html, ModelDownloadFile(html.name, "unused")))
    }

    @Test
    fun validatesTokenizerJson() {
        val invalid = temporaryFolder.newFile("invalid.json")
            .apply { writeText("not-json".repeat(32)) }
        assertNotNull(ModelArtifactInspector.findProblem(invalid, ModelDownloadFile(invalid.name, "unused")))

        val valid = temporaryFolder.newFile("tokenizer.json")
            .apply { writeText("{\"padding\":\"${"x".repeat(160)}\"}") }
        assertNull(ModelArtifactInspector.findProblem(valid, ModelDownloadFile(valid.name, "unused")))
    }
}
