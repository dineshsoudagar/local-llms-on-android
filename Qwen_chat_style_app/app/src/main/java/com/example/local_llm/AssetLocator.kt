package com.example.local_llm

import android.content.Context
import java.io.FileNotFoundException

object AssetLocator {

    fun resolvePath(context: Context, preferredPath: String): String {
        if (assetExists(context, preferredPath)) {
            return preferredPath
        }

        val fileName = preferredPath.substringAfterLast('/')
        val matches = mutableListOf<String>()
        collectMatches(context, "", fileName, matches)

        return when (matches.size) {
            1 -> matches.single()
            0 -> throw FileNotFoundException(
                "Could not find '$preferredPath' in app assets. " +
                    "Place it in app/src/main/assets/ or inside a single nested model folder."
            )
            else -> throw FileNotFoundException(
                "Found multiple assets named '$fileName': ${matches.joinToString()}. " +
                    "Keep only one copy or update the configured asset path."
            )
        }
    }

    private fun assetExists(context: Context, path: String): Boolean {
        return runCatching {
            context.assets.open(path).close()
            true
        }.getOrDefault(false)
    }

    private fun collectMatches(
        context: Context,
        directory: String,
        fileName: String,
        matches: MutableList<String>
    ) {
        val entries = runCatching { context.assets.list(directory).orEmpty() }
            .getOrDefault(emptyArray())

        entries.forEach { entry ->
            val fullPath = if (directory.isEmpty()) entry else "$directory/$entry"
            val children = runCatching { context.assets.list(fullPath).orEmpty() }
                .getOrDefault(emptyArray())

            if (children.isEmpty()) {
                if (entry == fileName) {
                    matches += fullPath
                }
            } else {
                collectMatches(context, fullPath, fileName, matches)
            }
        }
    }
}
