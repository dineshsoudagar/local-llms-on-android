pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "0.10.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        ivy {
            name = "SherpaOnnxAndroidReleases"
            url = uri("https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4")
            patternLayout {
                artifact("[artifact]-[revision].[ext]")
            }
            metadataSources {
                artifact()
            }
            content {
                includeGroup("com.k2fsa.sherpa.onnx")
            }
        }
    }
}

rootProject.name = "local_llm"
include(":app")
