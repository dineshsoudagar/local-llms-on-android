# F-Droid and IzzyOnDroid release checklist

The `fdroid` build flavor is the one meant for F-Droid and IzzyOnDroid. It contains no Google ML Kit, Google Play Services or Firebase code. CI checks this on every build (`.github/scripts/check_no_proprietary_classes.py` scans the dex files of the F-Droid debug APK).

## Build

```bash
cd pocket_llm_src
bash ./gradlew assembleFdroidRelease
```

- Output: `app/build/outputs/apk/fdroid/release/app-fdroid-release-unsigned.apk`. Sign it with your own release key (`apksigner sign --ks <keystore> ...`), or configure a `signingConfig` locally. Never commit the keystore.
- Always sign F-Droid/IzzyOnDroid APKs with the **same key**. Users cannot update across keys without uninstalling.
- Release builds fail until `src/main/res/values/legal_contact.xml` is filled in (see [docs/legal](../legal/README.md)).
- Debug builds install side by side with the other flavors as `io.github.dineshsoudagar.pocketllm.debug.fdroid`.

## What differs in the fdroid flavor

| | github / play | fdroid |
|---|---|---|
| OCR engine | Google ML Kit text recognition (proprietary, model bundled, sends diagnostics to Google) | Tesseract 5 via Tesseract4Android (Apache-2.0) |
| OCR language data | inside the ML Kit library | `eng.traineddata` from [tesseract-ocr/tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast) tag `4.1.0`, about 4 MB, downloaded on first OCR use into `filesDir/ocr/tessdata`, size and SHA-256 pinned |
| Code | `src/mlkit/java` (added to the github and play source sets) | `src/fdroid/java` |

Both implement the `OcrEngine` interface in `src/main`. Callers (`OcrInput` for camera/gallery images, `AttachmentImporter` for scanned PDF pages) are the same in every flavor.

If OCR data is missing and the phone is offline, OCR fails with "Text recognition needs its English language data (about 4 MB) once. Connect to the internet and try again." In the chat screen, download progress is shown as a short status message. The PDF importer runs in a background worker and shows the same error on the attachment if the download fails.

Tesseract is noticeably slower than ML Kit and somewhat weaker on photos of curved or angled text. It is English-only for now; more languages can be added by pinning more `*.traineddata` files in `TesseractLanguageData.kt`.

### Where Tesseract4Android is published

Tesseract4Android (`cz.adaptech.tesseract4android:tesseract4android:4.9.0`, Tesseract 5.5.1 + Leptonica, Apache-2.0) is **only published on JitPack**, not on Maven Central. The build adds `https://jitpack.io` in `settings.gradle.kts`, restricted to the group `cz.adaptech.tesseract4android`.

Trade-off:
- IzzyOnDroid does not build from source, so a JitPack dependency is fine there; it only scans the APK.
- F-Droid builds from source and dislikes binaries prebuilt by JitPack. For the main F-Droid repository, Tesseract4Android should be built from source as a `srclib` (it is a plain Gradle + CMake/NDK project, so this is feasible) and the JitPack repository removed in the recipe's `prebuild`.
- The older `com.rmtheis:tess-two` is on Maven Central but unmaintained (Tesseract 3/4 era), so it was not chosen.

## IzzyOnDroid (fastest route)

IzzyOnDroid takes APKs straight from GitHub Releases and reads the store listing from `fastlane/metadata/android/` in this repository.

1. Build and sign `assembleFdroidRelease`. Attach the APK to the GitHub release with a clear name, for example `pocket_llm_v1.6.0_fdroid.apk`. Keep the GitHub (ML Kit) APK as a separate asset; tell Izzy which file name pattern to pick.
2. Keep `fastlane/metadata/android/en-US/` up to date: `title.txt`, `short_description.txt` (max 80 characters), `full_description.txt`, `images/icon.png`, and `changelogs/<versionCode>.txt` for every release (current versionCode: 15). Screenshots go into `images/phoneScreenshots/` (none added yet).
3. Open a request at https://codeberg.org/IzzyOnDroid/repodata/issues with the repository URL, license, the APK asset name and a short description. Mention that the app downloads models from Hugging Face at runtime and that the F-Droid flavor uses Tesseract instead of ML Kit.
4. **APK size:** IzzyOnDroid's general rule is about 30 MB per APK; larger apps need an exception. This APK is far larger (see blockers), so ask for an exception in the request, or ship a per-ABI `arm64-v8a` APK.
5. Izzy's scanner checks for proprietary libraries and trackers. The CI check above covers ML Kit, Play Services and Firebase.

## Main F-Droid repository

1. Fork https://gitlab.com/fdroid/fdroiddata and add `metadata/io.github.dineshsoudagar.pocketllm.yml` with a build recipe: `subdir: pocket_llm_src/app`, `gradle: [fdroid]`, the release commit or tag, `AutoUpdateMode`/`UpdateCheckMode: Tags`, and `srclibs`/`prebuild` steps for every native dependency that must be built from source (see blockers).
2. Test locally with `fdroid build -v -l io.github.dineshsoudagar.pocketllm` (fdroidserver) and `fdroid lint`.
3. Open a merge request. Reviewers will run the scanner and ask about every prebuilt binary and non-standard Maven repository.

F-Droid can also offer reproducible builds signed with the developer's key, if the build output matches byte for byte.

## Open blockers for the main F-Droid repository

These are not fixed in this branch; the fdroid flavor is free of ML Kit/Play Services but still ships prebuilt native code.

| Dependency | Source | Problem for F-Droid |
|---|---|---|
| LiteRT-LM `com.google.ai.edge.litertlm:litertlm-android:0.17.1` | Google Maven | Prebuilt native AAR. Source is Apache-2.0 (google-ai-edge/LiteRT-LM) but built with Bazel; F-Droid would need to build it from source. Its transitive dependencies should also be checked for Google-only artifacts (the CI dex check would catch Play Services classes). This is the core inference engine, so it is the largest blocker. |
| ONNX Runtime `com.microsoft.onnxruntime:onnxruntime-android:1.27.0` | Maven Central | MIT, but prebuilt `libonnxruntime.so` for 4 ABIs. Building from source is possible but heavy. |
| sherpa-onnx `sherpa-onnx-static-link-onnxruntime-1.13.4.aar` | GitHub Releases via an Ivy repository in `settings.gradle.kts`, then stripped by `verifyOnnxRuntimeNativeCompatibility` in `app/build.gradle.kts` | Apache-2.0, but a prebuilt binary AAR from a non-Maven repository. The scanner flags both. Needs a from-source `srclib` build. |
| Tesseract4Android 4.9.0 | JitPack | Apache-2.0, prebuilt by JitPack; build from source as a srclib (see above). |
| APK size | | The universal APK is about 300 MB because of native libraries for 4 ABIs. Consider ABI splits for the fdroid flavor. |

Other points reviewers may raise:
- Models are not in the APK; they are downloaded at runtime from Hugging Face. Some built-in models (Gemma) use Google's Gemma terms rather than an OSI license, so reviewers may discuss an anti-feature label for downloaded non-free assets.
- No other proprietary libraries were found: AndroidX, CameraX, Material Components, Markwon, PdfBox-Android, WorkManager and org.json (public domain since 2022; version 20240303 is used) are free software.

## Test on a phone before submitting

Install the fdroid debug APK (`app/build/outputs/apk/fdroid/debug/`) and check:

1. **Gallery OCR** with mobile data/Wi-Fi on: the first use shows "Downloading OCR language data..." and then the recognized text.
2. **Gallery OCR offline** after clearing app data: a clear error asking to connect once, no crash.
3. **Camera OCR**: photo of a printed page, including a portrait photo (EXIF rotation).
4. **Scanned PDF** attachment (pages without embedded text): text is extracted, and an offline first run fails with the same clear error.
5. Repeat 1, 3 and 4 on the github or play debug build to confirm ML Kit behaviour did not change.
