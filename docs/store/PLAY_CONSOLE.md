# Google Play release checklist

The `play` build flavor is the one uploaded to Google Play.

## Build

```bash
cd pocket_llm_src
bash ./gradlew bundlePlayRelease
```

- Upload `app/build/outputs/bundle/playRelease/app-play-release.aab`. Google splits it per CPU type, so phones download only their own native libraries instead of the 300 MB universal APK.
- Use **Play App Signing**: Play keeps the app signing key, and you sign uploads with your own upload key.
- Release builds fail until `src/main/res/values/legal_contact.xml` is filled in (see [docs/legal](../legal/README.md)).

| Setting | Value |
|---|---|
| Package name | `io.github.dineshsoudagar.pocketllm` |
| Target SDK | 36 (Android 16), as required since 2026-08-31 |
| Min SDK | 24 |

## App content (Play Console → Policy → App content)

**Privacy policy URL:** the public URL of the filled-in `privacy_policy.md`.

**App access:** all features are available without login.

**Ads:** answer "No", until an ads build exists.

**Target audience:** 18 and over. This keeps the app out of the Families policy.

**Content rating (IARC questionnaire):**
- Category: Reference, News or Educational / Utility.
- Users do not interact or share content with each other.
- Answer "Yes" to the question about AI-generated content.

**Generative AI:** the app has an in-app report button on every AI answer. It opens a pre-filled email to the contact address, or a GitHub issue as a fallback. Mention this if asked.

**Data safety**, for the current `play` build: no ads, ML Kit text recognition included.

| Question | Answer |
|---|---|
| Does the app collect or share user data? | Yes, collected (ML Kit diagnostics) |
| App info and performance → Diagnostics | Collected, not shared. Purpose: app functionality and analytics (by Google ML Kit). Not linked to the user |
| Device or other IDs | Collected, not shared. Purpose: same as above. Per-installation ID from ML Kit |
| Data encrypted in transit | Yes |
| Can users request deletion? | No. The developer holds no user data, and chats stay on the phone |
| Everything else (messages, photos, audio, files, location, contacts…) | Not collected. They stay on the device and are never transmitted |

If ML Kit is replaced by the open-source OCR used in the F-Droid build, the answer becomes "No data collected".

## Foreground service declarations (Policy → App content → Foreground service permissions)

Each declaration needs a short description and a link to a video (an unlisted YouTube video is fine) showing the feature.

| Service | Type | Justification to paste |
|---|---|---|
| `LanServerService` | `specialUse` | The user starts a local network server from the LAN Server dialog so their own computer on the same Wi-Fi can send prompts to the on-device model. It must keep running while the screen is off. It shows a persistent notification with a Stop action and stops when the user taps Stop. |
| `ModelDownloadService` | `dataSync` | Downloads the AI model files the user selected (0.5–4 GB) from Hugging Face. The download must continue when the user leaves the app, and shows a progress notification with a Cancel action. |
| WorkManager `SystemForegroundService` | `dataSync` | User-started extraction of text from large PDFs (up to 500 pages) and transcription of long audio files, with progress and Cancel. |

Known risk: Play may consider `dataSync` the wrong type for local document/audio processing. If the WorkManager declaration is rejected, move those jobs to the `mediaProcessing` type on Android 15+.

## Testing track

New **personal** developer accounts must run a closed test with at least **12 testers for 14 days** before publishing to production. Recruit testers from GitHub users with a pinned issue or Discussion.

**Organization** accounts, which need a free D-U-N-S number, don't have this requirement.

## Store listing

- Short description (80 chars), for example: "Private AI chat that runs offline on your phone. Turn it into a LAN AI server."
- Screenshots: at least 2 phone screenshots, plus a 512×512 icon and a 1024×500 feature graphic.
- Contact email and website: your Impressum/privacy page.
