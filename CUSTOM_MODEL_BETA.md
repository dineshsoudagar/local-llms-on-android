# Custom LiteRT-LM image and audio beta

Import a local `.litertlm` file through **use your own model (beta)**. The app
copies it into private storage and inspects its declared text, image, audio and
video inputs using LiteRT-LM 0.17.1's `Capabilities` API.

Detected inputs appear in the model's description. After loading, the ready
message reports whether native image and audio backends are available on this
device. Declared support is not a device compatibility guarantee. Older imports
are reinspected when loaded. If inspection fails, text-only loading is attempted.
Models explicitly declaring no text input cannot use this chat pipeline.

For images, select **Native image input** in the model menu and send one image
at a time. OCR remains selectable. The selected chat model processes native
images; the app does not load a separate Gemma model.

Audio attachments use the selected model's native audio encoder when supported.
Custom audio is normalized through the existing audio importer and divided into
segments of at most 10 seconds with overlap for longer recordings. Longer audio
requires confirmation before segmentation and synthesis. Native audio receives
a short compatibility smoke test on load, cached by model content, backend,
runtime and device build. If a declared audio encoder cannot initialize, the app
reports native audio unavailable. Text-only models retain the existing optional
Whisper transcription flow.

Video, generated audio, custom thinking controls and custom tool calling are not
enabled by this change. GPU/CPU fallback can leave a loaded model with fewer
available inputs than its file declares.

## Device test candidates

- [LFM2.5-VL-450M files](https://huggingface.co/litert-community/LFM2.5-VL-450M/tree/main):
  choose **LFM2.5-VL-450M_int4_fixB.litertlm** (about 407 MB) for a small non-Gemma
  vision test. The file listing identifies this repaired vision build; use it
  rather than the original int4 file.
- [Google Gemma 3n E2B files](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/tree/main):
  choose **gemma-3n-E2B-it-int4.litertlm** (about 3.66 GB) for image and audio.
  Access requires accepting Google's license on Hugging Face. Choose the generic
  file, rather than the Web or MediaTek-specific variant.

These are test candidates, not models validated in this app on your device.

## First device pass

1. Import, check detected inputs, load and check the ready message.
2. Send a short text prompt.
3. Select Native image input. Attach one ordinary photo and ask what it shows.
   Switch to OCR and confirm text extraction still works.
4. For Gemma 3n, attach a 5–10 second recording and request a transcript. Then try
   a recording longer than 10 seconds to exercise overlapping segments and final
   synthesis. Cancel a longer operation and retry.
5. Switch to a text-only model, then back to the imported model. Restart the app
   and reload it to check persistence and encoder availability.

For a failure, record the selected filename, detected inputs, ready/error text,
phone model, attachment type and approximate size/duration. A native process
crash needs Android logcat evidence; a beta label cannot catch native crashes.
