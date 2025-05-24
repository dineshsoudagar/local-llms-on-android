# Local LLM Android App - Qwen QA

This Android application integrates a local ONNX-converted LLM for interactive, on-device question answering. It uses a Hugging Face-compatible tokenizer and supports streaming inference for efficient generation.

## Features

- On-device question answering using ONNX Runtime.
- Hugging Face-style BPE tokenizer (`tokenizer.json` compatible).
- Qwen-compatible prompt formatting with streaming generation.
- Coroutine-based inference pipeline.
- Runs offline, preserving privacy.

## Example Prompts

Use these prompts to test the app or create screenshots:

- **General Knowledge:**  
  *"Who was the first president of the United States?"*

- **Science:**  
  *"Explain the theory of relativity in simple terms."*

- **Math:**  
  *"What is the integral of x^2?"*

- **Programming Help:**  
  *"What is the difference between a list and a tuple in Python?"*

- **Language:**  
  *"Translate 'Hello, how are you?' into French."*

## Model

This app uses the **Qwen2.5-0.5B-Instruct** model for local QA tasks.

### Preconverted ONNX Model

- Download [Qwen2.5-0.5B-onnx](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/blob/main/onnx) and [tokenizer.json](https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/tree/main) and place it in `app\src\main\assets`.

### OR: Convert the Model Yourself

```
pip install optimum[onnxruntime]
```
or
```
python -m pip install git+https://github.com/huggingface/optimum.git
```
then export with, 
```
optimum-cli export onnx --model Qwen/Qwen2.5-0.5B-Instruct qwen2.5-0.5B-onnx/
```
- More about Optimum [here.](https://huggingface.co/docs/optimum/main/en/index)


### Prerequisites

- Android Studio. [Link](https://developer.android.com/studio)
- ONNX Runtime for Android. [Link](https://github.com/microsoft/onnxruntime-genai/releases). Already in the repository.
- Physical Android device (for performance)

### Steps to create Apk.
- Open Android studio.
- Create new project with "Empty Activity". Name your app "local_llm".
- Place [MainActivity.kt](Qwen_App/app/src/main/java/com/example/deen_translator/MainActivity.kt), [OnnxModel.kt](Qwen_App/app/src/main/java/com/example/deen_translator/OnnxModel.kt) and [SimpleTokenizer.kt](Qwen_App/app/src/main/java/com/example/deen_translator/SimpleTokenizer.kt) inside `main\java\com\example\yourappaname`.
- Copy [build.gradle.kts](Qwen_App/app/build.gradle.kts) contents.
- Copy [libs.versions.toml](../deen_translator/gradle/libs.versions.toml) contents.
- Add [activity_main.xml](Qwen_App/app/src/main/res/layout/activity_main.xml) inside `app/src/main/res/layout`.
- copy [themes.xml](Qwen_App/app/src/main/res/values/themes.xml) contents.
- Change the heap size to "8192m" in gradle.properties.
- Connect your phone via wireless debugging mode and install apk or create apk with build -> Generate App bundles or APKs.