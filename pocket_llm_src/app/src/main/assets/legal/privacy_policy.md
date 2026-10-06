# Privacy Policy

**Pocket LLM** · Effective date: {{EFFECTIVE_DATE}}

Pocket LLM runs AI models on your own phone. It has no user accounts, no ads, no analytics and no server operated by the developer. The developer does not receive your chats, prompts, files, recordings or model outputs.

## 1. Who is responsible

{{DEVELOPER_NAME}}{{POSTAL_ADDRESS}}

Email: {{CONTACT_EMAIL}}

## 2. Data that stays on your phone

Everything you create in the app is stored only in the app's private storage on your phone:

- chats, prompts and model responses
- attached text files, PDFs, images, photos and audio recordings, and text extracted from them
- downloaded or imported model files
- settings, including the LAN server password

You can delete single chats in the app. Uninstalling the app deletes all of it. The developer cannot access, recover or delete this data for you.

Android's own backup feature can include chat history and settings in your device backup (for example your Google account backup), depending on your phone's backup settings. Attachments are excluded from backups.

## 3. Permissions

- **Camera:** only when you take a photo to send to the model or to read text from it.
- **Microphone:** only when you record audio or use dictation. Speech is transcribed on your phone.
- **Notifications:** to show model download progress and that the LAN server is running.
- **Internet:** to download models and to run the LAN server when you start it.
- **Foreground service and wake lock:** to keep downloads and the LAN server running while the screen is off.

## 4. Network connections

The app connects to the internet only in these cases:

- **Model downloads:** when you download a built-in model or the speech-to-text model, the file is downloaded from Hugging Face (huggingface.co). Hugging Face receives the usual technical request data, such as your IP address. See the Hugging Face privacy policy: https://huggingface.co/privacy
- **LAN server:** when you start it, other devices on your local network can connect to the phone with the password you set. Requests and responses stay inside your network and are not sent to the developer.

Chat, image, document and audio processing happens on your phone and works offline once the models are installed.

## 5. Text recognition (GitHub and Google Play versions)

The GitHub and Google Play versions read text from images and scanned PDFs with Google ML Kit, which runs on your phone. Google states that ML Kit sends limited diagnostic data to Google: device information (manufacturer, model, Android version), the app's package name and version, performance metrics, API settings and event types, with a per-installation identifier that is not meant to identify you. Your images and the recognized text are not sent. See https://developers.google.com/ml-kit/android-data-disclosure

The F-Droid version does not include ML Kit.

## 6. Your rights

Under the GDPR you have the right to access, rectification, erasure, restriction, data portability and objection, and the right to lodge a complaint with a data protection supervisory authority. Because the developer does not hold any of your data, most requests can be fulfilled by you directly in the app. Contact {{CONTACT_EMAIL}} with any question.

## 7. Children

The app is not directed at children under 16.

## 8. Changes

If this policy changes, the new version is shown in the app and published with the app's source code.
