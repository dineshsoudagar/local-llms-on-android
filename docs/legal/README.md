# Legal texts

The app ships two documents and shows them on first launch and under **About**:

- [Privacy Policy](../../pocket_llm_src/app/src/main/assets/legal/privacy_policy.md)
- [Terms of Use](../../pocket_llm_src/app/src/main/assets/legal/terms_of_use.md)

Both read the developer's name, address, contact email and effective date from
[`legal_contact.xml`](../../pocket_llm_src/app/src/main/res/values/legal_contact.xml).
Release builds fail while any of those values is still `TODO`. Debug builds still work.

The contact email is also where the in-app "Report response" button sends reports. While it is `TODO`, reports open a pre-filled GitHub issue instead.

## Before publishing

1. Fill in `legal_contact.xml`.
2. The postal address may stay empty in the public repo; the Play build sets it in the private repo.
3. Publish the filled-in Privacy Policy at a public URL (for example GitHub Pages). Google Play asks for this URL.
4. Fill in [`impressum.md`](impressum.md) and publish it next to the policy. German law requires an Impressum for apps offered commercially.
5. When either document changes materially, bump `LegalDocuments.CURRENT_VERSION` so users accept the new version.

These texts are a careful starting point written for an offline app with no accounts. They have not been reviewed by a lawyer.
