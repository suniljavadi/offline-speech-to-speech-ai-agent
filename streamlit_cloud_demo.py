"""Cloud-safe, read-only showcase for the local-first voice assistant."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Offline Voice & Coding Assistant",
        layout="centered",
    )
    st.title("Offline Voice & Coding Assistant")
    st.caption(
        "A local-first speech and coding assistant, built to keep inference "
        "on your device."
    )

    st.info(
        "This is a read-only project showcase. Streamlit Community Cloud cannot "
        "access your computer's microphone, local Ollama server, models, or files. "
        "No voice or coding actions run on this page."
    )

    st.subheader("What runs locally")
    st.markdown(
        """
        - **Speech recognition:** Faster-Whisper with a model installed on your device.
        - **Responses:** Ollama running locally; no hosted inference provider.
        - **Speech output:** Piper using a locally installed voice.
        - **Coding tools:** Workspace-scoped file operations with explicit approval
          before writes or code execution.
        - **Conversation memory:** SQLite on the local machine; audio is not stored.
        """
    )

    st.subheader("Local architecture")
    st.code(
        "Microphone -> Silero VAD -> Faster-Whisper -> Ollama -> Piper -> Speakers\n"
        "                                      \\-> approved workspace tools",
        language="text",
    )

    with st.expander("Run the full assistant on your own computer"):
        st.markdown(
            """
            1. Clone the repository and follow the setup instructions in the README.
            2. Install the Whisper and Piper models and start your local Ollama service.
            3. Start the API with `python -m app.api.main`.
            4. In another terminal, run
               `streamlit run app/ui/streamlit_app.py`.

            The local UI and API intentionally accept loopback connections only.
            """
        )

    st.link_button(
        "View source and local setup",
        "https://github.com/suniljavadi/offline-speech-to-speech-ai-agent",
    )


if __name__ == "__main__":
    main()
