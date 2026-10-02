# audio specialist prompt

No model reads this file. The audio lane (`specialists/voice_message.py`)
marks the turn as a voice message; the assistant writes only the words to be
spoken, and the server speaks them with TTS and returns them as `message.audio`,
only when the request asked for audio output. The file exists because
certification binds a prompt version.
