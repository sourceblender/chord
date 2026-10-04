# Image capability

In chat and Responses, the chat model serving the request writes the render
prompt from the whole conversation, as a plain-text reply; Chord submits that
text to the configured image workflow unchanged. If the writer fails, times out
or returns no usable prompt, the user's own last message is rendered verbatim.
On the Images endpoint the caller's prompt is rendered as given. The operator's
workflow owns graph choice and rendering. Chord validates the resulting PNG
before delivery.
