# —– UI setup pehle —–
st.title("Ask anything about Musk")

# Placeholder bana lo yahan, mic recorder se pehle
spinner_placeholder = st.empty()

# Mic recorder
voice_recording = speech_to_text(
    language="en",
    use_container_width=True,
    just_once=True,
    key="STT"
)

# Text input + send button
query = st.text_input("Please enter a query", key="query", on_change=send_input)
st.button("Send", key="send_btn")

# —– Chat processing —–
if (query and st.session_state.send_input) or voice_recording:
    # Yahan placeholder ke andar spinner show hoga
    with spinner_placeholder.spinner("Processing... Please wait!"):
        response_stream = chain.stream({"question": query})
        response_text = ""
        for chunk in response_stream:
            if isinstance(chunk, dict):
                response_text += chunk.get("answer", "")
            else:
                response_text += str(chunk)

    # Jab processing khatam ho jaye, placeholder ko clear kar do
    spinner_placeholder.empty()

    # Messages update kar do
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("assistant", response_text))
    st.session_state.send_input = False

# —– Render chat log —–
with chat_container:
    for role, message in st.session_state.messages:
        st.chat_message(role).write(message)

