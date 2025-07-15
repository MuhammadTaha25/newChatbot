def get_user_inputs(companies: dict):
    # ... existing checkbox and text_input code ...
    voice_recording = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")
    if voice_recording:
        user_query = voice_recording
    return tickers_input, user_query, bool(voice_recording)  # Added 3rd return

def main():
    # ... initialization code ...
    tickers_input, user_query, voice_trigger = get_user_inputs(companies)
    if st.button("Send") or voice_trigger:  # Fixed condition
        # ... processing logic ...
