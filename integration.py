import streamlit as st
import pyaudio
import wave
import threading

def load_integration():
    # Audio recording function
    def record_audio(output_filename, stop_recording_event, sample_rate=44100, chunk_size=1024):
        audio_format = pyaudio.paInt16
        channels = 1
        rate = sample_rate
        chunk = chunk_size

        audio = pyaudio.PyAudio()
        
        try:
            # Start the audio stream
            stream = audio.open(format=audio_format, channels=channels,
                                rate=rate, input=True, frames_per_buffer=chunk)
        except Exception as e:
            st.error(f"Failed to open audio stream: {e}")
            return

        frames = []

        def record():
            try:
                while not stop_recording_event.is_set():
                    try:
                        data = stream.read(chunk, exception_on_overflow=False)  # Handle overflows
                        frames.append(data)
                    except IOError as e:
                        st.error(f"IOError during recording: {e}")
                        break
            except Exception as e:
                st.error(f"Unexpected error during recording: {e}")

        # Start recording in a separate thread
        record_thread = threading.Thread(target=record)
        record_thread.start()

        # Wait until stop event is set
        stop_recording_event.wait()
        record_thread.join()

        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recorded frames to a .wav file
        try:
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(audio.get_sample_size(audio_format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(frames))
        except Exception as e:
            st.error(f"Failed to save audio file: {e}")
            return

        st.write(f"Audio recording saved as {output_filename}")

    # Function to handle recording start/stop
    def handle_recording(start_recording):
        if start_recording:
            st.write("Recording started...")
            # Create an event to signal when to stop recording
            st.session_state.stop_recording_event = threading.Event()
            # Start recording in a new thread to avoid blocking the UI
            st.session_state.recording_thread = threading.Thread(
                target=record_audio,
                args=("output_test.wav", st.session_state.stop_recording_event)
            )
            st.session_state.recording_thread.start()
        else:
            st.write("Recording stopped.")
            # Set the stop event to stop the recording thread
            st.session_state.stop_recording_event.set()
            # Wait for the recording thread to finish
            st.session_state.recording_thread.join()

    # Streamlit UI for the record button
    def record_button():
        # Initialize session state if not already present
        if 'start_recording' not in st.session_state:
            st.session_state.start_recording = False

        # Display the appropriate button based on the recording state
        button_text = 'Start Recording' if not st.session_state.start_recording else 'Stop Recording'
        if st.button(button_text):
            st.session_state.start_recording = not st.session_state.start_recording
            handle_recording(st.session_state.start_recording)

    # Run the Streamlit UI
    record_button()
