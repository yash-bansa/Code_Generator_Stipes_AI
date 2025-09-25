import streamlit as st
import requests
import json
import random
import string
import time
from typing import Generator

# Configure Streamlit page
st.set_page_config(
    page_title="DevAgent Chat",
    page_icon="🤖",
    layout="wide"
)

def generate_chat_id() -> str:
    """Generate a 10-character chat ID with mix of numbers and letters separated by '-'"""
    # Generate 5 characters, then dash, then 4 characters
    part1 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    part2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{part1}-{part2}"

def parse_streaming_response(response_text: str) -> tuple[str, bool]:
    """Parse streaming response and return content and finish status"""
    content = ""
    is_finished = False

    lines = response_text.split('\n')
    for line in lines:
        if line.startswith('data:') or line.startswith('data :'):
            try:
                # Extract JSON part
                json_str = line.replace('data:', '').replace('data :', '').strip()
                if json_str:
                    data = json.loads(json_str)
                    if 'choices' in data and len(data['choices']) > 0:
                        choice = data['choices'][0]

                        # Check for finish reason
                        if choice.get('finish_reason') == 'stop':
                            is_finished = True
                        elif choice.get('finish_reason') and choice.get('finish_reason') != 'stop':
                            # Handle error cases
                            content += f"\n{choice.get('finish_reason')}"
                            is_finished = True

                        # Extract content from delta
                        if 'delta' in choice and 'content' in choice['delta']:
                            content += choice['delta']['content']
                        elif 'delta' in choice and 'role' in choice['delta']:
                            # Skip role messages
                            continue
            except json.JSONDecodeError:
                continue

    return content, is_finished

def send_chat_request(message: str, chat_id: str) -> Generator[str, None, None]:
    """Send chat request and yield streaming response"""
    url = "http://127.0.0.1:8000/chat"
    payload = {
        "messages": [{"content": message}],
        "config": {"chat_Id": chat_id}
    }

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=300
        ) as response:

            if response.status_code != 200:
                yield f"Error: HTTP {response.status_code}"
                return

            accumulated_content = ""

            for line in response.iter_lines(decode_unicode=True):
                if line:
                    content, is_finished = parse_streaming_response(line)
                    if content:
                        accumulated_content += content
                        yield accumulated_content

                    if is_finished:
                        break

    except requests.exceptions.RequestException as e:
        yield f"Connection Error: {str(e)}"
    except Exception as e:
        yield f"Error: {str(e)}"

# Initialize session state
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# UI Layout
st.title("🤖 DevAgent Chat")

# Sidebar with DevAgent button and session info
with st.sidebar:
    st.header("Session Control")

    if st.button("🚀 New DevAgent Session", type="primary", use_container_width=True):
        st.session_state.chat_id = generate_chat_id()
        st.session_state.messages = []
        st.success(f"New session created!")
        st.rerun()

    if st.session_state.chat_id:
        st.info(f"**Session ID:** `{st.session_state.chat_id}`")
    else:
        st.warning("No active session. Click 'New DevAgent Session' to start.")

    st.header("Chat History")
    if st.session_state.messages:
        st.write(f"Messages: {len(st.session_state.messages)}")
        if st.button("Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.write("No messages yet")

# Main chat interface
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if not st.session_state.chat_id:
        st.warning("⚠️ Please create a new DevAgent session to start chatting.")
    elif st.session_state.is_streaming:
        st.info("🔄 Processing your request... Please wait.")
    else:
        # User input
        user_input = st.chat_input(
            "Type your message here...",
            disabled=st.session_state.is_streaming
        )

        if user_input and not st.session_state.is_streaming:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)

            # Set streaming state
            st.session_state.is_streaming = True

            # Create placeholder for assistant response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                # Send request and handle streaming response
                try:
                    full_response = ""
                    for partial_response in send_chat_request(user_input, st.session_state.chat_id):
                        full_response = partial_response
                        response_placeholder.write(full_response)
                        time.sleep(0.1)  # Small delay for smooth streaming effect

                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    response_placeholder.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

                finally:
                    # Reset streaming state
                    st.session_state.is_streaming = False
                    st.rerun()

# Footer with connection status
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.chat_id:
        st.success("🟢 Session Active")
    else:
        st.error("🔴 No Session")

with col2:
    if st.session_state.is_streaming:
        st.warning("🟡 Streaming...")
    else:
        st.success("🟢 Ready")

with col3:
    st.info(f"📝 Messages: {len(st.session_state.messages)}")

# CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        margin-bottom: 1rem;
    }

    .stButton > button {
        width: 100%;
    }

    .chat-container {
        height: 400px;
        overflow-y: auto;
    }

    .status-indicator {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Auto-scroll to bottom (optional)
if st.session_state.messages:
    st.markdown("""
    <script>
        var element = document.querySelector('.main');
        element.scrollTop = element.scrollHeight;
    </script>
    """, unsafe_allow_html=True)