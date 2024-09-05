"""
## Main streamlit file

The main frontend file to run the application.

Execute :
    `streamlit run main_streamlit.py`
to run the application
"""

import streamlit as st
import embedding_generator as emb

# Main application
st.title("YouTube Assistant 🤖")

# Initialize session state for the YouTube URL and database
if "yt_url" not in st.session_state:
    st.session_state["yt_url"] = None
    st.session_state["db"] = None
    st.session_state["thumbnail"] = None

# Side bar
with st.sidebar:
    with st.form(key="form"):
        yt_url = st.text_input(
            label="📽️ YouTube Video URL",
        )

        query = st.text_area(label="Ask me about the video ❓", max_chars=60)

        submit_btn = st.form_submit_button()


# On submit button click!
if submit_btn:
    if not (query and yt_url):
        st.error("Please Enter Video URL and Question both.")
    else:
        # Check if the URL has changed
        if yt_url != st.session_state["yt_url"]:
            # If URL is new, update the session state and create a new vector database
            st.session_state["yt_url"] = yt_url
            st.session_state["db"], st.session_state["thumbnail"] = emb.create_vector_db_from_yt_url(yt_url)
        
        if st.session_state["thumbnail"]:
            st.image(st.session_state["thumbnail"], width=400)

        st.subheader("User :  " + query)

        # Use the stored db from session state
        response = emb.get_response(query, st.session_state["db"])
        # st.text("Answer :")
        st.write_stream(response)
