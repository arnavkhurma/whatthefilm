import streamlit as st
from propelauth import auth
import io
from PIL import Image
from inference_sdk import InferenceHTTPClient # type: ignore
from openai import OpenAI

st.set_page_config(layout="wide")

# PROPEL AUTH AUTHENTICATION
user = auth.get_user()
if user is None:
    st.write("Redirect to http//:localhost:8000")
    st.stop()
# SIDEBAR
with st.sidebar:
    st.image("static/logo.png")
    # st.header("WHATTHEFILM")
    st.markdown("---")
    if user is None:
        st.text("Not Signed in")
    else:
        st.text(f"Signed in as {user.email}")
        st.link_button('MY ACCOUNT', auth.get_account_url(), use_container_width=True)
        st.link_button('LOG OUT', auth.get_account_url(), use_container_width=True)

# MAIN PAGE
col1, col2 = st.columns(2)
with col1:
    st.header("Upload Image")
    model_picture = None
    picture = st.camera_input("")
    st.markdown("<p style='text-align: center;'>OR</p>", unsafe_allow_html=True)
    df = st.file_uploader(label='')

    # Determine if picture or uploaded file should be used
    if df is not None:
        model_picture = df
    elif picture is not None:
        model_picture = picture

with col2:
    if model_picture is not None:
        st.header("Your Results")
        if st.button('Find the Movie'):
            image_bytes = None
            try:
                if isinstance(model_picture, st.runtime.uploaded_file_manager.UploadedFile):
                    image = Image.open(model_picture)
                else:
                    image = Image.open(io.BytesIO(model_picture.getvalue()))

                img_byte_arr = io.BytesIO()
                image.convert("RGB").save(img_byte_arr, format='JPEG')
                image_bytes = img_byte_arr.getvalue()

                # SAVE IMAGE LOCALLY
                with open("processed_image.jpg", "wb") as f:
                    f.write(image_bytes)
                st.success("Image added to profile successfully!")

                # API CALL TO YOLO MODEL
                CLIENT = InferenceHTTPClient(
                    api_url="https://detect.roboflow.com",
                    api_key= st.secrets["YOLO_API_KEY"]
                )
                result = CLIENT.infer("processed_image.jpg", model_id="whatthefilm/1")
                st.write(result)
                ans = result['predictions']
                client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])
                prompt = """
Please provide the following information for the movie or TV series associated with the 'class' name of what they are playing from the json provided to you:
1. Name of the movie/TV series.
2. Popular actors that worked in it.
3. Streaming platforms where it is available.

Output in the format:
<name of movie/tv series> \\n <popular actors> \\n <streaming platforms>

If nothing is discernible, return an empty string and nothing else.
"""

                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": f"{prompt}. Content from Image: {result}"}
                    ]
                )
                ans = completion.choices[0].message.content
                if len(ans) <= 5:
                    st.write("length <= 5")
                    st.write(f'''
                        <body style="display: flex; justify-content: center; align-items: center; background-color: #262730; width: 100%;">
                            <div style="width: 100%; border-radius: 10px; padding: 20px; box-sizing: border-box; background: #262730;">
                                <div style="display: flex; flex-direction: column;">
                                    <p style="margin: 10px 0; font-size: 16px; color: #fff;">Could not find a matching title.</p>
                                </div>
                            </div>
                        </body>
                    ''', unsafe_allow_html=True)
                else:
                    st.write("length > 5")
                    st.write(f'''
                        <body style="display: flex; justify-content: center; align-items: center; background-color: #262730; width: 100%;">
                            <div style="width: 100%; border-radius: 10px; padding: 20px; box-sizing: border-box; background: #262730;">
                                <div style="display: flex; flex-direction: column;">
                                    <h2 style="margin: 0; font-size: 24px; color: #fff;">{ans}</h2>
                                    <p style="margin: 10px 0; font-size: 16px; color: #fff;">Played by: {ans}</p>
                                    <p style="margin: 10px 0; font-size: 16px; color: #fff;">Watch on: {ans}</p>
                                </div>
                            </div>
                        </body>
                    ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
