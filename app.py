# https://www.mlq.ai/gpt-4-vision-data-analysis/
# https://www.kdnuggets.com/how-to-access-and-use-gemini-api-for-free

import PIL.Image
import streamlit as st
from openai import OpenAI
import base64
import os
import json
import google.generativeai as genai
from IPython.display import Markdown

# pix2tex[gui]

# read json config
json_config = json.load(open("config_private.json", "r"))

OPENAI_API_KEY = json_config["openai_api_key"]
client = OpenAI(api_key=OPENAI_API_KEY)

#gemini_api_key = json_config["gemini_api_key"]
#genai.configure(api_key = gemini_api_key) # this requires access to Google AI Studio, not available in all regions

# use Vertex AI
# https://medium.com/google-cloud/a-pisceans-take-on-gemini-b9681a0fa04d

from google.cloud import aiplatform
import vertexai.preview

#Authenticate
#from google.colab import auth
#auth.authenticate_user()

#Restart Kernel
#import IPython
#app = IPython.Application.instance()
#app.kernel.do_shutdown(True)

#Set PROJECT_ID and REGION variables
import vertexai.preview
import vertexai
PROJECT_ID = "data-dragon-409706"  # your project id
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\mmocak\\PycharmProjects\\ransx-ai\\google_creds\\data-dragon-409706-f89f065f78c3.json'

from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, Image
#vision_model = GenerativeModel("gemini-pro-vision")

def encode_image(uploaded_file):
  return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def decode_image(image_data):
    return base64.b64decode(image_data)

def analyze_image(image_data_list, question, ai_input_model, is_url=False):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]

    if ai_input_model == 'GPT-4 Vision':
        for image_data in image_data_list:
            if is_url:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_data}})
            else:
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})

        response = client.chat.completions.create(model="gpt-4-vision-preview", messages=messages,max_tokens=4096)
        return response.choices[0].message.content
    elif ai_input_model == 'Gemini Pro Vision':
        for image_data in image_data_list:
            img = PIL.Image.open(image_data)
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content(img)

        return Markdown(response.text)
        #return "Gemini Pro Vision not available in your region"


st.set_page_config(page_title="AI Vision for Data Analysis", page_icon="🔍")
st.title('AI Vision for Data Analysis')

# User Inputs
ai_input_model = st.radio("Select AI Model",
                          ('GPT-4 Vision', 'Gemini Pro Vision'))
image_input_method = st.radio("Select Image Input Method",
                              ('Upload Image', 'Enter Image URL'))
user_question = st.text_input("Enter your question for the image",
                              value="Explain this image")

image_data_list_for_openai_gpt4_vision = []
image_data_list_for_google_gemini_vision = []

if image_input_method == 'Upload Image':
  uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
  if uploaded_files:
    for uploaded_file in uploaded_files:
      image_data_list_for_openai_gpt4_vision.append(encode_image(uploaded_file))
      image_data_list_for_google_gemini_vision.append(uploaded_file)
    if st.button('Analyze image(s)'):
        if ai_input_model == 'GPT-4 Vision':
          insights = analyze_image(image_data_list_for_openai_gpt4_vision, user_question, ai_input_model)
          # show image in the UI
          for image_data in image_data_list_for_openai_gpt4_vision:
            st.image(decode_image(image_data))

          print(insights)
          st.write(insights)
        elif ai_input_model == 'Gemini Pro Vision':
            insights = analyze_image(image_data_list_for_google_gemini_vision, user_question, ai_input_model)
            # show image in the UI
            for image_data in image_data_list_for_openai_gpt4_vision:
                st.image(image_data)
        else:
            st.write("Error: AI Model not supported")

elif image_input_method == 'Enter Image URL':
  image_urls = st.text_area("Enter the URLs of the images, one per line")
  if image_urls and st.button('Analyze image URL(s)'):
    url_list = image_urls.split('\n')
    insights = analyze_image(url_list, user_question, ai_input_model, is_url=True)
    print(insights)
    st.write(insights)

