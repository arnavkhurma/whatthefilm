FROM python:3.8
ENV auth_url="https://10549518275.propelauthtest.com"
ENV auth_api_key=st.secrets["PROPELAUTH_API_KEY"]
EXPOSE 8501
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . ./
ENTRYPOINT ["streamlit", "run", "app.py",  "--server.address=0.0.0.0", "--server.enableCORS=false"]