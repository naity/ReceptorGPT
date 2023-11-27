#Base Image to use
FROM python:3.10-slim

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt update && apt install -y git gcc g++

#Copy Requirements.txt file into app directory
COPY requirements.txt /usr/src/app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r /usr/src/app/requirements.txt

# stitchr data sets 
RUN stitchrdl -s human
RUN stitchrdl -s mouse

# replace the index.html file
COPY index.html /usr/local/lib/python3.10/site-packages/streamlit/static/index.html
COPY images/logo4.jpg /usr/local/lib/python3.10/site-packages/streamlit/static/favicon.png

#Copy app files in current directory into app directory
COPY app.py /usr/src/app/app.py
COPY utils /usr/src/app/utils
COPY images /usr/src/app/images

#Change Working Directory to app directory
WORKDIR /usr/src/app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]