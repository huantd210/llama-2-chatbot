# LLAMA 2 CHATBOT

## Run on localhost

1. Setup .venv environment > Using Python 3.9
2. Create .env file create .env file (reference to .env.example)
3. `pip install --upgrade -r requirements.txt`
4. `python app.py`
5. Go to http://localhost:7860

## Run on Docker

1. `docker build --platform=linux/amd64 -t llama-2-chatbot:v1.0.0 .`
2. `docker run -it -p 7860:7860 --platform=linux/amd64 llama-2-chatbot:v1.0.0 python app.py`
3. Go to http://localhost:7860

## Reference

1. https://www.docker.com/blog/llm-docker-for-local-and-hugging-face-hosting/
2. https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476
