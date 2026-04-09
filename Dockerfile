FROM python:3.10

# Set up a new user named "user" with user ID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements.txt and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY --chown=user . $HOME/app

# Run the FastAPI server on port 7860, which HF Spaces uses natively
CMD ["uvicorn", "src.web.server:app", "--host", "0.0.0.0", "--port", "7860"]
