# ----------------------------
# Base image
# ----------------------------
FROM python:3.10-slim

# ----------------------------
# Create a non-root user
# ----------------------------
RUN useradd -m -u 1000 user
USER user

# ----------------------------
# Environment variables
# ----------------------------
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR $HOME/app

# ----------------------------
# Copy requirements and install
# ----------------------------
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy all project files
# ----------------------------
COPY --chown=user . .

# ----------------------------
# Expose port for Streamlit
# ----------------------------
EXPOSE 7860

# ----------------------------
# Start the Streamlit app
# ----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--logger.level=debug"]