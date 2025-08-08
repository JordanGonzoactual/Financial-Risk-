FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Set environment variables to ensure Conda is in the PATH
ENV PATH=/opt/conda/bin:$PATH

# Install core packages directly into the base environment
# Grouped for clarity and better layer caching

# Data science packages
RUN conda install -c conda-forge -y pandas numpy scikit-learn matplotlib seaborn plotly

# Web framework packages
RUN conda install -c conda-forge -y streamlit flask flask-cors requests pydantic

# Additional utilities
RUN conda install -c conda-forge -y xgboost psycopg2 openpyxl

# Verify that all essential packages are installed and importable
RUN python -c "import pandas, numpy, sklearn, streamlit, flask, requests, xgboost, psycopg2, pydantic; print('All packages imported successfully')"

# Set container environment flag
ENV CONTAINER_ENV=true

# Set Python-related environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy the application code into the container
COPY . .

# Create necessary directories
RUN mkdir -p /app/Logs /app/Models /app/Data

# Expose ports for Streamlit and Flask
EXPOSE 8501
EXPOSE 5001

# Command to run the Streamlit application (which handles Flask backend internally)
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
