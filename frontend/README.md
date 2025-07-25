# Batch Loan Risk Assessment Streamlit Application

This application provides a user-friendly interface for assessing credit risk on a batch of loan applications. It's designed as a showcase project for a senior machine learning engineering portfolio, demonstrating best practices in frontend application structure, data handling, and user experience.

## Features

- **File Upload**: Supports both CSV and Excel formats for loan application data.
- **Data Validation**: Displays a preview of the uploaded data to ensure correctness.
- **Batch Processing**: Simulates a risk assessment model on the entire dataset with a progress bar.
- **Interactive Dashboard**: Presents results through summary metrics, charts, and a detailed table.
- **Downloadable Results**: Allows users to download the full assessment results as a CSV file.

## Setup and Installation

Follow these steps to set up and run the application locally.

### 1. Prerequisites

- Python 3.8+ installed on your system.
- `pip` and `venv` for package and environment management.

### 2. Create a Virtual Environment

From the `frontend` directory, create and activate a virtual environment:

```bash
# Navigate to the frontend directory
cd path\to\your\project\FinancialRisk\frontend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
.\venv\Scripts\activate

# On macOS/Linux
# source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Once the dependencies are installed, you can run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser.
