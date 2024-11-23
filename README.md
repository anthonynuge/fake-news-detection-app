# Fake News Detection App

A machine learning application that detects fake news based on article content. Built using **Python**, **scikit-learn**, and **Natural Language Processing (NLP)** techniques, this application provides a user-friendly interface to help users determine the credibility of news articles.

## Features

- **Fake News Detection:** Classify news articles as real or fake using machine learning models.
- **User-Friendly Interface:** Upload file or copy and paste text to analyze
- **Machine Learning Model:** Uses NLP techniques and a trained classification model to provide reliable results.

## Tech Stack

- Python

- **GUI: Tkinter**

- **Machine Learning:** scikit-learn, NLP (TF-IDF Vectorizer)

- **Jupyter Notebooks:** Used for model training and experimentation

- Visualizations: Matplotlib, Seaborn

## Installation and Setup

Follow the steps below to clone and run the application locally:

### Prerequisites

- [Python 3](https://www.python.org/downloads/) installed on your system.
- Git installed for cloning the repository.

### Step 1: Clone the Repository

1. Open your terminal.
2. Run the following command:
   ```bash
   git clone https://github.com/anthonynuge/fake-news-detection-app.git
   ```
3. Navigate to the project directory:
   ```bash
   cd fake-news-detection-app
   ```

### Step 2: Create a Virtual Environment and Install Dependencies

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Run the Application

1. Start the application by running the main Python script:
   ```bash
   cd src
   python main.py
   ```

## Usage

- Copy and paste the content of a news article into the provided text area and click the **Classify** button to see the results.
- The system will analyze the article and classify it as real or fake.
