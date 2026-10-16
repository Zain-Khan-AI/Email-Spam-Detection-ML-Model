# MailGuard | Email Spam Classifier

An intelligent email spam detection application built with Machine Learning and Natural Language Processing (NLP). MailGuard analyzes email or message text and classifies it as **Spam** or **Not Spam** through a clean, interactive Streamlit dashboard.

## Overview

MailGuard is a machine learning project designed to demonstrate how text preprocessing and machine learning can be used to identify unwanted messages. The application processes user-provided text and uses a trained classification model to generate predictions.

## Features

- **Spam Detection:** Classifies messages as Spam or Not Spam.
- **NLP Preprocessing:** Uses text cleaning, tokenization, stopword removal, and stemming.
- **TF-IDF Vectorization:** Converts text into numerical features for machine learning prediction.
- **Interactive Dashboard:** Provides a user-friendly interface built with Streamlit.
- **Trained Model Integration:** Uses a saved model and vectorizer for predictions.
- **Professional UI:** Features a custom-styled dashboard with a dark theme.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Scikit-learn | Machine learning and TF-IDF |
| NLTK | Natural language processing |
| Pandas | Data manipulation |
| Pickle | Loading saved model artifacts |
| CSS & SVG | Custom interface styling |

## Project Structure

```text
MailGuard/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Update this structure if your repository contains additional folders or files.*

## How It Works

1. **Input:** The user enters an email or message.
2. **Preprocessing:** The text is cleaned and processed using the configured NLP steps.
3. **Vectorization:** TF-IDF transforms the processed text into numerical features.
4. **Prediction:** The trained machine learning model predicts the message category.
5. **Output:** The dashboard displays the classification result.

## Model Artifacts

- `model.pkl`: Stores the trained machine learning model.
- `vectorizer.pkl`: Stores the fitted TF-IDF vectorizer.

Both artifacts must be compatible with the preprocessing pipeline used by the application.

## Project Goals

MailGuard demonstrates practical skills in:

- Machine Learning model deployment
- Natural Language Processing
- Text classification
- Model serialization and integration
- Interactive dashboard development
- Python application development

## Future Improvements

- Display model confidence when supported by the classifier.
- Add a prediction history feature.
- Evaluate the model using precision, recall, F1-score, and a confusion matrix.
- Improve input validation and error handling.

## Author

**Zain Khan**

- GitHub: [Zain-Khan-AI](https://github.com/Zain-Khan-AI)
- LinkedIn: [[Add Your LinkedIn Profile URL](YOUR_LINKEDIN_URL](https://www.linkedin.com/in/m-zainkhan/))

---

*MailGuard is an educational machine learning project. Its predictions depend on the training data and model performance and should not be treated as a guarantee that a message is safe.*
