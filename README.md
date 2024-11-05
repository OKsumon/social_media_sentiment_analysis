# Social Media Sentiment Analysis for Early Mental Health Detection
<div align="center">
  <img src="images/university_logo.png" alt="University of Bedfordshire Logo" width="150"/>
  <h6>A University of Bedfordshire MSc Computer Science Project</h6>
</div>
## University of Bedfordshire Project
This project was developed as part of the MSc Computer Science program at the University of Bedfordshire. It was conducted under the supervision of Gregory Beacher, as a final MSc project focusing on building a machine learning system for identifying early signs of mental health disorders through social media sentiment analysis .

## Overview
This project aims to develop a complete system for identifying early signs of mental health disorders using social media sentiment analysis. Leveraging deep learning models such as CNN, LSTM, BERT, and hybrid models, the system analyzes user-generated content to classify sentiments and provide early detection insights. A web-based application was also developed for real-time analysis of social media data.

<img src="images/projectlogo.webp" alt="Project Logo" width="300"/>


## Code Hierarchy
The project code is organized into several folders, each containing scripts for specific tasks:

- `1. Data`: Contains data handling scripts, including data loading, cleaning, and initial pre-processing steps.
- `2. Pre-processing`: Scripts for advanced data pre-processing such as tokenization, lemmatization, spell correction, and augmentation.
- `3. Model`: Implementation of machine learning and deep learning models, including CNN, LSTM, BERT, and hybrid models (CNN-LSTM, BERT-BiLSTM). Each model has its dedicated Python script for training and evaluation.
- `4. META LEARNER`: Scripts for the meta-learner model that combines predictions from individual models for improved accuracy.
- `5. Website Building`: Code for developing the Flask-based web application for real-time sentiment analysis.
- `6. visu`: Contains scripts and output files for visualizations of model performance, evaluation metrics, and results comparison. More visualizations will be added here as the project progresses.

## Abstract
With rising concerns about mental health disorders, early detection and intervention have become increasingly important. This project aims to develop a system that can identify early signs of mental health issues by analyzing social media posts using sentiment analysis. The project involves multiple models, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, Bidirectional Encoder Representations from Transformers (BERT), and hybrid models like CNN-LSTM and BERT-BiLSTM.

## Background and Motivation
Mental health disorders, such as anxiety and depression, are on the rise, especially among young adults. Social media platforms have become a rich source of data where users openly express their emotions. This project aims to utilize this data to detect mental health issues early using machine learning and natural language processing techniques.

## Project Goals and Features
- Develop a robust pre-processing pipeline to clean social media data, including lemmatization, spell correction, and augmentation.
- Implement sentiment analysis models: CNN, LSTM, BERT, and hybrid models (CNN-LSTM, BERT-BiLSTM).
- Apply data augmentation techniques such as back-translation, synonym replacement, and random insertion.
- Use feature engineering techniques with GloVe, VADER, TextBlob, and AFINN embeddings to enrich input data.
- Develop a web-based application to allow real-time sentiment analysis for early detection of mental health disorders.

## Dataset
The dataset used for this project consists of over 165,000 tweets collected from Kaggle. These tweets represent different mental health-related sentiments and are pre-processed to ensure high-quality analysis. The dataset is publicly available and contains no personally identifiable information.

## Installation Instructions
To use this repository, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/OKsumon/social_media_sentiment_analysis.git
   cd social_media_sentiment_analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run pre-processing and model training scripts as per the folder structure below.

## Folder Structure
- `1. Data`: Data handling and pre-processing scripts.
- `2. Pre-processing`: Scripts for data cleaning, tokenization, and augmentation.
- `3. Model`: Deep learning models (CNN, LSTM, BERT, hybrids).
- `4. META LEARNER`: Meta-learner model for combining individual model predictions.
- `5. Website Building`: Code for developing the web-based application for real-time analysis.
- `6. visu`: Visualizations for performance evaluation and results comparison. More visualizations will be added here as the project progresses.

## Usage Guide
To run the models:
1. Pre-process the data using the scripts in the `2. Pre-processing` folder.
2. Train the models in the `3. Model` folder by running the respective Python files.
3. To test real-time sentiment analysis, deploy the web application in the `5. Website Building` folder.

## Results and Performance
- The models were evaluated based on accuracy, precision, recall, and F1-score.
- Hybrid models (CNN-LSTM and BERT-BiLSTM) provided the best performance for sentiment classification.
- A meta-learner was used to combine predictions from individual models, enhancing accuracy and robustness.

## Meta-Learner and Real-Time Application
The meta-learner uses logistic regression to combine model predictions for improved results. The web-based application allows users to input text and get real-time sentiment analysis to identify potential mental health issues.

## Technical Details
- **Models Used**: CNN, LSTM, BERT, Hybrid Models (CNN-LSTM, BERT-BiLSTM).
- **Embedding Techniques**: GloVe, Word2Vec, TF-IDF, FastText.
- **Augmentation Techniques**: Back-translation, synonym replacement, random insertion.
- **Frameworks and Tools**: TensorFlow, Hugging Face Transformers, spaCy, NLTK, nlpaug, Flask (for the web app).

## Ethical Considerations
This project adheres to ethical guidelines for using social media data. The dataset contains no personally identifiable information, and data pre-processing steps removed potential identifiers like usernames and URLs.

## Contributing
Feel free to submit issues and pull requests if you'd like to contribute to the project.

## License
This project is licensed under the MIT License.



You can also reach me at: sumonahmedjubayer@gmail.com
