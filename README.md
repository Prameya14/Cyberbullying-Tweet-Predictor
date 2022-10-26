
# Cyberbullying Tweet Predictor

This is a Tweet Prediction System which would allow users to get the Category of any tweet. It can classify tweets in the form of 5 categories: Age, Ethnicity, Gender, Not Cyberbullying, Other Cyberbullying and Religion.

## About Project

**Context of the Project:**

Non Techy People regularly face a problem of being victims of Cyberbullying and Cyber Crimes. This is because sometimes they are unable to the classify the messages or texts which they receive.

My Idea for this project is that we can create an AI Enabled App which can take the text or tweet as input and classify it based on some categories.

I transformed my idea into a solution by applying the Basic NLP Concepts which I have learnt during the Bootcamp and use my own Web Development Skills to make a website on the trained model.

**Working of the Project:**

I have used the LinearSVC Algorithm of Scikit Learn in this AI App. This would help us to classify the tweets. 

I have used the trained model in a website which I have made using flask where the users can enter their tweet and get the category of the tweet as output.

In this future this can be expanded by simultaneously expanding the dataset which I have used from Kaggle.

## Tech Stack

**Exploratory Data Analysis:** Pandas, NumPy, SeaBorn, WordCloud, Pickle and Matplotlib

**AI Model:** ScikitLearn and NLTK (Natural Language Toolkit)

**Server:** Flask

**Client:** Bootstrap 5 and HTML (Hyper Text Markup Language)


## Installation

I. Clone Project

```bash
git clone https://github.com/Prameya14/Cyberbullying-Tweet-Predictor.git
```

II. Install Requirements

```bash
pip install -r ./requirements.txt
```

III. EDA and Model

```bash
For this you need to execute all the cells of the iPython Notebook i.e. "[Notebook] Tweet Predictor.ipynb"
```

IV. Start Flask Sever

```bash
python ./main.py
```

V. After that open the URL which pops up in the console and enjoy the application.
    