# I-Comprehend
 Thesis Project | Laguna State Polytechnic University - Computer Science

## Overview
This project aims to design, develop, implement, and evaluate a
system with a multimodal approach, incorporating quiet reading, loud reading, question
generation, and advanced recognition technologies using machine learning (ML) approaches
to enhance college students’ reading comprehension.

## Prerequisites
Before you begin, ensure you have the following prerequisites:
- Version of Python (Python 3.12.3)
- Virtual environment
- Required Python Packages (listed in `requirements.txt`)

## Installation
You can download the necessary files from this Google Drive Link.

[Google Drive Link](https://drive.google.com/drive/folders/1vvXvlYrBWtJxfcqEK0a4BerMbsno3lcx?usp=sharing) (t5_model, t5_tokenizer, qg_model, qgtokenizer)

after downloading make or move the model to models directory 


## Install Dependencies

- before installing the dependencies create a virtual environment using virtualenv
to pip install virtual env
```bash
pip install virtualenv
```
to create a virtual environment
```bash
virtualenv env
```
to activate environment
```bash
env\Scripts\activate
```

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt

```

## Run the application
To run the server, run the following command (go to the directory first):
```bash
jupyter notebook main.ipynb
```

## Run the evaluation
To run the evaluation, run the following command (go to the directory first):
```bash
jupyter notebook model_evaluation.ipynb
```
