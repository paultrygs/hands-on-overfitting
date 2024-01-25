# Hands-on Overfitting

**This learning material was originally developed for a coding event at Bouvet Norge.**

This repository contains two primary sections:

1. ``hands_on_demo.ipynb``: An interactive demo for learning about overfitting using simple models that are easy to visualize.
2. ``exercise.ipynb``: A coding exercise that complements the material from part 1.

## Prerequisites

Make sure that **Python 3.11** is installed. 

### Setup on a UNIX system

To set up the project on a UNIX system:

1. Navigate to the project root directory.
2. Run the following commands:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```

3. Now you can run the Jupyter Notebook with the command:

```bash
jupyter lab
```

### Setup on a Windows system

For Windows system users, the easiest way is probably to install a WinPython distribution that already includes Python 3.11 and Jupyter Notebooks. 

This version is tested to work with the demo and exercise: https://github.com/winpython/winpython/releases/tag/7.0.20231126final

1. Download WinPython and extract the files to a folder of your choice.
2. Open the extracted folder and run `WinPython Command Prompt.exe`.
3. Navigate to your GitHub folder with the exercise by typing `cd C:\[....]\GitHub\hands-on-overfitting`.
4. Type `jupyter notebook` to start Jupyter Notebooks. Select the appropriate notebook in the browser window that opens.

Note: If you prefer Jupyter Lab over Jupyter Notebook, you can install it by typing `pip install jupyterlab` in your command terminal and then following similar steps to launch it as done with Jupyter Notebook.


## Additional Information

This material provides an introduction to overfitting using Jupyter Notebook. It covers overfitting concepts and demonstrates how you can use various techniques such as regularization to mitigate overfitting. The provided exercises will complement your understanding of overfitting and help strengthen your practical skills in performing tasks like model selection, parameter tuning, and prediction.

We hope you find this material useful, and if you have any questions or issues setting up your environment or running the notebook, please don't hesitate to reach out for assistance.