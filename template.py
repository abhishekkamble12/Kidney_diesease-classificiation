import os 
from pathlib import Path
import logging

# project name 
project_name = "tweet_sentiment_classifier"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    ".github/workflows/.gitkeep",
    ".github/workflows/github-actions.yml",
    ".github/workflows/github-actions.yml",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml"
    "templates/index.html",
    "setup.py",
    "research/trials.ipynb",
    
]

for filepath in list_of_files :
    filepath = Path(filepath) #system will give me file dir and 
    filedir , filename = os.path.split(filepath) #os.path.split will give me file dir and filename
    if filedir != "" :
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0) :
        with open(filepath, "w") as f :
            pass
            logging.info(f"Creating empty file: {filepath}")
    else :
        logging.info(f"File already exists: {filepath}")