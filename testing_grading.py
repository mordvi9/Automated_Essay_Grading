import argparse
import os
import feature_extract
import xgboost_regression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="The prompt for the essay")
    parser.add_argument("essay_file", help="The essay file to be graded")
    args = parser.parse_args()
    
    essay_path = os.path.join(os.getcwd(), args.essay_file)
    if not os.path.exists(essay_path):
        print(f"Essay file {essay_path} does not exist.")
        return
    with open(essay_path, 'r') as file:
        essay = file.read()
    
    prompt = args.prompt
    feature_extract()
    print("Extracted features:")
    
    print("Model returns following grade: ")
    xgboost_regression(prompt, essay)
    
    