import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *

def generate_ranges(total_length, clip_size=20):
    ranges = []
    start = 0
    while start < total_length:
        end = min(start + clip_size, total_length)
        ranges.append([start, end])
        start = end
    return ranges


def main():
    root_dir = "/home/zl3466/Documents/github/SuperX"
    data_dir = "/home/zl3466/Documents/dataset/mini"

    out_dir = f"{root_dir}/outputs"

    question_list = [
        f"This is a image from a robot exploring an indoor space. Determine if this space belongs to one of "
        f"{space_list}; if not, give me the space it appears to be in "
        f"(e.g. kitchen, bedroom, livingroom, garage, basement, etc.).\n"
        f"Please provide your text answer within the <answer> </answer> tags."
    ]

    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()
    results = model.analyze_single_image(f"{data_dir}/{filename}", question_list)
    ''' ================== save questions and answers ================== '''
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")

    ''' ================== save questions and answers in md and json ================== '''



if __name__ == "__main__":
    main()
