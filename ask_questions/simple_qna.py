import sys
import os
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gemini_utils import *

pre_prompt = "Think about the question as if you are a human pondering deeply.\n"
post_prompt = "Give your answer between <answer></answer> tags.\n"

# list of metrices
rule_list = ["nearest", "farthest", "largest", "smallest", "widest", "narrowest", "longest", "shortest"]


def main():
    # R = "livingroom"
    # Q = "Get me my delivery box"

    # R = "Office"
    # Q = "Get me a cup from the kitchen"

    # R = "Office"
    # Q = "Find me a place to sit"

    # R = "Office"
    # Q = "I am tired and would like to take some rest. My favorite color is blue."

    # R = "Laboratory"
    # Q = "I need to loose a nut that is overly tight."

    R = "office"
    Q = "Find me a cup as soon as possible. My favorite color is red."

    question_list = [
        f"I am at a {R}. I have an inquiry: {Q}. \n"
        f"What is the object I am looking for? \n"
        f"Where is this item likely located at? \n"
        f"Name one large object that is likely nearby my target object. \n"
        f"Your answer should be three words or phrases separated by comma. \n",

        f"I am at a {R}. I have an inquiry: {Q}. \n"
        f"What should be the color of the object I am looking for? "
        f"If I did not specify a color, answer None. \n"
        f"Your answer should be one word or phrase. \n",

        f"I am at a {R}. I have an inquiry: {Q}. \n"
        f"For the object I am looking for, is one of the following properties implied in my inquiry: {rule_list}? "
        f"If I did not specify any of these properties, answer None. \n"
        f"Your answer should be one word or phrase from the list. \n"
    ]
    # question_list = [
    #     f"I am at a {R}. I have an inquiry {Q}. \n"
    #     f"What is the object I am trying to get? \n"
    #     f"Where is this item likely located at? \n"
    #     f"Name one large object that is likely nearby my target object. \n"
    #     f"Your answer should be three words or phrases separated by comma. "
    #     f"Give your answer between <answer></answer> tags."
    # ]

    for i in range(len(question_list)):
        question_list[i] = f"{pre_prompt}{question_list[i]}{post_prompt}"


    ''' ================== Inference ================== '''
    ''' upload images to model and ask questions '''
    model = GeminiModel()
    results = model.analyze_text(question_list)
    ''' ================== save questions and answers ================== '''
    if results:
        print("\nAnalysis Summary:")
        for question, answer in results.items():
            print(f"\n{question}")
            print(f"{answer}")

    ''' ================== save questions and answers in md and json ================== '''


if __name__ == "__main__":
    main()
