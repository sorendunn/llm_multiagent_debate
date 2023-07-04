import threading
from threading import Lock
from glob import glob
import pandas as pd
import json
import random
import time
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')
cot_prompt = [
    {"role":"user",
     "content":"Can you answer the following question as accurately as possible? Which of the following statements best explains why magnets usually stick to a refrigerator door?: A) The refrigerator door is smooth. B) The refrigerator door contains iron. C) The refrigerator door is a good conductor. D) The refrigerator door has electric wires in it. Explain your answer, putting the answer in the form (X) at the end of your response."
    },
    {"role":"assistant",
     "content":"Magnets include a piece of iron or other material that has its component atoms ordered to exhibit magnetism. Magnetism attracts other iron-containing objects. Thus the magnetic probably sticks to the refrigerator door since the door contains iron which attracts the magnet (B)"},
     
     {"role":"user",
      "content":"Can you answer the following question as accurately as possible? What would most likely be measure during an investigation of the water cycle?: A) Wind speed B) Ozone layer C) Natural gas emissions D) Precipitation amounts. Explain your answer, putting the answer in the form (X) at the end of your response."},
    {"role":"assistant",
     "content":"The water cycle is the cycle of processes by which water circulates between the earth's oceans, atmosphere, and land. It involves water precipitating as rain or snow, collecting, and returning to the atmosphere by evaporation and transpiration. Since water precipitating is a key component of the water cycle precipiration amounts would be the most likely choice to be measured in an investigation of the water cycle (D)"},

     {"role":"user",
      "content":"Can you answer the following question as accurately as possible? When a person is beginning work, what are they building?: A) Momentum B) Tiredness C) Working D) Time. Explain your answer, putting the answer in the form (X) at the end of your response."},
      {"role":"assistant",
       "content":"The answer must be something that is built. Of the above choices, only Momentum is built. So the answer is (A)"}
]

# Set the indexes of the questions to use to make sure they match the ones used in the original paper
idxs = [49, 861, 66, 124, 155, 91, 27, 35, 71, 194, 204, 116, 90, 77, 79, 93, 
        169, 143, 45, 80, 163, 70, 113, 133, 7, 70, 11, 204, 0, 126, 222, 62, 
        166, 32, 145, 61, 18, 69, 93, 81, 130, 27, 141, 63, 42, 236, 26, 77, 
        75, 227, 98, 73, 148, 24, 47, 66, 17, 33, 38, 20, 276, 400, 180, 70, 
        60, 27, 603, 243, 148, 57, 169, 717, 365, 83, 29, 150, 42, 24, 8, 138, 
        90, 95, 174, 109, 15, 18, 89, 11, 146, 68, 87, 13, 81, 155, 147, 50, 94, 29, 5, 47]

def extract_text(completion):
    content = completion["choices"][0]["message"]["content"]
    return content

def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=answer_context,
                  n=1)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        time.sleep(10)
        return generate_answer(answer_context)

    return completion

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = "Can you answer the following question as accurately as possible? {}: A) {}, B) {}, C) {}, D) {} Explain your answer, putting the answer in the form (X) at the end of your response.".format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def process_question(f, lock, dfs):
    df = random.choice(dfs)
    ix = len(df)
    idx = random.randint(0, ix-1)

    question, answer = parse_question_answer(df, idx)

    responses = []
    agent_context = cot_prompt.copy()
    agent_context.append({"role": "user", "content": question})
    completion = generate_answer(agent_context)
    responses.append(construct_assistant_message(completion))

    with lock:  # Ensure only one thread writes to the file at a time
        json.dump({question: (responses, answer)}, f)
        f.write("\n")  # write newline character to keep each entry on separate line

def main():

    tasks = glob("C:/Users/soren/Desktop/data/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]
    
    threads = []
    lock = Lock()  # Lock to prevent simultaneous writes to the file

    with open("mmlu_cot_baseline.json", "w") as f:
        random.seed(0)
        for _ in range(100):
            thread = threading.Thread(target=process_question, args=(f, lock, dfs))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()
