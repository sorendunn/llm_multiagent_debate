from glob import glob
import pandas as pd
import json
import time
import random
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

def construct_message(agents, question):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Put your answer in the form (X) at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def generate_answer(answer_context):
    try:
        completion = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo-16k",
                  messages=answer_context,
                  n=1)
    except Exception as e:  # Store the exception in variable e
        print(f"An error occurred: {e}")
        print("retrying due to an error......")
        time.sleep(20)
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


def main():
    agents = 4
    hint_ans = ["A", "B", "C", "D"]

    tasks = glob("C:/Users/soren/Desktop/data/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    
    threads = []
    lock = Lock()  # Lock to prevent simultaneous writes to the file

    with open("mmlu_{}_fast.json".format(agents), "w") as f:
        for _ in range(100):
            thread = threading.Thread(target=process_question, args=(f, lock, hint_ans, dfs))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()

def process_question(f, lock, hint_ans, dfs):
    df = random.choice(dfs)
    ix = len(df)
    idx = random.randint(0, ix-1)

    question, answer = parse_question_answer(df, idx)

    agent_contexts = [cot_prompt + [{"role": "user", "content": question}] for agent in range(agents)]

    for round in range(rounds):
        for i, agent_context in enumerate(agent_contexts):

            if round != 0:
                agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                message = construct_message(agent_contexts_other, question)
                agent_context.append(message)

            completion = generate_answer(agent_context)

            assistant_message = construct_assistant_message(completion)
            agent_context.append(assistant_message)

    response_dict[question] = (agent_contexts, answer)

json.dump(response_dict, open("mmlu_{}_{}.json".format(agents, rounds), "w"))

if __name__ == "__main__":
    agents = 4
    rounds = 1

    tasks = glob("C:/Users/soren/Desktop/data/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    
    threads = []
    lock = Lock()  # Lock to prevent simultaneous writes to the file

    with open("mmlu_{}_fast.json".format(agents), "w") as f:
        for _ in range(100):
            thread = threading.Thread(target=process_question, args=(f, lock, hint_ans, dfs))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()