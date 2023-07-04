from glob import glob
import json
import time
import random
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')
advocate_prompt = " (Hint: the correct answer choice is "
arbiter_prompt = " Four students proposed different answers to the above question. Based on the students' reasoning below, what is the correct answer?\n"
arbiter_format_prompt = "\n\nExplain each of the students' reasoning step by step. Put your answer in the form (X) at the end of your response."

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

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

if __name__ == "__main__":
    agents = 4
    hint_ans = ["A", "B", "C", "D"]

    tasks = glob("C:/Users/soren/Desktop/data/data/test/*.csv")

    dfs = [pd.read_csv(task) for task in tasks]

    random.seed(0)
    response_dict = {}

    for i in range(100):
        df = random.choice(dfs)
        ix = len(df)
        idx = random.randint(0, ix-1)

        question, answer = parse_question_answer(df, idx)

        responses = []

        for i in range(agents):
            # Each advocate is prodded towards advocating for a particular answer choice
            agent_context_with_hint = [{"role": "user", "content": question + advocate_prompt + hint_ans[i] + ")"}]
            completion = generate_answer(agent_context_with_hint)
            completion_text = construct_assistant_message(completion)
            responses.append(completion_text)


        # Now the arbiter judges the four previous responses and comes to their own conclusion
        advocate_responses_formatted = "\nStudent 1:\n {},\n Student 2:\n {},\n Student 3:\n Student 4:\n {}".format(responses[0], responses[1], responses[2], responses[3])

        arbiter_context = [{"role": "user", "content": question + arbiter_prompt + advocate_responses_formatted + arbiter_format_prompt}]

        completion = generate_answer(arbiter_context)

        arbiter_response = construct_assistant_message(completion)

        responses.append(arbiter_response)

        response_dict[question] = (responses, answer)

    json.dump(response_dict, open("mmlu_{}.json".format(agents), "w"))
