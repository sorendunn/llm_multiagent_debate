import json
import numpy as np

def compute_accuracy(gt, pred_solutions):
    # Assuming gt and pred_solution are both strings.
    # Extract the option within parentheses from the predicted solution
    pred = pred_solutions[0].split("(")[-1].split(")")[0]
    return gt == pred

if __name__ == "__main__":
    with open("mmlu_cot_baseline.json", "r") as f:
        lines = f.readlines()

    questions, responses, gts = [], [], []
    for line in lines:
        data = json.loads(line.strip())
        question, response, gt = list(data.items())[0]
        questions.append(question)
        responses.append(response[0][-1]['content']) # taking the content of the last assistant's response
        gts.append(gt)

    accuracies = []

    for question, response, gt in zip(questions, responses, gts):

        accurate = compute_accuracy(gt, [response])

        if accurate is not None:
            accuracies.append(float(accurate))
        else:
            import pdb
            pdb.set_trace()
            print(gt)

        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))