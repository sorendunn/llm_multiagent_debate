import json
import numpy as np
import re

def compute_accuracy(gt, pred_solution):
    # This finds all the instances of characters within parentheses
    all_preds = re.findall(r'\((.*?)\)', pred_solution[0])

    # If there are no matches, or the last match is not a single character, we return False
    if not all_preds or len(all_preds[-1]) != 1:
        return False

    # Otherwise, we return whether the ground truth matches the last occurrence
    return gt == all_preds[-1]


if __name__ == "__main__":
    with open("mmlu_cot_baseline.json", "r") as f:
        data = [json.loads(line) for line in f]

    accuracies = []

    for item in data:
        for question, content in item.items():
            responses, gt = content

            pred_solutions = []
            for response in responses:
                pred_solution = response['content']
                pred_solutions.append(pred_solution)

            accurate = compute_accuracy(gt, [pred_solutions[-1]])

            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                import pdb
                pdb.set_trace()
                print(gt)

        print("accuracies:", np.mean(accuracies), np.std(accuracies) / (len(accuracies) ** 0.5))