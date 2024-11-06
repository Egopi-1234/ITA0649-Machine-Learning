import csv

def load_data(training_data):
    with open(training_data, 'r') as file:
        return list(csv.reader(file))

def find_s(data):
    hypothesis = None
    for instance in data:
        if instance[-1] == 'Yes':
            if hypothesis is None:
                hypothesis = instance[:-1]
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != instance[i]:
                        hypothesis[i] = '?'
    return hypothesis
data = load_data(r"C:\Users\gopie\Documents\MACHINE LEARNING\findS_algorithm.csv")
specific_hypothesis = find_s(data)

print("Most specific hypothesis:", specific_hypothesis)