import csv
def load_data(file_path):
    with open(file_path, 'r') as file:
        return list(csv.reader(file))
def candidate_elimination(data):
    num_attributes = len(data[0]) - 1
    specific_h = ['ϕ'] * num_attributes
    general_h = [['?'] * num_attributes]
    for instance in data:
        if instance[-1] == 'Yes':
            for i in range(num_attributes):
                if specific_h[i] == 'ϕ':
                    specific_h[i] = instance[i]
                elif specific_h[i] != instance[i]:
                    specific_h[i] = '?'
            general_h = [g for g in general_h if all(g[i] == '?' or g[i] == specific_h[i] for i in range(num_attributes))]
        else:
            new_general_h = []
            for g in general_h:
                if all(g[i] == '?' or g[i] == instance[i] for i in range(num_attributes)):
                    for i in range(num_attributes):
                        if g[i] == '?' and specific_h[i] != instance[i]:
                            new_hypothesis = g[:]
                            new_hypothesis[i] = specific_h[i]
                            new_general_h.append(new_hypothesis)
            general_h.extend(new_general_h)
    general_h = [list(h) for h in set(tuple(h) for h in general_h)]
    return specific_h, general_h
data = load_data(r"C:\Users\gopie\Documents\MACHINE LEARNING\findS_algorithm.csv")
specific_h, general_h = candidate_elimination(data)
print("Most specific hypothesis:", specific_h)
print("General hypotheses:", general_h)
