
def correctly_clustered(labels, clusters):
    count = 0
    for i in range(len(labels) - 1):
        for j in range(i+1, len(labels)):
            if i != j:
                if labels[i] == labels[j] and clusters[i] == clusters[j]:
                    count += 1
    return count

def correctly_unclustered(labels, clusters):
    count = 0
    for i in range(len(labels) - 1):
        for j in range(i+1, len(labels)):
            if i != j:
                if labels[i] != labels[j] and clusters[i] != clusters[j]:
                    count += 1
    return count

def rand_index(labels, clusters):
    a = correctly_clustered(labels, clusters)
    b = correctly_unclustered(labels, clusters)
    n = len(labels)
    return 2*(a+b)/(n*(n-1))

# y = [1,1,1,0,0,0]
# labels = [1,1,0,0,0,0]

# print(correctly_clustered(y, labels))
# print(correctly_unclustered(y, labels))
# print(rand_index(y, labels))