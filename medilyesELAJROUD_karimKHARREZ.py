import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


def page_rank(trans, alpha=0.85, err=0.001):
    size = trans.shape[0]
    x1 = np.ones(size) / size
    x2 = np.ones(size)
    G = np.ones((size, size))
    while np.sum(np.abs(x1 - x2)) > err:
        x2 = x1.copy()
        x1 = (
                alpha * x1.dot(trans)
                + (1 - alpha) * x1.dot(G) / size
        )
    return np.argsort(-x1)


def adjtotrans(adj):
    ri, ci = adj.nonzero()
    res = np.zeros(adj.shape)
    res[ri, ci] = adj[ri, ci] / adj.sum(1)[ri]
    return res


def import_adjacente(input_file, delimiter=" ", header=False):
    with open(input_file) as file:
        reader = csv.reader(file, delimiter=delimiter)
        if header:
            next(reader, None)
        edges = [(int(row[0]), int(row[1])) for row in reader]
    size = max(max(v1, v2) for v1, v2 in edges) + 1
    adj = np.zeros((size, size), dtype=np.int8)
    print("graph name :",input_file)
    print("number of nodes :",size)
    print("number of edges ", len(edges))
    for (v1, v2) in tqdm(edges):
        adj[v1, v2] = adj[v2, v1] = 1
    return adj


def iterate(adj, initinfect=0.05, contamprob=0.3, cureprob=0.1, ratiovacc=0.25, random=False, countit=200):
    original_size = adj.shape[0]
    if random:
        vacc = np.random.choice(original_size, int(original_size * ratiovacc), replace=False )
    else:
        vacc = page_rank(adjtotrans(adj))[: int(original_size * ratiovacc)]
    adj = np.delete(adj, vacc, 0)
    adj = np.delete(adj, vacc, 1)
    contam = np.random.choice(adj.shape[0],int(original_size * initinfect),replace=False,)
    result = [(0, len(contam) / original_size)]
    for i in tqdm(range(1, countit)):
        # print(i)
        contam = np.delete(
            contam,
            np.where(np.random.rand(contam.shape[0]) < cureprob)[0],
        )
        adjacent_to_contam = adj[contam, :].nonzero()[1]
        n_contam = np.extract(
            np.random.rand(adjacent_to_contam.shape[0])
            < contamprob,
            adjacent_to_contam,
        )
        contam = np.concatenate((contam, n_contam))
        contam = np.unique(contam)
        result += [(i, len(contam) / original_size)]
    return result


if __name__ == "__main__":
    A = import_adjacente(
        #"p2p-Gnutella09.txt",
        #"email-Eu-core.txt",
        "soc-sign-bitcoinalpha.csv",
        delimiter=",",
        header=True,
    )
    P = adjtotrans(A)
    print("génération de la courbe de propagation de la maladie sans Vaccination:")
    plt.plot(*zip(*iterate(A, ratiovacc=0,countit=500)), label="Sans Vaccination", color="red")
    print("génération de la courbe de propagation de la maladie avec vaccination aléatoire: ")
    plt.plot(*zip(*iterate(A, random=True,countit=500)), label="Vaccination aléatoire",color="grey")
    print("génération de la courbe de propagation de la maladie avec vaccination pageRank:")
    plt.plot(*zip(*iterate(A,countit=500)), label="Vaccination pageRank",color="green")
    plt.xlabel("temps (nbr d'itérations)")
    plt.ylabel("Proportion d'individus inféctés  ")
    plt.title("Simulation de propagation du virus")
    plt.legend()
#    print("génération de la courbe de propagation de la maladie sans Vaccination:")
#    plt.plot(*zip(*iterate(A, ratiovacc=0, countit=200, contamprob=0.3)), label="Sans Vaccination", color="black")
    # print("génération de la courbe de propagation de la maladie avec vaccination aléatoire: ")
    # plt.plot(*zip(*iterate(A, random=True)), label="Vaccination aléatoire",color="grey")
#    print("génération de la courbe de propagation de la maladie avec pageRank ratio vaccination=0.25:")
#    plt.plot(*zip(*iterate(A, countit=200, ratiovacc=0.25, contamprob=0.3)), label="pageRank ratio vaccination=0.25",
#             color="green")
#    print("génération de la courbe de propagation de la maladie avec pageRank ratio vaccination=0.18:")
#    plt.plot(*zip(*iterate(A, countit=200, ratiovacc=0.18, contamprob=0.3)), label="pageRank ratio vaccination=0.18",
#             color="blue")
#    print("génération de la courbe de propagation de la maladie avec pageRank ratio vaccination=0.08:")
#    plt.plot(*zip(*iterate(A, countit=200, ratiovacc=0.08, contamprob=0.3)), label="pageRank ratio vaccination=0.08",
#             color="red")
    plt.savefig('bitcoinalpha.png')
