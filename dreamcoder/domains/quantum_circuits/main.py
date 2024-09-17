import datetime
import os
from dreamcoder.domains.quantum_circuits.tasks import QuantumTask, makeTasks
from dreamcoder.domains.quantum_circuits.primitives import primitives, grammar, full_grammar
import dreamcoder as dc
import sys
sys.path.append("../")


try:  # pypy will fail
    from dreamcoder.recognition import variable
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import train_test_split
    import dill as pickle
    import numpy as np
except:
    pass


def get_tasks(enumeration_timeout, label):
    if os.path.exists(f"experimentOutputs/quantum/{label}_train_tasks"):
        dc.utilities.eprint(f"Loading task dataset {label}.")
        with open(f"experimentOutputs/quantum/{label}_train_tasks", "rb") as f:
            train_tasks = pickle.load(f)

        with open(f"experimentOutputs/quantum/{label}_test_tasks", "rb") as f:
            test_tasks = pickle.load(f)

        with open(f"experimentOutputs/quantum/{label}_tasks", "rb") as f:
            tasks = pickle.load(f)

    else:
        dc.utilities.eprint("No task dataset found, generating a new one.")

        tasks = makeTasks(enumeration_timeout)
        n_train = 5000  # int(len(tasks)/30)  # TODO: this should be a flag

        total_indices = np.arange(len(tasks))
        # TODO: weight should be assigned in a different way, perhaps a variable in the task
        probs = np.array([task.name[6:].count("(") for task in tasks], dtype=float)
        for i in np.arange(np.max(probs)+1):
            m = probs[probs == i].sum()
            probs[probs == i] /= m
        weight = probs/probs.sum()

        indices = set(np.random.choice(total_indices, n_train, p=weight, replace=False))
        # remaining_indices = set(total_indices) - indices

        train_tasks = [tasks[i] for i in indices]
        test_tasks = [task for task in tasks if task not in train_tasks]

        with open(f"experimentOutputs/quantum/{label}_train_tasks", "wb") as f:
            pickle.dump(train_tasks, f)

        with open(f"experimentOutputs/quantum/{label}_test_tasks", "wb") as f:
            pickle.dump(test_tasks, f)

        with open(f"experimentOutputs/quantum/{label}_tasks", "wb") as f:
            pickle.dump(tasks, f)

    return tasks, train_tasks, test_tasks


def test_grammar(grammar, tasks, arguments, path_and_label):
    arguments = arguments.copy()
    dc.utilities.eprint(f"Testing extracted grammar on {len(tasks)} tasks.")
    arguments["noConsolidation"] = True
    arguments["iterations"] = 1
    del arguments["taskBatchSize"]
    del arguments["taskReranker"]
    del arguments["resume"]

    # or load from file
    # with open("experimentOutputs/quantum/long_grammar","rb") as f:
    #     g0 = pickle.load(f)

    generator = dc.dreamcoder.ecIterator(grammar, tasks,
                                         testingTasks=[],
                                         outputPrefix=path_and_label,
                                         **arguments)
    for result in generator:
        ...
    return result


def main(arguments):
    """
    Takes the return value of the `commandlineArguments()` function as input and
    trains/tests the model on a set of quantum-algorithm tasks.
    """

    # Create experiment directory
    timestamp = datetime.datetime.now().isoformat()
    if arguments["resume"] is None:
        outputDirectory = "experimentOutputs/quantum/%s" % timestamp
    else:
        outputDirectory = "/".join(arguments["resume"].split("/")[:-1])
    del arguments["outputDirectory"]
    os.system("mkdir -p %s" % outputDirectory)
    # Dump arguments
    with open(f"{outputDirectory}/arguments.pickle", "wb") as f:
        pickle.dump(arguments, f)

    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    dc.domains.quantum_circuits.primitives.GLOBAL_NQUBIT_TASK = int(arguments["nqubit"])
    del arguments["nqubit"]

    # Get quantum task dataset
    # TODO: this is hard coded!
    tasks, train_tasks, test_tasks = get_tasks(50, "medium_3qubit_2023_fastphase")
    # tasks, train_tasks, test_tasks = get_tasks(150, "medium_5qubit_2023_fastphase")

    # check LIMITED_CONNECTIVITY
    if arguments["limitedConnectivity"]:
        dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = True
        dc.utilities.eprint("Limited qubit connectivity enforced")
    del arguments["limitedConnectivity"]

    # TRAIN
    g0 = grammar
    if arguments["fromGrammar"] is not None:
        with open(f"experimentOutputs/quantum/{arguments['fromGrammar']}", "rb") as f:
            g0 = pickle.load(f)
    del arguments["fromGrammar"]

    generator = dc.dreamcoder.ecIterator(g0, train_tasks,
                                         testingTasks=[],
                                         outputPrefix=f"{outputDirectory}/quantum_train",
                                         **arguments)

    for result in generator:
        # TEST at each step
        train_result = test_grammar(result.grammars[-1],
                                    train_tasks,
                                    arguments,
                                    f"{outputDirectory}/quantum_test-train_{len(result.grammars)-1}")
        with open(f"{outputDirectory}/solved_train.txt", "a") as f:
            f.write(f"{train_result.learningCurve[-1]}\n")

        test_result = test_grammar(result.grammars[-1],
                                   test_tasks,
                                   arguments,
                                   f"{outputDirectory}/quantum_test_{len(result.grammars)-1}")
        with open(f"{outputDirectory}/solved_test.txt", "a") as f:
            f.write(f"{test_result.learningCurve[-1]}\n")

    # FULL GRAMMAR on TEST
    dc.utilities.eprint("Consistency check: enumerating full grammar (should solve all tasks)")
    dc.domains.quantum_circuits.primitives.GLOBAL_LIMITED_CONNECTIVITY = False
    test_grammar(full_grammar, test_tasks, arguments, f"{outputDirectory}/quantum_full")

    return outputDirectory
