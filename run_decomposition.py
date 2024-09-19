#! /bin/env python
import argparse
import dill
from datetime import datetime
import numpy as np
import pickle
import qiskit as qk
from qiskit.quantum_info import Operator, Statevector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from typing import Tuple

from akash.TFIM_ham_gen import construct_hamiltonian

import dreamcoder as dc
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.fragmentGrammar import FragmentGrammar
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.program import Abstraction
from dreamcoder.task import Task
from dreamcoder.utilities import numberOfCPUs
import dreamcoder.domains.quantum_ground_state.primitives as pr
from dreamcoder.domains.quantum_ground_state.primitives import (
    circuit_to_mat,
    full_op_names,
    mat_contraction,
    mat_to_tensor,
    execute_program,
    normalize_unitary,
    get_qiskit_circuit,
    get_instructions_from_qiskit,
    get_code_from_instructions,
    qiskit_full_op_names,
    tcircuit,
    tensor_contraction,
    no_op,
    n_qubit_gate,
    QiskitTester,
)
from dreamcoder.domains.quantum_ground_state.primitives import execute_quantum_algorithm
from dreamcoder.domains.quantum_ground_state.tasks import GroundStateTask,get_energy
from dreamcoder.program import Program, Primitive, EtaLongVisitor
from dreamcoder.utilities import eprint, Curried

decomposed_list = [0, 1]


class args:
    n_qubits = 2
    J = 1
    hh = 0.05
    decomposed = 1
    arity = 2
    structurePenalty = 1
    pseudoCounts = 10


# Read the command line arguments
parser = argparse.ArgumentParser(
    description="Example implementation of Regularized Mutual Information Feature Selector on a solid drop.",
    epilog="Results will be saved in files with the OUTPUT tag in the 'outputs/' folder.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "-n_qubits", type=int, default=args.n_qubits, help="Number of qubits"
)
parser.add_argument("-J", type=float, default=args.J, help="Interaction strength")
parser.add_argument("-hh", type=float, default=args.hh, help="External field strength")
parser.add_argument(
    "-decomposed",
    type=int,
    default=args.decomposed,
    help="Either 0=parametrized gates, 1=qiskit hardware basis",
)
parser.add_argument(
    "-arity",
    type=int,
    default=args.arity,
    help="Number of arguments of extracted gates",
)
parser.add_argument(
    "-structurePenalty", type=int, default=args.structurePenalty, help="hyperparameter"
)
parser.add_argument(
    "-pseudoCounts", type=int, default=args.pseudoCounts, help="hyperparameter"
)

try:
    args = parser.parse_args()
except SystemExit as e:
    eprint("Running from interactive session. Loading default parameters")
path = f"akash/solved_RL_circuits/circ_list_TFIM_qubit{args.n_qubits}_J{args.J}_h{args.hh}_decomposed{args.decomposed}.pickle"
name = f"ground_{args.n_qubits}_J{args.J}_h{args.hh}_dec{args.decomposed}"
with open(path, "rb") as handle:
    b = dill.load(handle)
eprint(f"Loading solutions from {path}")
# Unfortunately these flags are set globally
dc.domains.quantum_ground_state.primitives.GLOBAL_NQUBIT_TASK = args.n_qubits
dc.domains.quantum_ground_state.primitives.GLOBAL_LIMITED_CONNECTIVITY = False

library_settings = {
    "topK": 2,  # how many solutions to consider
    "arity": args.arity,  # how many arguments
    "structurePenalty": args.structurePenalty,  # increase regularization 3 4 (it was 1), look at a few in [1,15]
    "pseudoCounts": args.pseudoCounts,  # increase to 100, test a few values
}

primitives = [pr.p_sx, pr.p_x, pr.p_rz, pr.p_cz]
grammar = Grammar.uniform(primitives)
eprint(f"Library building settings: {library_settings}")
# Generate a few example tasks
solutions = {}  # dict of task:solution
# NOTE: we have a task for each decomposition because they have various different real parameters
# We cannot have solutions with different requests for a task,
# and it is not clear how to use real numbers as primitives (just for evaluation, we cannot enumerate them)
for idx, circuit in enumerate(b):
    H = construct_hamiltonian(args.J, args.hh, args.n_qubits)
    instructions = get_instructions_from_qiskit(circuit)
    code, arguments = get_code_from_instructions(instructions)
    program = Program.parse(code)
    task = GroundStateTask(
        f"J_{args.J:2.2f}_h_{args.hh:2.2f}_N_{args.n_qubits}_v{idx}",
        hamiltonian=H,
        arguments=arguments,
        request=program.infer(),
    )
    likelihood = task.logLikelihood(program)
    prior = grammar.logLikelihood(program.infer(), program)

    frontier_entry = FrontierEntry(
        program=program, logLikelihood=likelihood, logPrior=prior
    )

    solutions[task] = Frontier(
        frontier=[frontier_entry],  # multiple solutions are allowed
        task=task,
    )
    eprint(f"#{idx:3}, Energy = {likelihood:2.6f}")
tasks = list(solutions.keys())
frontiers = [f for f in solutions.values()]

unique_frontiers_set = set()
unique_frontiers = []
for frontier in frontiers:
    program = frontier.entries[0].program
    if program not in unique_frontiers_set:
        unique_frontiers_set.add(program)
        unique_frontiers.append(frontier)
eprint(
    f"We have {len(unique_frontiers)}/{len(frontiers)} frontiers. The others are duplicate solutions"
)

unique_frontiers
new_grammar, new_frontiers = FragmentGrammar.induceFromFrontiers(
    g0=grammar,
    frontiers=unique_frontiers[:],
    **library_settings,
    CPUs=numberOfCPUs() - 2
)

new_grammar, new_frontiers
timestamp = datetime.now().isoformat()
with open(f"experimentOutputs/{timestamp}_{name}_grammar.pickle", "wb") as f:
    pickle.dump(new_grammar, f)

with open(f"experimentOutputs/{timestamp}_{name}_frontiers.pickle", "wb") as f:
    pickle.dump(new_frontiers, f)
eprint(f"Results saved in experimentOutputs/{timestamp}_{name}_...")