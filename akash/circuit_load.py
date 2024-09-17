#!/bin/env python
import pickle
N = 2
J=1
decomposed_list = [0,1]
mag_field_strength_list = [0.001,0.1,1,1.5]
mag_field_strength = [mag_field_strength_list[0]]
for decomposed in decomposed_list:
    print('decomposed?', decomposed)
    for h in mag_field_strength:
        with open(f'akash/solved_RL_circuits/circ_list_TFIM_qubit{N}_J{J}_h{h}_decomposed{decomposed}.pickle', 'rb') as handle:
            b = pickle.load(handle)
        print(J, len(b))
        for circ in b:
            print(circ)
        print('-x-x-x-x-x-')
        print()
