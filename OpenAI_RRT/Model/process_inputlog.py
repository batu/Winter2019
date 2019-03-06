import sys
import numpy as np
f = open("Input_Log.txt", "r")

meta_data_lines = 4
for _ in range(meta_data_lines):
    print(f.readline())
   
# 0     1     2         3      4        5       6        7     8    9   10   11
#[Up, Down,  Left,     Right, Select, Start,    Y,       B,    X,   A,   L,   R]
#['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']

start = 4
end = start + 12


dist = np.zeros(4096)
def map2retro(bizhawk_rep):
    retro_mapping = [0,0,0,0,0,0,0,0,0,0,0,0]

    for idx, value in enumerate(bizhawk_rep):
        if idx == 0:
            retro_mapping[4] = value
        if idx == 1:
            retro_mapping[5] = value
        if idx == 2:
            retro_mapping[6] = value
        if idx == 3:
            retro_mapping[7] = value
        if idx == 4:
            retro_mapping[2] = value
        if idx == 5:
            retro_mapping[3] = value
        if idx == 6:
            retro_mapping[1] = value
        if idx == 7:
            retro_mapping[0] = value
        if idx == 8:
            retro_mapping[9] = value
        if idx == 9:
            retro_mapping[8] = value
        if idx == 10:
            retro_mapping[10] = value
        if idx == 11:
            retro_mapping[11] = value

    number = 0
    if 1 in retro_mapping:
        for idx, digit in enumerate(retro_mapping):
            number += 2**idx * digit
        print(retro_mapping, number)
        dist[number] += 1 

            
for line in f:
    p1_buttons = list(line[start:end])
    binary_bizhawk = [0 if button == "." else 1 for button in p1_buttons]
    map2retro(binary_bizhawk)

print(list(dist))
print(np.argmax(list(dist)))
dist /= sum(dist)
np.save("retro_dist.npy", np.array(dist))