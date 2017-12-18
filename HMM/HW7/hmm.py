import math
import collections


def get_neighbours(location):
    x = location[0]
    y = location[1]
    n = []

    if x+1 < 10:
        n.append((x+1, y))
    if y+1 < 10:
        n.append((x, y+1))
    if x-1 > 0:
        n.append((x-1, y))
    if y-1 > 0:
        n.append((x, y-1))

    return n

def get_distance(a, b):
    s = pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)
    return math.sqrt(s)


g = t = n = -1
pos_dr = [] # Possible distance ranges
prob_locs = [] # possible locations per time stamp
avail_grid_locs = [] # All the grid locations, robot can stay in
full_grid = [[0 for col in range(10)] for row in range(10)]  # Entire grid world
t_loc = [] # location of towers
dist = [] # distance from noisy data
row = 0


# Read input data
with open("hmm-data.txt") as f:
    for lines in f:
        lines = lines.strip()
        if lines == 'Grid-World:':
            g += 1
        elif lines == 'Tower Locations:':
            t += 1
            g = -1
        elif lines.startswith('Noisy Distances'):
            n += 1
            t = -1
        else:
            if g == 0:
                g += 1
            elif g == 1:
                if lines == '':
                    continue
                # Read Grid Data
                items = lines.split()
                for col, ele in enumerate(items):
                    full_grid[row][col] = int(ele)
                    if int(ele) == 1:
                        avail_grid_locs.append([row, col])
                row += 1
                if row == 10:
                    continue
            elif t == 0:
                t += 1
            elif t == 1:
                if lines == '':
                    continue
                # read tower data
                items = lines.split(':')
                items = items[1].split()
                t_loc.append([int(items[0]), int(items[1])])
                if len(t_loc) == 4: 
                    continue
            elif n == 0:
                n += 1
            elif n == 1:
                if lines == '':
                    continue
                # read distance data
                items = lines.split()
                temp = []
                for ele in items:
                    temp.append(float(ele))
                dist.append(temp)


# calculate possibel distance ranges 
for item in avail_grid_locs:
    temp = []
    for t in t_loc:
        d = get_distance(item, t)
        temp.append([.7 * d, 1.3 * d])
    pos_dr.append(temp)


# Get all possible locations for each time stamp
for d in dist:
    temp = []
    for i, locs in enumerate(avail_grid_locs):
        flag = True
        for j, ele in enumerate(d):
            if pos_dr[i][j][0] <= ele <= pos_dr[i][j][1] :
                continue
            else:
                flag = False
                break
        if flag:
            temp.append(locs)
    prob_locs.append(temp)



# Probability of reaching a location from another location

d = collections.defaultdict(list)
for i, lis in enumerate(prob_locs):
    for tup in lis:
        d[tuple(tup)].append(i)


probabs = collections.defaultdict(dict)
prob_c = collections.defaultdict(dict)

for tup in d:
    indices = d[tup]
    neigh = get_neighbours(tup)
    for ind in indices:
        ind += 1
        for ele in neigh:
            if ele in d:
                if ind in d[ele]:
                    if ele not in probabs[tup]:
                        probabs[tup][ele] = 0
                    probabs[tup][ele] += 1
                    if tup not in prob_c:
                        prob_c[tup] = 0
                    prob_c[tup] += 1


for tup in probabs:
    for ele in probabs[tup]:
        probabs[tup][ele] /= prob_c[tup]
        


# Viterbi
l = 0
final = {}
final[l] = {}
for tup in prob_locs[l]:
    tup = tuple(tup)
    final[l][tup] = {}
    final[l][tup]['percent'] = 1
    final[l][tup]['father'] = ''

for l in range(1, 11):
    final[l] = {}
    for tup in final[l - 1]:
        if tup in probabs:
            for avail in probabs[tup]:
                contends_list = list(avail)
                if contends_list in prob_locs[l]:
                    if avail not in final[l]:
                        final[l][avail] = {}
                        final[l][avail]['father'] = tup
                        present_prob = final[l - 1][tup]['percent'] * probabs[tup][avail]
                        final[l][avail]['percent'] = present_prob
                    else:
                        present_prob = final[l - 1][tup]['percent'] * probabs[tup][avail]
                        if present_prob > final[l][avail]['percent']:
                            final[l][avail]['father'] = tup
                            final[l][avail]['percent'] = present_prob

f = ''
l = 10
top = -1
for tup in final[l]:
    if top < final[l][tup]['percent']:
        top = final[l][tup]['percent']
        f = tup

path = []
path.append(f)
while True:
    if final[l][f]['father'] == '':
        break
    new_final = final[l][f]['father']
    path.append(new_final)
    l -= 1
    f = new_final

print "Path - ", path[::-1]
