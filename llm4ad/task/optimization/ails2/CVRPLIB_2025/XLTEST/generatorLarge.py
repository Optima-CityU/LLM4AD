import sys, random, math, os

# To plot the instance, uncomment the next five lines and the twelve last lines (it deteriorates performance)
# import matplotlib as mpl
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
# import matplotlib.pyplot as plt

if len(sys.argv) < 7:
    print('Missing arguments:\n\t python generate.py n depotPos custPos demandType avgRouteSize randSeed')
    help="""

    n (number of customers)

    Depot positioning
        1 = Random				
        2 = Centered				
        3 = Cornered				
                        
    Customer positioning
        1 = Random				
        2 = Clustered				
        3 = Random-clustered		
                        
    Demand distribution	
        1 = Unitary		
        2 = Small, large var		
        3 = Small, small var		
        4 = Large, large var		
        5 = Large, small var		
        6 = Large, depending on quadrant	
        7 = Few large, many small

    Average route size
        1 = Ultra short
        2 = Very short
        3 = Short
        4 = Medium
        5 = Long
        6 = Very long
        7 = Ultra long
        
    Output: instance file XL-nA-kB.vrp

    This script was used to generate the instances of the CVRPLIB Best Known Solution Challenge. For more details about the generation process read:
        Uchoa et al (2017). New benchmark instances for the Capacitated Vehicle Routing Problem. European Journal of Operational Research
        Queiroga, Eduardo, et al. (2022). 10,000 optimal CVRP solutions for testing machine learning based heuristics.
        """
    print(help) 
    exit(0)

def distance(x,y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

# constants
maxCoord = 1000
decay = 40

# read input argmuments
n = int(sys.argv[1])
rootPos = int(sys.argv[2])
custPos = int(sys.argv[3])
demandType = int(sys.argv[4])
randSeed = int(sys.argv[6]) # random seed for reproducibility
if demandType > 7:
    print("Demant type out of range!")
    exit(0)

random.seed(randSeed)

nSeeds = random.randint(2,6) 

In = {1:(3,5), 2:(5,8), 3:(8,12), 4:(12,16), 5:(16,25), 6:(25,50), 7:(50,200)}
avgRouteSize = int(sys.argv[5])
if avgRouteSize > 7:
    print("Average route size out of range!")
    exit(0)	
r = random.uniform(In[avgRouteSize][0], In[avgRouteSize][1])

depot = (-1,-1) # depot position
S = set() # set of coordinates for the customers

x_,y_ = (-1,-1)
#Root positioning
if rootPos == 1:
    x_ = random.randint(0,maxCoord)
    y_ = random.randint(0,maxCoord)
elif rootPos == 2:
    x_ = y_ = int(maxCoord/2.0)
elif rootPos == 3:
    x_ = y_ = 0
else:
    print("Depot Positioning out of range!")
    exit(0)
depot = (x_,y_)

#Customer positioning
nRandCust = -1
if custPos == 3:
    nRandCust = int(n/2.0)
elif custPos == 2:
    nRandCust = 0
elif custPos == 1:
    nRandCust = n
    nSeeds = 0
else:
    print("Costumer Positioning out of range!")
    exit(0)

nClustCust = n - nRandCust

#Generating random customers
for i in range(1, nRandCust+1):
    x_ = random.randint(0,maxCoord)
    y_ = random.randint(0,maxCoord)
    while (x_,y_) in S or (x_,y_) == depot:
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
    S.add((x_,y_))

nS = nRandCust

seeds = []
# Generation of the clustered customers
if nClustCust > 0:
    if nClustCust < nSeeds:
        print("Too many seeds!")
        exit(0)
    
    #Generate the seeds
    for i in range(nSeeds):
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
        while (x_,y_) in S or (x_,y_) == depot:
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
        S.add((x_,y_))
        seeds.append((x_,y_))
    nS = nS + nSeeds
    
    # Determine the seed with maximum sum of weights (w.r.t. all seeds)
    maxWeight = 0.0
    for i,j in seeds:
        w_ij = 0.0
        for i_,j_ in seeds:
            w_ij += 2**(-distance((i,j), (i_,j_)) / decay)
        if w_ij > maxWeight:
            maxWeight = w_ij

    norm_factor = 1.0/maxWeight

    # Generate the remaining customers using Accept-reject method
    while nS < n:
        x_ = random.randint(0,maxCoord)
        y_ = random.randint(0,maxCoord)
        while (x_,y_) in S or (x_,y_) == depot:
            x_ = random.randint(0,maxCoord)
            y_ = random.randint(0,maxCoord)
        
        weight = 0.0
        for i_,j_ in seeds:
            weight += 2**(-distance((x_,y_), (i_,j_)) / decay)
        weight *= norm_factor
        rand = random.uniform(0,1)

        if rand <= weight: # Will we accept the customer?
            S.add((x_,y_))
            nS = nS + 1

V = [depot] + list(S) # set of vertices (from now on, the ids are defined)

# Demands
demandMinValues = [1,1,5,1,50,1,51,50,1]
demandMaxValues = [1,10,10,100,100,50,100,100,10]
demandMin = demandMinValues[demandType-1]
demandMax = demandMaxValues[demandType-1]
demandMinEvenQuadrant = 51
demandMaxEvenQuadrant = 100
demandMinLarge = 50
demandMaxLarge = 100
largePerRoute = 1.5
demandMinSmall = 1
demandMaxSmall = 10

D = [] # demands
sumDemands = 0
maxDemand = 0

for i in range(2,n + 2):
    j = int((demandMax - demandMin + 1) * random.uniform(0,1) + demandMin)
    if demandType == 6:
        if (V[i - 1][0] < maxCoord/2.0 and V[i - 1][1] < maxCoord/2.0) or (V[i - 1][0] >= maxCoord/2.0 and V[i - 1][1] >= maxCoord/2.0):
            j = int((demandMaxEvenQuadrant - demandMinEvenQuadrant + 1) * random.uniform(0,1) + demandMinEvenQuadrant)
    if demandType == 7:
        if i < (n / r) * largePerRoute:
            j = int((demandMaxLarge - demandMinLarge + 1) * random.uniform(0,1) + demandMinLarge)
        else:
            j = int((demandMaxSmall - demandMinSmall + 1) * random.uniform(0,1) + demandMinSmall)
    D.append(j)
    if j > maxDemand:
        maxDemand = j
    sumDemands = sumDemands + j

# Generate capacity
capacity = -1
if sumDemands == n:
    capacity = math.floor(r)
else:
    capacity = max(maxDemand, math.ceil(r * sumDemands / n))

k = int(math.ceil(sumDemands/float(capacity)))

instanceName = 'XLTEST-n'+str(n+1)+'-k'+str(k)
f = open(instanceName+'.vrp', 'w')
f.write('NAME : ' + instanceName + '\n')
f.write('COMMENT : \"This is not an official CVRPLIB BKS Challenge instance; the official ones will be released on Jan 12, 2026, 15:00:00 GMT.\"\n')
f.write('TYPE : CVRP\n')
f.write('DIMENSION : ' + str(n+1) + '\n')
f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
f.write('CAPACITY : ' + str(int(capacity)) + '\n')
f.write('NODE_COORD_SECTION\n')

for i,v in enumerate(V):
    f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(v[0])+' '+'{:<4}'.format(v[1])+'\n')

f.write('DEMAND_SECTION\n')
if demandType != 6:
    random.shuffle(D)
D = [0] + D
for i,d in enumerate(V):
    f.write('{:<4}'.format(i+1)+' '+'{:<4}'.format(D[i])+'\n')

f.write('DEPOT_SECTION\n1\n-1\nEOF\n')

f.close()

# x = [v[0] for v in V]
# y = [v[1] for v in V]
# x_s = [v[0] for v in seeds]
# y_s = [v[1] for v in seeds]
# plt.figure(figsize=(20, 20), dpi=80)
# plt.scatter(x, y, marker='o', color='blue',edgecolor='blue', s=40)
# plt.scatter(x_s, y_s, marker='o', color='magenta',edgecolor='magenta', s=40)
# plt.scatter([x[0]], [y[0]], marker='s', edgecolor='black', color='yellow', s=200)
# plt.xticks([]) # Turn off x labels
# plt.yticks([]) # Turn off y labels
# plt.savefig(instanceName+'.png')
# plt.close()

