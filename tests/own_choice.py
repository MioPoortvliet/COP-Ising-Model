from src.utils import choice
import numpy as np

N=int(1e6)
p = 0.55

ownchoice = choice(p, N)
npchoice = np.random.choice([False, True], N, p=[p, 1-p])

#print(np.sum(ownchoice==True)/N)
#print(np.sum(npchoice==True)/N)