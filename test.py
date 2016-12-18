import random
from collections import deque
import numpy as np
import pickle

def loadTable():

	fp = open('savedAI/secondAI', 'rb')
	AI = pickle.load(fp)
	q_table = AI['qt']
	timestep = AI['t']
	replay_memory = AI['replay']
	reward_memory = AI['reward_mem']
	fp.close()

	print(q_table['lowerside'][134][32])


loadTable()