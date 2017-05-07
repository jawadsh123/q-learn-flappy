
import random
from collections import deque
import pickle
import math


q_table = {}

INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EPSILON = INITIAL_EPSILON
EXPLORE = 2000000
LAMBDA = 1
ALPHA = 0.7
REWARD = 1
PENALTY = -1000
MEMORY_LENGTH = 50000
REWARD_MEMORY_LENGTH = 4
PENALTY_MEMORY_LENGTH = 1
RESOLUTION = 20

# update AI or new AI
NEW_AI = False
AI_NAME = 'high_penalty_heavy_rewarding'



replay_memory = deque()
reward_memory = deque()
last_state = {}


def mapInResolution(number):
	return math.floor(number/RESOLUTION)

def actionSelect(x_distance, y_distance, side):
	x_distance = mapInResolution(x_distance)
	y_distance = mapInResolution(y_distance)

	new_state = q_table[side][y_distance][x_distance]

	up = new_state['flap']
	down = new_state['do_nothing']
	if up == down:
		action = 'do_nothing'
	else:
		randNum = random.random()
		if randNum < EPSILON:
			action = randomSelect()
		else:
			if up > down:
				action = 'flap'
			else:
				action = 'do_nothing'

	reward_memory.append({'x':x_distance, 'y':y_distance, 's':side, 'a':action})

	if len(reward_memory) > REWARD_MEMORY_LENGTH:
		reward_memory.popleft()

	return action



def updateRewards():

	for state in reward_memory:
		x = state['x']
		y = state['y']
		s = state['s']
		a = state['a']
		q_table[s][y][x][a] += REWARD

def reward(heavy=False):

	if len(reward_memory) >= 2:
		from_state = reward_memory[-2]
		from_x = from_state['x']
		from_y = from_state['y']
		from_s = from_state['s']
		from_a = from_state['a']

		to_state = reward_memory[-1]
		to_x = to_state['x']
		to_y = to_state['y']
		to_s = to_state['s']

		if not heavy:
			curr_reward = REWARD
		else:
			curr_reward = REWARD * 100

		q_table[from_s][from_y][from_x][from_a] = q_table[from_s][from_y][from_x][from_a] + ALPHA*(curr_reward + LAMBDA*(max(q_table[to_s][to_y][to_x]['flap'], q_table[to_s][to_x][to_y]['do_nothing'])) - q_table[from_s][from_y][from_x][from_a] )

def penalize(timestep):

	if len(reward_memory) >= 2:
		from_state = reward_memory[-2]
		from_x = from_state['x']
		from_y = from_state['y']
		from_s = from_state['s']
		from_a = from_state['a']

		to_state = reward_memory[-1]
		to_x = to_state['x']
		to_y = to_state['y']
		to_s = to_state['s']

		q_table[from_s][from_y][from_x][from_a] = q_table[from_s][from_y][from_x][from_a] + ALPHA*(PENALTY + LAMBDA*(max(q_table[to_s][to_y][to_x]['flap'], q_table[to_s][to_x][to_y]['do_nothing'])) - q_table[from_s][from_y][from_x][from_a] )

		print("Dying State {}".format(from_state))
		print("Values {}, TimeStep {}".format(q_table[from_s][from_y][from_x], timestep))
	

def backtrackRewards(x_distance, y_distance, side, action):
	x_distance = mapInResolution(x_distance)
	y_distance = mapInResolution(y_distance)

	new_state = q_table[side][y_distance][x_distance]	

	if len(last_state) > 0:
		x = last_state['x_distance']
		y = last_state['y_distance']
		s = last_state['side']
		a = last_state['action']

		q_table[s][y][x][a] = q_table[s][y][x][a] + ALPHA*(REWARD + LAMBDA*(max(new_state['flap'], new_state['do_nothing'])) - q_table[s][y][x][a] )


		replay_memory.append({'from':{'x':x, 'y':y, 's':s, 'a':a}, 'to':{'x':x, 'y':y, 's':s} })

		# if a == 'flap':
		# 	print(q_table[s][y][x], x, y, s, a, flappy.timestep)

	if len(replay_memory) > MEMORY_LENGTH:
		replay_memory.popleft()

	last_state.clear()
	last_state['x_distance'] = x_distance
	last_state['y_distance'] = y_distance
	last_state['side'] = side
	last_state['action'] = action


def updatePenalty():

	for x in range(len(reward_memory)):

		if len(reward_memory)-x <= PENALTY_MEMORY_LENGTH:
			state = reward_memory[x]
			x = state['x']
			y = state['y']
			s = state['s']
			a = state['a']
			q_table[s][y][x][a] -= PENALTY
			print(q_table[s][y][x], x, y, s, a, flappy.timestep)

	


def updateEpsilon():
	global EPSILON
	
	if EPSILON > FINAL_EPSILON:
		EPSILON -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


def train():
	minibatch = random.sample(replay_memory, 32)

	for transition in minibatch:
		f = transition['from']
		s = transition['to']

		new_state = q_table[s['s']][s['y']][s['x']]
		old_state = q_table[f['s']][f['y']][f['x']][f['a']]

		q_table[f['s']][f['y']][f['x']][f['a']] = q_table[f['s']][f['y']][f['x']][f['a']] + ALPHA*( LAMBDA*(max(new_state['flap'], new_state['do_nothing'])) - q_table[f['s']][f['y']][f['x']][f['a']] )

	

def saveTable(timestep, max_score):
	global replay_memory, reward_memory

	fp = open('savedAI/' + AI_NAME, 'wb')
	pickle.dump({'qt': q_table, 't': timestep, 'replay': replay_memory, 'reward_mem': reward_memory, 'max_score': max_score}, fp)
	fp.close()
	print('AI Saved to :  savedAI/'+AI_NAME)


def loadTable():
	global replay_memory, reward_memory, q_table

	if NEW_AI == False:
		fp = open('savedAI/' + AI_NAME, 'rb')
		AI = pickle.load(fp)
		q_table = AI['qt']
		timestep = AI['t']
		max_score = AI['max_score']
		replay_memory = AI['replay']
		reward_memory = AI['reward_mem']
		fp.close()

		print('AI Loaded from :  savedAI/'+AI_NAME)
	else:
		timestep = 0
		max_score = 0
		q_table['upperside'] = [ [ {'flap': 0, 'do_nothing': 0} for _ in range(mapInResolution(288)) ] for _ in range(mapInResolution(512)) ]
		q_table['lowerside'] = [ [ {'flap': 0, 'do_nothing': 0} for _ in range(mapInResolution(288)) ] for _ in range(mapInResolution(512)) ]

	return timestep, max_score


def randomSelect():
	x = random.randrange(1, 100)
	if x > 90:
		return 'flap'
	else:
		return 'do_nothing'

if __name__ == '__main__':
	print("running")