import flappy
import random
from collections import deque
import pickle



q_table = {}
q_table['upperside'] = [ [ {'flap': 0, 'do_nothing': 0} for _ in range(288) ] for _ in range(512) ]
q_table['lowerside'] = [ [ {'flap': 0, 'do_nothing': 0} for _ in range(288) ] for _ in range(512) ]


INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EPSILON = INITIAL_EPSILON
EXPLORE = 2000000
LAMBDA = 0.8
ALPHA = 0.5
REWARD = 1000
PENALTY = 50
MEMORY_LENGTH = 50000
REWARD_MEMORY_LENGTH = 50
PENALTY_MEMORY_LENGTH = 10

# update AI or new AI
NEW_AI = False
AI_NAME = 'secondAI'



replay_memory = deque()
reward_memory = deque()
last_state = {}



def actionSelect(x_distance, y_distance, side):

	new_state = q_table[side][y_distance][x_distance]

	up = new_state['flap']
	down = new_state['do_nothing']
	if up == down:
		action = randomSelect()
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

	

def backtrackRewards(x_distance, y_distance, side, action):	

	new_state = q_table[side][y_distance][x_distance]	

	if len(last_state) > 0:
		x = last_state['x_distance']
		y = last_state['y_distance']
		s = last_state['side']
		a = last_state['action']

		q_table[s][y][x][a] = q_table[s][y][x][a] + ALPHA*( LAMBDA*(max(new_state['flap'], new_state['do_nothing'])) - q_table[s][y][x][a] )


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

		if len(reward_memory)-x < PENALTY_MEMORY_LENGTH:
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

	

def saveTable():
	global replay_memory, reward_memory, q_table

	fp = open('savedAI/' + AI_NAME, 'wb')
	pickle.dump({'qt': q_table, 't': flappy.timestep, 'replay': replay_memory, 'reward_mem': reward_memory}, fp)
	fp.close()
	print('AI Saved to :  savedAI/'+AI_NAME)


def loadTable():
	global replay_memory, reward_memory, q_table

	if NEW_AI == False:
		fp = open('savedAI/' + AI_NAME, 'rb')
		AI = pickle.load(fp)
		q_table = AI['qt']
		flappy.timestep = AI['t']
		replay_memory = AI['replay']
		reward_memory = AI['reward_mem']
		fp.close()

		print('AI Loaded from :  savedAI/'+AI_NAME)


def randomSelect():
	x = random.randrange(1, 100)
	if x > 90:
		return 'flap'
	else:
		return 'do_nothing'


if __name__ == '__main__':
	flappy.main()