#Curio Simulator Physics test:

#imports
import simpleguitk as simple
#from PIL import Image as IM
#import math
import random
import matplotlib.path as mplPath
import numpy as np
from matplotlib import pyplot as plt

WIDTH = 400

scaleVal = 500/10

HEIGHT = 400#10 * scaleVal

X_min = 0
X_max = WIDTH
Y_min = 0
Y_max = HEIGHT
delta_tt = 0.1

wall_list = list()

start = (40,40)

#end = (200,WIDTH-100)
end = (WIDTH-40,HEIGHT-80)

stepSize = 20
epidode = 0
X = start[0]
Y = start[1]
maxSteps = 2000
actions = ['rt', 'lt', 'up', 'dw']
stateAction = dict()
epsilon = 0.05
p_correct = 0.95
rewardSum = 0        
stepCount = 0
epi_count = 0
current_state = start
alpha_q = 0.05
gamma = 0.95
reward_dict = {'o': -100, 'u': -0.1, 'g': 100 }
max_epi = 2000
stateAction_f = open('sa_q_pi.txt','a+')
aveRlist = list()

q = open('sa_q.txt','r')
stateAction_old = dict()

reuseProbability = 1
v = 0.95

def read_oldQ():
    global stateAction_old
    
    line = 'x'
    while (line != ''):
        line = q.readline()
        line_l = line.split(',')
        if len(line_l) == 4:
            state = (int(line_l[0].strip('(')),int(line_l[1].strip(')')))
            action = line_l[2]
            value = float(line_l[3].strip('\n'))
            stateAction_old[(state,action)] = value


def scale(humanVal):
    """
    Returns the scaled value to pixel.

    @humanVal: list, or 2-tuple: [x, y] in meter 
    Return: 2-tuple: [pix_x, pix_y] in pixels

    [Assumes: scale factor is previously generated]
    """
    if not scaleVal == None:
        if type(humanVal) is list:
            return (humanVal[0] * scaleVal, humanVal[1] * scaleVal)
        else:
            return humanVal * scaleVal
    else:
        return None

class Obstacle(object):
    def __init__(self,vertices):
        dv = 2
        self.vertices = vertices
        self.expand = list()
        
        self.expand.append([vertices[0][0]-stepSize/dv,vertices[0][1]-stepSize/dv])
        self.expand.append([vertices[1][0]+stepSize/dv,vertices[1][1]-stepSize/dv])
        self.expand.append([vertices[2][0]+stepSize/dv,vertices[2][1]+stepSize/dv])
        self.expand.append([vertices[3][0]-stepSize/dv,vertices[3][1]+stepSize/dv])
        
    def collision(self,robot):
        o_path = mplPath.Path(np.array(self.expand))
        return o_path.contains_point(robot)

o1 = Obstacle([[0,0],[10,0],[10,HEIGHT],[0,HEIGHT]])
wall_list.append(o1)
o2 = Obstacle([[WIDTH-10,0],[WIDTH,0],[WIDTH,HEIGHT],[WIDTH-10,HEIGHT]])
wall_list.append(o2)
o3 = Obstacle([[0,0],[WIDTH,0],[WIDTH,10],[0,10]])
wall_list.append(o3)
o4 = Obstacle([[0,HEIGHT-10],[WIDTH,HEIGHT-10],[WIDTH,HEIGHT],[0,HEIGHT]])
wall_list.append(o4)

o5 = Obstacle([[60,60],[100,60],[100,120],[60,120]])
wall_list.append(o5)
o8 = Obstacle([[150,HEIGHT-200],[100,HEIGHT-200],[100,HEIGHT],[150,HEIGHT]])
wall_list.append(o8)
o7 = Obstacle([[WIDTH-100,100],[WIDTH,100],[WIDTH,120],[WIDTH-100,120]])
wall_list.append(o7)
o6 = Obstacle([[240,200],[340,200],[340,300],[240,300]])
wall_list.append(o6)

#o5 = Obstacle([[150,150],[200,150],[200,200],[150,200]])
#wall_list.append(o5)
#o6 = Obstacle([[300,200],[500,200],[500,250],[300,250]])
#wall_list.append(o6)
#o7 = Obstacle([[WIDTH-300,300],[WIDTH,300],[WIDTH,350],[WIDTH-300,350]])
#wall_list.append(o7)
#o8 = Obstacle([[200,HEIGHT-200],[250,HEIGHT-200],[250,HEIGHT],[200,HEIGHT]])
#wall_list.append(o8)
#o9 = Obstacle([[WIDTH-200,0],[WIDTH-150,0],[WIDTH-150,150],[WIDTH-200,150]])
#wall_list.append(o9)


def initialise_Qsa():
    global stateAction
    
    for i in range(0,WIDTH+1,stepSize):
        for j in range(0,HEIGHT+1,stepSize):
            
            for a in actions:
                stateAction[((i,j),a)] = 0
                
def isOccupied(r):
    for w in wall_list:
        if w.collision(r):
            return True
    return False
    
def isOutofBound(node):
    """
    Takes in a node and returns if node is out of image
    @param node <-- tuple
    @return bool
    """
    if node[0] > X_max or node[0] < X_min or node[1] > Y_max or node[1] < Y_min:
        #print "Node out of bound \n"        
        return True
    else:
        return False
        
def winnerNeighbour(s,a):
    '''
    Takes in state: return Max Q valued neighbour with Q value
    @param: state --> tuple position
    @return: s', value --> tuple, float
    '''
    winningN = None
    winningAction = None
    values = dict()
    
    for ele in actions:
        values[stateAction[(s,ele)]] = (s,ele)

    #print values
    
    maxVal = max(values.keys())
    #print maxVal
    
    winningN, winningAction = values[maxVal]
    
    return winningN, winningAction

def winnerNeighbour_old(s,a):
    '''
    Takes in state: return Max Q valued neighbour with Q value
    @param: state --> tuple position
    @return: s', value --> tuple, float
    '''
    winningN = None
    winningAction = None
    values = dict()
    
    for ele in actions:
        values[stateAction_old[(s,ele)]] = (s,ele)

    #print values
    
    maxVal = max(values.keys())
    #print maxVal
    
    winningN, winningAction = values[maxVal]
    
    #print "Old Winning action, Max value", s, winningAction,maxVal
    return winningN, winningAction
        
def e_greedy(c_state):
    '''
    Takes in current state and return action
    Policy is e-greedy so return random action with probability e
    and return greedy action with probability 1-e
    '''
    flag = 'N'
    a = 'N'
    
    if random.random() <= epsilon:
        flag = 'random'
    else:
        flag = 'greedy'
        
    if flag == 'greedy':
        n, win_a = winnerNeighbour(c_state,'n')
        a = win_a
    
    elif flag == 'random':
        index = random.randint(0,len(actions) - 1)
        a = actions[index]
    else:
        print 'Error \n'
        a = None
    
    #print a
    return a
    
def e_greedy_old(c_state):
    '''
    Takes in current state and return action
    Policy is e-greedy so return random action with probability e
    and return greedy action with probability 1-e
    '''
    flag = 'N'
    a = 'N'
    
    if random.random() <= epsilon:
        flag = 'random'
    else:
        flag = 'greedy'
        
    if flag == 'greedy':
        n, win_a = winnerNeighbour_old(c_state,'n')
        a = win_a
    
    elif flag == 'random':
        index = random.randint(0,len(actions) - 1)
        a = actions[index]
    else:
        print 'Error \n'
        a = None
    
    #print a
    return a
    
def take_action(a):
    global X, Y
    old = (X,Y)
    reward = 0
    #print "Old and action: ", (X,Y),a
    if a == 'lt':
        X_n = X - stepSize
        Y_n = Y
    elif a == 'rt':
        X_n = X + stepSize
        Y_n = Y
    elif a == 'up':
        X_n = X 
        Y_n = Y - stepSize
    elif a == 'dw':
        X_n = X 
        Y_n = Y + stepSize
        

    if  ((not isOccupied((X_n,Y_n))) and (not isOutofBound((X_n,Y_n)))):
        X = X_n
        Y = Y_n
        reward = reward_dict['u']
    else:
        X = X
        Y = Y
        reward = reward_dict['o']
    
    if (X,Y) == end:
        print "GOAL REACHED \n"
        reward = reward_dict['g']
    
    new = (X,Y)
    
    if ((new[0] - old[0] > 25) or (new[0] - old[0] < -25) or (new[1] - old[1] > 25) or (new[1] - old[1] < -25)):
        print "Take action did something fishy: ",old,new
    
    #print "NEW: ", (X,Y)
    return (X,Y),reward
        
        
def update_Q(s,a,s_next,r):
    global stateAction
    
    n, win_a = winnerNeighbour(s,'n')
    maxQ = stateAction[s_next,win_a]
    stateAction[s,a] = stateAction[s,a] + alpha_q*(r + gamma*maxQ-stateAction[s,a])
    
            
def drawHandler(canvas):
    canvas.draw_circle([X,Y], stepSize/2, 1, 'Blue','Blue')    
    
    for w in wall_list:
             canvas.draw_polygon(w.vertices, 1, 'Red','Red')
             
    canvas.draw_circle(start, 2, 1, 'Blue','Blue')    
    canvas.draw_circle(end, 2, 1, 'Green','Green')   
    
    
def timerHandler():
    global X, Y, stepCount,current_state, rewardSum, epi_count, stepCount, aveRlist,reuseProbability
    
    if not(stepCount >= maxSteps or (current_state == end)):
        #print current_state
        
        if random.random() <= reuseProbability:
            #print "Using Old Q table with probability: ", reuseProbability
            current_action = e_greedy_old(current_state)
        else:
            current_action = e_greedy(current_state)
        
#        if random.random() <= (1-p_correct):
#                index = random.randint(0,len(actions) - 1)
#                taken_action = actions[index]
#        else:
#                taken_action = current_action
        taken_action = current_action
        
        next_state, reward = take_action(taken_action)
        rewardSum += reward
        
        #print next_state
        update_Q(current_state,current_action,next_state,reward)
        
        current_state = next_state
        reuseProbability = reuseProbability*v
        stepCount += 1
    else:
        if not (epi_count >= max_epi):
            epi_count += 1
            print '******* Episode Count ******* :', epi_count
            current_state = start
            X = start[0]
            Y = start[0]
            reuseProbability = 1
            
            averageReward = float(rewardSum) / stepCount
            aveRlist.append(averageReward)
            
            aveR = open('aveR_pi.txt','a+')
            aveR.write(str(averageReward) + ',')
            aveR.close()
            
            rewardSum = 0        
            stepCount = 0
        else:
            print "RUN OVER \n"
            
            for k in stateAction.keys():
                stateAction_f.writelines(str(k[0]) + ',' + str(k[1]) + ',' + str(stateAction[k]) + '\n')

            stateAction_f.close()
            
            
# Create frame for rendering
myFrame =  simple.create_frame('Curio Simulator', WIDTH, HEIGHT)
# Create timer for updating the positions of robot(s)
tmrUpdate = simple.create_timer(5, timerHandler)
# Attach handler for graphics rendering
#myFrame.set_draw_handler(drawHandler)
# Start the timer and rendering

initialise_Qsa()
read_oldQ()

tmrUpdate.start()
myFrame.start()        
