#!/usr/bin/env python
import sys
import traceback
import random
import time
import math
from collections import defaultdict
from math import sqrt


MY_ANT = 0
ANTS = 0
DEAD = -1
LAND = -2
FOOD = -3
WATER = -4

PLAYER_ANT = 'abcdefghij'
HILL_ANT = string = 'ABCDEFGHIJ'
PLAYER_HILL = string = '0123456789'
MAP_OBJECT = '?%*.!'
MAP_RENDER = PLAYER_ANT + HILL_ANT + PLAYER_HILL + MAP_OBJECT

AIM = {'n': (-1, 0),
       'e': (0, 1),
       's': (1, 0),
       'w': (0, -1)}
RIGHT = {'n': 'e',
         'e': 's',
         's': 'w',
         'w': 'n'}
LEFT = {'n': 'w',
        'e': 'n',
        's': 'e',
        'w': 's'}
BEHIND = {'n': 's',
          's': 'n',
          'e': 'w',
          'w': 'e'}

class Ants():
    def __init__(self):
        self.cols = None
        self.rows = None
        self.map = None
        self.hill_list = {}
        self.ant_list = {}
        self.dead_list = defaultdict(list)
        self.food_list = []
        self.turntime = 0
        self.loadtime = 0
        self.turn_start_time = None
        self.vision = None
        self.viewradius2 = 0
        self.attackradius2 = 0
        self.spawnradius2 = 0
        self.turns = 0
        



    def setup(self, data):
        'parse initial input and setup starting game state'
        for line in data.split('\n'):
            line = line.strip().lower()
            if len(line) > 0:
                tokens = line.split()
                key = tokens[0]
                if key == 'cols':
                    self.cols = int(tokens[1])
                elif key == 'rows':
                    self.rows = int(tokens[1])
                elif key == 'player_seed':
                    random.seed(int(tokens[1]))
                elif key == 'turntime':
                    self.turntime = int(tokens[1])
                elif key == 'loadtime':
                    self.loadtime = int(tokens[1])
                elif key == 'viewradius2':
                    self.viewradius2 = int(tokens[1])
                elif key == 'attackradius2':
                    self.attackradius2 = int(tokens[1])
                elif key == 'spawnradius2':
                    self.spawnradius2 = int(tokens[1])
                elif key == 'turns':
                    self.turns = int(tokens[1])
        self.map = [[LAND for col in range(self.cols)]
                    for row in range(self.rows)]

    def update(self, data):
        'parse engine input and update the game state'
        # start timer
        self.turn_start_time = time.time()
        
        # reset vision
        self.vision = None
        
        # clear hill, ant and food data
        self.hill_list = {}
        for row, col in self.ant_list.keys():
            self.map[row][col] = LAND
        self.ant_list = {}
        for row, col in self.dead_list.keys():
            self.map[row][col] = LAND
        self.dead_list = defaultdict(list)
        for row, col in self.food_list:
            self.map[row][col] = LAND
        self.food_list = []
        
        # update map and create new ant and food lists
        for line in data.split('\n'):
            line = line.strip().lower()
            if len(line) > 0:
                tokens = line.split()
                if len(tokens) >= 3:
                    row = int(tokens[1])
                    col = int(tokens[2])
                    if tokens[0] == 'w':
                        self.map[row][col] = WATER
                    elif tokens[0] == 'f':
                        self.map[row][col] = FOOD
                        self.food_list.append((row, col))
                    else:
                        owner = int(tokens[3])
                        if tokens[0] == 'a':
                            self.map[row][col] = owner
                            self.ant_list[(row, col)] = owner
                        elif tokens[0] == 'd':
                            # food could spawn on a spot where an ant just died
                            # don't overwrite the space unless it is land
                            if self.map[row][col] == LAND:
                                self.map[row][col] = DEAD
                            # but always add to the dead list
                            self.dead_list[(row, col)].append(owner)
                        elif tokens[0] == 'h':
                            owner = int(tokens[3])
                            self.hill_list[(row, col)] = owner
                            
                            
#    def A_Star(self, start, goal):
#        closedset = []  # The set of nodes already evaluated.
#        openset = []
#        openset.append(start)# The set of tentative nodes to be evaluated, initially containing the start node
#        came_from = {} # The map of navigated nodes.
#        tentative_g_score = 0
#        g_score = {}
#        h_score = {}
#        f_score = {}
#        g_score[start] = 0    # Cost from start along best known path.
#        h_score[start] = self.heuristic_cost_estimate(start, goal)
#        f_score[start] = g_score[start] + h_score[start]
#        udie = start
#        while (len(openset) > 0):
#            lowest = 10000
#            x = udie
#            i = self.find(x,openset)
#            if x == goal:
#                
#                return self.reconstruct_path(came_from,came_from[goal])
#            
#            del openset[i]
#            closedset.append(x)
#            for y in self.neigbor_nodes(x):
#                if y in closedset:
#                    continue
#                if y not in openset:
#                    openset.append(y)
#                    tentative_is_better = True
#                elif tentative_g_score < g_score[y]:
#                    tentative_is_better = True
#                else:
#                    tentative_is_better = False
# 
#                if tentative_is_better == True:
#                    came_from[y] = x
#                    g_score[y] = tentative_g_score
#                    h_score[y] = self.heuristic_cost_estimate(y, goal)
#                    f_score[y] = g_score[y] + h_score[y]
#                if lowest > g_score[y] + h_score[y]:
#                    udie = y
#        return False
    
    def find(self,f, seq):
        i = 0
        for item in seq:
            if f[0] == item[0] and f[1] == item[1]:
                return i
        return False  
      
    def neigbor_nodes(self,x):
        nodes= []
        for direction in ('s','e','w','n'):
            if self.passable(self.destination(x, direction)):
                if self.no_dead_end_neigbor(self.destination(x, direction),x):
                    nodes.append(self.destination(x, direction))
        return nodes
    
    def no_dead_end_neigbor(self, x, back):
        i=0
        for direction in ('s','e','w','n'):
            if self.passable(self.destination(x, direction)) and self.destination != back:
                i+=1
        if x in self.food_list:
            i+4
        if i >1:
            return True
        else:
            return False
    
    def reconstruct_path(self,came_from,current_node):
            if current_node in came_from:
                return self.reconstruct_path(came_from,came_from[current_node]) + [current_node]
            else:
                return [current_node]
    
    
        
        
    def heuristic_cost_estimate(self, start, goal):
        minusI = start[0]-goal[0]
        minusIi = math.pow(minusI,2.0)

        minusJ = (start[1]-goal[1])
        minusIj = math.pow(minusJ,2.0)

        return minusIj + minusIi;

#    def neigbor_nodes(self,x):
#        nodes= []
#        for direction in ('s','e','w','n'):
#            if self.passable(self.destination(x, direction)):
#                if self.no_dead_end_neigbor(self.destination(x, direction),x):
#                    nodes.append(self.destination(x, direction))
#        return nodes
#    
#    def no_dead_end_neigbor(self, x, back):
#        i=0
#        for direction in ('s','e','w','n'):
#            if self.passable(self.destination(x, direction)) and self.destination != back:
#                i+=1
#        if i >1:
#            return True
#        else:
#            return False

    def A_Star(self,start,goal):
        DebugOutput=0
        closedset={} #the empty set - set of nodes already evaluated.
        openset={}
        openset[start]=None #set containing the initial node    // The set of tentative nodes to be evaluated.
        came_from={} #the empty map // The map of navigated nodes.
    
        g_score={}
        g_score[start]=0 #// Cost from start along best known path.
        h_score={}
        h_score[start]=self.heuristic_cost_estimate(start,goal)
        f_score={}
        f_score[start] = h_score[start] #// Estimated total cost from start to goal through y.
     
        while len(openset.keys())>0:#while openset is not empty
            x=openset.keys()[0]
            for xi in openset.keys():
                if f_score[xi]<f_score[x]:
                    x=xi
            del openset[x]
            #(x,value)=openset.popitem()#the node in openset having the lowest f_score[] value
            
            if x == goal:
                return self.reconstruct_path(came_from,came_from[goal])
            
            #remove x from openset//already done with popitem()
            closedset[x]=None #add x to closedset
        
            for y in self.neigbor_nodes(x): #foreach y in neighbor_nodes(x)
                if DebugOutput==1:
                    print y
                if y in closedset.keys():#if y in closedset
                    continue
                tentative_g_score=g_score[x]+self.dist_between(x,y)#tentative_g_score := g_score[x] + dist_between(x,y)
    
                if y not in openset.keys():#if y not in openset
                    openset[y]=None#add y to openset
                    tentative_is_better = True
                elif tentative_g_score < g_score[y]:
                    tentative_is_better = True
                else:
                    tentative_is_better = False
                if tentative_is_better == True:
                    came_from[y] = x
                    g_score[y] = tentative_g_score
                    h_score[y] = self.heuristic_cost_estimate(y, goal)
                    f_score[y] = g_score[y] + h_score[y]
        

        return False
 
#    def reconstruct_path(self, came_from, current_node):
#        next_node=current_node
#        path=[]
#        
#        while 1:
#            if next_node in came_from.keys():
#                path.append(next_node)
#                next_node=came_from[next_node]
#            else:
#                path.append(next_node)
#                break
#        return path
        
    #    if current_node in came_from.keys():
    #        p = reconstruct_path(came_from, came_from[current_node])
    #        asNode=[]
    #        asNode.append(current_node)
    #        return (p + asNode)
    #    else:
    #        asNode=[]
    #        asNode.append(current_node)
    #        return asNode

    def dist_between(self,nodeA,nodeB):
        return self.heuristic_cost_estimate(nodeA,nodeB)
#    
#    def heuristic_cost_estimate(self, nodeA,nodeB):
#        n=100
#        rowA=(nodeA[1]/n)+nodeA[1]%n
#        colA=nodeA[0]%n+n
#   
#
#        rowB=(nodeB[0]/n)+nodeB[0]%n
#        colB=nodeB[1]%n+n
#        return abs(rowB-rowA)+abs(colB-colA)
#  
    #    print "rowA=",rowA
    #    print "colA=",colA
    #    
    #    print "rowB=",rowB
    #    print "colB=",colB
    #    
    #    print "rowB-rowA=",rowB-rowA
    #    print "colB-colA=",colB-colA
        
#        return abs(rowB-rowA)+abs(colB-colA)

    def time_remaining(self):
        return self.turntime - int(1000 * (time.time() - self.turn_start_time))
    
    def issue_order(self, order):
        'issue an order by writing the proper ant location and direction'
        (row, col), direction = order
        sys.stdout.write('o %s %s %s\n' % (row, col, direction))
        sys.stdout.flush()
        
    def finish_turn(self):
        'finish the turn by writing the go line'
        sys.stdout.write('go\n')
        sys.stdout.flush()
    
    def my_hills(self):
        return [loc for loc, owner in self.hill_list.items()
                    if owner == MY_ANT]

    def enemy_hills(self):
        return [(loc, owner) for loc, owner in self.hill_list.items()
                    if owner != MY_ANT]
        
    def my_ants(self):
        'return a list of all my ants'
        return [(row, col) for (row, col), owner in self.ant_list.items()
                    if owner == MY_ANT]

    def enemy_ants(self):
        'return a list of all visible enemy ants'
        return [((row, col), owner)
                    for (row, col), owner in self.ant_list.items()
                    if owner != MY_ANT]

    def food(self):
        'return a list of all food locations'
        return self.food_list[:]

    def passable(self, loc):
        'true if not water'
        row, col = loc
        return self.map[row][col] != WATER
    
    def get_current_ants_values(self):
        return self.current_ants.values()
    
    def unoccupied(self, loc):
        'true if no ants are at the location'
        row, col = loc
        return self.map[row][col] in (LAND, DEAD)

    def destination(self, loc, direction):
        'calculate a new location given the direction and wrap correctly'
        row, col = loc
        d_row, d_col = AIM[direction]
        return ((row + d_row) % self.rows, (col + d_col) % self.cols)        

    def distance(self, loc1, loc2):
        'calculate the closest distance between to locations'
        row1, col1 = loc1
        row2, col2 = loc2
        d_col = min(abs(col1 - col2), self.cols - abs(col1 - col2))
        d_row = min(abs(row1 - row2), self.rows - abs(row1 - row2))
        return d_row + d_col

    def direction(self, loc1, loc2):
        'determine the 1 or 2 fastest (closest) directions to reach a location'
        row1, col1 = loc1
        row2, col2 = loc2
        height2 = self.rows # 2
        width2 = self.cols # 2
        d = []
        if row1 < row2:
            if row2 - row1 >= height2:
                d.append('n')
            if row2 - row1 <= height2:
                d.append('s')
        if row2 < row1:
            if row1 - row2 >= height2:
                d.append('s')
            if row1 - row2 <= height2:
                d.append('n')
        if col1 < col2:
            if col2 - col1 >= width2:
                d.append('w')
            if col2 - col1 <= width2:
                d.append('e')
        if col2 < col1:
            if col1 - col2 >= width2:
                d.append('e')
            if col1 - col2 <= width2:
                d.append('w')
        return d


    def visible(self, loc):
        ' determine which squares are visible to the given player '

        if self.vision == None:
            if not hasattr(self, 'vision_offsets_2'):
                # precalculate squares around an ant to set as visible
                self.vision_offsets_2 = []
                mx = int(sqrt(self.viewradius2))
                for d_row in range(-mx, mx + 1):
                    for d_col in range(-mx, mx + 1):
                        d = d_row ** 2 + d_col ** 2
                        if d <= self.viewradius2:
                            self.vision_offsets_2.append((
                                # Create all negative offsets so vision will
                                # wrap around the edges properly
                                (d_row % self.rows) - self.rows,
                                (d_col % self.cols) - self.cols
                            ))
            # set all spaces as not visible
            # loop through ants and set all squares around ant as visible
            self.vision = [[False] * self.cols for row in range(self.rows)]
            for ant in self.my_ants():
                a_row, a_col = ant
                for v_row, v_col in self.vision_offsets_2:
                    self.vision[a_row + v_row][a_col + v_col] = True
        row, col = loc
        return self.vision[row][col]
    
    def render_text_map(self):
        'return a pretty string representing the map'
        tmp = ''
        for row in self.map:
            tmp += '# %s\n' % ''.join([MAP_RENDER[col] for col in row])
        return tmp

    # static methods are not tied to a class and don't have self passed in
    # this is a python decorator
    @staticmethod
    def run(bot):
        'parse input, update game state and call the bot classes do_turn method'
        ants = Ants()
        map_data = ''
        while(True):
            try:
                current_line = sys.stdin.readline().rstrip('\r\n') # string new line char
                if current_line.lower() == 'ready':
                    ants.setup(map_data)
                    bot.do_setup(ants)
                    ants.finish_turn()
                    map_data = ''
                elif current_line.lower() == 'go':
                    ants.update(map_data)
                    # call the do_turn method of the class passed in
                    bot.do_turn(ants)
                    ants.finish_turn()
                    map_data = ''
                else:
                    map_data += current_line + '\n'
            except EOFError:
                break
            except KeyboardInterrupt:
                raise
            except:
                # don't raise error or return so that bot attempts to stay alive
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
