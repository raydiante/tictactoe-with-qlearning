#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 7 11:16:17 2019

@author: rayssarosa
"""
import numpy as np
import sys
import pickle
from copy import deepcopy
     
ROLES = "For insert your move on the board enter the row number followed by a space followed by the column number. Example: `3 3` "

#Constants of the game
X=1
O=2
DIMENSION = 3 

isTheEnd = False
board = []
states = []
moves=0 
winner=0 
# 0-> No winner
# 1-> X won
# 2-> O won
boardHash = None
reward = 0

#-------------------------------Tic tac toe Base
#Function to finish th game
def endGame():
    global winner
    global DIMENSION
    global reward
    if winner==0 and moves==DIMENSION:
        print("It`s a tie")
        reward=0
        sys.exit()
    elif winner==2:
        print("Player O won!")
        reward=-1
        sys.exit()
    elif winner==1:
        print("Player X won!")
        reward=1
        sys.exit()
    

#Print the board of the game
def printboard():
    global board
    global DIMENSION
    for i in range(DIMENSION):
        rowi = ""
        for j in range(DIMENSION):
            rowi += " " +board[i][j]+" "
            if j<DIMENSION-1:
                rowi+= "|" 
        print(rowi)
        if i<DIMENSION-1:
            print("---+---+---")
        

#Create the board game
def createboard():
    global board
    global DIMENSION
    for i in range(DIMENSION):
        row = []
        for j in range(DIMENSION):
            row.append(' ')
        board.append(row)
        
#Check if there is a winner
def checkboard():
    global DIMENSION
    global winner
    global board
    global isTheEnd
    global moves
    if moves ==DIMENSION*DIMENSION:
        isTheEnd=True
        return
        
    #Check rows
    for i in range(DIMENSION):
        plx=0
        g=True
        #For x
        for j in range(DIMENSION):
            if board[i][j]!= "X":
                g=False
                break
        if g:
            winner=X
            isTheEnd=True
        g=True
        
        #For O
        for j in range(DIMENSION):
            if board[i][j]!= "O":
                g=False
                break
        if g:
            winner=O
            isTheEnd=True
        g=True
    
    #Check columns
    for j in range(DIMENSION):
        g=True
        #For x
        for i in range(DIMENSION):
            if board[i][j]!= "X":
                g=False
                break
        if g:
            winner=X
            isTheEnd=True
        g=True
        
        #For O
        for i in range(DIMENSION):
            if board[i][j]!= "O":
                g=False
                break
        if g:
            winner=O
            isTheEnd=True
        g=True
            
    #Check diagonal 1 For x
    g=True
    for i in range(DIMENSION):
        if board[i][i]!= "X":
            g=False
            break
    if g:
        winner=X
        isTheEnd=True
        
    #Check diagonal 1 For O
    g=True
    for i in range(DIMENSION):
        if board[i][i]!= "O":
                g=False
                break
    if g:
        winner=O
        isTheEnd=True
        
    #Check diagonal 2 For x
    g=True
    for i in range(DIMENSION):
        if board[i][DIMENSION-1-i]!= "X":
            g=False
            break
    if g:
        winner=X
        isTheEnd=True
            
    #Check diagonal 2 For x
    g=True
    for i in range(DIMENSION):
        if board[i][DIMENSION-1-i]!= "O":
            g=False
            break
    if g:
        winner=O
        isTheEnd=True
        
        
#-------------------------------Reinforcement Learning   

#Creating the agent
class Agent:
    def __init__(self, name,symbol, exp_rate=0.3):
        self.name = name
        self.symbol = symbol
        self.states = []  # store all the states reached by the actions
        self.lr = 0.2#alfa
        self.exp_rate = exp_rate #E-greedy
        self.decay_gamma = 0.9 #gama
        self.state_value = {}  # Map the estado in a value



#return array  with free positions for the agent to go (S)
def availablePositions(DIMENSION,board):
    positions = []
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            if board[i][j] == " ":
                positions.append((i, j))
    return positions
      
#agent do the action (put x in the position [i][j] of the board)
def updateState(board,i,j,symbol):
    global moves
    board[i][j] =symbol
    moves+= 1
    
    
#Resize the board
def getHash(board,DIMENSION):
    global boardHash
    boardHash = str(np.reshape(board,DIMENSION*DIMENSION))
    return boardHash

#agent choose the action
def chooseAction(positions,board,agent):
    # if the rand number is less than exp_rate -> random position

    idx = np.random.choice(len(positions))
    action = positions[idx]
    if np.random.uniform(0, 1) <= agent.exp_rate:
        return action     
    #choose the best action
    else:
        value_max = -999
        for p in positions:
            next_board = deepcopy(board)
            next_board[p[0]][p[1]] = agent.symbol 
            next_boardHash = getHash(next_board,DIMENSION) #create a hash value for the next board
            #if the dictionary has this hash number -> value=hash
            #else -> value = 0
            if (next_boardHash not in agent.state_value):
                    value = 0 
            else:
                value = agent.state_value[next_boardHash]
            if value >= value_max: 
                value_max = value
                action = p
            
        return action            
    
    
    

def feedReward(reward,agent,next_action_value):
    # add in the qtable (agent.state_value)
    global states
    for st in (agent.states):
        if agent.state_value.get(st) is None:
            agent.state_value[st] = 0
            
        if(next_action_value == "none"):
            #if the game is finished-> there is no future q value
            agent.state_value[st] += agent.lr * (reward - agent.state_value[st])
        else:
            #if the game isn`t finished
            agent.state_value[st] += agent.lr * (reward + (agent.decay_gamma * next_action_value) - agent.state_value[st])

        
            
def addState(state,agent):
    #add state to the array of states
    if(state not in agent.states):
        agent.states.append(state) 
            
            
            
def play(agent1,agent2, rounds=50000):
    #game between the agents
    global winner
    global DIMENSION
    global moves
    global board
    global state
    global isTheEnd
    for i in range(rounds):
        resetBoard()
        if i % 500 == 0:
            print("Rounds {}".format(i+1))
        while (moves < DIMENSION*DIMENSION):
            # Player 1
            positions = availablePositions(DIMENSION,board)
            p1_action = chooseAction(positions, board, agent1)
            # take action and upate board state
            updateState(board,p1_action[0],p1_action[1],agent1.symbol)
            board_hash = getHash(board,DIMENSION)
            addState(board_hash,agent1)
            #printboard()
            # check board status if it is end
            checkboard()
            if (isTheEnd==True):
                giveReward(agent1,agent2)
                resetBoard()
                resetAgent(agent1)
                resetAgent(agent2)
                break
            else:
                #update the qtable
                # reward=0 bc the game continues
                st=board_hash
                if agent1.state_value.get(st) is None:
                    agent1.state_value[st] = 0
                feedReward(0,agent1,agent1.state_value.get(st))
                # Player 2
                positions = availablePositions(DIMENSION,board)
                p2_action = chooseAction(positions, board, agent2)
                # take action and upate board state
                updateState(board,p2_action[0],p2_action[1],agent2.symbol)
                board_hash = getHash(board,DIMENSION)
                addState(board_hash,agent2)
                
                # check board status if it is end
                checkboard()
                if (isTheEnd==True):
                    giveReward(agent1,agent2)
                    resetBoard()
                    resetAgent(agent1)
                    resetAgent(agent2)
                    break
                #else:
                 #   if(agent2.state_value.get(board_hash) is None):
                  #      agent2.state_value[board_hash] = 0
                   # feedReward(0,agent2,agent2.state_value.get(board_hash))
         
def humanPlay(agent1):
    # play with human
    global winner
    global DIMENSION
    global moves
    global states
    global board
    global isTheEnd
    
    
    while (moves < DIMENSION*DIMENSION):
        # Agent action
        positions = availablePositions(DIMENSION,board)
        p1_action = chooseAction(positions, board, agent1)
        updateState(board,p1_action[0],p1_action[1],agent1.symbol)
        board_hash = getHash(board,DIMENSION)
        addState(board_hash,agent1)
        printboard()
        checkboard()
    
        if (isTheEnd==True):
            #reward if is the end of the game
            humanGiveReward(agent1)
            return
        else:
            #reward if the game didn`t end yet
            if agent1.state_value.get(board_hash) is None:
                agent1.state_value[board_hash] = 0
            feedReward(0,agent1,agent1.state_value.get(board_hash))
            
            #human action
            positions = availablePositions(DIMENSION,board)
            p2_action = humanChooseAction( positions)
            updateState(board,p2_action[0],p2_action[1],"O")
            printboard()
            checkboard()
            if (isTheEnd==True):
                humanGiveReward(agent1)
                return
    
def humanChooseAction( positions):
    #let the player choose the action
    while True:
        row = int(input("Input your action row:"))
        col = int(input("Input your action col:"))
        action = (row, col)
        if action in positions:
            return action
            
def giveReward(agent1,agent2):
    #give the reward for the agents
    global winner
    result = winner
    if result == 1:
        feedReward(1,agent1,"none")
        feedReward(-1,agent2,"none")
    elif result == 2:
        feedReward(-1,agent1,"none")
        feedReward(1,agent2,"none")
    else:
        feedReward(0.5,agent1,"none")
        feedReward(0.5,agent2,"none")
    
        
def humanGiveReward(agent1):
    #give the reward for the agent that played with the human
    global winner
    result = winner
    if result == 1:
        feedReward(1,agent1,"none")
        print(agent1.name+" won! \n ")
    elif result == 2:
        feedReward(-1,agent1,"none")
        print("Human won! \n ")
    else:
        feedReward(0.5,agent1,"none")
        print("tie! \n ")
        
def savePolicy(agent):
    global states
    i=0
    print("Saving...")
    #print(states)
    dic = open("dictionary_value.txt", "w")
    dicKey = open("dictionary_key.txt", "w")
    for key in agent.state_value:
        i+=1
        strin=str(agent.state_value[key]) + "\n"
        dic.write(strin)
        strin=key + "\n"
        dicKey.write(strin)
    dic.close()
    dicKey.close()
    print("Saved ",i ,"senteces " )

    
def loadPolicy(agent):
    global states
    print("loading...")
    dic = open("dictionary_value.txt", "r")
    dicKey = open("dictionary_key.txt", "r")
    key=dicKey.readline()
    value=dic.readline()
    i=0
    while(key and value):
        if key.count('\n')==1:
            key=key.replace('\n', '')
        if value.count('\n')==1:
            value=float(value.replace('\n', ''))
        i+=1
        agent.state_value[key] = value
        states.append(key)
        key=dicKey.readline()
        value=dic.readline()
    dic.close()
    dicKey.close()
    print("loaded ",i ,"senteces " )

    
def resetBoard():
    global board
    global moves
    global winner
    global isTheEnd
    board = []
    moves=0 
    winner=0 
    isTheEnd=False
    createboard()
    
def resetAgent(agent):
    agent.states = []

def main():
    #trainning the agent         
    agent1 = Agent("agent1","X",exp_rate=0.3) 
    loadPolicy(agent1)
    agent2 = Agent("agent2","O",exp_rate=0.4)        
    createboard()
    play(agent1,agent2)
    savePolicy(agent1)
    
    #playing withe the human
    agent1 = Agent("agent1","X",exp_rate=0.1) 
    loadPolicy(agent1)
    menu="s"
    while(menu == "s"):
        resetBoard()
        humanPlay(agent1)
        menu = input("Press: \ns to start another game \nq to quit\n")
        #print(agent1.state_value)
        print(" ")
    savePolicy(agent1)
    
    
main()
