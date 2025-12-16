from bandits import Bandit
import random
import math
import numpy as np
import matplotlib.pyplot as plt

bandits = [Bandit(random.random()*4-2) for _ in range(10)]

def run_greedy():

    Q = np.zeros(10)
    N = np.zeros(10)
    rewards = []
    cumavgrewards = []
    T = 1000
    for i in range (T):
        maxmlist=[]
        maxm = np.max(Q)
        for j in range(10):
            if(Q[j]==maxm): maxmlist.append(j)
        action = np.random.choice(maxmlist)
        reward = bandits[action].pullLever()
        rewards.append(reward)
        N[action]+=1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        cumavgrewards.append(sum(rewards)/len(rewards))
    return cumavgrewards

    

greedy=run_greedy()
plt.plot(greedy)
plt.xlabel("Time steps")
plt.ylabel("Cumulative Average of rewards")
plt.title("greedy")
plt.show()

def run_epsilon_greedy(epsilon):
    Q = np.zeros(10)
    N = np.zeros(10)
    rewards = []
    cumavgrewards = []
    T = 1000
    for i in range(T):


        if random.random()<epsilon:
            action=random.randint(0,9)
        else:
            maxmlist=[]
            maxm = np.max(Q)
            for j in range(10):
                if(Q[j]==maxm): maxmlist.append(j)
            action = np.random.choice(maxmlist)
        reward = bandits[action].pullLever()
        rewards.append(reward)
        N[action]+=1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        cumavgrewards.append(sum(rewards)/len(rewards))
    
    return cumavgrewards

    

epsilongreedy = run_epsilon_greedy(0.01)
plt.plot(epsilongreedy)
plt.xlabel("Time steps")
plt.ylabel("Cumulative Average of rewards")
plt.title("epsilon-greedy")
plt.show()

def plot_multiple_epsilon_greedy(epsilonlist):
    for i in range(len(epsilonlist)):
        plt.plot(run_epsilon_greedy(epsilonlist[i]), label="epsilon = "+ str(epsilonlist[i]))
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative Average of rewards")
    plt.title("epsilon-greedy")
    plt.legend()
    plt.show()

    pass

plot_multiple_epsilon_greedy([0,0.01,0.1,1])

def run_optimistic_greedy(Q1):

    Q = np.full(10,Q1)
    N = np.zeros(10)
    rewards = []
    cumavgrewards = []
    T = 1000
    for i in range (T):
        maxmlist=[]
        maxm = np.max(Q)
        for j in range(10):
            if(Q[j]==maxm): maxmlist.append(j)
        action = np.random.choice(maxmlist)
        reward = bandits[action].pullLever()
        rewards.append(reward)
        N[action]+=1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        cumavgrewards.append(sum(rewards)/len(rewards))
    return cumavgrewards

    

plt.plot(run_optimistic_greedy(10),label="optimistic greedy Q1=10")
plt.plot(run_epsilon_greedy(0.1),label="epsilon greedy for 0.1")
plt.legend()
plt.ylabel("cumulative reward average")
plt.xlabel("time steps")
plt.show()


def run_ucb(c):

    Q = np.zeros(10)
    N = np.zeros(10)
    rewards = []
    cumavgrewards = []
    T = 1000
    for i in range (T):
        
        ucb=np.zeros(10)
        for k in range(10):
            if N[k]==0:
                ucb[k]=float('inf')
            else:
                 ucb[k] = Q[k] + c * np.sqrt(np.log(i+1)/N[k])
        action=np.argmax(ucb)
        reward = bandits[action].pullLever()
        rewards.append(reward)
        N[action]+=1
        Q[action] = Q[action] + (reward - Q[action])/N[action]
        cumavgrewards.append(sum(rewards)/len(rewards))
    return cumavgrewards
