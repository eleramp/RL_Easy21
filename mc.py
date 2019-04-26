#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import gym

import utils
import numpy as np
import dill as pickle

from config import mc_params as cf

## --- Define policy functions --- ##
# exploration probability
def exploration(N0):
    return lambda Nsa: N0/(N0+np.sum(Nsa))

# epsilon-greedy policy
def e_greedy_policy(Q, actions, epsilon):
    if np.random.random() < epsilon:
        #random action: exploration
        a = actions.sample()
    else:
        #greedy action: exploitation
        a = np.argmax([Q[a] for a in range(actions.n)])
    return a
## --- ##

## --- ##

def mc_control(env,):

    actions = env.action_space

    # Init action-value function
    Q = np.zeros(env.dim)
    # Init variable to store number of visit of each state/action
    Nsa = np.zeros(env.dim)

    # instantiate epsilon_t for e-greedy exploration strategy
    epsilon_t = exploration(cf['N0'])

    printEvery = int(10e4)
    meanReturn, wins = 0,0
    for episode in range(cf['n_episodes']): #terminal state for exploration?
        SAR = list() # state, action, reward
        #init state
        s = env.reset()

        G,t,done = 0,0,0
        while not done:
            #Pick an action with e-greedy policy
            a = e_greedy_policy(Q[s], actions, epsilon_t(Nsa[s]))
            #Forward a step
            s_next,reward,done,_ = env.step(a)

            #Update Nsa
            Nsa[s[0],s[1],a] += 1
            #Add s(t),a(t),r(t+1) to the episode history
            SAR.append([s,a,reward])
            # Update return
            G += (cf['r_gamma']**t)*reward

            s = s_next
            t+=1
        #update action-vlue function Q
        for (s,a,r) in SAR:
            Q[s[0],s[1],a] += 1/Nsa[s[0],s[1],a] * (G - Q[s[0],s[1],a])

        ## --- bookkeeping --- ##
        meanReturn = meanReturn + 1/(episode+1) * (G - meanReturn)
        if reward == 1:
            wins += 1
        if episode % printEvery == 0:
            print("Episode %i, Mean-Return %.3f, Wins %.2f"%(episode, meanReturn, wins/(episode+1)))
        ## --- ##
    return Q

def main():
    env = gym.make("gym_easy21:easy21-v0")
    mc_Q = mc_control(env)
    # Save best Q
    pickle.dump(mc_Q, open('Q.dill', 'wb'))
    _ = pickle.load(open('Q.dill', 'rb')) # sanity check

    utils.plot(mc_Q, range(env.action_space.n))


if __name__ == "__main__":
    main()
