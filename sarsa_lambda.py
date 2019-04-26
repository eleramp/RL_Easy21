#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import dill as pickle
import utils
from config import td_params as cf

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
def reset(env_dim):
    # state-action value & counter
    Q = np.zeros(env_dim)
    Nsa = np.zeros(env_dim)
    wins = 0
    return Q, Nsa, wins

def td_control(env,trueQ):

    actions = env.action_space
    lambdas = cf['lambdas']
    mselambdas = np.zeros((len(lambdas), cf['n_episodes']))
    finalMSE = np.zeros(len(lambdas))

    # instantiate epsilon_t for e-greedy exploration strategy
    epsilon_t = exploration(cf['N0'])

    for i_lambda,lambda_decay in enumerate(lambdas):
        Q, Nsa, wins = reset(env.dim)

        for episode in range(cf['n_episodes']): #terminal state for exploration?
            done = 0
            E = np.zeros(env.dim)
            SA = list() # state, action

            #init state
            s = env.reset()
            #Pick an action with e-greedy policy
            a = e_greedy_policy(Q[s], actions, epsilon_t(Nsa[s]))

            while not done:
                #Forward a step
                s_next,reward,done,_ = env.step(a)

                if done: # Sarsa lambda Update
                    td_error = reward - Q[s[0],s[1],a]
                else: #Pick an action with e-greedy policy
                    a_next = e_greedy_policy(Q[s_next], actions, epsilon_t(Nsa[s_next]))
                    td_error = reward + cf['r_gamma']*Q[s_next[0],s_next[1],a_next] - Q[s[0],s[1],a]

                #Add s(t),a(t) to the episode history
                SA.append([s,a])
                E[s[0],s[1],a] += 1
                Nsa[s[0],s[1],a] += 1

                #update action-value function Q
                for (s,a) in SA:
                    Q[s[0],s[1],a] += 1/Nsa[s[0],s[1],a] * td_error * E[s[0],s[1],a]
                    E[s[0],s[1],a] *= cf['r_gamma'] * lambda_decay

                if not done:
                    s = s_next
                    a = a_next

             # bookkeeping
            if reward == 1:
                wins += 1
            mse = np.sum(np.square(Q-trueQ)) / (env.dim[0]*env.dim[1]*env.dim[2])
            mselambdas[i_lambda, episode] = mse

        finalMSE[int(i_lambda)] = mse
        print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lambda_decay, episode, mse, wins/(episode+1)))
        print("--------")

    utils.plotMseLambdas(finalMSE, lambdas)
    utils.plotMseEpisodesLambdas(mselambdas)

def main():
    env = gym.make("gym_easy21:easy21-v0")
    trueQ = pickle.load(open('Q.dill', 'rb'))
    td_control(env, trueQ)

if __name__ == "__main__":
    main()
