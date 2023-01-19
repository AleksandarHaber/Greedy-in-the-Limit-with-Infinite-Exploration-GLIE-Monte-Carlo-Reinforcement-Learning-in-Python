# -*- coding: utf-8 -*-
"""
Python Implementation of 
the Greedy in the Limit with Infinite Exploration (GLIE) Monte Carlo Control Method

Author: Aleksandar Haber 
Date: December 2023
"""


###############################################################################
# this function learns the optimal policy by using the GLIE Monte Carlo Control Method
###############################################################################
# inputs: 
##################
# env - OpenAI Gym environment 
# stateNumber - number of states
# numberOfEpisodes - number of simulation episodes
# discountRate - discount rate 
# initialEpsilon - initial value of epsilon for the epsilon greedy approach
##################
# outputs:
##################
# finalPolicy - learned policy 
# actionValueMatrixEstimate - estimate of the action value function matrix
##################

def MonteCarloControlGLIE(env,stateNumber,numberOfEpisodes,discountRate,initialEpsilon):
    ###########################################################################
    #                   START - parameter definition
    ###########################################################################
    import numpy as np
    # number of actions in every state 
    actionNumber=4
    # initial epsilon-greedy parameter
    epsilon=initialEpsilon
    # this matrix stores state-action visits
    # that is in state i, action j is selected, then (i,j) entry of this matrix
    # incremented
    numberVisitsForEveryStateAction=np.zeros((stateNumber,actionNumber))
    # estimate of the action value function matrix 
    # initial value
    actionValueMatrixEstimate=np.zeros((stateNumber,actionNumber))
    # final learned policy, it is an array of stateNumber entries
    # every entry of the list is the action selected in that state
    finalPolicy=np.zeros(stateNumber)
    ###########################################################################
    #                   END - parameter definition
    ###########################################################################
    
    ###########################################################################
    #    START - function selecting an action: epsilon-greedy approach
    ###########################################################################
    # this function selects an action on the basis of the current state 
    def selectAction(state,indexEpisode):
        
        # first 5 episodes we select completely random actions to avoid being stuck
        if indexEpisode<5:
            return np.random.choice(actionNumber)   
            
             
        # Returns a random real number in the half-open interval [0.0, 1.0)
        randomNumber=np.random.random()
            
        
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(actionNumber)            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where actionValueMatrixEstimate[state,:] has the max value
            return np.random.choice(np.where(actionValueMatrixEstimate[state,:]==np.max(actionValueMatrixEstimate[state,:]))[0])
            # here we need to return the minimum index since it can happen
            # that there are several identical maximal entries, for example 
            # import numpy as np
            # a=[0,1,1,0]
            # np.where(a==np.max(a))
            # this will return [1,2], but we only need a single index
            # that is why we need to have np.random.choice(np.where(a==np.max(a))[0])
            # note that zero has to be added here since np.where() returns a tuple
    ###########################################################################
    #    END - function selecting an action: epsilon-greedy approach
    ###########################################################################
    
    
    ###########################################################################
    # START - outer loop for simulating episodes
    ###########################################################################
    for indexEpisode in range(numberOfEpisodes):
        
        # this list stores visited states in the current episode
        visitedStatesInEpisode=[]
        # this list stores the return in every visited state in the current episode
        rewardInVisitedState=[]
        # selected actions in an episode 
        actionsInEpisode=[]
        # reset the environment at the beginning of every episode
        (currentState,prob)=env.reset()
        visitedStatesInEpisode.append(currentState)
        print("Simulating episode {}".format(indexEpisode))
         
         
        ###########################################################################
        # START - single episode simulation
        ###########################################################################
        # here we simulate an episode
        while True:
             
            # select an action on the basis of the current state
            selectedAction = selectAction(currentState, indexEpisode)
            # append the selected action
            actionsInEpisode.append(selectedAction)
                         
            # explanation of "env.action_space.sample()"
            # Accepts an action and returns either a tuple (observation, reward, terminated, truncated, info)
            # https://www.gymlibrary.dev/api/core/#gym.Env.step
            # format of returnValue is (observation,reward, terminated, truncated, info)
            # observation (object)  - observed state
            # reward (float)        - reward that is the result of taking the action
            # terminated (bool)     - is it a terminal state
            # truncated (bool)      - it is not important in our case
            # info (dictionary)     - in our case transition probability
            # env.render()
             
            # here we step and return the state, reward, and boolean denoting if the state is a terminal state
            (currentState, currentReward, terminalState,_,_) = env.step(selectedAction)          
            #print(currentState,selectedAction)
            # append the reward
            rewardInVisitedState.append(currentReward)
             
            # if the current state is NOT terminal state 
            if not terminalState:
                visitedStatesInEpisode.append(currentState)   
            # if the current state IS terminal state 
            else: 
                break
            # explanation of IF-ELSE:
            # let us say that a state sequence is 
            # s0, s4, s8, s9, s10, s14, s15
            # the vector visitedStatesInEpisode is then 
            # visitedStatesInEpisode=[0,4,8,10,14]
            # note that s15 is the terminal state and this state is not included in the list
             
            # the return vector is then
            # rewardInVisitedState=[R4,R8,R10,R14,R15]
            # R4 is the first entry, meaning that this is the reward going
            # from state s0 to s4. That is, the rewards correspond to the reward
            # obtained in the destination state
         
        ###########################################################################
        # END - single episode simulation
        ###########################################################################
         
        ###########################################################################
        # START - updates of the action value function and number of visits
        ########################################################################### 
        # how many states we visited in an episode    
        numberOfVisitedStates=len(visitedStatesInEpisode)
             
        # this is Gt=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} + ...
        Gt=0
        # we compute this quantity by using a reverse "range" 
        # below, "range" starts from len-1 until second argument +1, that is until 0
        # with the step that is equal to the third argument, that is, equal to -1
        # here we do everything backwards since it is easier and faster 
        # if we go in the forward direction, we would have to 
        # compute the total return for every state, and this will be less efficient
         
        for indexCurrentState in range(numberOfVisitedStates-1,-1,-1):
            
            stateTmp=visitedStatesInEpisode[indexCurrentState] 
            returnTmp=rewardInVisitedState[indexCurrentState]
            actionTmp=actionsInEpisode[indexCurrentState]
            # this is an elegant way of summing the returns backwards 
            Gt=discountRate*Gt+returnTmp
            # below is the first visit implementation 
            if stateTmp not in visitedStatesInEpisode[0:indexCurrentState]:
                
                numberVisitsForEveryStateAction[stateTmp,actionTmp]+=1
                actionValueMatrixEstimate[stateTmp,actionTmp]=actionValueMatrixEstimate[stateTmp,actionTmp]+(1/numberVisitsForEveryStateAction[stateTmp,actionTmp])*(Gt-actionValueMatrixEstimate[stateTmp,actionTmp])
        ###########################################################################
        # END - updates of the action value function and number of visits
        ########################################################################### 
        
        # first 100 episodes, we keep a fixed value of 
        # epsilon to ensure that we explore enough
        # after that, we decrease the epsilon value
        if indexEpisode < 1000:
            epsilon=initialEpsilon
        else:
            epsilon=0.8*epsilon
        
            
    ###########################################################################
    # END - outer loop for simulating episodes
    ###########################################################################
        
    # now we compute the final learned policy
    for indexS in range(stateNumber):
        # we use np.random.choice() because in theory, we might have several identical maximums
        finalPolicy[indexS]=np.random.choice(np.where(actionValueMatrixEstimate[indexS]==np.max(actionValueMatrixEstimate[indexS]))[0])
        
     
    return finalPolicy,actionValueMatrixEstimate      

