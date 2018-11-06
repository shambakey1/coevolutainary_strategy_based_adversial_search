
from sample_players import DataPlayer

import random, numpy

from workspace import tools
from workspace import base, creator, algorithms
from math import floor, pow
from isolation.isolation import Action


atr_min=0   # Minimum bound for attribute value
atr_max=8   # Maximum bound for attribute value
ind_size=3  # Individual size (i.e., depth) corresponding to current and opponent moves. Thus, the first gene in individual is the next step for current player, the next gene is the second gene for the opponent, the third gene is the second next step for current player, and so on  
mut_pb=0.05 # Mutation probability
gen_size=3  # Number of generations (i.e., iterations)
pop_size=floor(0.5*pow(atr_max,ind_size)) # Population size per each generation
sel_size=10 # Number of individuals to be selected
CXPB, MUTPB = 0.5, 0.2  # Crossover and mutation probabilities
attrAction={0:Action.NNE,1:Action.ENE,2:Action.ESE,3:Action.SSE,4:Action.SSW,5:Action.WSW, \
                6:Action.WNW,7:Action.NNW}

def score(state,player_id):
    own_loc = state.locs[player_id]
    opp_loc = state.locs[1 - player_id]
    own_liberties = state.liberties(own_loc)
    opp_liberties = state.liberties(opp_loc)
    return len(own_liberties) - len(opp_liberties)

# the goal ('fitness') function to be maximized
def evaluate(individual,state,player_id,attrAction,idx=0):
    '''
    The return value of current individual (i.e., the set of suggested movements for active player and opponent)
    '''
    if attrAction[individual[idx]] not in state.actions():
        value=float("-inf")    # Treat illegal move as lose
    elif state.terminal_test():
        value=state.utility(player_id)
    elif idx>=len(individual)-1:
        value=score(state,player_id)
    else:
        value = evaluate(individual, state.result(attrAction[individual[idx]]), player_id, attrAction, idx+1)
    
    # Return tuble if idx==0 (as required by deap library). Otherwise, return single fitness value
    if idx==0:
        return value,
    else:
        return value


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action1(self, state):
        self.queue.put(random.choice(state.actions()))
        print(self.queue)
        
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #import random
        #self.queue.put(random.choice(state.actions()))
        
        # randomly select a move as player 1 or 2 on an empty board, otherwise
        # return the sub-optimal move, found by optimization algorithms like GA, at a fixed search 
        # depth of individual size
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            toolbox = base.Toolbox()
            # Attribute generator 
            toolbox.register("attr_action", random.randrange, atr_max)
            # Structure initializers
            toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attr_action, ind_size)
            # define the population to be a list of individuals
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Define statistics and hall of fame objects
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", numpy.mean)
            stats.register("std", numpy.std)
            stats.register("min", numpy.min)
            stats.register("max", numpy.max)
            
            #----------
            # Operator registration
            #----------
            # register the goal / fitness function
            toolbox.register("evaluate", evaluate)
            
            # register the crossover operator
            toolbox.register("mate", tools.cxTwoPoint)
            
            # register a mutation operator with a specified probability
            toolbox.register("mutate", tools.mutUniformInt, low=atr_min, up=atr_max-1, indpb=mut_pb)
            
            # Select the specified number of best individuals
            toolbox.register("select", tools.selTournament, tournsize=3)
            # Select the final best individual
            toolbox.register("select_fin",tools.selBest, k=1)
            # create an initial population of individuals (where each individual is a list of integers. Each integer corresponds to an action)
            random.seed(64)
            
            pop = toolbox.population(n=pop_size)  # Initial random actions of specified depth(=individual size)
            #for i in pop: print(i)

            pop,log=algorithms.eaSimple_mod(pop, toolbox, CXPB, MUTPB, gen_size,\
                                        state, stats=stats,player_id=self.player_id, \
                                        attrAction=attrAction,halloffame=hof, verbose=True)
            '''
            with open("log.out","w") as f:
                for i in pop:
                    f.write('Individual: '+str(i)+', fitness: '+str(i.fitness.values[0])+'\n')
                f.write("hof: "+str(hof[0])+", fitness: "+str(hof[0].fitness.values[0])+'\n')
            '''
            if len(hof)!=0 and hof[0].fitness.values[0]!=float("-inf"):
                self.queue.put(attrAction.get(hof[0][0])) # Add the sub-optimal individual (i.e., expected best next move) to the queu
            else:
                if state.actions:
                    self.queue.put(random.choice(state.actions()))
                else:
                    print("No more solutions left")
            
