import numpy as np
import helper
import random

#   This class has all the functions and variables necessary to implement snake game
#   We will be using Q learning to do this


class SnakeAgent:

    #   This is the constructor for the SnakeAgent class
    #   It initializes the actions that can be made,
    #   Ne which is a parameter helpful to perform exploration before deciding next action,
    #   LPC which ia parameter helpful in calculating learning rate (lr)
    #   gamma which is another parameter helpful in calculating next move, in other words
    #            gamma is used to blalance immediate and future reward
    #   Q is the q-table used in Q-learning
    #   N is the next state used to explore possible moves and decide the best one before updating
    #           the q-table
    def __init__(self, actions, Ne, LPC, gamma):
        self.actions = actions
        self.Ne = Ne # Epsilon
        self.LPC = LPC # Alpha
        self.gamma = gamma
        self.reset()

        # Create the Q and N Table to work with
        self.Q = helper.initialize_q_as_zeros() # Q table (index like self.Q[idx])
        self.N = helper.initialize_q_as_zeros() # N table (index like self.N[idx])

    #   This function sets if the program is in training mode or testing mode.

    def set_train(self):
        self._train = True

     #   This function sets if the program is in training mode or testing mode.
    def set_eval(self):
        self._train = False

    #   Calls the helper function to save the q-table after training
    def save_model(self):
        helper.save(self.Q)

    #   Calls the helper function to load the q-table when testing
    def load_model(self):
        self.Q = helper.load()

    #   resets the game state
    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    #   This is a function you should write.
    #   Function Helper:IT gets the current state, and based on the
    #   current snake head location, body and food location,
    #   determines which move(s) it can make by also using the
    #   board variables to see if its near a wall or if  the
    #   moves it can make lead it into the snake body and so on.
    #   This can return a list of variables that help you keep track of
    #   conditions mentioned above.
    def helper_func(self, state):
        print("IN helper_func")
        # https://edstem.org/us/courses/67912/discussion/5825400
        
        # Enum to represent idx:
        ADJOINING_WALL_X = 0
        ADJOINING_WALL_Y = 1
        FOOD_DIR_X = 2
        FOOD_DIR_Y = 3
        ADJOINING_BODY_TOP = 4
        ADJOINING_BODY_BOTTOM = 5
        ADJOINING_BODY_LEFT = 6
        ADJOINING_BODY_RIGHT = 7
        
        # Enum to represent state:
        SNAKE_HEAD_X = 0
        SNAKE_HEAD_Y = 1
        SNAKE_BODY = 2
        FOOD_X = 3
        FOOD_Y = 4
        
        # Return vector
        idx: list[int] = [0] * 8
        
        # Adjoining wall inference is taken from board.py Snake.move() hit wall clauses
        
        # ADJOINING_WALL_X, 0 if wall to the left, 1 if no wall to the left or right (in the same row), 2 if wall to the right
        if state[SNAKE_HEAD_X] - helper.GRID_SIZE < helper.GRID_SIZE:
            idx[ADJOINING_WALL_X] = 0
        elif state[SNAKE_HEAD_X] + helper.GRID_SIZE > helper.DISPLAY_SIZE - helper.GRID_SIZE:
            idx[ADJOINING_WALL_X] = 2
        else:
            idx[ADJOINING_WALL_X] = 1
            
        # ADJOINING_WALL_Y, 0 if wall above, 1 if no wall above or below (in the same column), 2 if wall below
        if state[SNAKE_HEAD_Y] - helper.GRID_SIZE < helper.GRID_SIZE:
            idx[ADJOINING_WALL_Y] = 0
        elif state[SNAKE_HEAD_Y] + helper.GRID_SIZE > helper.DISPLAY_SIZE - helper.GRID_SIZE:
            idx[ADJOINING_WALL_Y] = 2
        else:
            idx[ADJOINING_WALL_Y] = 1
            
        # FOOD_DIR_X, 0 if food in column to the left, 1 if food in same column, 2 if food in column to the right
        if state[SNAKE_HEAD_X] < state[FOOD_X]:
            # Food is in a column to the right
            idx[FOOD_DIR_X] = 0
        elif state[SNAKE_HEAD_X] > state[FOOD_X]:
            # Food is in a column to the left
            idx[FOOD_DIR_X] = 2
        else:
            # Food is in the same column
            idx[FOOD_DIR_X] = 1
            
        # FOOD_DIR_Y, 0 if food in row above, 1 if food in same row, 2 if food in row below
        # don't forget y is inverted
        if state[SNAKE_HEAD_Y] < state[FOOD_Y]:
            # Food is in a row below
            idx[FOOD_DIR_Y] = 0
        elif state[SNAKE_HEAD_Y] > state[FOOD_Y]:
            # Food is in a row above
            idx[FOOD_DIR_Y] = 2
        else:
            # Food is in the same row
            idx[FOOD_DIR_Y] = 1
            
        # ADJOINING_BODY_TOP, 0 if there is no body above in any row, 1 if there is a body above in any row
        # Iterate each piece of the body (besides the head) and check if it is above the head
        idx[ADJOINING_BODY_TOP] = 0
        for body_piece in state[SNAKE_BODY][1:]:
            if body_piece[1] < state[SNAKE_HEAD_Y]:
                idx[ADJOINING_BODY_TOP] = 1
                break
            
        # ADJOINING_BODY_BOTTOM, 0 if there is no body below in any row, 1 if there is a body below in any row
        idx[ADJOINING_BODY_BOTTOM] = 0
        for body_piece in state[SNAKE_BODY][1:]:
            if body_piece[1] > state[SNAKE_HEAD_Y]:
                idx[ADJOINING_BODY_BOTTOM] = 1
                break
            
        # ADJOINING_BODY_LEFT, 0 if there is no body to the left in any column, 1 if there is a body to the left in any column
        idx[ADJOINING_BODY_LEFT] = 0
        for body_piece in state[SNAKE_BODY][1:]:
            if body_piece[0] < state[SNAKE_HEAD_X]:
                idx[ADJOINING_BODY_LEFT] = 1
                break
            
        # ADJOINING_BODY_RIGHT, 0 if there is no body to the right in any column, 1 if there is a body to the right in any column
        idx[ADJOINING_BODY_RIGHT] = 0
        for body_piece in state[SNAKE_BODY][1:]:
            if body_piece[0] > state[SNAKE_HEAD_X]:
                idx[ADJOINING_BODY_RIGHT] = 1
                break
            
        # Convert to tuple and return
        return tuple(idx)
            
            
    # Computing the reward, need not be changed.

    def compute_reward(self, points, dead):
        if dead:
            return -1
        elif points > self.points:
            return 1
        else:
            return -0.1

    #   This is the code you need to write.
    #   This is the reinforcement learning agent
    #   use the helper_func you need to write above to
    #   decide which move is the best move that the snake needs to make
    #   using the compute reward function defined above.
    #   This function also keeps track of the fact that we are in
    #   training state or testing state so that it can decide if it needs
    #   to update the Q variable. It can use the N variable to test outcomes
    #   of possible moves it can make.
    #   the LPC variable can be used to determine the learning rate (lr), but if
    #   you're stuck on how to do this, just use a learning rate of 0.7 first,
    #   get your code to work then work on this.
    #   gamma is another useful parameter to determine the learning rate.
    #   based on the lr, reward, and gamma values you can update the q-table.
    #   If you're not in training mode, use the q-table loaded (already done)
    #   to make moves based on that.
    #   the only thing this function should return is the best action to take
    #   ie. (0 or 1 or 2 or 3) respectively.
    #   The parameters defined should be enough. If you want to describe more elaborate
    #   states as mentioned in helper_func, use the state variable to contain all that.
    def agent_action(self, state: list[int], points, dead):
        print("IN AGENT_ACTION")
        idx = self.helper_func(state)
        actions = self.Q[idx] # Weighted actions to take
        print(idx)
        
        action = 0
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE
        # YOUR CODE HERE

        # UNCOMMENT THIS TO RETURN THE REQUIRED ACTION.
        return action
