import numpy as np
import Minecraft_Manipulate as Minecraft
from mcpi.minecraft import Minecraft

# mcpi
mc = Minecraft.create()

# Movements
FORWARD = 0
LEFT = 1
RIGHT = 2
BACK = 3
BIGGER_FORWARD = 4
BIGGER_LEFT = 0
BIGGER_RIGHT = 0
BIGGER_BACK = 0


class qtable:
    def __init__(self, old_position, expected_position, percentage):
        self.old_position = old_position
        self.expected_position = expected_position
        self.percentage = percentage
        self.alpha = 0
        self.gamma = 0

    def get_state(self):
        if self.percentage < 35:
            state = 0
        elif self.percentage < 40:
            state = 1
        elif self.percentage < 60:
            state = 2
        elif self.percentage < 85:
            state = 3
        else:
            state = 4

        return state

    def get_reward(self):
        new_position = mc.player.getTilePos()
        a = np.linalg.norm(new_position-self.old_position)
        b = np.linalg.norm(self.expected_position-self.old_position)

        return a/b * 200 - 100
    
    def update(self, qtable, latest_percentage, last_percentage):
        reward = self.get_reward()
        latest_state = self.get_state()
        last_state = self.get_state()

    

    

        



    















# def get_action(state, action):


def main():
    # Set up the dimensions of the Q-table
    num_states = 5  # number of states
    num_actions = 8  # number of actions

    # Initialize the Q-table with zeros
    q_table = np.zeros((num_states, num_actions))

    # # Set the first column of the Q-table with percentages
    # percentages = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    # q_table[:, 0] = percentages

    # Print the Q-table
    print(q_table)

main()