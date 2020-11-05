print('Importing functions...', end='')

import time
import math
import random
import numpy as np

# plotting, interactives & animations
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
matplotlib.rcParams['animation.embed_limit'] = 120
rc('animation', html='jshtml')
import seaborn as sns
sns.set()

 # for gameplay in Jupyter
from IPython.display import HTML
from IPython.display import Video
from IPython.core import display

# supress warnings for public release
global_testing = False
if global_testing:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=AttributeError)
    

# function to display looping gif of Q-Learning diagram
def ql_flow(): return Video("ql_flow.mov", width=900, html_attributes='autoplay loop')



# environment globals
size=3
min_contig=3

# training globals
lrate=.1
discount=.9

# for setting gobals in a session. warning, resets (clears) the existing q-table
def game_settings(size=3, min_contig=3):
    if min_contig > size: return "ERROR: Minimum winning length cannot be greater than board size." 
    print('Settings updated:')
    globals()['size'] = size
    print(f'> New board size: {size} x {size} tiles')
    globals()['min_contig'] = min_contig
    print(f'> Mininum length to win: {min_contig}')
    globals()['q_table'] = {}
    print(f'Q Table reset.')

# —————— BASIC ENVIRONMENT AND ENV. UTILITY FUNCTIONS ——————

def new_board(board_size=size): # returns a blank board of size*size.
    return [' ']*size**2
 
def check_for_win(line, min_contig=min_contig): # check for a win along a single line of the board
                                                # (the line will passed from the evaluate() function,r below)
    for p in 'XO':
        for i in list(range(len(line)-min_contig+1)):
            if line[i:i+min_contig].count(p) == min_contig: return p
    return False

   
# SHOW FUNCTION (for human appreciation)
def show(b, helpers=False):
    board = list(b).copy() # accepts board as list or string
    size = int(math.sqrt(len(board)))  
    
    # placeholder values (starting at 1)  
    if helpers:   
        board = [str(e+1) if i==' ' else i for e,i in enumerate(board)]
    
    # add spacers for single-digit tiles if board has double-digits
    board = [b if len(b)>1 else f' {b}' for b in board]
        
    # recolor player tiles (x=red & o=blue) for visibility:
    for e,b in enumerate(board):
        if 'X' in b: board[e] = f"\x1b[31m{b}\x1b[0m" # 31=red
        elif 'O' in b: board[e] = f"\x1b[34m{b}\x1b[0m" # 34=blue
        else: board[e] = f"\x1b[37m{b}\x1b[0m" # 37=gray
            
    # print grid and values
    for row in range(0, len(board), size): # start of each row
        print('—'*(4*size+1))
        for col in range(size):  # add column to row start
            print(f'|{board[(row+col)]} |', end='\b') # backspace extra | (horizontal line) char.
        print('|')
        print('—'*(4*size+1), end='\r')
    print()

    
# for giving the agent a singe dual-perspective Q-Table
def flip_board(b_key): 
    return b_key.replace('X','o').replace('O','X').upper()

 

# BOARD STATE EVALUATOR FUNCTION
    
# check lines for a win - function ends if a win is found, returning the state as a string...
def evaluate(b, min_contig=min_contig, n_players=2):
                                        # uses the check_for_win() function from above to check
                                        # across all possible lines across the board. 
    size = int(math.sqrt(len(b))) # define length of a row (n columns)
    
     # check if a win is even possible yet, if not, trigger another move
    n_occupied_positions = len(b) - b.count(' ')
    moves_so_far = round(n_occupied_positions/n_players) # round up to get max moves (by X)
    if moves_so_far < (min_contig-1) :
        return 'Continue'
    
    # start checking lines for a win. return if found.
    
    # verticals
    for col in range(size):
        v_line = [b[row+col] for row in range(0, len(b), size)]
        winner = check_for_win(v_line, min_contig)
        if winner: return winner+' Wins!'

    # horizontals
    for row in range(0, len(b), size):
        h_line = [b[row+col] for col in range(size)]
        winner = check_for_win(h_line, min_contig)
        if winner: return winner+' Wins!'

    # down-right diagonal
    dr_line = [b[int(row+row/size)] for row in range(0, len(b), size)]
    winner = check_for_win(dr_line, min_contig)
    if winner: return winner+' Wins!'
    
    # up-right diagonal
    ur_line = [b[int(size*col-col)] for col in range(1, size+1)[::-1]]   
    winner = check_for_win(ur_line, min_contig)
    if winner: return winner+' Wins!'
    
    # if it is a "normal board" (3x3, 4x4) — aka min_contig==size — evaluation stops here
    
    # if min_contig < size, we need to check smaller diagonal lines as well - not just the big ones in the middle
    # (on normal boards, these loops have nothing to iterate over)
    # the "over" iterator shifts the diagonal search over to scan the smaller sections
    
    # shifted down-right diagonals
    for over in range(1, size-min_contig+1): # how far can we move over to still start a long enough line?
        dr_line = [b[int(row+row/size)+over] for row in range(0, len(b), size)[:-over]] # get tiles in this line
        winner = check_for_win(dr_line, min_contig) # is there a win in this line?
        if winner: return winner+' Wins!'
    for over in range(1, size-min_contig+1):
        dr_line = [b[int(row+row/size)-over] for row in range(0, len(b), size)[over:]]
        winner = check_for_win(dr_line, min_contig)
        if winner: return winner+' Wins!'
    
    # shifted up-right diagonals
    for over in range(1, size-min_contig+1):
        ur_line = [b[int(size*col-col)+over] for col in range(1, size+1)[over:][::-1]]   
        winner = check_for_win(ur_line, min_contig)
        if winner: return winner+' Wins!'
    for over in range(1, size-min_contig+1):
        ur_line = [b[int(size*col-col)-over] for col in range(1, size+1)[:-over][::-1]]   
        winner = check_for_win(ur_line, min_contig)
        if winner: return winner+' Wins!'

        
    # all possible winning lines were checked, but no win was found:
    if b.count(' ')>0: # empty spaces remain:
        return 'Continue'
    else: return 'Tie!' # no empty spaces
        

     
# E-GREEDY ANIMATION FUNCTION

# animate a pie chart of random vs. greedy moves over n simulated moves.
def simulate_e_greedy(e_init, e_terminal, games=100):
    globals()['sims'] = 0
    print('Launching simulator...', end='')
    
    
    # define the range of epsilon values for each moves
    # (normalize & scale a range to generate the descending epsilon climb)
    e_terminal = int(e_terminal*100) # convert given values to integers
    e_init = int(e_init*100)
    delta = e_init-e_terminal-1
    if delta<=0:
        print('\rInitial epsilon should be greater than terminal epsilon!')
        #return
        print('Simulating anyway...')
    e_factor = games/delta
    e_terminal *= e_factor
    e_init *= e_factor
    # generate list of epsilon values, one for each move
    e_range = [i/100/e_factor for i in range(int(e_terminal), int(e_init)+1)]
    e_range =  e_range[::-1]     
    
    def animator(i, e_range=e_range): # creates one frame of the animation
        epsilon = e_range[i]
        
        # moves = [n_random, n_greedy]
        
        # make e-greedy decision using random.uniform
        if random.uniform(0,1) > epsilon: moves[0] += 1 
        else: moves[1] += 1
            
        plt.axis('equal')
        pct_smart = round(moves[0]/sum(moves)*100)
        pct_rand = round(moves[1]/sum(moves)*100)
        dynamic_legend =[
                    mpatches.Patch(color='blue', label=f'Random moves: {pct_rand}%'),
                    mpatches.Patch(color='orange', label=f'Greedy moves: {pct_smart}%'),
                    mpatches.Patch(fill=False, linewidth=0, label=f'Current ε: {round(epsilon, 2)}'),
                        ]
        labels = [f'Random: {moves[1]}', f'Q-Informed: {moves[0]}']
        colors = ['orange', 'blue']
        patches, texts = plt.pie(moves, colors=colors, startangle=225)
        legend = plt.legend(patches, labels, handles=dynamic_legend, loc=1)
        title = plt.title(f'{sum(moves)-2} moves over ε range ({round(e_range[0], 2)}–{round(epsilon, 2)})')
        return patches
    
    def simgen(): # generates the iteration (i) number
        global sims
        i = 0
        while sims < games:
            i += 1
            yield i
            
    moves = [0, 0] # [n_random, n_greedy] - global
    print(' done.\nLaunching animation...', end='')
    fig = plt.figure(figsize=(7,3))
    anim = animation.FuncAnimation(fig,
                                   animator,
                                   frames=simgen,
                                   interval=70,
                                    blit=True);
    return anim
   
    # END E-GREEDY ANIMATOR


   
# initialize global q table
q_table = {}
qt_update_counter = {}


# CHOOSE MOVE FUNCTION
# returns the integer index of the chosen position on the board
def get_move(b, epsilon=.5, player='X', init_q=.3):
    global qt_update_counter
    global q_table
    
    b_key = ''.join(b) # accept b as a string or a list
    
    # reverse tiles if player is O
    if player =='O': b_key = flip_board(b_key)
    
    # list possible moves
    opts = [i for i in range(len(b)) if b[i]==' ']

    # if new state, initialize in q_table
    if b_key not in q_table.keys():
        # nested dicts Q1 and Q2 = for every board state
        q_table[b_key] = {v: {o:init_q for o in opts} for v in ['Q1', 'Q2']}
        qt_update_counter[b_key] = 0
    else: qt_update_counter[b_key] += 1
    
    # get average Q values from both versions of the table
    q_vals = {o: sum([q[o] for q in q_table[b_key].values() ])/2 for o in opts}
     
    # e-greedy decision
    random_move = epsilon > random.uniform(0, 1)
    if random_move: # random move
        return random.choice(opts)
    else: # smart move
        return max(q_vals, key=q_vals.get)

    
# SIMULATE ONES GAME
    
def simulate_game(epsilon_x=1, epsilon_o=1, verb=False, slow_down=False, size=size):
    global min_contig
    
    steps = [] # to store the history of states + actions for this game
    
    b=new_board(size)
    while True: # iterate between players with provided e-values attached
        for player, epsilon in zip(['X', 'O'], [epsilon_x, epsilon_o]):
            
            # before each move, check current state
            result = evaluate(b, min_contig=min_contig)
                         
            # non-terminal state
            if 'C' in result: 
                
                # use this player's e-value to choose the next move 
                move = get_move(b, epsilon, player)
                
                # store the current board + the chosen action in steps (a history object)
                steps.append({'state':''.join(b.copy()), 'move': move,})
                
                if slow_down: # visualizing in the kernel
                    show(b, helpers=True)
                    print(f'{player} -> {move+1} (ε={epsilon})\n')
                    time.sleep(slow_down*.2)
                    display.clear_output()
                    
                # update the board
                b[move] = player
                
            else: # terminal state
                if verb or slow_down: show(b), print(result)
                return steps, result[0]
            
            
# Q-UPDATE FORMULA

# lrate and discount are global, set above (after imports)

def get_new_q(current_q, reward, max_future_q):
    return (1-lrate)*current_q + lrate*(reward + discount * max_future_q)
    
    
    
# BACKPROPAGATION FUNCTION

# updates the Q-Table using the "steps" output from simulate_game() after one match
def backpropagate(steps, winner, alpha=.9, wait_seconds=False):
   
    global q_table
    
    # if visualizing in the kernel:
    if wait_seconds and wait_seconds>0: 
        verb = True
        time.sleep(2)
    else: verb=False
    if wait_seconds>60: return 'wait_seconds cannot exceed 60'
        
    # backprop through all moves by each player
    for player in ['X', 'O']:
        
        if verb: # for visualizing in the kernel
            display.clear_output()
            print('—'*34)
            print(f'Starting backpropagation for {player} ...')
            print('—'*34)
            time.sleep(min(wait_seconds, 2))
            
        # isolate this player's moves from the steps object:
        p_steps = steps.copy() # working steps object for this specific player
        
        if player=='O': # if O, drop X's first move
            p_steps = p_steps[1:]
            
        # check if a slice is needed due to tie
        if winner == 'T': # the need to slice for this player depends if board size is odd or even
            if player =='O' and size**2%2==1:
                p_steps = p_steps[:-1]
            if player =='X' and size**2%2==0:
                p_steps = p_steps[:-1]
            
        # if hard loss, drop opponent's last move:
        if player!=winner and winner!='T': # drop opponent's last move
            p_steps = p_steps[:-1] 
               
        # drop the other players' remaining moves
        p_steps = p_steps[::-2]
        

        # iterate backwards over steps where player moved
        for n_steps_left, step in enumerate(p_steps):
            
             # select random q table to update in this step for Double-Q Learning
            qv = random.choice(['Q1', 'Q2']) 

            state, move = step['state'], step['move']

            if player=='O': state = flip_board(state) # if player O, reverse tiles
                
            if verb: # for kernel visualization, show previous state with selected move as '*'
                display.clear_output()
                future = [i for i in step['state']]
                future[move] = '*'
                print(f"{player}'s move ({len(p_steps) - n_steps_left}/{len(p_steps)})")
                show(''.join(future))

        # define key variables for get_new_q(), the Q update formula:
        
            old_q = q_table[state][qv][move] # solely for printing
            
            # define reward amount (negative if loss, positive if win)
            if winner==player:
                reward = alpha**(n_steps_left+1)
            elif winner=='T':
                reward = -.1*alpha**(n_steps_left+1) # slight negative if Tie
            else:
                reward= -alpha**(n_steps_left+1)
            
            # if steps remain, get the max_future_q
            if n_steps_left>0:
                future = p_steps[n_steps_left-1]['state']
                if player=='O': future = flip_board(future)
                max_future_q = max(q_table[future][qv].values())
                
            # these are arguable    
            elif player==winner: max_future_q = 1
            elif winner=='T': max_future_q = .5
            else: max_future_q = 0
            
            # run the updater formula to nudge the Q-value for this move: 
            new_q = get_new_q(old_q, reward, max_future_q)
            
            # overwrite target table with new q (Q1 or Q2, selected randomly above for double-q learning)
            q_table[state][qv][move] = new_q 
            
            # print step details if visualizing in kernel:
            if verb:
                print(f'  > Move: [{move}]', )
                print(f'  > Old Q value for [{move}]:', old_q)
                if player==winner:
                    print(f"    > \x1b[01mReward\x1b[0m for [{move}]: {reward}")
                else: print(f"    > \x1b[01mPenalty\x1b[0m for [{move}]: {reward}")
                print('    > Max future Q:', max_future_q)
                print(f'  > New Q value for [{move}]:', new_q)
                print(f"\n>>> Updated Q Table for '{state}': ", q_table[state][qv])
                time.sleep(wait_seconds)
                
    if verb:
        display.clear_output()
        print('Done.')

    return # safety closer


all_games_counter = 0

def clear_q_table():
    confirm = input(prompt='Are you sure? (y/n): ')
    if 'y' in confirm.lower():
        print('Q Table reset.')
        globals()['q_table'] = {}
        globals()['qt_update_counter'] = {}
        globals()['all_games_counter'] = 0
    else: print('Cancelled.')


def visualize_learning(boards='default',
                   batches=4, sims_per_batch=50, init_e=1,
                   lrate=.1, discount=.9, min_contig=min_contig):
    if boards=='default':
        if size%2==0:
            boards = [size**2*' ',
                      int(size**2/2-1)*' '+'X'+' '*int(size**2/2)]
        else:
            boards = [size**2*' ',
                      int(size**2/2)*' '+'X'+' '*int(size**2/2)]

    print('Launching simulator...', end='')
    
    globals()['sims']=0
    global q_table
 
    globals()['games'] = {'Training as X (O moves randomly)': {
            'evals': (True, False),
            'wins': {'X': 0, 'O': 0, 'T': 0},
            'stats': [],
             },
         'Training as O (X moves randomly)': {
            'evals': (False, True),
            'wins': {'X': 0, 'O': 0, 'T': 0},
            'stats': [],
             },
         'Training as both (versus)': {
            'evals': (True, True),
            'wins': {'X': 0, 'O': 0, 'T': 0},
            'stats': [],
             }}
    global games
    globals()['game_rotator'] = list(games.keys())
    global game_rotator

    globals()['lrate'] = lrate
    globals()['discount'] = discount
    
    # identify player to move and flip board if O
    b_players = []
    for i, b in enumerate(boards):
        if b.count('X')==b.count('O'):
            b_players += ['X'] 
        else:
            b_players += ['O']
            boards[i] = flip_board(b)


# —————————————————    
# SIMULATE ONE GAME
    def train_agent(boards, i, init_e=init_e, min_contig=min_contig):
        

        # set game index info
        g_type = game_rotator[ int(((i+1)/sims_per_batch))%3 ]
        g_count = (i)%sims_per_batch # which game (in this batch) 
        batch_n = int((int((i-1)/sims_per_batch))/3) # which batch?
        
        # set epsilon values for player (tell one or both players to move using the Q Table)
        e_vals = games[g_type]['evals'] # get (T/F, T/F) epsilon bools for (X, O)
        with_low_e = (1-g_count/sims_per_batch)*(1-batch_n/batches)*init_e
        if g_type == 'Training as both (versus)': with_low_e*=.5 # decrease epsilon for training against each other,
                                                                 # otherwise it takes too long to visualize changes
                                                                 # should be addressed if formally published
        # simulate game & update q table
        steps, winner = simulate_game(
            with_low_e if e_vals[0] else 1, # epsilon X 
            with_low_e if e_vals[1] else 1,  # epsilon O 
            )
        backpropagate(steps, winner)
        
        games[g_type]['wins'][winner] += 1
        stats =  {p: games[g_type]['wins'][p] / sum(games[g_type]['wins'].values()) for p in 'XOT'}
        stats.update({'e': with_low_e})
        games[g_type]['stats'] += [stats]
            # total wins divided by total games for each player

        
        # make target heatmaps
        qval_arrays = []
        for b in boards:
            # open positions
            opts = [i for i in range(len(b)) if b[i]==' ']
            # if new state initialize q table 
            if b not in q_table.keys(): q_vals =  {i: .6 for i in opts}
            # else get average q vals 
            else:  q_vals = {i: sum([q[i] for q in q_table[b].values() ])/2 for i in opts}
            # custom color for occupied spaces:
            q_vals.update( {i: -2 for i in range(len(b)) if i not in list(q_vals.keys()) } )
            q_vals = [v for k,v in sorted(q_vals.items())]

            q_map = []
            for row in range(0, len(b), size):
                h_line = [q_vals[row+col] for col in range(size)]
                q_map.append(h_line)
            qval_arrays += [np.asarray(q_map)]
        return qval_arrays, games, g_type

    print(' done.\nLaunching animation...', end='')
# —————————————————    
# FIGURE GRID......   
    fig = plt.figure(figsize=(9,5), constrained_layout=True)
    gs = GridSpec(6, 16, figure=fig)

    
# —————————————————    
# TRAINING PLOTS...    
    ax1 = fig.add_subplot(gs[0:2, :-5], title='ax1')
    ax2 = fig.add_subplot(gs[2:4, :-5], title='ax2')
    ax3 = fig.add_subplot(gs[4:, :-5], title='ax3')
    
    for ax, g_type in zip([ax1, ax2, ax3], game_rotator):
        ax.set_title(g_type)
        ax.set_ylim(0, 1)
    im1, = ax1.plot([], [], color='r', label='X win rate')
    im2, = ax1.plot([], [], color='orange', label='O win rate')
    im3, = ax1.plot([], [], color='g', label='Tie rate')
    ime1, = ax1.plot([], [], color='b', linestyle='dashed', label='ε for X')
    ax1.legend(loc=2)
    im4, = ax2.plot([], [], color='r', label='X win rate')
    im5, = ax2.plot([], [], color='orange', label='O win rate')
    im6, = ax2.plot([], [], color='g', label='Tie rate')
    ime2, = ax2.plot([], [], color='b', linestyle='dashed', label='ε for O')
    ax2.legend(loc=2)
    im7, = ax3.plot([], [], color='r', label='X win rate')
    im8, = ax3.plot([], [], color='orange', label='O win rate')
    im9, = ax3.plot([], [], color='g', label='Tie rate')
    ime3, = ax3.plot([], [], color='b', linestyle='dashed', label='ε for both')
    ax3.legend(loc=2)

    
# —————————————————    
# Q VAL HEATMAPS....
    ax4 = fig.add_subplot(gs[0:3, -5:-1], title='ax4') # qmap1
    ax5 = fig.add_subplot(gs[3:, -5:-1], title='ax5') # qmap2
    # initial heatmap figure
    init_q_map = np.asarray([[.6,]*size]*size)
    # custom colormap
    coolwarm = cm.get_cmap('PiYG', 256)
    coolwarm_lin = coolwarm(np.linspace(0, 1, 256))
    unavail = np.array([180/256, 180/256, 180/256, 1]) # white
    coolwarm_lin[:25, :] = unavail
    custom_cmap = ListedColormap(coolwarm_lin)
    # set figure options
    qmap1 = ax4.imshow(init_q_map, cmap=custom_cmap, interpolation=None,
                    vmin=-1.1, vmax=1)
    qmap2 = ax5.imshow(init_q_map, cmap=custom_cmap, interpolation=None,
                    vmin=-1.1, vmax=1)
    ax6 = fig.add_subplot(gs[0:, -1:]) # qmap legend
    colbar = fig.colorbar(qmap1, cax=ax6, shrink=.6)
    colbar.set_label('Action Q-Value')
    # format heatmaps:
    for ax, b, player in zip([ax4, ax5], boards, b_players):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Player: {player}  | State:")
        ax.grid(False, which='both')
        # annotations
        r_idx = 0
        b_idx = -1
        for row in range(0, len(b), size):
            for col in range(size):
                b_idx +=1
                if player=='O': tile = flip_board(b)[b_idx]
                else: tile = b[b_idx]
                if tile=='X': p_color='red'
                elif tile=='O': p_color='blue'
                else: p_color='purple'
                ax.annotate(b_idx+1 if tile == ' ' else tile,
                             xy=(col, r_idx),
                             color=p_color,
                             fontsize=10)
            r_idx+=1
        
# —————————————————    
# ANIMATOR FUNCTION
    def board_animator(i, boards=boards,
                       sims_per_batch=sims_per_batch,
                       batches=batches, min_contig=min_contig):

        qval_arrays, games, g_type = train_agent(boards, i, min_contig=min_contig)
        
        if g_type == 'Training as X (O moves randomly)':
            ims = [im1, im2, im3, ime1]
            focax = ax1   
        elif g_type == 'Training as O (X moves randomly)':
            ims = [im4, im5, im6, ime2]
            focax = ax2
        elif g_type == 'Training as both (versus)':
            ims = [im7, im8, im9, ime3]
            focax = ax3
        for im, p in zip(ims, 'XOTe'):
            y_data = [stats[p] for stats in games[g_type]['stats']]
            y_data = np.asarray(y_data)
            x_data = np.arange(len(y_data))
            im.set_data(x_data, y_data)

        focax.set_xlim(0, len(y_data))

        # update heatmap
        qmap1.set_array(qval_arrays[0])
        qmap2.set_array(qval_arrays[1])

        fig.suptitle(f'Player winning rates over {sims_per_batch*3*batches} games |  {i+2} completed')
        return im1,im2,im3,im4,im5,im6,im7,im8,im9,qmap1,qmap2

    def simgen():
        global sims
        i = -1
        while sims < sims_per_batch*3*batches:
            sims += 1
            i += 1
            yield i
    
    anim = animation.FuncAnimation(fig, board_animator, 
                                   frames = simgen,
                                   interval=1, blit=True);
    return anim


   

# FOR TRAINING WITH**OUT** LIVE ANIMATION (better for bigger simulations)
def efficient_trainer(iters=1000, batches=3, min_contig=globals()['min_contig']):
    s = time.perf_counter()
    global q_table
    # for each batch, run these game types:
    game_types = {  # (epsilon_x/o)
        'Training X (random O)': (True, False), # random moves by O
        'Training O (random X)': (False, True), # random moves by X
        "Training both (versus)": (True, True),} # smart moves by both

    print(f'STARTING {iters*batches*len(game_types)} SIMULATIONS...')
    
    # to store data from sequential batches in each game type:
    results = {g_type:[] for g_type in game_types.keys()}  
    e_values = {g_type:[] for g_type in game_types.keys()}  
    
    for batch in range(batches):
        batch_start = time.perf_counter()

        # set initial epsilon value based on batch number
        init_e=round(1-(batch)/batches, 3)
        print(f'  Starting session {batch+1}/{batches}... | initial epsilon: {init_e}')
        # iterate through game times (set which player moves randomly)
        for game_type, player_e_vals in game_types.items():
            
            print(f'\r  > {game_type}... | {iters} games', end=''+' '*30)
            
            # initialize game type stats for batch
            wins = {'X':0, 'O':0, 'T':0 }
            # simulate games
            for i in range(iters):
                steps, winner = simulate_game( # random moves if e not specified:
                            init_e*(1-i/iters) if player_e_vals[0] else 1,
                            init_e*(1-i/iters) if player_e_vals[1] else 1)
                # backpropagate
                backpropagate(steps, winner)
                e_values[game_type].append(init_e*(1-i/iters))
                # save stats data by game type
                wins[winner] += 1
                results[game_type].append(
                    {p: ws/sum(list(wins.values())) for p, ws in wins.items()} )        
            
        batch_dur = round(time.perf_counter()-batch_start, 1)
        print(f'\r  > Session completed | {iters*len(game_types)} games in {batch_dur} s'+' '*20)
    dur = round(time.perf_counter()-s, 1)
    display.clear_output()
    print(f'>>> {iters*batches*len(game_types)} games completed ', end='')
    print(f'in {dur} s (avg {round(iters*batches*len(game_types)/dur)} games/second)')
    
    return results, e_values # stats for plotting AFTER the simulation is complete...

# ...plot results from the efficient trainer (static line charts of winning rates over time) 
def plot_learning(results, e_values, game_type, row):
    plt.subplot(3,1,row+1)
    plt.plot(range(len(e_values)),
                 e_values, linestyle='dashed',
                 label = "Trainer epsilon")
    for player in ['X', 'T', 'O']:
        plt.plot(range(len(results)),
                 [stats[player] for stats in results],
                 label = f'{player}', )
    plt.subplots_adjust(hspace=0.5)
    plt.legend(fontsize=9), plt.title(game_type), plt.show();
    
    
# COMPLETE TRAINING FUNCTION
# this is for ease of use in the kernel
# to automatically train a decent number of sumalations and the plot results
# running this once should let the agent see around 99% of all possible board states
def full_training(iters=1000, batches=15, min_contig=globals()['min_contig']):
    results, e_values = efficient_trainer(iters=iters, batches=batches, min_contig=min_contig)
    print('Plotting results:')
    fig=plt.figure(figsize=(9,9), constrained_layout=False)
    plt.suptitle(f'Winning rates over {iters*batches*3} simulations')
    for row, game_type in enumerate(list(results.keys())):
        plot_learning(results[game_type], e_values[game_type], game_type, row)

     
    
# INTERACTIVE PLAY AGAINST THE AGENT
    

# ARE ANY NON-TIE RESULTS STILL POSSIBLE?
# during a human-vs-computer match, later in the game, check futures to see if all paths lead to a tie
def check_futures(board, player, results=[]):
    global min_contig
    state = evaluate(board, min_contig=min_contig)[0]
    if 'C' not in state: return state
    else:
        for o in [e for e,b in enumerate(board) if b==' ']:
            future = list(board)
            future[o] = player
            results += check_futures(future, 'XO'.replace(player, ''), results)
    return set(results)



# LAUNCH THE INTERACTIVE PLAYER:
def versus_agent(custom_q_table=False): # accepts a custom q_table for comparing multiple agents

    global min_contig
    global q_table
    if custom_q_table: # custom agent
        cache_table = globals()['q_table'] # save existing table
        globals()['q_table'] = custom_q_table

    
    # title sequence
    print(' welcome to…')
    for t in range(13):
        print('•', end=''), time.sleep(.04)
    print('\n TIC-TAC-TOE')
    
    # stats
    wins = 0
    ties = 0
    losses = 0
    
    # start game "engine"
    exit = 0
    while True:  
        b = new_board()
        show(b)
        
        # define user with input
        tries = 0
        while True: 
            user = input(prompt='Choose a player (X or O): ').replace('0', 'o').upper()
            if user=='X' or user=='O':
                agent = 'XO'.replace(user, '')
                break
            tries+=1
            print(f'Try again ({3-tries})!')
            if tries>2:
                exit = True 
                break
        display.clear_output()
        if exit: break # exit if 3 attempts failed 
        
        # start moving
        restart = 0 
        while True:
            for player in 'XO': # iterate between players
                
                show(b, helpers=True)
                state = evaluate(b, min_contig) # evaluate current board state
                
                if b.count(' ')<3: # if near the end, see if any wins are still possible
                    if check_futures(b, player)=={'T'}:
                        state = 'Tie!'
                
                if 'C' in state and not restart: # get next move + update board
                    
                    # make the last move automatically (only one choice)
                    if b.count(' ')==1:  move = get_move(b, 0, player)
                    
                    # user's turn
                    elif player==user:
                        tries = 0
                        while True: # try to get user's choice
                            move = input(prompt=f'Your move, {user}: ')
                            if 'q' in move.lower():
                                restart=True # manual quit
                                break
                            try:
                                move = int(move)-1 # is input integer?
                                try: # does the position exist on this board?
                                    if b[move] in 'OX': # is the position available?
                                        print('Position occupied!')
                                    else: break
                                except:
                                    print('Position out of range!')
                            except: print('Enter a number!')
                            tries+=1
                            if tries>2:
                                exit = True # 3 failed tries --> quit
                                break
                    
                    # agent's turn
                    elif player==agent: move = get_move(b, 0, player)
                    
                    # end game if user entered 'q' or input failed
                    if exit or restart: break
                    
                    # make selected move (update the board)
                    b[move] = player               
                
                # game over
                elif 'C' not in state:
                    winner = state[0]
                  
                    # print match result
                    if user == winner:
                        wins +=1
                        print('You won!')
                    elif winner == 'T':
                        ties +=1
                        print(state)
                    else:
                        losses +=1
                        print(state)
                    print('—'*13) 
                    
                    # print overall stats for this session
                    for stat, n in zip(['Won','Tied','Lost'], [wins,ties,losses]):
                        if n>0: print(f'{stat}: {n}')
                    time.sleep(.5)
                    
                    # ask if play again
                    play_again = input(prompt=f'Play again? (y/n): ')
                    if 'y' in play_again.lower(): restart = True
                    else: exit = True 
                    break
                
                # end step
                display.clear_output() 
                
            # end game if user selected restart
            if restart or exit:
                display.clear_output()
                break  
               
        # end the outer loop
        if exit: break
            
    # if a custom table was used, return default to original q_table
    if custom_q_table: # custom agent
        globals()['q_table'] = cache_table
            
    print('Goodbye!')
    
        

print('\rFunctions imported successfully.'+50*' ')
