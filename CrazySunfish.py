#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import namedtuple
import chess
import chess.variant
import chess.polyglot
import re, sys, time
import numpy as np
import copy
from numpy.core.fromnumeric import sort
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from CrazyAra.DeepCrazyhouse.src.domain.agent.player.raw_net_agent import RawNetAgent
from CrazyAra.DeepCrazyhouse.src.domain.variants.game_state import GameState
from CrazyAra.DeepCrazyhouse.src.domain.util import get_check_move_mask



###############################################################################
# Global constants
###############################################################################


INITIAL = chess.variant.CrazyhouseBoard()


# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# An Entry from the Transposition Table
# lower and upper are the bounds of the position
# Score is the predicted Value from the NN_eval 
# moveList contains the ordered tuples (move, P(move)) from the NN_eval ordered from highest to lowest or in case we had a bestmove in the TT it has been moved to the first position
# depth is the depth to which the position has been searched 
Entry = namedtuple('Entry', 'lower upper Score moveList depth' )


# Constants for tuning search
DRAW_TEST = True
ENHANCE_CHECKS = False
ENHANCE_CAPTURES = False
#at depth 0 the quantil doesnt matter
QUANTILS = [0,0.75,0.95,0.99,1]
# Constants for the NeuralNet-Evaluation
batch_size = 1
threads = 1
#
nets = []
nets.append(NeuralNetAPI(ctx='cpu', batch_size=batch_size))

raw_agent = RawNetAgent(nets[0])


###############################################################################
# Chess logic
###############################################################################

class StateKeyPair(namedtuple('StateKeyPair', 'state, key')):
    """ A state of a chess game
    state -- GameState
    key -- zobrist key of the position
    """
    def apply_move(self, move):
        #print(move)
        self.state.apply_move(move)
        newpos= StateKeyPair(self.state, self.state.get_pythonchess_board().fen())# chess.polyglot.zobrist_hash(self.state.get_pythonchess_board()))
        return newpos

    def undo_move(self):
        self.state.board.pop()
        newpos = StateKeyPair(self.state,self.state.get_pythonchess_board().fen())# chess.polyglot.zobrist_hash(self.state.get_pythonchess_board()))
        return newpos

###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = list()
        self.nodes = 0
        self.tablehits =0
        self.NN_evals = 0


    def bound(self, pos, gamma, depth, root=True):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1
        #print('Pocket Black:')
        #print(pos.state.get_pythonchess_board().pockets[chess.BLACK])
        #print('_______________')
        #print(pos.state.get_pythonchess_board())
        #print('Pocket White:')
        #print(pos.state.get_pythonchess_board().pockets[chess.WHITE])
        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)
        
        if pos.state.is_loss():
            return -1
        #we check for 3-fold repetition by comparing the zobrist hashes of our previously actually played positions 
        # if there is a match we call the actual draw check which would be very costly if called every time 
        repetition = False
        if DRAW_TEST:
            if repTest(pos, self.history):
                #print('repetition')
                repetition = True
                if pos.state.is_draw():
                    return 0
                

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        # Entry ( Score, Move_order from NN, depth_of_search)TT only saves the highest depth search 
        entry = self.tp_score.get((pos.key,repetition))
        histmove = self.tp_move.get((pos.key,repetition))

        if entry is not None:
            self.tablehits +=1
            if entry.depth >= depth:
                if entry.lower >= gamma:
                    #print('lower:{} upper: {} move:{}'.format(entry.lower, entry.upper, histmove))
                    #print('lower bigger gamma')
                    return entry.lower
                if entry.upper < gamma:
                    #print('lower:{} upper: {} move:{}'.format(entry.lower, entry.upper, histmove))
                    #print('upper smaller gamma')
                    return entry.upper
            pred_value = entry.Score
            moveList = entry.moveList

        # There was no entry in the TT for this position so the NN is used to evaluate the board and return a policy of which moves to look at first 
        #
        else:
            pred_value, legal_moves, p_vec_small , cp, depthnet, nodes, time_elapsed_s, nps, pv = raw_agent.evaluate_board_state(pos.state)
            self.NN_evals += 1
            #print(pos.board)
            #print(pos.key)
            #print(pred_value)

            if ENHANCE_CHECKS:
                check_mask, nb_checks = get_check_move_mask(pos.state.get_pythonchess_board() , legal_moves)
                if nb_checks > 0:
                    # increase chances of checking
                    p_vec_small[np.logical_and(check_mask, p_vec_small < 0.1)] += 0.1
                # normalize back to 1.0
                p_vec_small /= p_vec_small.sum()

            if ENHANCE_CAPTURES:
                for capture_move in pos.state.get_pythonchess_board().generate_legal_captures():
                    index = legal_moves.index(capture_move)
                    if p_vec_small[index] < 0.04:
                        p_vec_small[index] += 0.04
                # normalize back to 1.0
                p_vec_small /= p_vec_small.sum()


            
            mergedlist = list(zip(legal_moves, p_vec_small))
            #print(mergedlist)
            moveList = sorted(mergedlist, key=lambda x:x[1], reverse=True)
            #print(sortedlist)



        # insert the best move fron the previous searches found in the TT 
        if histmove is not None:
            for idx, tuple in enumerate(moveList):
                if tuple[0]==histmove:
                    moveList.pop(idx)
                    moveList.insert(0, tuple)



        def moves(sortedlist, position, history, Quantil):
            P_sum= 0   
            for move in sortedlist:
                if(P_sum<Quantil):
                    P_sum=P_sum + move[1]
                    #print('Psum:',P_sum)
                    position = position.apply_move(move[0])
                    history.append(position)
                    score =-self.bound(position, -gamma, depth-1, root=False)
                    position.undo_move()
                    history.pop()
                    yield move[0], score
                else:
                    break


        #leaf node => return the value of the position estimated by the NN      
        if depth == 0:
            self.tp_score[(pos.key,repetition)] = Entry(-1,1,pred_value,moveList,0)
            return pred_value
        # Run through the moves, shortcutting when possible
        printmoves = []
        bestmove = None
        best = -1
        qidx = min(depth,QUANTILS.__len__()-1)

        for move, score in moves(moveList, pos, self.history, QUANTILS[qidx]):
            #print('move:', move , 'score',score)
            printmoves.__add__([(move, score)])
            best = max(best, score)
            if best == score:
                bestmove = move
            if score >gamma:
                #print('newbest', move, ':', score, 'depth',depth)
                
                break
                
        # Clear before setting, so we always have a value
        if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
        # Save the move for pv construction and killer heuristic
        #print('move:', move, 'is better than gamma with ', score, ' / ', gamma)
        self.tp_move[(pos.key,repetition)] = bestmove        
                

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        tmpdepth = 0
        upper = 1
        lower = -1
        if entry is not None:
            tmpdepth = entry.depth
            lower = entry.lower
            upper = entry.upper
        #if best>upper:
        #    best = upper
        #if best < lower: 
        #    best = lower
        if depth>= tmpdepth:
            if best >= gamma:
                self.tp_score[(pos.key,repetition)] = Entry(best, upper, pred_value, moveList, depth)
            if best < gamma:
                self.tp_score[(pos.key,repetition)] = Entry(lower, best, pred_value, moveList, depth)

        return best

    def search(self, pos, history=()):
        """ Iterative deepening MTD-bi search """
        self.nodes = 0
        self.NN_evals = 0
        print(pos.key)
        if DRAW_TEST:
            self.history = history
            # print('# Clearing table due to new history')
            self.tp_score.clear()
        repetition = False
        if repTest(pos, self.history):
            repetition = True
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 200):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            lower, upper = -1, +1
            self.tablehits =0
            while lower < upper:
                gamma = (lower+upper)/2 
                score = self.bound(pos, gamma, depth)
                print('Before: score',score,'upper', upper, 'lower', lower,'gamma', gamma)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                print('After: score',score,'upper', upper, 'lower', lower,'gamma', gamma)
                
            # We want to make sure the move to play hasn't been kicked out of the table,
            # So we make another call that must always fail high and thus produce a move.
            self.bound(pos,lower,depth)
            #print('Lower:', lower, 'Upper:', upper, 'Gamma:', gamma, 'score:', score, 'bounds: ',bounds)
            # If the game hasn't finished we can retrieve our move from the
            # transposition table.
            yield depth, self.tp_move.get((pos.key,repetition)), self.tp_score.get((pos.key,repetition)).lower, self.nodes, self.tablehits, self.NN_evals

def repTest(position, History):
    for histpos in History:
        if histpos.key == position.key:
            return True
    return False

def printPosition(position):
    print('Pocket Black:')
    print(position.state.get_pythonchess_board().pockets[chess.BLACK])
    print('_______________')
    print(position.state.get_pythonchess_board())
    print('Pocket White:')
    print(position.state.get_pythonchess_board().pockets[chess.WHITE])

###############################################################################
# User interface
###############################################################################
def main():
    hist = [StateKeyPair(GameState(INITIAL),INITIAL)]# chess.polyglot.zobrist_hash(INITIAL))]
    searcher = Searcher()
    while True:
        printPosition(hist[-1])

        if hist[-1].state.is_loss():
            print("You lost")
            break

        # We query the user until she enters a (pseudo) legal move.
        move = None
        pymove = None
        while pymove not in hist[-1].state.get_legal_moves():
            match = re.match('([a-h][1-8]|[PQNBR]@)([a-h][1-8])', input('Your move: '))
            if match:
                move = match.group(1) + match.group(2)
                pymove = chess.Move.from_uci(move)
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        
        hist.append(copy.deepcopy(hist[-1]).apply_move(pymove))

        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        printPosition(hist[-1])
        print(hist.__len__())

        if hist[-1].state.is_loss():
            print("You won")
            break

        # Fire up the engine to look for a move.
        start = time.time()
        oldtime = start
        f= 1
        f_new =0
        for _depth, move, score, nodes ,TT_hits ,NN_evals in searcher.search(hist[-1], hist):
            newtime = time.time()
            it_time = newtime -oldtime
            if _depth > 1:
                f_new = it_time/it_time_old
                f = max(f,f_new)
            oldtime = newtime
            it_time_old = it_time


            print('depth:',_depth, 'hits', TT_hits, 'Iteration_factor:', f_new)
            print('Nodes Searched:',nodes, 'score:', score, 'NN evals:', NN_evals, 'time', time.time()-start)

            #if (time.time() - start)*f > 30 or score==1:
            if(_depth==5):
                break
        


        print("My move:", move)
        hist.append(copy.deepcopy(hist[-1]).apply_move(move))

if __name__ == '__main__':
    main()

