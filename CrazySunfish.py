#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import namedtuple
import chess
import chess.variant
import chess.polyglot
import re, sys, time
import numpy as np
from numpy.core.fromnumeric import sort
sys.path.insert(0,'CrazyAra/')
from DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from DeepCrazyhouse.src.domain.agent.player.raw_net_agent import RawNetAgent
from DeepCrazyhouse.src.domain.variants.game_state import GameState
from DeepCrazyhouse.src.domain.util import get_check_move_mask



###############################################################################
# Global constants
###############################################################################


initial = chess.variant.CrazyhouseBoard()


# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6


# The table size is the maximum number of elements in the transposition table.
TABLE_SIZE = 1e7

# Constants for tuning search
QS_LIMIT = 219
EVAL_ROUGHNESS = 13
DRAW_TEST = True
ENHANCE_CHECKS = False
ENHANCE_CAPTURES = False
QUANTIL = 0.99

batch_size = 1
threads = 1

nets = []
for idx in range(2):
    nets.append(NeuralNetAPI(ctx='cpu', batch_size=batch_size))

raw_agent = RawNetAgent(nets[0])


###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board, key')):
    """ A state of a chess game
    board -- a pychess board
    key -- zobrist key of the position
    """

    def gen_moves(self):
        for x in self.board.legal_moves:
            yield x

    def move(self, move):
        newboard = self.board.copy(stack=False)
        newboard.push(move)
        return Position(newboard, chess.polyglot.zobrist_hash(newboard))


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple('Entry', 'lower upper')

class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = list()
        self.nodes = 0
        self.tablehits =0
        self.NN_evals = 0


    def bound(self, pos, gamma, depth, root=True, move_given=''):
        """ returns r where
                s(pos) <= r < gamma    if gamma > s(pos)
                gamma <= r <= s(pos)   if gamma <= s(pos)"""
        self.nodes += 1
        print('nodes: ',self.nodes, ' move given: ', move_given)
        print('Pocket Black:')
        print(pos.board.pockets[chess.BLACK])
        print('_______________')
        print(pos.board)
        print('Pocket White:')
        print(pos.board.pockets[chess.WHITE])
        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.board.is_checkmate():
            return -1
        #if pos.board.is_stalemate():
        #    return 0

        # We detect 3-fold captures by comparing against previously
        # _actually played_ positions.
        # Note that we need to do this before we look in the table, as the
        # position may have been previously reached with a different score.
        # This is what prevents a search instability.
        # FIXME: This is not true, since other positions will be affected by
        # the new values for all the drawn positions.
        if DRAW_TEST:
            if(pos.key in histpos.key for histpos in self.history):
                if pos.board.can_claim_draw():
                    return 0
                

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos.key, depth, root), Entry(-1, 1))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos.key) is not None):
            self.tablehits +=1
            print('tablehit')
            return entry.lower
        if entry.upper < gamma:
            self.tablehits +=1
            print('tablehit')
            return entry.upper

        # Here extensions may be added
        # Such as 'if in_check: depth += 1'

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            state = GameState(pos.board)
            
            pred_value, legal_moves, p_vec_small, cp, depthnet, nodes, time_elapsed_s, nps, pv = raw_agent.evaluate_board_state(state)
            self.NN_evals += 1
            #print(pos.board)
            #print(pos.key)
            #print(pred_value)
            if depth == 0:

                yield None, pred_value
            else:
                if ENHANCE_CHECKS:
                    check_mask, nb_checks = get_check_move_mask(pos.board, legal_moves)
                    if nb_checks > 0:
                        # increase chances of checking
                        p_vec_small[np.logical_and(check_mask, p_vec_small < 0.1)] += 0.1
                    # normalize back to 1.0
                    p_vec_small /= p_vec_small.sum()

                if ENHANCE_CAPTURES:
                    for capture_move in pos.board.generate_legal_captures():
                        index = legal_moves.index(capture_move)
                        if p_vec_small[index] < 0.04:
                            p_vec_small[index] += 0.04
                    # normalize back to 1.0
                    p_vec_small /= p_vec_small.sum()


                P_sum= 0
                mergedlist = list(zip(legal_moves, p_vec_small))
                #print(mergedlist)
                sortedlist = sorted(mergedlist, key=lambda x:x[1], reverse=True)
                #print(sortedlist)
                move = self.tp_move.get(pos.key)
                #bestmove from previous iteration 
                if move is not None:
                    sortedlist.insert(0, (move,0))
                for move in sortedlist:
                    if(P_sum<QUANTIL):
                        P_sum=P_sum + move[1]
                        print('Psum:',P_sum)
                        newpos = pos.move(move[0])
                        yield move[0], -self.bound(newpos, gamma, depth-1, root=False, move_given= move[0])
                    else:
                        #print('quantil reached')
                        break

        # Run through the moves, shortcutting when possible
        best = -1
        for move, score in moves():
            #print('move:', move , 'score',score)
            best = max(best, score)
            if best >= gamma:
                #print('newbest', move, ':', score, 'depth',depth)
                # Clear before setting, so we always have a value
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                # Save the move for pv construction and killer heuristic
                print('move:', move, 'is better than gamma with ', score, ' / ', gamma)
                self.tp_move[pos.key] = move
                break
                
                
                
                

        # Clear before setting, so we always have a value
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        # Table part 2
        if best >= gamma:
            self.tp_score[pos.key, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos.key, depth, root] = Entry(entry.lower, best)

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

        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply.
        for depth in range(1, 2):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but play tests show a margin of 20 plays
            # better.
            lower, upper = -1, +1
            self.tablehits =0
            while lower < upper:
                gamma = (lower+upper)/2 
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                print('score',score,'upper', upper, 'lower', lower,'gamma', gamma)
                
            # We want to make sure the move to play hasn't been kicked out of the table,
            # So we make another call that must always fail high and thus produce a move.
            self.bound(pos,lower,depth)
            #print('Lower:', lower, 'Upper:', upper, 'Gamma:', gamma, 'score:', score, 'bounds: ',bounds)
            # If the game hasn't finished we can retrieve our move from the
            # transposition table.
            yield depth, self.tp_move.get(pos.key), self.tp_score.get((pos.key, depth, True)).lower, self.nodes, self.tablehits, self.NN_evals


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input


def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)


def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.split()):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')


def main():
    hist = [Position(initial, chess.polyglot.zobrist_hash(initial))]
    searcher = Searcher()
    while True:
        print('Pocket Black:')
        print(hist[-1].board.pockets[chess.BLACK])
        print('_______________')
        print(hist[-1].board)
        print('Pocket White:')
        print(hist[-1].board.pockets[chess.WHITE])

        if hist[-1].board.is_checkmate():
            print("You lost")
            break

        # We query the user until she enters a (pseudo) legal move.
        move = None
        pymove = None
        while pymove not in hist[-1].gen_moves():
            match = re.match('([a-h][1-8]|[PQNBR]@)([a-h][1-8])', input('Your move: '))
            if match:
                move = match.group(1) + match.group(2)
                pymove = chess.Move.from_uci(move)
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        hist.append(hist[-1].move(pymove))

        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        print('Pocket Black:')
        print(hist[-1].board.pockets[chess.BLACK])
        print('_______________')
        print(hist[-1].board)
        print('Pocket White:')
        print(hist[-1].board.pockets[chess.WHITE])

        if hist[-1].board.is_checkmate():
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

            if (time.time() - start)*f > 15 or score==1:
                print('Nodes Searched:',nodes, 'score:', score, 'NN evals:', NN_evals, 'time', time.time()-start)
                break



        print("My move:", move)
        hist.append(hist[-1].move(move))


if __name__ == '__main__':
    main()

