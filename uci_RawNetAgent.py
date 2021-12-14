#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import importlib
import re
import sys
import time
import logging
import argparse

import tools
from tools import WHITE, BLACK, Unbuffered
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from CrazyAra.DeepCrazyhouse.src.domain.agent.player.raw_net_agent import RawNetAgent
from CrazyAra.DeepCrazyhouse.src.domain.variants.game_state import GameState

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('module', help='crazysunfish.py file (without .py)', type=str, default='CrazySunfish', nargs='?')
    args = parser.parse_args()

    Crazysunfish = importlib.import_module(args.module)

    logging.basicConfig(filename='Crazysunfish.log', level=logging.DEBUG)
    out = Unbuffered(sys.stdout)
    def output(line):
        print(line)
        logging.debug(line)
    pos = tools.parseCrazyFEN(tools.FEN_INITIAL_CRAZYHOUSE)
    searcher = Crazysunfish.Searcher()
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = True

    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else:
            smove = input()

        logging.debug(f'>>> {smove} ')

        if smove == 'quit':
            break

        elif smove == 'uci':
            output('id name rawnetagent')
            output('id author Jannik Holmer using the CrazyAra NN by Johannes Czech')
            output('option name UCI_Variant type combo default crazyhouse var crazyhouse')
            output('uciok')

        elif smove == 'isready':
            output('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + tools.FEN_INITIAL_CRAZYHOUSE)

        # syntax specified in UCI
        # position [fen  | startpos ]  moves  ....

        elif smove.startswith('position'):
            params = smove.split(' ')
            idx = smove.find('moves')

            if idx >= 0:
                moveslist = smove[idx:].split()[1:]
            else:
                moveslist = []

            if params[1] == 'fen':
                if idx >= 0:
                    fenpart = smove[:idx]
                else:
                    fenpart = smove

                _, _, fen = fenpart.split(' ', 2)
                print(fen)

            elif params[1] == 'startpos':
                fen = tools.FEN_INITIAL

            else:
                pass

            pos = tools.parseCrazyFEN(fen)

            for move in moveslist:
                pos = pos.apply_move(tools.parseMove(move))

        elif smove.startswith('go'):
            #  default options
            depth = 1000
            movetime = -1

            _, *params = smove.split(' ')
            for param, val in zip(*2*(iter(params),)):
                if param == 'depth':
                    depth = int(val)
                if param == 'movetime':
                    movetime = int(val)
                if param == 'wtime':
                    our_time = int(val)
                if param == 'btime':
                    opp_time = int(val)

            moves_remain = 40

            start = time.time()
            oldtime = start
            f = 1
            ponder = None
            batch_size = 1
            threads = 1

            nets = []
            for idx in range(2):
                nets.append(NeuralNetAPI(ctx='cpu', batch_size=batch_size))

            raw_agent = RawNetAgent(nets[0])
            pred_value, legal_moves, p_vec_small , cp, depthnet, nodes, time_elapsed_s, nps, pv = raw_agent.evaluate_board_state(pos.state)
            mergedlist = list(zip(legal_moves, p_vec_small))
            moveList = sorted(mergedlist, key=lambda x:x[1], reverse=True)



            if show_thinking:
                score = pred_value
                usedtime = int((time.time() - start) * 1000)
                moves_str = ''
                output('info depth {} score cp {} time {} nodes {} pv {}'.format(0, score, usedtime, searcher.nodes, moves_str))

            # We only resign once we are mated.. That's never?
        
            output('bestmove ' + str(moveList[0][0]))

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()

