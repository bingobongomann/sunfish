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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('module', help='crazysunfishIterativeWidening.py file (without .py)', type=str, default='CrazySunfishIterativeWidening5_c', nargs='?')
    args = parser.parse_args()

    Crazysunfish = importlib.import_module(args.module)

    logging.basicConfig(filename='CrazysunfishIterativeWidening.log', level=logging.DEBUG)
    out = Unbuffered(sys.stdout)
    def output(line):
        print(line)
        logging.debug(line)
    pos = tools.parseCrazyFENIW(tools.FEN_INITIAL_CRAZYHOUSE)
    searcher = Crazysunfish.Searcher()
    color = WHITE
    our_time, opp_time = 1000, 1000 # time in centi-seconds
    show_thinking = True
    hist = []
    repetition = False
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
            output('id name CrazySunfishIterativeWidening')
            output('id author Jannik Holmer, based on Sunfish by Thomas Ahle &contributers')
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

            pos = tools.parseCrazyFENIW(fen)
            hist=[pos]
            repetition = False
            for move in moveslist:
                pos = pos.apply_move(tools.parseMove(move))
                hist.append(pos)
            for histpos in hist[:-1]:
                if(histpos.key == pos.key):
                    repetition = True

        elif smove.startswith('go'):
            #  default options
            depth = 1000
            movetime = -1
            our_time = -1
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
            for sdepth, _move, _score , nodes, T_hit, NN_evals in searcher.search(pos, hist):
                moves = tools.crazypvIW(searcher, pos, sdepth, repetition, hist)

                newtime = time.time()
                it_time = newtime -oldtime
                if sdepth > 1:
                    f_new = it_time/it_time_old
                    f = max(f,f_new)
                oldtime = newtime
                it_time_old = it_time

                if show_thinking:
                    usedtime = int((time.time() - start)*1000)
                    moves_str = moves if len(moves) < 15 else ' '.join(moves.split(' ')[:3])
                    output('info depth {} score cp  {} time {}ms nodes {} pv {}'.format(sdepth, _score, usedtime, searcher.nodes, moves_str))

                if len(moves) > 5:
                    ponder = moves[1]

                if movetime > 0 and (time.time() - start)*f * 1000 > movetime:
                    break

                if our_time>0 and ((time.time() - start)*f * 1000) > our_time/moves_remain:
                    break

                if sdepth >= depth:
                    break

            m, s = _move, _score
            # We only resign once we are mated.. That's never?
            
            moves = moves.split(' ')
            if len(moves) > 1:
                output(f'bestmove {moves[0]} ponder {moves[1]}')
            else:
                output('bestmove ' + moves[0])

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()

