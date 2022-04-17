#!/usr/bin/env pypy -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import importlib
from platform import node
import re
import sys
import time
import logging
import argparse
import chess.variant
import chess

variant = "standard"

Quantil = [0,0.99,0.995,1]

def main():
    initial = True
    def output(line):
                print(line)
                logging.debug(line)
    stack = []
    while True:
        if stack:
            smove = stack.pop()
        else:
            smove = input()

        logging.debug(f'>>> {smove} ')

        if smove == 'quit':
            searcher.stop_helpers()
            break

        elif smove == 'uci':
            output('id name CrazySunfishIterativeWidening')
            output('id author Jannik Holmer')
            output('option name UCI_Variant type combo default crazyhouse var crazyhouse')
            output('uciok')

        elif smove == 'isready':
            if initial:
                initial=False
                parser = argparse.ArgumentParser()
                parser.add_argument('module', help='crazysunfishIterativeWidening.py file (without .py)', type=str, default='searcher_MTDbi', nargs='?')
                args = parser.parse_args()

                Crazysunfish = importlib.import_module(args.module)
                logging.basicConfig(filename='CrazysunfishIterativeWidening.log', level=logging.DEBUG)
                
                searcher = Crazysunfish.Searcher(2, 1, net=None, variant=variant, quantils = Quantil, ctx="gpu")
                if variant == "standard":
                    board = chess.Board()
                else:
                    board = chess.variant.CrazyhouseBoard()
                
                our_time, opp_time = 1000, 1000 # time in centi-seconds
                show_thinking = True
                hist = []
                fen = None
                repetition = False
            output('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + board.starting_fen)

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
                output("new position: "+fen)

            elif params[1] == 'startpos':
                
                fen = board.starting_fen
                output("startpos: "+fen)
            else:
                pass
            
            #hist=[fen.split(" ")[0]]
            #repetition = False
            #for move in moveslist:
            #    pos = pos.apply_move(tools.parseMove(move))
            #    hist.append(pos)
            #for histpos in hist[:-1]:
            #    if(histpos.key == pos.key):
            #        repetition = True

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
            if fen is not None:
                for sdepth, _move, _score , searchingtime, nodes , nn_evals in searcher.searchPosition(fen, moveslist):

                    newtime = time.time()
                    it_time = newtime -oldtime
                    if sdepth > 1:
                        f_new = it_time/it_time_old
                        f = max(f,f_new)
                    oldtime = newtime
                    it_time_old = it_time

                    if show_thinking:
                        usedtime = int((time.time() - start)*1000)
                        output(f'info depth {sdepth} score cp  {_score} time {usedtime}ms nodes {nodes}')

                        #ponder = moves[1]

                    if movetime > 0 and (time.time() - start)*f * 1000 > movetime:
                        break

                    if our_time>0 and ((time.time() - start)*f * 1000) > our_time/moves_remain:
                        break

                    if sdepth >= depth:
                        break

                m, s = _move, _score
                # We only resign once we are mated.. That's never?
                

                output(f'bestmove {m} Score {s}')
            else: output("Please input a position first!")

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()

