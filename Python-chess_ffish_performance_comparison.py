# -*- coding: utf-8 -*-

import random
import time
import chess.variant
from FFishBoard import FairyBoard




def main():
    pychessboard = chess.variant.CrazyhouseBoard()
    ffishboard = FairyBoard("crazyhouse")

    PyPTime = PyMGTime = PyNrLegalMoves =0
    ffPTime = ffMGTime = ffNrLegalMoves = 0
    PTotaltime = ffTotaltime = 0
    for i in range(10000):
        pstart = time.time()
        start = time.time()
        legalmoves = [*pychessboard.legal_moves]
        MGtime = time.time() -start
        
        move = legalmoves[random.randint(0,legalmoves.__len__()-1)]
        start = time.time()
        pychessboard.push(move)
        Ptime = time.time() -start
        PyNrLegalMoves += legalmoves.__len__()
        PyPTime += Ptime
        PyMGTime += MGtime
        PTotaltime += time.time() - pstart
        ffstart = time.time()
        start = time.time()
        legalmoves = ffishboard.legal_moves()
        MGtime = time.time() -start
        #print(legalmoves.__len__())
        if legalmoves.__len__()==0:
            ffishboard = FairyBoard("crazyhouse")
            legalmoves= ffishboard.legal_moves()
            #print(legalmoves.__len__())
        move = legalmoves[random.randint(0,legalmoves.__len__()-1)]
        start = time.time()
        ffishboard.push(move)
        Ptime = time.time() -start
        ffNrLegalMoves += legalmoves.__len__()
        ffPTime += Ptime
        ffMGTime += MGtime
        ffTotaltime += time.time() - ffstart

        if pychessboard.is_game_over():
            pychessboard.reset()
        
    print(f"PythonchessBoard: Time for MoveGen:{PyMGTime}s Time for Pushing moves: {PyPTime}s Number of LegalMoves generated in total: {PyNrLegalMoves} TotalTime: {PTotaltime}")
    print(f"FFishBoard: Time for MoveGen: {ffMGTime}s Time for Pushing moves: {ffPTime}s Number of LegalMoves generated in total: {ffNrLegalMoves} TotalTime: {ffTotaltime}")
        


if __name__ == '__main__':
    main()