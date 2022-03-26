from re import search
from rtpt.rtpt import RTPT
import chess
from searcher_MTDbi import Searcher
import sys
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI

rtpt = RTPT(name_initials="JH", experiment_name="Engine Rapid Test", max_iterations=4)
netAPI = NeuralNetAPI(ctx="cpu", select_policy_form_planes=True)
Quantils = {[0, .5, .75, 1],
            [0, .6, .8, 1],
            [0, .75, .9, 1],
            [0, .75, .95, 1]}
rtpt.start()
for quantil in Quantils:
    S = Searcher(2,1,netAPI, "standard", quantil)
    file = open('Eigenmann Rapid Engine Chess.epd', 'r')
    lines = file.readlines()
    fens = []
    bestmoves_san =[]
    epds = []
    operators =[]
    ids = []
    lines[0] = lines[0].replace(" -", "",1)
    lines[4] = lines[4].replace(" -", "",1)
    for line in lines:
        parts = line.split(";")
        epds.append(parts[0])
        ids.append(parts[1])
    for i ,epd in enumerate(epds): 
        parts = epd.split(" ")
        fens.append(" ".join(parts[:4]))
        operators.append(parts[4])
        bestmoves_san.append(parts[5])
    #print(operators)
    num_correct = 0 

    for i ,fen in enumerate(fens):
        chosenmove = None
        for depth, move, score, searchtime, nodes in S.searchPosition(fen, None):
            print(f"time: {searchtime}, depth {depth}, move: {move}, score: {score}")
            if searchtime > 15:
                print(searchtime)
                print(f"move before time ran out {chosenmove}")
                break
            chosenmove= S.board.san(move)
            if score == 1: break
        correct = False
        chosenmove = chosenmove.replace("+", "")
        chosenmove = chosenmove.replace("-","")
        print(operators[i])
        if operators[i] == "bm":
            print(f"best move: {bestmoves_san[i]}, our move: {chosenmove}")
            if bestmoves_san[i] == chosenmove:
                correct = True
        if operators[i] == "am":
            print(f"avoid move: {bestmoves_san[i]}, our move: {chosenmove}")
            if not bestmoves_san[i] == chosenmove:
                correct = True
                
        if correct:
            print("correct")
            num_correct +=1
        else: print("wrong")
        
    print(f"correct solved: {num_correct}, thats {num_correct/111} ")
    S.stop_helpers()
    rtpt.step()
