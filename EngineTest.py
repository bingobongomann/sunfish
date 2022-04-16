import logging
from rtpt.rtpt import RTPT
import chess
from searcher_MTDbi import Searcher
import sys
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
logging.basicConfig(filename="EngineTestwithAverage.log",filemode="w",level=logging.DEBUG)
rtpt = RTPT(name_initials="JH", experiment_name="Engine Rapid Test", max_iterations=666)
netAPI = NeuralNetAPI(ctx="cpu", select_policy_form_planes=True)
Quantils = [[0, 0.9, 0.995, 1]]
            
Results = []
rtpt.start()
file = open('Eigenmann Rapid Engine Chess.epd', 'r')
lines = file.readlines()
fens = []
bestmoves_san =[]
epds = []
operators =[]
ids = []
lines[0] = lines[0].replace(" -", "",1)
lines[4] = lines[4].replace(" -", "",1)
nodesperdepth = [0]*200
hitsperdepth = [0]*200
for line in lines:
    parts = line.split(";")
    epds.append(parts[0])
    ids.append(parts[1])
for i ,epd in enumerate(epds): 
    parts = epd.split(" ")
    fens.append(" ".join(parts[:4]))
    operators.append(parts[4])
    bestmoves_san.append(parts[5])

for quantil in Quantils:
    S = Searcher(2,1,netAPI, "standard", quantil)
    
    #print(operators)
    num_correct = 0 
    for i ,fen in enumerate(fens):
        chosenmove = None
        for depth, move, score, searchtime, nodes , nn_evals in S.searchPosition(fen, None):
            logging.info(f"time: {searchtime}, depth {depth}, move: {move}, score: {score}")
            nodesperdepth[depth] += nn_evals
            hitsperdepth[depth] +=1
            if searchtime > 15:
                logging.info(f"move before time ran out {chosenmove}")
                break
            chosenmove= S.board.san(move)
            if score == 1: break
        correct = False
        chosenmove = chosenmove.replace("+", "")
        chosenmove = chosenmove.replace("-","")
        #print(operators[i])
        if operators[i] == "bm":
            logging.info(f"best move: {bestmoves_san[i]}, our move: {chosenmove}")
            if bestmoves_san[i] == chosenmove:
                correct = True
        if operators[i] == "am":
            logging.info(f"avoid move: {bestmoves_san[i]}, our move: {chosenmove}")
            if not bestmoves_san[i] == chosenmove:
                correct = True
                
        if correct:
            logging.info("correct")
            num_correct +=1
        else: logging.info("wrong")
        rtpt.step()

    Results.append((num_correct, num_correct/111))
    logging.info(f"correct solved: {num_correct}, thats {num_correct/111} ")
    S.stop_helpers()
for idx, result in enumerate(Results):
    logging.info(f"Quantil-set Nr.: {idx} {Quantils[idx]} num solved: {result[0]}, thats {result[1]*100}%")
for i, nodes in enumerate(nodesperdepth):
    hits = hitsperdepth[i]
    if hits>0:        
        logging.info(f"depth {i}: average of {hits} times is {nodes/hits}")
