from rtpt.rtpt import RTPT
import logging
import subprocess
import sys
import chess
from searcher_MTDbi import Searcher

sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
logging.basicConfig(filename="EngineTestLog.log",filemode="w",level=logging.DEBUG)
rtpt = RTPT(name_initials="JH", experiment_name="Engine Rapid Test", max_iterations=666)
netAPI = NeuralNetAPI(ctx="gpu", select_policy_form_planes=True)
quantil = [0, 0.99, 0.995, 1]
rtpt.start()
file = open('Eigenmann Rapid Engine Chess.epd', 'r')
lines = file.readlines()
fens = []
bestmoves_san =[]
epds = []
operators =[]
ids = []
evalnumbers = [0]*111
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

#S = Searcher(2,1,netAPI, "standard", quantil)
    
    #print(operators)
num_correct = 0 
#for i ,fen in enumerate(fens):
#    chosenmove = None
#    for depth, move, score, searchtime, nodes, nn_evals in S.searchPosition(fen, None):
#        logging.info(f"time: {searchtime}, depth {depth}, move: {move}, score: {score}")
#        if searchtime > 15:
#            logging.info(f"move before time ran out {chosenmove}")
#            break
#        chosenmove= S.board.san(move)
#        evalnumbers[i] = nn_evals
#        if score == 1: break
#    correct = False
#    chosenmove = chosenmove.replace("+", "")
#    chosenmove = chosenmove.replace("-","")
#    #print(operators[i])
#    if operators[i] == "bm":
#        logging.info(f"best move: {bestmoves_san[i]}, our move: {chosenmove}")
#        if bestmoves_san[i] == chosenmove:
#            correct = True
#    if operators[i] == "am":
#        logging.info(f"avoid move: {bestmoves_san[i]}, our move: {chosenmove}")
#        if not bestmoves_san[i] == chosenmove:
#            correct = True
#              
#    if correct:
#        logging.info("correct")
#        num_correct +=1
#    else: logging.info("wrong")
#    rtpt.step()
#logging.info(f"correct solved: {num_correct}, thats {num_correct/111} ")
#S.stop_helpers()


    #open ClassicAra and iterate through the testpositions with the amount of nodes used by the minimax engine.
with open("ClassicArastdout.txt", "w") as out, open("ClassicArastderr","w") as err:
    with subprocess.Popen("./ClassicAra", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1) as CrazyAra:
        CAin = CrazyAra.stdin
        CAout = CrazyAra.stdout
        CAerr = CrazyAra.stderr
        out = None
        for i in range(21):
                out = CAout.readline()
                print(out)
        CAin.write("uci\n")
        CAin.flush()
        while True:
            out = CAout.readline()
            print(out)
            if out == "uciok\n":
                break
        CAin.write("isready\n")
        CAin.flush()
        while True:
            out = CAout.readline()
            print(out)
            if out =="readyok\n":
                break
        CAin.write("setoption name Batch_size value 1\n setoption name Use_Raw_Network value True\n")
        CAin.flush()
        print(CAout.readline())
        board = chess.Board()
        num_correct = 0
        for i, fen in enumerate(fens):
            correct = False
            inputstring = "position fen "+fen+" moves\n"
            board.set_fen(fen)
            print(f"number {i}: fen: {fen}")
            print(inputstring)
            CAin.write("ucinewgame\n")
            CAin.flush()
            print(CAout.readline())
            CAin.write("isready\n")
            CAin.flush()
            print(CAout.readline())
            CAin.write(inputstring)
            CAin.flush()
            print(CAout.readline())
            CAin.write(f"go\n")
            CAin.flush()
            while True:
                out =CAout.readline()
                print(out)
                parts = out.split(" ")
                if parts[0] =="bestmove":
                    print(out)
                    movestring = parts[1]
                    movestring = movestring.replace("\n", "")
                    chosenmove = board.san(chess.Move.from_uci(movestring))
                    break
            CAout.readline()
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
            print(f"num correct: {num_correct}")
        logging.info(f"number of correct solved: {num_correct}")
        CAin.write("quit\n")
        CAin.flush()
