from rtpt.rtpt import RTPT
import logging
import subprocess
import sys
import chess

rtpt = RTPT(name_initials="JH", experiment_name="Engine Rapid Test", max_iterations=111)
logging.basicConfig(filename="CrazyAraFirstTest.log",filemode="w",level=logging.DEBUG)


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
for line in lines:
    parts = line.split(";")
    epds.append(parts[0])
    ids.append(parts[1])
for i ,epd in enumerate(epds): 
    parts = epd.split(" ")
    fens.append(" ".join(parts[:4]))
    operators.append(parts[4])
    bestmoves_san.append(parts[5])

with subprocess.Popen("./ClassicAra", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as CrazyAra:
    CAin = CrazyAra.stdin
    CAout = CrazyAra.stdout
    CAerr = CrazyAra.stderr
    out = CAout.readlines()
    print(out)
    CAin.write('uci\n')
    out = CAout.readlines()
    print(out)
    
    CAin.write("setoption name Batch_size value 1\n")
    board = chess.Board()
    num_correct = 0
    for i, fen in enumerate(fens):
        inputstring = "position "+fen
        board.set_board_fen(fen)
        CAin.write(inputstring)
        CAin.write("go nodes 1000\n")
        while True:
            if CAout.readable():
                out =CAout.readline()  
                parts = out.split(" ")
                if parts[0] == "bestmove":
                    movestring = parts[1]
                    chosenmove = board.san(chess.Move.from_uci(movestring))
                    break
        
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
    logging.info(f"number of correct solved: {num_correct}")
    CAin.write("quit\n")