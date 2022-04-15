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
with open("ClassicArastdout.txt","w") as out, open("ClassicArastderr.txt", "w") as err:
    with subprocess.Popen("./ClassicAra", stdin=subprocess.PIPE, stdout=out, stderr=err, universal_newlines=True) as CrazyAra:
        CAin = CrazyAra.stdin
        CAout = CrazyAra.stdout
        CAerr = CrazyAra.stderr
        input = ""
        input.__add__("uci\n")
        input.__add__("isready\n")
        #set engine options
        input.__add__("setoption name Batch_size value 1\n")
        for i, fen in enumerate(fens):
            inputstring = "position "+fen+"\n"
            input.__add__(inputstring)
            input.__add__("go nodes 1000\n")
        input.__add__("quit\n")
        CrazyAra.communicate(input)