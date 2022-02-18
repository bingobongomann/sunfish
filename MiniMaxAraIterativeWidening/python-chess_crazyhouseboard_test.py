import chess.variant
def printboard(pos):
        print('Pocket Black:')
        print(pos.pockets[chess.BLACK])
        print('_______________')
        print(pos)
        print('Pocket White:')
        print(pos.pockets[chess.WHITE])

firstBoard = chess.variant.CrazyhouseBoard()
printboard(firstBoard)
firstBoard.push_uci("e2e4")
printboard(firstBoard)
firstBoard.push_uci("d7d5")
printboard(firstBoard)
secondboard=firstBoard.copy()
firstBoard.push_uci("e4d5")
printboard(secondboard)
printboard(firstBoard)
thirdboard = firstBoard.copy()
printboard(thirdboard)



