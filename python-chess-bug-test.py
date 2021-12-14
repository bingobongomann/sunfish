import chess.variant
import chess

def main():
    fen = 'rn1qkbnr/ppp1pppp/8/3p4/3P1Bb1/8/PPP1PPPP/RN1QKBNR[] w KQkq - 0 1'
    board = chess.variant.CrazyhouseBoard(fen)
    board.push_uci('f2f3')
    board.push_uci('e8d7')
    board.push_uci('b1c3')
    board.push_uci('d7e8')
    board.push_uci('e2e3')
    board.push_uci('b8d7')
    board.push_uci('d1c1')
    board.push_uci('h7h5')
    board.push_uci('g1e2')
    board.push_uci('d7f6')
    print('Pocket Black:')
    print(board.pockets[chess.BLACK])
    print('_______________')
    print(board)
    print('Pocket White:')
    print(board.pockets[chess.WHITE])
    newboard = board.copy()
    newboard.push_uci('f3g4')
    print('Pocket Black:')
    print(newboard.pockets[chess.BLACK])
    print('_______________')
    print(newboard)
    print('Pocket White:')
    print(newboard.pockets[chess.WHITE])
    newnewboard = board.copy()
    newnewboard.push_uci('f3g4')
    newnewboard.pockets[chess.BLACK].add(5)
    newnewboard.pockets[chess.BLACK].remove(5)
    print('Pocket Black:')
    print(newnewboard.pockets[chess.BLACK])
    print('_______________')
    print(newnewboard)
    print('Pocket White:')
    print(newnewboard.pockets[chess.WHITE])

if __name__ == '__main__':
    main()
    