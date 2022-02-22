import threading
class SplitPoint():
    def __init__(self, board, movelist, depth, alpha, beta, masterID, bestmove, history) -> None:
        self.board = board.copy()
        self.moveList = movelist
        self.moveLock = threading.Lock()
        self.boundsLock = threading.Lock()
        self.depth = depth 
        self.alpha = alpha
        self.beta = beta 
        self.masterID = masterID
        self.slaveIDs = []
        self.activeThreadCount = 0
        self.bestmove = bestmove
        self.history = history