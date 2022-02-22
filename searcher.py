from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
import threading
from time import time
import chess.variant
import chess
import numpy as np
from splitpoint import SplitPoint
import sys
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from CrazyAra.DeepCrazyhouse.src.domain.variants.game_state import GameState
from CrazyAra.DeepCrazyhouse.src.domain.agent.player.util.net_pred_service import NetPredService
from CrazyAra.DeepCrazyhouse.src.domain.variants.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_TOTAL, NB_LABELS
from CrazyAra.DeepCrazyhouse.src.domain.variants.output_representation import get_probs_of_move_list, value_to_centipawn

QUANTILS = [0,0.5,0.75,0.9,0.99,1]
DTYPE = np.float32 

#TODO: include self written history of positions for easy and cheap repetiton and draw detection
#       potetially rework the TT key for more Transposition Hits on the NN_Eval 
#       

NN_Entry = namedtuple('NN_Entry','score policy' )
Score_Entry = namedtuple('Score_Entry', 'score move depth')



class Searcher():

    def __init__(
        self,
        threads=128,
        batch_size=64,
        net = NeuralNetAPI
        ):
        # Transpostion Table for the NeuralNet evaluations 
        self.TT_NN = {}
        # Transpostion Table for the best score found during search
        self.TT_Score = {}
        self.nodes = 0
        self.Tablehits_NN = 0
        self.Tablehits_Score = 0
        self.IdleThreads = []
        self.work = {}
        self.stop = {}
        self.ThreadAccess = threading.Lock()
        self.net = net
        self.threads = threads
        self.batch_size = batch_size
        self.my_pipe_endings = []  # create pipe endings for itself and the prediction service
        pipe_endings_external = []
    
        for i in range(threads):
            ending1, ending2 = Pipe()
            self.my_pipe_endings.append(ending1)
            pipe_endings_external.append(ending2)

        self.batch_state_planes = np.zeros((self.threads,NB_CHANNELS_TOTAL,BOARD_HEIGHT, BOARD_WIDTH),DTYPE)
        self.batch_value_results = np.zeros(self.threads, DTYPE)
        self.batch_policy_results = np.zeros((self.threads, NB_LABELS),DTYPE)
        self.net_pred_service = NetPredService(
            pipe_endings_external,
            net,
            batch_size,
            self.batch_state_planes,
            self.batch_value_results,
            self.batch_policy_results)


    # This is the function being called by the UCI Interface
    def searchPosition(self, initialFen, movelist):
        self.t_start_eval = time()

        #start the net_pred_service 
        if not self.net_pred_service.running:
            self.net_pred_service.start()

        #clear Transposition Tables before search 
        self.TT_NN.clear()
        self.TT_Score.clear()
        self.nodes = 0

        #fix FEN because there is no convention for empty pockets in FENs
        initialFen = initialFen.replace("[-]","[]")
        #create board from movelist and fen 
        board = chess.variant.CrazyhouseBoard(fen=initialFen)
        if movelist is not None:
            for move in movelist:
                board.push(move)
        TTKey = self.getTTKey(board)
        #start worker threads 
        futures = []
        executor = ThreadPoolExecutor(max_workers=self.threads-1)
        for i in range(1,self.threads):
            futures.append(
                executor.submit(self.idle_loop,threadID=i, splitpoint=None )
            )
        print("hello")
        # iterative deepening search 
        for It_depth in range(1, 200):

            alpha, beta = -1, 1
            score = self.search(board, alpha, beta, It_depth,0)
            entry = self.TT_Score.get(TTKey)
            move = entry.move

            yield It_depth, move, score, time()-self.t_start_eval, self.nodes



    def search(self, board, alpha, beta, depth, threadID):    
        self.nodes += 1
        #check for Terminal Positions
        if board.is_checkmate():
            return -1
        #include  repetiton check through history here 
        if board.is_stalemate():
            return 0 

        # The FEN uniquely identifies the position
        TTkey = self.getTTKey(board)
        # Check Transposition Table for Neural Network Evaluation
        nn_entry = self.TT_NN.get(TTkey)
        # Check Transposition for results from previous searches
        score_Entry = self.TT_Score.get(TTkey)
        if nn_entry is not None:
            #Transpositon Table found a NN_eval for the position
            self.Tablehits_NN
            pred_value = nn_entry.score
            pred_Policy = nn_entry.policy
        else:
            #Transposition Table found no NN_eval entry for the Position 
            pred_value, pred_Policy = self.evaluate_Position(board, threadID)
            self.TT_NN[TTkey] = NN_Entry(pred_value, pred_Policy)

        if score_Entry is not None:
            # we have a previous search result in the Transposition Table
            if score_Entry.depth >= depth:
                # the found search result was to greater or equal depth so we dont have to search 
                return score_Entry.score
            histmove = score_Entry.move
            # if the best move from a previous search differs from the NN_prediction we swap
            # the history move into the first position to be searched. 
            if histmove is not None:
                for idx, tuple in enumerate(pred_Policy):
                    if tuple[0]==histmove:
                        pred_Policy.pop(idx)
                        pred_Policy.insert(0, tuple)
        # SearchDepth has been reached and Value estimated by the NN is propagated back
        if depth == 0: 
            return pred_value

        # generate Quantil index
        Qidx = min(depth,QUANTILS.__len__()-1)
        best_score = -1
        best_move = None
        moveList = []
        # generate the movelist from the Quantil and all the moves
        for move in self.moveQuantilGen(pred_Policy,Qidx):
            moveList.append(move)
        
        bestmove = None
        #while the list is not empty keep searching
        while moveList:
            move = moveList.pop(0)
            board.push(move)
            score = -self.search(board, -beta, -alpha, depth-1, threadID)
            board.pop()
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
                bestmove = move

            #if other threads are available to help, make a splitnode from this position and split the work
            #print(self.ThreadAccess.locked())
            if not self.ThreadAccess.locked():
                self.ThreadAccess.acquire()
                IDs = self.idle_threads(threadID)
                #print(IDs)
                if(IDs):
                    s = SplitPoint(board,moveList,depth,alpha,beta, threadID, bestmove)
                    self.split(s, IDs)
                    self.TT_Score[TTkey] = Score_Entry(s.alpha, s.bestmove, depth)
                    return s.alpha #??? should be right, just break should return the old alpha and disregard the splitsearch
                self.ThreadAccess.release()
                    
        self.TT_Score[TTkey] = Score_Entry(alpha, bestmove, depth)
        return alpha



    def getTTKey(self, board):
        return board.fen()
    #  Split assignes work to all the idle threads 
    def split(self, splitpoint, IDs):
        for threadID in IDs :
            self.assign_work(threadID, splitpoint)
        self.ThreadAccess.release()
        self.assign_work(splitpoint.masterID, splitpoint)
        self.idle_loop(splitpoint.masterID, splitpoint)
    # idle_loop in which threads go when they have nothing to do,
    # constantly looking for new assigned work 
    def idle_loop(self, threadID, splitpoint):
        if splitpoint is None:
            self.set_idle(threadID,None)
            #print(f"thread {threadID} idle")
        while True:
            if self.assigned_work(threadID):
                self.splitpoint_search(self.retrieve_work(threadID), threadID)
                self.reset_stop(threadID)
                self.set_idle(threadID, splitpoint)

            
            if splitpoint is not None and splitpoint.activeThreadCount == 0:
                break



    def set_idle(self, threadID, splitpoint):
        self.ThreadAccess.acquire()
        self.IdleThreads.append((threadID,splitpoint))
        self.ThreadAccess.release()

    def assign_work(self, threadID, splitpoint):
        splitpoint.slaveIDs.append(threadID)
        self.work[threadID] = splitpoint

    
    def assigned_work(self, threadID):
        work = self.work.get(threadID)
        if work:
            return True
        return False


    def retrieve_work(self, threadID):
        work = self.work.get(threadID)
        self.work[threadID] = None
        return work
    #checks the list of ilde threads for viable slaves for ThreadID
    # removes all viable ones from the list and returns their ids to split() 
    # idle masters of splitpoints can only help their own slaves "helpful master concept"
    def idle_threads(self, threadID):
        viableIDs = []
        remainingThreads= []
        #print(f"idle_threads: {self.IdleThreads}")
        for i, tuple in enumerate(self.IdleThreads):
            splitpoint = tuple[1]
            if splitpoint:
                if threadID in splitpoint.slaveIDs:
                    viableIDs.append(tuple[0])
                else:
                    remainingThreads.append(tuple)
            else:
                viableIDs.append(tuple[0])
        self.IdleThreads = remainingThreads.copy()
        return viableIDs

    # takes a sorted movelist and the Quantil-array index and 
    # returns all moves until the quantil is reached    
    def moveQuantilGen(self, Policy , Qidx):
        P_sum = 0
        quantil = QUANTILS[Qidx]
        for tuple in Policy: 
            if P_sum<quantil:
                P_sum += tuple[1]
                yield tuple[0]
        #handles the communication with the evaluator thread and returns the sorted policy and value prediction
    def evaluate_Position(self, board, pipeID):
        #add batch commmunication with GPU eval (net_pred_service)
        state = GameState(board)
        my_pipe = self.my_pipe_endings[pipeID]
        state_planes = state.get_state_planes()
        self.batch_state_planes[pipeID] = state_planes
        my_pipe.send(pipeID)
        result_channel = my_pipe.recv()
        value = np.array(self.batch_value_results[result_channel])
        policy_vec = np.array(self.batch_policy_results[result_channel])
        legal_moves = state.get_legal_moves()
        p_vec_small = get_probs_of_move_list(policy_vec, legal_moves ,state.mirror_policy(),normalize=True)

        mergedlist = list(zip(legal_moves, p_vec_small))
        moveList = sorted(mergedlist, key=lambda x:x[1], reverse=True)
        return value, moveList


    def splitpoint_search(self, splitpoint, threadID):
        splitpoint.activeThreadCount +=1
        # search new moves until the list is empty
        while splitpoint.moveList and not self.told_to_stop(threadID):
            move = self.pickMove(splitpoint)
            if move is None: break
            newBoard = splitpoint.board.copy()
            newBoard.push(move)
            score = -self.search(newBoard,-splitpoint.beta, -splitpoint.alpha, splitpoint.depth -1, threadID)
            if score > splitpoint.beta:
                self.tell_all_threads_stop(splitpoint)
                break
            splitpoint.boundsLock.acquire()
            if score > splitpoint.alpha:
                splitpoint.alpha = score
                splitpoint.bestmove = move
            splitpoint.boundsLock.release()

        if threadID == splitpoint.masterID and self.told_to_stop(threadID):
            self.tell_all_threads_stop(splitpoint)

        splitpoint.activeThreadCount -=1

    def tell_all_threads_stop(self, splitpoint):
        for threadID in splitpoint.slaveIDs:
            self.stop[threadID] = True
        self.stop[splitpoint.masterID] = True

    def reset_stop(self, threadID):
        self.stop[threadID]= False
    
    def told_to_stop(self, threadID):
        stop = self.stop.get(threadID)
        if stop:
            return True
        return False

    def pickMove(self, splitpoint):
        splitpoint.moveLock.acquire()
        move = splitpoint.moveList.pop(0)
        splitpoint.moveLock.release()
        return move

#testscript when this file is run by itself and not imported
if __name__ == "__main__":
    board = chess.variant.CrazyhouseBoard()
    movelist = None
    netapi = NeuralNetAPI(ctx="gpu",batch_size=8)
    s = Searcher(16,8,netapi)
    for depth, move, score, searchtime, nodes in s.searchPosition(board.fen(),movelist):
        print(f"depth: {depth}, selected move: {move}, score: {score}, time needed: {searchtime}, nodes searched: {nodes}")