from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
import threading
from time import time
import chess
import chess.variant
import numpy as np
import queue
import yappi
from splitpoint import SplitPoint
import sys
sys.path.insert(0,'CrazyAra/')
from CrazyAra.DeepCrazyhouse.src.domain.agent.neural_net_api import NeuralNetAPI
from CrazyAra.DeepCrazyhouse.src.domain.variants.game_state import GameState
from CrazyAra.DeepCrazyhouse.src.domain.agent.player.util.net_pred_service import NetPredService
from CrazyAra.DeepCrazyhouse.src.domain.variants.constants import BOARD_HEIGHT, BOARD_WIDTH, NB_CHANNELS_TOTAL, NB_LABELS
from CrazyAra.DeepCrazyhouse.src.domain.variants.output_representation import get_probs_of_move_list, value_to_centipawn

QUANTILS = [0,0.5,0.75,1]
DTYPE = np.float32 
FINISHED = object()
EXACT = 0 
LOWER = 1
UPPER = 2

#TODO: include self written history of positions for easy and cheap repetiton and draw detection
#       potetially rework the TT key for more Transposition Hits on the NN_Eval 
#       

NN_Entry = namedtuple('NN_Entry','score policy' )
Score_Entry = namedtuple('Score_Entry', 'flag value move depth')



class Searcher():

    def __init__(
        self,
        threads=128,
        batch_size=64,
        net = NeuralNetAPI,
        variant = "Crazyhouse"
        ):
        # Transpostion Table for the NeuralNet evaluations 
        self.TT_NN = {}
        self.variant = variant
        # Transpostion Table for the best score found during search
        self.TT_Score = {}
        self.board = None
        self.nodes = 0
        self.theoretical_nodes = [(0,0)]*80
        self.depthLocks = [threading.Lock()]*80
        self.splits = 0 
        self.Tablehits_NN = 0
        self.Tablehits_Score = 0
        self.Tablehits_Score_greater_depth = 0
        self.IdleThreads = [None]*threads
        self.work_queues = []
        self.stop = {}
        self.stability = True
        self.ThreadAccess = threading.Lock()
        self.net = net
        self.threads = threads
        self.batch_size = batch_size
        self.my_pipe_endings = []  # create pipe endings for itself and the prediction service
        pipe_endings_external = []

        for i in range(threads):
            #create work Queues for splitting work between threads
            workqueue = queue.Queue()
            self.work_queues.append(workqueue)

            #create pipe endings for us and the prediction service 
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
        #timing analysis
        self.t_TT_NN = np.zeros(threads)
        self.t_TT_score = np.zeros(threads)
        self.t_NN_eval = np.zeros(threads)
        self.t_split = np.zeros(threads)
        #start worker threads 
        self.futures = []
        self.executor = ThreadPoolExecutor(max_workers=self.threads-1)
        for i in range(1,self.threads):
            self.futures.append(
                self.executor.submit(self.idle_loop,threadID=i, splitpoint=None )
            )


    # This is the function being called by the UCI Interface
    def searchPosition(self, initialFen, movelist):
        self.t_start_eval = time()
        history = []
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
        TTKey = None 
        if self.variant == "Crazyhouse":
            self.board = chess.variant.CrazyhouseBoard(fen=initialFen)
        else: self.board = chess.Board(initialFen)
        if movelist is not None:
            for move in movelist:
                board.push(move)
                HistKey = self.getHistKey(self.board)
                history.append(HistKey)
        orig_repetition = False
        Occ = self.getOccurrences(self.board, history)
        if Occ == 2:
            orig_repetition= True
        TTKey = self.getTTKey(self.board, orig_repetition)
        
        print("hello")
        # iterative deepening search 
        for It_depth in range(1, 200):
            self.theoretical_nodes = [(0,0)]*80
            alpha, beta = -1, 1
            while alpha< beta:
                gamma = (alpha+beta)/2
                #print(f"gamma: {gamma}")
                score = self.search(self.board, gamma, gamma,  It_depth,0, history)
                if score >=gamma:
                    alpha = score
                if score <gamma:
                    
                    beta =score

            entry = self.TT_Score.get(TTKey)
            move = entry.move
            #print(alpha==beta)

            yield It_depth, move, score, time()-self.t_start_eval, self.nodes



    def search(self, board, alpha, beta, depth, threadID, history):    
        self.nodes += 1
        #check for Terminal Positions
        if board.is_checkmate():
            return -1
        #include  repetiton check through history here 
        repetition = False
        Occ = self.getOccurrences(board, history)
        if Occ == 2:
            repetition= True
            #print("rep")
        if Occ >2:
            #print("draw by rep")
            return 0 
        a = alpha
        # The FEN uniquely identifies the position
        #generate Transpostion key 
        TTkey = self.getTTKey(board, repetition)
        TT_NN_start = time()
        # Check Transposition Table for Neural Network Evaluation
        nn_entry = self.TT_NN.get(TTkey)
        if nn_entry is not None:
            #Transpositon Table found a NN_eval for the position
            self.Tablehits_NN +=1
            pred_value = nn_entry.score
            pred_Policy = nn_entry.policy
            t_NN_eval= 0
        else:
            #Transposition Table found no NN_eval entry for the Position 
            nn_eval_start = time()
            pred_value, pred_Policy = self.evaluate_Position(board, threadID)
            t_NN_eval = time()- nn_eval_start
            self.t_NN_eval[threadID] += t_NN_eval
            self.TT_NN[TTkey] = NN_Entry(pred_value, pred_Policy)
        # timing analysis
        self.t_TT_NN[threadID] += time()-TT_NN_start - t_NN_eval
        TT_score_start = time()
        # Check Transposition for results from previous searches
        score_Entry = self.TT_Score.get(TTkey)
        if score_Entry is not None:
            self.Tablehits_Score +=1
            # we have a previous search result in the Transposition Table
            if score_Entry.depth >= depth:
                self.Tablehits_Score_greater_depth +=1
                # the found search result was to greater or equal depth so we dont have to search 
                if score_Entry.flag is EXACT:
                    #print(f"exact score:{score_Entry.value}")
                    return score_Entry.value
                elif score_Entry.flag is LOWER:
                    #print(f"lower bound TT hit at depth {depth}: {score_Entry.value}")
                    a = max(a, score_Entry.value)
                elif score_Entry.flag is UPPER:
                    #print(f"uper bound TT hit at depth {depth}: {score_Entry.value}")

                    beta = min(beta, score_Entry.value)

                if a > beta:
                    #print(f"cutoff from TT at depth {depth}")
                    return score_Entry.value
            histmove = score_Entry.move
            # if the best move from a previous search differs from the NN_prediction we swap
            # the history move into the first position to be searched. 
            if histmove is not None:
                for idx, tuple in enumerate(pred_Policy):
                    if tuple[0]==histmove:
                        pred_Policy.pop(idx)
                        pred_Policy.insert(0, tuple)

        self.t_TT_score[threadID] += time() - TT_score_start
        # SearchDepth has been reached and Value estimated by the NN is propagated back
        if depth == 0: 
            #if pred_value==0.0:
                #print(f"predicted value zero: ")
                #print(board)
            return pred_value

        # generate Quantil index
        Qidx = min(depth,QUANTILS.__len__()-1)
        moveList = []
        # generate the movelist from the Quantil and all the moves
        for move in self.moveQuantilGen(pred_Policy,Qidx):
            moveList.append(move)

        #average branching factor for each depth, to estimate the size of the tree
        self.depthLocks[depth].acquire()
        tuple = self.theoretical_nodes[depth]
        avg = tuple[0]
        n = tuple[1]
        avg = avg*(n/(n+1))+ (moveList.__len__()/(n+1))
        self.theoretical_nodes[depth] = (avg,n+1)
        self.depthLocks[depth].release()
        bestmove = None
        best = -1
        #while the list is not empty keep searching
        if not moveList: print("no Moves!!!")
        while moveList:
            move = moveList.pop(0)
            len = moveList.__len__()
            board.push(move)
            history.append(self.getHistKey(board))
            score = -self.search(board, -beta, -a, depth-1, threadID, history)
            
            history.pop(-1)
            board.pop()
            #keep track of the best move 
            if score > best:
                best = score
                bestmove = move
                a = max(a,best)
            
            # if the score is bigger than gamma return immediately to restart with the new bounds 
            if score > beta: 
                break

            #if other threads are available to help, make a splitnode from this position and split the work
            #print(self.ThreadAccess.locked())
            if depth >=1 and len >2 and self.ThreadAccess.acquire(False):
                #print("lock acquired")
                IDs = self.idle_threads(threadID, len-1)
                #print(IDs)
                self.ThreadAccess.release()
                if(IDs):
                    t_split_start = time()
                    s = SplitPoint(board,moveList,depth,a,beta, threadID, bestmove, history, best)
                    self.split(s, IDs, t_split_start)
                    if s.alpha <= alpha:
                        self.TT_Score[TTkey] = Score_Entry(UPPER, s.best, s.bestmove, depth)
                    elif s.alpha >= beta:
                        self.TT_Score[TTkey] = Score_Entry(LOWER, s.best, s.bestmove, depth)
                    else: self.TT_Score[TTkey] = Score_Entry(EXACT, s.best, s.bestmove, depth)
                    return s.best #??? should be right, just break should return the old alpha and disregard the splitsearch
                
                #print("lock released")
        self.stabilitycheck(best, alpha, beta)
        if best <= alpha:
            self.TT_Score[TTkey] = Score_Entry(UPPER, best, bestmove, depth)
        elif best >= beta:
            self.TT_Score[TTkey] = Score_Entry(LOWER, best, bestmove, depth)
        else: self.TT_Score[TTkey] = Score_Entry(EXACT, best, bestmove, depth)
        return best

    def stabilitycheck(self,score,alpha,beta):
        if score < alpha or score > beta:
            self.stability = False
    def getTTKey(self, board, repetition):
        fen = board.fen()
        fenparts = fen.split(" ")
        fenparts[5]= ""
        key = "".join(fenparts)
        if repetition:
            return key+"1"
        else: 
            return key
    
    def getHistKey(self, board):
        return board.fen().split(" ")[0]
    #  Split assignes work to all the idle threads 
    def split(self, splitpoint, IDs, starttime):
        self.splits +=1
        #print(f"split work to these threads {IDs}")
        for threadID in IDs :
            self.assign_work(threadID, splitpoint)
        self.assign_work(splitpoint.masterID, splitpoint)
        self.t_split[splitpoint.masterID] += time() - starttime
        self.idle_loop(splitpoint.masterID, splitpoint)

    #retruns how often a position has occured in the current game history
    def getOccurrences(self, board, history):
        Occ = 0
        bHistkey = self.getHistKey(board)
        for histkey in history:
            if bHistkey == histkey:
                Occ +=1
        return Occ


    # idle_loop in which threads go when they have nothing to do,
    # constantly looking for new assigned work 
    def idle_loop(self, threadID, splitpoint):
        if splitpoint is None:
            self.set_idle(threadID,None)
            #print(f"thread {threadID} idle")
        while True:

            if splitpoint is not None:
                try:
                    work = self.work_queues[threadID].get(False,timeout=0.0001)
                    self.splitpoint_search(work, threadID)
                except queue.Empty:
                    pass
                if splitpoint.activeThreadCount == 0:
                    break
            else:
                work = self.work_queues[threadID].get()
                if work is FINISHED:
                    break
                self.splitpoint_search(work, threadID)
                self.reset_stop(threadID)
                self.set_idle(threadID, splitpoint)




    def set_idle(self, threadID, splitpoint):
        #self.ThreadAccess.acquire()
        #print("Lock acquired")
        self.IdleThreads[threadID] = (threadID,splitpoint)
        #self.ThreadAccess.release()
        #print("lock released")

    def assign_work(self, threadID, splitpoint):
        splitpoint.slaveIDs.append(threadID)
        self.work_queues[threadID].put(splitpoint)
    #checks the list of ilde threads for viable slaves for ThreadID
    # removes all viable ones from the list and returns their ids to split() 
    # idle masters of splitpoints can only help their own slaves "helpful master concept"
    def idle_threads(self, threadID, maxnum):
        viableIDs = []
        #print(f"idle_threads: {self.IdleThreads}")
        for i in range(self.threads):
            if maxnum==0: break
            tuple = self.IdleThreads[i]
            if tuple:
                if tuple[1]:
                    if threadID in tuple[1].slaveIDs:
                        self.IdleThreads[i] = None
                        viableIDs.append(i)
                        maxnum -=1
                else:
                    self.IdleThreads[i] = None
                    viableIDs.append(i)
                    maxnum-=1
        #print(viableIDs)
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
        history = splitpoint.history.copy()
        # search new moves until the list is empty
        while splitpoint.moveList and not self.told_to_stop(threadID):
            move = self.pickMove(splitpoint)
            if move is None: break
            newBoard = splitpoint.board.copy()
            newBoard.push(move)
            history.append(self.getHistKey(newBoard))
            score = -self.search(newBoard,-splitpoint.beta, -splitpoint.alpha, splitpoint.depth -1, threadID, history)
            history.pop(-1)
            newBoard.pop()
            splitpoint.boundsLock.acquire()
            if score > splitpoint.best:
                splitpoint.best = score
                splitpoint.alpha = max(splitpoint.alpha, splitpoint.best)
                splitpoint.bestmove = move
            splitpoint.boundsLock.release()
            if score >= splitpoint.beta:
                self.tell_all_threads_stop(splitpoint)
                break

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
    def stop_helpers(self):
        for i in range(1,self.threads):
            self.work_queues[i].put(FINISHED)

#testscript when this file is run by itself and not imported
if __name__ == "__main__":
    #yappi.start()
    board = chess.variant.CrazyhouseBoard()
    movelist = None
    netapi = NeuralNetAPI(ctx="cpu",batch_size=1)
    s = Searcher(2,1,netapi)
    lastscore = 0
    pushmove = None
    while not lastscore ==1:
        for depth, move, score, searchtime, nodes in s.searchPosition(board.fen(),movelist):
            print(f"depth: {depth}, selected move: {move}, score: {score}, time needed: {searchtime}, nodes searched: {nodes}")
            print(f"t_NN_eval: {s.t_NN_eval},\nt_TT_NN: {s.t_TT_NN},\nt_TT_Score: {s.t_TT_score}")
            print(f"Transposition hits NNeval: {s.Tablehits_NN}, Transposition hits Score: {s.Tablehits_Score}, hits >= depth {s.Tablehits_Score_greater_depth}")
            print(f"splits: {s.splits}, splitting times: {s.t_split}")
            print("\n")
       # for i in range(1,depth+1):
        #    print(f"average branches at depth {i} was {s.theoretical_nodes[i][0]}")
        #print(f"stability is {s.stability}")
            pushmove = move
            lastscore = score
            if depth == 6: break
        board.push(pushmove)
        print(board.move_stack)
    s.stop_helpers()
    #threads = yappi.get_thread_stats()
    #for thread in threads:
    #    print(f"stats for yappi-thread: {thread.id}")
    #    yappi.get_func_stats(ctx_id=thread.id).print_all()