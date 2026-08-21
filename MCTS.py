import py333
import numpy as np
import math
import constants
import multiprocessing

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']
NUM_MOVES = len(moves)
STATE_SIZE = constants.kNumStickers * constants.kNumCubes

def infer(model, state_array):
    outputs = model(state_array, training=False)
    return outputs[0].numpy(), outputs[1].numpy()

def _cube_key(cube):
    return cube.tobytes()

def _get_children(cube):
    children = []
    child_states = np.empty((NUM_MOVES, STATE_SIZE))
    rewards = np.empty(NUM_MOVES)
    for j, move in enumerate(moves):
        child = py333.doAlgStr(cube, move)
        children.append(child)
        child_states[j] = py333.getState(child).flatten()
        rewards[j] = 1 if py333.isSolved(child, True) else -1
    return children, child_states, rewards

def reward(cube):
    return 1 if py333.isSolved(cube, True) else -1

# ── Greedy ──

def solveSingleCubeGreedy(model, cube, maxMoves):
    movePath = []
    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py333.isSolved(cube, convert=True):
            return True, numMovesTaken, movePath
        state = np.array([py333.getState(cube).flatten()])
        _, policies = infer(model, state)
        bestMove = policies[0].argmax()
        movePath.append(moves[bestMove])
        cube = py333.doAlgStr(cube, moves[bestMove])
        numMovesTaken += 1
    return False, maxMoves + 1, movePath

# ── Vanilla MCTS ──

def solveSingleCubeVanillaMCTS(model, cube, maxMoves, maxDepth):
    movePath = []
    numMovesTaken = 0
    q = {}
    counts = {}
    while numMovesTaken <= maxMoves:
        if py333.isSolved(cube, convert=True):
            return True, numMovesTaken, movePath
        bestMoveIdx = selectActionVanillaMCTS(model, cube, maxDepth, q, counts)
        movePath.append(moves[bestMoveIdx])
        cube = py333.doAlgStr(cube, moves[bestMoveIdx])
        numMovesTaken += 1
    return False, maxMoves + 1, movePath

def selectActionVanillaMCTS(model, state, depth, q, counts):
    stateKey = _cube_key(state)
    seenStates = set()
    for _ in range(constants.kMCTSSimulateIterations):
        simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateKey)
    allVals = np.array([q[stateKey][i] for i in range(NUM_MOVES)])
    return allVals.argmax()

def simulateVanillaMCTS(model, state, depth, q, counts, seenStates, stateKey):
    if depth == 0:
        return 0
    if stateKey not in seenStates:
        _, child_states, rewards = _get_children(state)
        values, _ = infer(model, child_states)
        values = values.flatten() + rewards
        q[stateKey] = {}
        counts[stateKey] = {}
        for i in range(NUM_MOVES):
            q[stateKey][i] = values[i]
            counts[stateKey][i] = 1
        seenStates.add(stateKey)
        return rolloutVanillaMCTS(model, state, depth)
    totalStateCounts = sum(counts[stateKey][i] for i in range(NUM_MOVES))
    allQuantities = np.array([
        q[stateKey][i] + constants.kMCTSExploration * math.sqrt(math.log(totalStateCounts) / counts[stateKey][i])
        for i in range(NUM_MOVES)
    ])
    bestActionIndex = allQuantities.argmax()
    nextState = py333.doAlgStr(state, moves[bestActionIndex])
    r = reward(nextState)
    newQ = r + constants.kDiscountFactor * simulateVanillaMCTS(model, nextState, depth - 1, q, counts, seenStates, _cube_key(nextState))
    counts[stateKey][bestActionIndex] += 1
    q[stateKey][bestActionIndex] += (newQ - q[stateKey][bestActionIndex]) / counts[stateKey][bestActionIndex]
    return newQ

def rolloutVanillaMCTS(model, cube, depth):
    if depth == 0:
        return 0
    state = np.array([py333.getState(cube).flatten()])
    _, policies = infer(model, state)
    actionIndex = selectActionSoftmax(policies)
    nextState = py333.doAlgStr(cube, moves[actionIndex])
    r = reward(nextState)
    return r + constants.kDiscountFactor * rolloutVanillaMCTS(model, nextState, depth - 1)

def selectActionSoftmax(probabilities):
    probs = probabilities[0]
    weights = np.exp(constants.kLambda * probs)
    totals = np.cumsum(weights)
    throw = np.random.rand() * totals[-1]
    return np.searchsorted(totals, throw)

# ── Full MCTS (AlphaGo-style) ──

def solveSingleCubeFullMCTS(model, cube, maxMoves):
    simulatedPath = []
    simulatedActions = []
    treeStates = set()
    seenStates = set()
    currentCube = cube
    currentKey = _cube_key(cube)
    counts = {}
    maxVals = {}
    priorProbs = {}
    virtualLosses = {}
    movePath = []

    state = np.array([py333.getState(currentCube).flatten()])
    _, probs = infer(model, state)
    _initStateVals(currentKey, counts, maxVals, priorProbs, virtualLosses, probs[0])
    seenStates.add(currentKey)
    simulatedPath.append(currentKey)

    numMovesTaken = 0
    while numMovesTaken <= maxMoves:
        if py333.isSolved(currentCube, convert=True):
            return True, numMovesTaken, movePath
        if currentKey not in treeStates:
            children, child_states_batch, _ = _get_children(currentCube)
            unseen_indices = []
            unseen_keys = []
            for j, child in enumerate(children):
                ck = _cube_key(child)
                if ck not in seenStates:
                    unseen_indices.append(j)
                    unseen_keys.append(ck)

            if unseen_indices:
                unseen_states = child_states_batch[unseen_indices]
                _, unseen_probs = infer(model, unseen_states)
                for idx, ck in enumerate(unseen_keys):
                    _initStateVals(ck, counts, maxVals, priorProbs, virtualLosses, unseen_probs[idx])
                    seenStates.add(ck)

            cur_state = np.array([py333.getState(currentCube).flatten()])
            value, _ = infer(model, cur_state)
            value = value[0][0]
            for i, pathKey in enumerate(simulatedPath):
                if i < len(simulatedActions):
                    act = simulatedActions[i]
                    maxVals[pathKey][act] = max(maxVals[pathKey][act], value)
                    counts[pathKey][act] += 1
                    virtualLosses[pathKey][act] -= constants.kVirtualLoss
            treeStates.add(currentKey)
        else:
            totalStateCounts = sum(counts[currentKey][i] for i in range(NUM_MOVES))
            actionVals = np.array([
                (maxVals[currentKey][i] - virtualLosses[currentKey][i]) +
                constants.kMCTSExploration * priorProbs[currentKey][i] * math.sqrt(totalStateCounts) / (1 + counts[currentKey][i])
                for i in range(NUM_MOVES)
            ])
            bestIdx = actionVals.argmax()
            virtualLosses[currentKey][bestIdx] += constants.kVirtualLoss
            simulatedActions.append(bestIdx)
            movePath.append(moves[bestIdx])
            currentCube = py333.doAlgStr(currentCube, moves[bestIdx])
            currentKey = _cube_key(currentCube)
            simulatedPath.append(currentKey)
            numMovesTaken += 1
    return False, maxMoves + 1, movePath

def _initStateVals(stateKey, counts, maxVals, priorProbs, virtualLosses, probs):
    counts[stateKey] = {}
    maxVals[stateKey] = {}
    priorProbs[stateKey] = {}
    virtualLosses[stateKey] = {}
    for i in range(NUM_MOVES):
        counts[stateKey][i] = 0
        maxVals[stateKey][i] = 0
        virtualLosses[stateKey][i] = 0
        priorProbs[stateKey][i] = probs[i]

# ── Parallel MCTS ──

_worker_model = None

def _workerInit(model_path):
    global _worker_model
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import load_model
    _worker_model = load_model(model_path)
    infer(_worker_model, np.zeros((1, STATE_SIZE)))

def _solveWorker(args):
    cube_bytes, cube_dtype, strategy, maxMoves, maxDepth, simIters = args
    cube = np.frombuffer(cube_bytes, dtype=cube_dtype).copy()
    if simIters is not None:
        constants.kMCTSSimulateIterations = simIters
    if strategy == "greedy":
        return solveSingleCubeGreedy(_worker_model, cube, maxMoves)
    elif strategy == "vanillamcts":
        return solveSingleCubeVanillaMCTS(_worker_model, cube, maxMoves, maxDepth)
    elif strategy == "fullmcts":
        return solveSingleCubeFullMCTS(_worker_model, cube, maxMoves)

def solveParallel(model_path, cubes, strategy="greedy", maxMoves=20, maxDepth=1, numWorkers=4):
    simIters = constants.kMCTSSimulateIterations
    tasks = [
        (cube.tobytes(), cube.dtype.str, strategy, maxMoves, maxDepth, simIters)
        for cube in cubes
    ]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=numWorkers, initializer=_workerInit, initargs=(model_path,)) as pool:
        results = pool.map(_solveWorker, tasks)
    return results

# ── Benchmark helpers ──

def simulateCubeSolvingGreedy(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance + 1)
    for dist in range(maxSolveDistance + 1):
        numSolved = 0
        for _ in range(numCubes):
            scrambledCube = py333.createScrambledCube(dist)
            result, numMoves, path = solveSingleCubeGreedy(model, scrambledCube, 6 * dist + 1)
            if result:
                numSolved += 1
        data[dist] = numSolved / numCubes
        print(f"  d={dist}: {numSolved}/{numCubes} ({data[dist]:.0%})")
    print(f"Overall: {data}")

def simulateCubeSolvingVanillaMCTS(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance + 1)
    solveLengths = []
    for dist in range(maxSolveDistance + 1):
        numSolved = 0
        for _ in range(numCubes):
            scrambledCube = py333.createScrambledCube(dist)
            result, numMoves, path = solveSingleCubeVanillaMCTS(model, scrambledCube, 6 * dist + 1, 1)
            if result:
                solveLengths.append(numMoves)
                numSolved += 1
        data[dist] = numSolved / numCubes
        print(f"  d={dist}: {numSolved}/{numCubes} ({data[dist]:.0%})")
    print(f"Overall: {data}")
    if solveLengths:
        solveLengths.sort()
        print(f"Median solve length: {solveLengths[len(solveLengths)//2]}")

def simulateCubeSolvingFullMCTS(model, numCubes, maxSolveDistance):
    data = np.zeros(maxSolveDistance + 1)
    for dist in range(maxSolveDistance + 1):
        numSolved = 0
        for _ in range(numCubes):
            scrambledCube = py333.createScrambledCube(dist)
            result, numMoves, path = solveSingleCubeFullMCTS(model, scrambledCube, 20 * dist + 1)
            if result:
                numSolved += 1
        data[dist] = numSolved / numCubes
        print(f"  d={dist}: {numSolved}/{numCubes} ({data[dist]:.0%})")
    print(f"Overall: {data}")
