import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import py333
from random import randint
import numpy as np
import tensorflow as tf
import sys
from scipy.sparse import coo_matrix
import collections
import math
import gc
from CubeModel import buildModel, compileModel
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import MCTS
import constants
import time

tf.get_logger().setLevel("ERROR")

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

# TODO: Add the loss weight to each sample?
def generateSamples(k, l):
    N = k * l
    samples = np.empty((N, constants.kNumStickers), dtype=bytes)
    states = np.empty((N, constants.kNumCubes * constants.kNumStickers))
    for i in range(l):
        currentCube = py333.initState()
        for j in range(k):
            scrambledCube = py333.doAlgStr(currentCube, getRandomMove())
            samples[k * i + j] = scrambledCube
            states[k * i + j] = py333.getState(scrambledCube).flatten()
            currentCube = scrambledCube
    return samples, coo_matrix(states)

def reward(cube):
    return 1 if py333.isSolved(cube, True) else -1

def evaluateSolveRate(model, numCubes, maxScramble):
    results = {}
    for dist in range(1, maxScramble + 1):
        solved = 0
        for _ in range(numCubes):
            cube = py333.createScrambledCube(dist)
            result, _, _ = MCTS.solveSingleCubeGreedy(model, cube, 6 * dist + 1)
            if result:
                solved += 1
        results[dist] = solved / numCubes
    return results

def doADI(k, l, M):
    model = buildModel(constants.kNumStickers * constants.kNumCubes)
    compileModel(model, constants.kLearningRate)
    numMoves = len(moves)
    stateSize = constants.kNumStickers * constants.kNumCubes
    os.makedirs(constants.kCheckpointDir, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(constants.kLogDir, run_id)
    writer = tf.summary.create_file_writer(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    for iterNum in range(M):
        t0 = time.time()
        samples, _ = generateSamples(k, l)
        N = len(samples)
        t_sample = time.time() - t0

        t0 = time.time()
        rewards_arr = np.empty(N * numMoves)
        childStates = np.empty((N * numMoves, stateSize))
        for i, sample in enumerate(samples):
            for j, move in enumerate(moves):
                child = py333.doAlgStr(sample, move)
                idx = i * numMoves + j
                childStates[idx] = py333.getState(child).flatten()
                rewards_arr[idx] = reward(child)

        values, _ = model.predict(childStates, batch_size=N * numMoves, verbose=0)
        values = values.flatten() + rewards_arr
        values = values.reshape(N, numMoves)

        optimalVals = values.max(axis=1, keepdims=True)
        optimalPolicies = values.argmax(axis=1).astype(np.int32)

        states = np.empty((N, stateSize))
        for i, sample in enumerate(samples):
            states[i] = py333.getState(sample).flatten()
        t_target = time.time() - t0

        t0 = time.time()
        history = model.fit(states, {"PolicyOutput": optimalPolicies,
                           "ValueOutput": optimalVals}, epochs=constants.kNumMaxEpochs,
                           verbose=0, steps_per_epoch=1)
        t_train = time.time() - t0

        val_loss = history.history.get("ValueOutput_loss", [None])[-1]
        pol_loss = history.history.get("PolicyOutput_loss", [None])[-1]
        total_loss = history.history.get("loss", [None])[-1]

        with writer.as_default():
            tf.summary.scalar("loss/total", total_loss, step=iterNum)
            if val_loss is not None:
                tf.summary.scalar("loss/value", val_loss, step=iterNum)
            if pol_loss is not None:
                tf.summary.scalar("loss/policy", pol_loss, step=iterNum)
            tf.summary.scalar("targets/mean_optimal_value", optimalVals.mean(), step=iterNum)
            tf.summary.scalar("targets/std_optimal_value", optimalVals.std(), step=iterNum)
            tf.summary.scalar("timing/sample_gen_s", t_sample, step=iterNum)
            tf.summary.scalar("timing/target_comp_s", t_target, step=iterNum)
            tf.summary.scalar("timing/training_s", t_train, step=iterNum)

        print(f"[iter {iterNum}] loss: {total_loss:.4f} (val: {val_loss:.4f}, pol: {pol_loss:.4f}) | "
              f"time: {t_sample+t_target+t_train:.3f}s")

        if (iterNum + 1) % constants.kEvalInterval == 0:
            solve_rates = evaluateSolveRate(model, constants.kEvalNumCubes, constants.kEvalMaxScramble)
            with writer.as_default():
                for dist, rate in solve_rates.items():
                    tf.summary.scalar(f"solve_rate/scramble_{dist}", rate, step=iterNum)
            rates_str = " ".join(f"d{d}={r:.0%}" for d, r in solve_rates.items())
            print(f"[iter {iterNum}] solve rates: {rates_str}")

        if (iterNum + 1) % constants.kCheckpointInterval == 0 or iterNum == M - 1:
            path = os.path.join(constants.kCheckpointDir, f"model_iter_{iterNum+1}.keras")
            model.save(path)
            print(f"[iter {iterNum}] checkpoint saved: {path}")

        writer.flush()
        gc.collect()
    writer.close()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rubik's Cube ADI Training & Solving")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train a new model")
    train_p.add_argument("--k", type=int, default=30, help="Scramble depth per trajectory (default: 30)")
    train_p.add_argument("--l", type=int, default=10, help="Number of trajectories per iter (default: 10)")
    train_p.add_argument("--M", type=int, default=1000, help="Number of ADI iterations (default: 1000)")
    train_p.add_argument("--model", default="default", help="Model save path prefix (default: constants.kModelPath)")

    solve_p = sub.add_parser("solve", help="Restore model and solve cubes")
    solve_p.add_argument("--model", default="default", help="Model path prefix to restore")
    solve_p.add_argument("--strategy", choices=["greedy", "vanillamcts", "fullmcts"], default="greedy")
    solve_p.add_argument("--cubes", type=int, default=50, help="Number of cubes to test")
    solve_p.add_argument("--max-dist", type=int, default=7, help="Max scramble distance")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    model_prefix = args.model
    if model_prefix == "default":
        model_prefix = constants.kModelPath

    if args.command == "train":
        print(f"Training: k={args.k}, l={args.l}, M={args.M} ({args.k * args.l} samples/iter)")
        model = doADI(k=args.k, l=args.l, M=args.M)
        save_path = f"{model_prefix}.keras"
        model.save(save_path)
        print(f"Model saved: {save_path}")

    elif args.command == "solve":
        model = load_model(f"{model_prefix}.keras")
        print(f"Model restored from {model_prefix}")
        if args.strategy == "greedy":
            MCTS.simulateCubeSolvingGreedy(model, numCubes=args.cubes, maxSolveDistance=args.max_dist)
        elif args.strategy == "vanillamcts":
            MCTS.simulateCubeSolvingVanillaMCTS(model, numCubes=args.cubes, maxSolveDistance=args.max_dist)
        elif args.strategy == "fullmcts":
            MCTS.simulateCubeSolvingFullMCTS(model, numCubes=args.cubes, maxSolveDistance=args.max_dist)



