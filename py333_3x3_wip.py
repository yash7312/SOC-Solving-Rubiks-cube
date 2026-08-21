import numpy as np
from random import randint
'''


sticker indices:
          ┌──┬──┬──┐
          │ 0│ 1│ 2│
          ├──┼──┤──┤
          │ 3│ 4│ 5│
          ├──┼──┤──┤
          │ 6│ 7│ 8│
 ┌──┬──┬──┼──┼──┼──┼──┬──┬──┬──┬──┬──┐
 │36│37│38│18│19│20│ 9│10│11│45│46│47│
 ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤──┤
 │39│40│41│21│22│23│12│13│14│48│49│50│
 ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤──┤
 │42│43│44│24│25│26│15│16│17│51│52│53│
 └──┴──┴──┼──┼──┼──┼──┴──┴──┴──┴──┴──┘
          │27│28│29│
          ├──┼──┼──┤
          │30│31│32│
          ├──┼──┤──┤
          │33│34│35│
          └──┴──┴──┘

face colors:
    ┌──┐
    │ 0│
 ┌──┼──┼──┬──┐
 │ 4│ 2│ 1│ 5│
 └──┼──┼──┴──┘
    │ 3│
    └──┘

moves:
[ U , U', U2, R , R', R2, F , F', F2, D , D', D2, L , L', L2, B , B', B2, x , x', x2, y , y', y2, z , z', z2]






'''
# move definition for each move we can do
moveDefs = np.array[1]

moveInds = { \
  "U": 0, "U'": 1, "U2": 2, "R": 3, "R'": 4, "R2": 5, "F": 6, "F'": 7, "F2": 8, \
  "D": 9, "D'": 10, "D2": 11, "L": 12, "L'": 13, "L2": 14, "B": 15, "B'": 16, "B2": 17, \
  "x": 18, "x'": 19, "x2": 20, "y": 21, "y'": 22, "y2": 23, "z": 24, "z'": 25, "z2": 26 \
}

# corner information, piece definition
# pieceDefs = np.array([ \
#   [  0, 21, 16], \
#   [  2, 17,  8], \
#   [  3,  9,  4], \
#   [  1,  5, 20], \
#   [ 12, 10, 19], \
#   [ 13,  6, 11], \
#   [ 15, 22,  7], \
# ])

# pieceInds = np.zeros([58, 2], dtype=np.int)
# pieceInds[50] = [0, 0]; pieceInds[54] = [0, 1]; pieceInds[13] = [0, 2]
# pieceInds[28] = [1, 0]; pieceInds[42] = [1, 1]; pieceInds[ 8] = [1, 2]
# pieceInds[14] = [2, 0]; pieceInds[21] = [2, 1]; pieceInds[ 4] = [2, 2]
# pieceInds[52] = [3, 0]; pieceInds[15] = [3, 1]; pieceInds[11] = [3, 2]
# pieceInds[47] = [4, 0]; pieceInds[30] = [4, 1]; pieceInds[40] = [4, 2]
# pieceInds[25] = [5, 0]; pieceInds[18] = [5, 1]; pieceInds[35] = [5, 2]
# pieceInds[23] = [6, 0]; pieceInds[57] = [6, 1]; pieceInds[37] = [6, 2]

# pieceCols = np.zeros([7, 3, 3], dtype=np.int)
# pieceCols[0, 0, :] = [0, 5, 4]; pieceCols[0, 1, :] = [4, 0, 5]; pieceCols[0, 2, :] = [5, 4, 0]
# pieceCols[1, 0, :] = [0, 4, 2]; pieceCols[1, 1, :] = [2, 0, 4]; pieceCols[1, 2, :] = [4, 2, 0]
# pieceCols[2, 0, :] = [0, 2, 1]; pieceCols[2, 1, :] = [1, 0, 2]; pieceCols[2, 2, :] = [2, 1, 0]
# pieceCols[3, 0, :] = [0, 1, 5]; pieceCols[3, 1, :] = [5, 0, 1]; pieceCols[3, 2, :] = [1, 5, 0]
# pieceCols[4, 0, :] = [3, 2, 4]; pieceCols[4, 1, :] = [4, 3, 2]; pieceCols[4, 2, :] = [2, 4, 3]
# pieceCols[5, 0, :] = [3, 1, 2]; pieceCols[5, 1, :] = [2, 3, 1]; pieceCols[5, 2, :] = [1, 2, 3]
# pieceCols[6, 0, :] = [3, 5, 1]; pieceCols[6, 1, :] = [1, 3, 5]; pieceCols[6, 2, :] = [5, 1, 3]

# hashOP = np.array([1, 2, 10])
# pow3 = np.array([1, 3, 9, 27, 81, 243, 729])
# pow7 = np.array([1, 7, 49, 343, 2401, 16807, 117649])
# fact6 = np.array([720, 120, 24, 6, 2, 1, 1])

moves = ['F', 'F\'', 'B', 'B\'', 'R', 'R\'', 'L', 'L\'', 'D', 'D\'', 'U', 'U\'']

def getRandomMove():
    return moves[randint(0, len(moves) - 1)]

# # get FC-normalized solved state
# def initState():
#   return np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x'])
#   #return np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])


def doMove(s, move):
  return s[moveDefs[move]]

def generateCubeletMappings():
    cubeletMappings = {}
    # corner edges being marked
    cubeletMappings[0] = 0
    # cubeletMappings[16] = 0
    # cubeletMappings[21] = 0
    # cubeletMappings[1] = 1
    # cubeletMappings[5] = 1
    # cubeletMappings[20] = 1
    # cubeletMappings[2] = 2
    # cubeletMappings[8] = 2
    # cubeletMappings[17] = 2
    # cubeletMappings[3] = 3
    # cubeletMappings[9] = 3
    # cubeletMappings[4] = 3
    # cubeletMappings[18] = 4
    # cubeletMappings[14] = 4
    # cubeletMappings[23] = 4
    # cubeletMappings[15] = 5
    # cubeletMappings[7] = 5
    # cubeletMappings[22] = 5
    # cubeletMappings[19] = 6
    # cubeletMappings[10] = 6
    # cubeletMappings[12] = 6
    # cubeletMappings[6] = 7
    # cubeletMappings[11] = 7
    # cubeletMappings[13] = 7
    return cubeletMappings

cubeletMappings = generateCubeletMappings()



def getState(cube):
  state = np.zeros((8, 24)) # TODO: constants
  for i in range(24):
    state[cubeletMappings[i]][ord(cube[i]) - ord('a')] = 1
  return state

# apply a string sequence of moves to a state
def doAlgStr(s, alg):
  moves = alg.split(" ")
  for m in moves:
    if m in moveInds:
      s = doMove(s, moveInds[m])
  return s

def getNumerical(cube):
  return (np.array(list(map(ord, cube))) - ord('a')) // 4

# check if state is solved
def isSolved(s, convert=False):
  if convert:
    s = getNumerical(s)
  for i in range(6):
    if not (s[4 * i:4 * i + 4] == s[4 * i]).all():
      return False
  return True

# normalize stickers relative to a fixed DLB corner
def normFC(s):
  normCols = np.zeros(6, dtype=np.int32)
#   print(normCols)
#   print(s)
  normCols[s[18] - 3] = 1
  normCols[s[23] - 3] = 2
  normCols[s[14]] = 3
  normCols[s[18]] = 4
  normCols[s[23]] = 5
  return normCols[s]

# get OP representation given FC-normalized sticker representation
def getOP(s):
  return pieceInds[np.dot(s[pieceDefs], hashOP)]

# get sticker representation from OP representation
def getStickers(sOP):
  s = np.zeros(24, dtype=np.int)
  s[[14, 18, 23]] = [3, 4, 5]
  for i in range(7):
    s[pieceDefs[i]] = pieceCols[sOP[i, 0], sOP[i, 1], :]
  return s

# get a unique index for the piece orientation state (0-2186)
def indexO(sOP):
  return np.dot(sOP[:, 1], pow3)

# get a unique index for the piece permutation state (0-823542)
def indexP(sOP):
  return np.dot(sOP[:, 0], pow7)

# get a (gap-free) unique index for the piece permutation state (0-5039)
def indexP2(sOP):
  ps = np.arange(7)
  P = 0
  for i, p in enumerate(sOP[:, 0]):
    P += fact6[i] * np.where(ps == p)[0][0]
    ps = ps[ps != p]
  return P

# get a unique index for the piece orientation and permutation state (0-11017439)
def indexOP(sOP):
  return indexO(sOP) * 5040 + indexP2(sOP)

def createScrambledCube(numScrambles):
    cube = initState()
    for i in range(numScrambles):
        cube = doAlgStr(cube, getRandomMove())
    return cube

# print state of the cube
def printCube(s):
  print("      ┌──┬──┐")
  print("      │ {}│ {}│".format(s[0], s[1]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[2], s[3]))
  print("┌──┬──┼──┼──┼──┬──┬──┬──┐")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[16], s[17], s[8], s[9], s[4], s[5], s[20], s[21]))
  print("├──┼──┼──┼──┼──┼──┼──┼──┤")
  print("│ {}│ {}│ {}│ {}│ {}│ {}│ {}│ {}│".format(s[18], s[19], s[10], s[11], s[6], s[7], s[22], s[23]))
  print("└──┴──┼──┼──┼──┴──┴──┴──┘")
  print("      │ {}│ {}│".format(s[12], s[13]))
  print("      ├──┼──┤")
  print("      │ {}│ {}│".format(s[14], s[15]))
  print("      └──┴──┘")

if __name__ == "__main__":
  # get solved state
  s = initState()
  printCube(getNumerical(s))

  s = doAlgStr(s, "U")
  printCube(s)
  # do some moves
  s = doAlgStr(s, "x y R U' R' U' F2 U' R U R' U F2")
  printCube(s)
  # normalize stickers relative to DLB
  s = normFC(s)
  printCube(s)
