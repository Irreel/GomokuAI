import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import re
import itertools
import copy
from queue import PriorityQueue
from collections import Counter

pp.infotext = 'name="pbrain-abpruning", version="1.0", country="China"'
global MAX_EXPAND
MAX_EXPAND = 78
MAX_BOARD = 20
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]

opposite_player = lambda x: 3 - x if (0 < x < 3) else None

def probable_position(board):
    """
    :param
        board: 当前棋盘状态
    :return: 状态为free且四周有棋子的位置列表
    """
    height, width = len(board), len(board[0])
    probable_list = []
    scale = 1
    for (pos_x, pos_y) in itertools.product(range(width), range(height)):
        if not board[pos_x][pos_y] == 0:
            continue
        for (i,j) in itertools.product(range(2 * scale + 1), range(2 * scale + 1)):
            x, y = pos_x + i - scale, pos_y + j - scale
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if not board[x][y] == 0:
                probable_list.append((pos_x, pos_y))
                break
    if probable_list == []:
        return None
    return probable_list

# def renew_probable_position(action, probable_list, board):
#     """
#     :param
#         action: 该步落子动作
#         probable_list: 当前未更新的位置列表
#         board: 当前棋盘状态
#     :return: 更新的位置列表
#     """
#     x, y = action[0], action[1]
#     height, width = len(board), len(board[0])
#     scale = 1
#
#     for (i, j) in itertools.product(range(2 * scale + 1), range(2 * scale + 1)):
#         new_x = x + i - scale
#         new_y = y + j - scale
#
#         if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height:
#             if board[new_x][new_y] == 0:
#                 if (new_x, new_y) not in probable_list:
#                     probable_list.append((new_x, new_y))
#
#     if (x, y) in probable_list:
#         probable_list.remove((x, y))
#
#     return probable_list

def renew_probable_position(action, probable_list):
    """
    renew the probable list
    :param
        action: the position AI or player put at
        probable_list: the list needed to be renewed
    :returns
        a new list
    """
    x, y = action[0], action[1]
    scale = 1

    for (i, j) in itertools.product(range(2 * scale + 1), range(2 * scale + 1)):
        new_x = x + i - scale
        new_y = y + j - scale
        if (new_x, new_y) not in probable_list:
            probable_list.append((new_x, new_y))

    if (x, y) in probable_list:
        probable_list.remove((x, y))

    return probable_list


def chess_type():
    return {"11111": "Long",  # 长连：取得胜利
              "011110": "H4",  # 活四：有两个连五点

              # 冲四：有一个连五点
              "011112": "C4", "211110": "C4", "10111": "C4", "11101": "C4", "11011": "C4",

              # 活三：可以形成活四
              "011100": "H3", "001110": "H3", "010110": "H3", "011010": "H3",

              # 眠三：只能形成冲四
              "001112": "M3", "010112": "M3", "011012": "M3",
              "211100": "M3", "211010": "M3", "210110": "M3",
              "10011": "M3", "11001": "M3", "10101": "M3", "2011102": "M3",

              # 活二：能形成活三
              "001100": "H2", "010100": "H2", "011000": "H2",
              "001010": "H2", "000110": "H2", "010010": "H2",

              # 眠二：能形成眠三
              "000112": "M2", "001012": "M2", "010012": "M2",
              "211000": "M2", "210100": "M2", "210010": "M2",
              "2001102": "M2", "2011002": "M2", "2010102": "M2",

              "211112": "S4",  # 死四：两头都被堵死的四
              "21112": "S3",   # 死三：两头都被堵死的三
              "2112": "S2",    # 死二：两头都被堵死的二
              "0210": "Other", "0120": "Other"}


def combchesstype(typecount):
    '''
    根据每种棋形的个数确定组合棋形，输入参数为一个字典typecount
    e.g. typecount = {"Long":0, "H4":1, "C4":0, "H3":0, "M3":0, "H2":0, "M2":0, "S4":0, "S3":0, "S2":0, "Other":0}
    '''
    if typecount["Long"]:
        return "Long"
    if typecount["H4"]:
        return "H4"
    if typecount["C4"]:
        if typecount["C4"] > 1:
            return "Double_C4"
        if typecount["H3"]:
            return "C4_H3"
        else:
            return "C4"
    if typecount["H3"]:
        if typecount["H3"] > 1:
            return "Double_H3"
        if typecount["M3"]:
            return "H3_M3"
        else:
            return "H3"
    if typecount["M3"]:
        return "M3"
    if typecount["H2"]:
        if typecount["H2"] > 1:
            return "Double_H2"
        if typecount["M2"]:
            return "H2_M2"
        else:
            return "H2"
    if typecount["M2"]:
        return "M2"
    if typecount["Other"]:
        return "Other"
    if typecount["S4"]:
        return "S4"
    if typecount["S3"]:
        return "S3"
    if typecount["S2"]:
        return "S2"
    return "Notype"

# def score_map():
#     return {"Long": 100000,
#              "H4": 10000,
#              "Double_C4": 10000,
#              "C4_H3": 10000,
#              "Double_H3": 2000,
#              "H3_M3": 1000,
#              "C4": 5000,
#              "H3": 200,
#              "Double_H2": 100,
#              "M3": 50,
#              "H2_M2": 10,
#              "H2": 5,
#              "M2": 3,
#              "Other": 1,
#              "S4": -5,
#              "S3": -5,
#              "S2": -5,
#              "Notype": 0}

# def score_map():
#     return {"Long": 100000,
#              "H4": 10000,
#              "C4": 5000,
#              "H3": 200,
#              "M3": 50,
#              "H2": 5,
#              "M2": 3,
#              "Other": 1,
#              "S4": -5,
#              "S3": -5,
#              "S2": -5,
#              "Notype": 0}

def boardtype(board, player):
    '''
    :param board: 当前棋盘状态
    :param player: 行棋方，黑方为1，白方为2
    :return: 返回一个字典typecount，记录每种棋型的个数
    '''
    typecount = {"Long": 0, "H4": 0, "C4": 0, "H3": 0, "M3": 0, "H2": 0, "M2": 0, "Other": 0, "S4": 0, "S3": 0, "S2": 0}

    # 如果是对手方行棋，则切换对手方视角
    def opponent_player(board):
        height, width = len(board), len(board[0])
        for i in range(height):
            for j in range(width):
                board[i][j] = (3 - board[i][j]) % 3
        return board

    if player == 2:
        board = opponent_player(board)

    height, width = len(board), len(board[0])

    # 阳线-行
    for row_idx, row in enumerate(board):
        list_str = "".join(map(str, row))
        for item in chess_type().items():
            typecount[item[1]] += len(re.findall(item[0], list_str))

    # 阳线-列
    for col_idx in range(width):
        col = [b[col_idx] for b in board]
        list_str = "".join(map(str, col))
        for item in chess_type().items():
            typecount[item[1]] += len(re.findall(item[0], list_str))

    # 阴线-左上右下对角线
    for dist in range(-width + 1, height):
        row_ini, col_ini = (0, -dist) if dist < 0 else (dist, 0)
        diag = [board[i][j] for i in range(
            row_ini, height) for j in range(col_ini, width) if i - j == dist]
        list_str = "".join(map(str, diag))
        for item in chess_type().items():
            typecount[item[1]] += len(re.findall(item[0], list_str))

    # 阴线-右上左下对角线
    for dist in range(0, width + height - 1):
        row_ini, col_ini = (dist, 0) if dist < height else (
            height - 1, dist - height + 1)
        diag = [board[i][j] for i in range(
            row_ini, -1, -1) for j in range(col_ini, width) if i + j == dist]
        list_str = "".join(map(str, diag))
        for item in chess_type().items():
            typecount[item[1]] += len(re.findall(item[0], list_str))

    # 切换回己方
    if player == 2:
        board = opponent_player(board)

    return typecount


def is_special_class(array, color):
    """
    judge whether the several chess given in the list form a special class
    :param
        array: the board of gomoku
        color: the index of color, 1: black, 2: white
    :return:
        Counter: ({class: num of this class}, ...)
    """

    # add judgement here. Details in 'http://zjh776.iteye.com/blog/1979748'

    def _black_color(array):
        height, width = len(array), len(array[0])
        for i in range(height):
            for j in range(width):
                array[i][j] = (3 - array[i][j]) % 3
        return array

    if color == 2:
        list_str = _black_color(array)

    class_dict = {("WIN", (), ()): "11111",
                  ("H4", (0, 5), ()): "011110",
                  ("C4", (0), (5)): "011112",
                  ("C4", (5), (0)): "211110",
                  ("C4", (4), ()): r"^11110",
                  ("C4", (0), ()): r"01111$",
                  ("C4", (0, 2, 6), ()): "0101110",
                  ("C4", (0, 4, 6), ()): "0111010",
                  ("C4", (0, 3, 6), ()): "0110110",
                  ("H3", (0, 4), ()): "01110",
                  ("H3", (0, 2, 5), ()): "010110",
                  ("H3", (0, 3, 5), ()): "011010",
                  ("M3", (0, 1), (5)): "001112",
                  ("M3", (0, 1), ()): r"00111$",
                  ("M3", (4, 5), (0)): "211100",
                  ("M3", (4, 5), ()): r"^11100",
                  ("M3", (0, 2), (5)): "010112",
                  ("M3", (0, 2), ()): r"01011$",
                  ("M3", (3, 5), (0)): "211010",
                  ("M3", (3, 5), ()): r"^11010",
                  ("M3", (0, 3), (5)): "011012",
                  ("M3", (0, 3), ()): r"01101$",
                  ("M3", (2, 5), (0)): "210110",
                  ("M3", (2, 5), ()): r"^10110",
                  ("M3", (1, 2), ()): "10011",
                  ("M3", (2, 3), ()): "11001",
                  ("M3", (1, 3), ()): "10101",
                  ("M3", (1, 4), (0, 6)): "2011102",
                  ("M3", (1, 4), (6)): r"^011102",
                  ("M3", (1, 4), (0)): r"201110$",
                  ("H2", (0, 1, 4), ()): "00110",
                  ("H2", (0, 3, 4), ()): "01100",
                  ("H2", (0, 2, 4), ()): "01010",
                  ("H2", (0, 2, 3, 5), ()): "010010",
                  ("M2", (0, 1, 2), (5)): "000112",
                  ("M2", (0, 1, 2), ()): r"00011$",
                  ("M2", (3, 4, 5), (0)): "211000",
                  ("M2", (3, 4, 5), ()): r"^11000",
                  ("M2", (0, 1, 3), (5)): "001012",
                  ("M2", (0, 1, 3), ()): r"00101$",
                  ("M2", (2, 4, 5), (0)): "210100",
                  ("M2", (2, 4, 5), ()): r"^10100",
                  ("M2", (0, 2, 3), (5)): "010012",
                  ("M2", (0, 2, 3), ()): r"01001$",
                  ("M2", (2, 3, 5), (0)): "210010",
                  ("M2", (2, 3, 5), ()): r"^10010",
                  ("M2", (1, 2, 3), ()): "10001",
                  ("M2", (1, 3, 5), (0, 6)): "2010102",
                  ("M2", (1, 3, 5), (0)): r"201010$",
                  ("M2", (1, 3, 5), (6)): r"^010102",
                  ("M2", (1, 4, 5), (0, 6)): "2011002",
                  ("M2", (1, 4, 5), (6)): r"^011002",
                  ("M2", (1, 4, 5), (0)): r"201100^",
                  ("M2", (1, 2, 5), (0, 6)): "2001102",
                  ("M2", (1, 2, 5), (0)): r"200110$",
                  ("M2", (1, 2, 5), (6)): r"^001102",
                  ("S4", (), (0, 5)): "211112",
                  ("S4", (), (0)): r"21111$",
                  ("S4", (), (5)): r"^11112",
                  ("S3", (), (0, 4)): "21112",
                  ("S3", (), (0)): r"2111$",
                  ("S3", (), (4)): r"^1112",
                  ("S2", (), (0, 3)): "2112",
                  ("S2", (), (3)): r"^112",
                  ("S2", (), (0)): r"211$",
                  }

    height, width = len(array), len(array[0])
    class_counter = Counter()

    # scan by row
    for row_idx, row in enumerate(array):
        list_str = "".join(map(str, row))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by col
    for col_idx in range(width):
        col = [a[col_idx] for a in array]
        list_str = "".join(map(str, col))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by diag_1, from TL to BR
    for dist in range(-width + 1, height):
        row_ini, col_ini = (0, -dist) if dist < 0 else (dist, 0)
        diag = [array[i][j] for i in range(
            row_ini, height) for j in range(col_ini, width) if i - j == dist]
        list_str = "".join(map(str, diag))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    # scan by diag_2, from BL to TR
    for dist in range(0, width + height - 1):
        row_ini, col_ini = (dist, 0) if dist < height else (
            height - 1, dist - height + 1)
        diag = [array[i][j] for i in range(
            row_ini, -1, -1) for j in range(col_ini, width) if i + j == dist]
        list_str = "".join(map(str, diag))
        for key in class_dict:
            class_counter[key[0]] += len(re.findall(class_dict[key], list_str))

    return class_counter


def class_to_score():
    """
    define the reward of some specific class of chess
    :return:
        score_map: a map from the special class(a string) to score(a real number)
    """
    score_map = {"WIN": 200000,
                 "H4": 10000,
                 "C4": 1000,
                 "H3": 200,
                 "M3": 50,
                 "H2": 5,
                 "M2": 3,
                 "S4": -5,
                 "S3": -5,
                 "S2": -5
                 }
    return score_map


def board_evaluation(board):
    """
    evaluate the situation of the brain.
    :param
        board:
    :return:
        score: a real number, indicating how good the condition is
    """
    score = 0

    for a_class, num in is_special_class(board, 1).items():
        score = score + class_to_score()[a_class] * num
    for a_class, num in is_special_class(board, 2).items():
        if a_class in ['H4', 'C4', 'WIN']:
            score = score - 10 * class_to_score()[a_class] * num
        else:
            score = score - class_to_score()[a_class] * num

    return score



# def step_reward(board):
#     '''
#     :param board: 当前棋盘状态
#     :return: 得分 = 己方得分 - 对手方得分
#     '''
#     # score_map1 = {"Long": 100000,
#     #          "H4": 10000,
#     #          "Double_C4": 10000,
#     #          "C4_H3": 10000,
#     #          "Double_H3": 5000,
#     #          "H3_M3": 1000,
#     #          "C4": 500,
#     #          "H3": 200,
#     #          "Double_H2": 100,
#     #          "M3": 50,
#     #          "H2_M2": 10,
#     #          "H2": 5,
#     #          "M2": 3,
#     #          "Other": 1,
#     #          "S4": -5,
#     #          "S3": -5,
#     #          "S2": -5,
#     #          "Notype": 0}
#     score_map2 = {"Long": 100000,
#                  "H4": 10000,
#                  "Double_C4": 10000,
#                  "C4_H3": 10000,
#                  "Double_H3": 2000,
#                  "H3_M3": 1000,
#                  "C4": 5000,
#                  "H3": 200,
#                  "Double_H2": 100,
#                  "M3": 50,
#                  "H2_M2": 10,
#                  "H2": 5,
#                  "M2": 3,
#                  "Other": 1,
#                  "S4": -5,
#                  "S3": -5,
#                  "S2": -5,
#                  "Notype": 0}
#     score = 0
#     typecount1 = boardtype(board, 1)
#     chesstype1 = combchesstype(typecount1)
#     score += score_map2[chesstype1]
#     typecount2 = boardtype(board, 2)
#     chesstype2 = combchesstype(typecount2)
#     score -= score_map2[chesstype2]
#
#     return score

# def step_reward(board):
#     '''
#     :param board: 当前棋盘状态
#     :return: 得分 = 己方得分 - 对手方得分
#     '''
#
#     score = 0
#     typecount1 = boardtype(board, 1)
#     for item in typecount1.items():
#         score += item[1] * score_map()[item[0]]
#
#     typecount2 = boardtype(board, 2)
#     for item in typecount2.items():
#         score -= item[1] * score_map()[item[0]]
#
#     return score


def brain_init():
	if pp.width < 5 or pp.height < 5:
		pp.pipeOut("ERROR size of the board")
		return
	if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
		pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
		return
	pp.pipeOut("OK")

def brain_restart():
	for x in range(pp.width):
		for y in range(pp.height):
			board[x][y] = 0
	pp.pipeOut("OK")

def isFree(x, y):
	return x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] == 0

def brain_my(x, y):
	if isFree(x,y):
		board[x][y] = 1
	else:
		pp.pipeOut("ERROR my move [{},{}]".format(x, y))

def brain_opponents(x, y):
	if isFree(x,y):
		board[x][y] = 2
	else:
		pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))

def brain_block(x, y):
	if isFree(x,y):
		board[x][y] = 3
	else:
		pp.pipeOut("ERROR winning move [{},{}]".format(x, y))

def brain_takeback(x, y):
	if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
		board[x][y] = 0
		return 0
	return 2

def brain_turn():
	if pp.terminateAI:
		return
	i = 0
	while True:
		x = random.randint(0, pp.width)
		y = random.randint(0, pp.height)
		i += 1
		if pp.terminateAI:
			return
		if isFree(x,y):
			break
	if i > 1:
		pp.pipeOut("DEBUG {} coordinates didn't hit an empty field".format(i))
	pp.do_mymove(x, y)

def brain_end():
	pass

def brain_about():
	pp.pipeOut(pp.infotext)


class Node:
    """Node of the tree"""

    def __init__(self, is_leaf=False, player=1, children=None, value=None, action=None):
        """

        :param is_leaf: bool, whether the node is a leaf or not
        :param player: int, 1 or 2, 1 for MAX node and 2 for MIN node, whose turn in the following step
                        Assume that during my turn, player = 1
        :param children: list of Node representing the children
        :param value: bool, whether the node is visited or not
        :param action: action leading to current node
        """
        self.is_leaf = is_leaf
        self.rule = 'max' if player == 1 else 'min'
        if children is None:
            children = []
        self.children = children
        self.value = value
        self.visited = False
        self.action = action


def get_step(node, alpha=float("-inf"), beta=float("inf")):
    """returns [next node's value under the strategy, action to reach the former node]"""
    if node.is_leaf:
        node.visited = True  # TODO: delete?
        return [node.value, node.action]
    if node.rule == 'max':
        return max_step(node, alpha, beta)
    elif node.rule == 'min':
        return min_step(node, alpha, beta)
    else:
        raise Exception


def max_step(node, alpha=float("-inf"), beta=float("inf")):
    val = float('-inf')
    action = None
    for s in node.children:
        if get_step(s, alpha, beta)[0] > val:
            val = get_step(s, alpha, beta)[0]
            action = s.action
        if val >= beta:  # return the value being pruned
            return [val, None]
        alpha = max(alpha, val)
    return [val, action]


def min_step(node, alpha=float("-inf"), beta=float("inf")):
    val = float('inf')
    action = None
    for s in node.children:
        if val > get_step(s, alpha, beta)[0]:
            val = get_step(s, alpha, beta)[0]
            action = s.action
        if val <= alpha:  # return the value being pruned
            return [val, None]
        beta = min(beta, val)
    return [val, action]


def construct_tree(depth, node_player, board, action=None, max_expand=78, next_steps=None):
    """
    Assumption:
    step_to_win() function returns all the possible-to-win next steps for a given board
    The worst case is that next_step() returns every free position on the board
    status_evaluation() function returns a score to current board
    """
    node = Node(player=node_player, action=action)
    children = []
    if next_steps is None:
        next_steps = probable_position(board)
        if next_steps is None:
            return None
    if len(next_steps) > MAX_EXPAND: # Pruning by evaluation score
        top_eval = PriorityQueue() # TODO: to be optimize, not need to use pq
        for action in next_steps:
            board_new = copy.deepcopy(board)
            board_new[action[0]][action[1]] = node_player
            top_eval.put((-board_evaluation(board_new), board_new)) # sort by descendent evaluation score
        for i in range(MAX_EXPAND):
            top_board = top_eval.get()
            if depth == 1:  # is leaf
                children.append(
                    Node(is_leaf=True, player=opposite_player(node_player), value=-top_board[0],
                         action=action))
            else:
                children.append(construct_tree(depth - 1, opposite_player(node_player), top_board[1], action,
                                               renew_probable_position(action, next_steps)))
    else:
        for action in next_steps:
            board_new = copy.deepcopy(board)
            board_new[action[0]][action[1]] = node_player
            if depth == 1:  # is leaf
                # TODO: Maybe set value = None during construction
                children.append(Node(is_leaf=True, player=opposite_player(node_player), value=board_evaluation(board_new),
                                     action=action))
            else:
                children.append(construct_tree(depth - 1, opposite_player(node_player), board_new, action,
                                               renew_probable_position(action, next_steps)))

    node.children = children
    return node


# def brain_abpruning():
#     # depth is a parameter
#     try:
#         root_node = construct_tree(depth=1, node_player=1, board=board, action=None, max_expand=28, next_steps=None)
#         if root_node is None:
#             pp.do_mymove(10, 10)
#         else:
#             action = get_step(root_node)[1]
#             x, y = action[0], action[1]
#             assert action is not None
#             pp.do_mymove(x, y)
#     except:
#         logTraceBack()


def brain_abpruning():
    # depth is a parameter
    root_node = construct_tree(depth=1, node_player=1, board=board, action=None, max_expand=78, next_steps=None)
    if root_node is None:
        pp.do_mymove(10, 10)
    else:
        action = get_step(root_node)[1]
        x, y = action[0], action[1]
        assert action is not None
        pp.do_mymove(x, y)


# def brain_abpruning():
#     # depth is a parameter
#     root_node = construct_tree(depth=1, node_player=1, board=board, max_expand = 1)
#     if root_node is None:
#         pp.do_mymove(10, 10)
#     else:
#         action = get_step(root_node)[1]
#         x, y = action[0], action[1]
#         assert action is not None
#         pp.do_mymove(x, y)


# if DEBUG_EVAL:
# 	import win32gui
# 	def brain_eval(x, y):
# 		# TODO check if it works as expected
# 		wnd = win32gui.GetForegroundWindow()
# 		dc = win32gui.GetDC(wnd)
# 		rc = win32gui.GetClientRect(wnd)
# 		c = str(board[x][y])
# 		win32gui.ExtTextOut(dc, rc[2]-15, 3, 0, None, c, ())
# 		win32gui.ReleaseDC(wnd, dc)
#
# # define a file for logging ...
# DEBUG_LOGFILE = "D:/新建文件夹/大四/人工智能2021/Final_PJ/GOMOKU/abp4/pbrain-py4.log"
# # ...and clear it initially
# with open(DEBUG_LOGFILE,"w") as f:
# 	pass
#
# # define a function for writing messages to the file
# def logDebug(msg):
# 	with open(DEBUG_LOGFILE,"a") as f:
# 		f.write(msg+"\n")
# 		f.flush()
#
# # define a function to get exception traceback
# def logTraceBack():
# 	import traceback
# 	with open(DEBUG_LOGFILE,"a") as f:
# 		traceback.print_exc(file=f)
# 		f.flush()
# 	raise


# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_abpruning
pp.brain_end = brain_end
pp.brain_about = brain_about
# if DEBUG_EVAL:
# 	pp.brain_eval = brain_eval

def main():
	pp.main()

if __name__ == "__main__":
	main()
