import torch
import numpy as np

from game import Connect2Game
from model import Connect2Model

def main():
    device = torch.device("cpu")

    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)
    checkpoint = torch.load('latest.pth')
    model.load_state_dict(checkpoint['state_dict'])

    board = game.get_init_board()
    player = 1

    while True:
        print(f"board: {board}, player: {player}")
        canonical_board = game.get_canonical_board(board, player)
        # print(canonical_board)

        pi, v = model.predict(canonical_board)
        print(f"pi: {pi}, v: {v}")
        action = np.random.choice(action_size, p=pi)
        print(f"action: {action}")

        new_board, new_player = game.get_next_state(board, player, action)

        board = new_board
        if game.is_win(board, player):
            print(f"player {player} wins")
            break

        if not game.has_legal_moves(board):
            print("game ends in a draw")
            break
        player = new_player

        print("==============")


if __name__ == '__main__':
    main()
