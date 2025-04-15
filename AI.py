import numpy as np
import pygame
import sys
import math
import random

# Initialize pygame
pygame.init()

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Game constants
ROWS = 6
COLS = 7
SQUARESIZE = 100  # Size of each board square in pixels
RADIUS = int(SQUARESIZE / 2 - 5)  # Radius of game pieces
WIDTH = COLS * SQUARESIZE  # Window width
HEIGHT = (ROWS + 1) * SQUARESIZE  # Window height (extra row for piece dropping animation)
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4  # Number of pieces in a row to win
DIFFICULTY_LEVELS = {
    "Easy": 2,
    "Medium": 4,
    "Hard": 6
}

# Fonts
FONT = pygame.font.SysFont("monospace", 30)
LARGE_FONT = pygame.font.SysFont("monospace", 50)


def create_board():
    """Create an empty board for the game"""
    return np.zeros((ROWS, COLS), dtype=int)


def drop_piece(board, row, col, piece):
    """Place a piece on the board"""
    board[row][col] = piece


def is_valid_location(board, col):
    """Check if a column has an open spot"""
    return board[0][col] == 0


def get_next_open_row(board, col):
    """Find the next available row in the given column"""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return -1


def winning_move(board, piece):
    """Check if the last move resulted in a win"""
    # Check horizontal locations
    for c in range(COLS - 3):
        for r in range(ROWS):
            if (board[r][c] == piece and board[r][c + 1] == piece and
                    board[r][c + 2] == piece and board[r][c + 3] == piece):
                return True, [(r, c), (r, c + 1), (r, c + 2), (r, c + 3)]

    # Check vertical locations
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (board[r][c] == piece and board[r + 1][c] == piece and
                    board[r + 2][c] == piece and board[r + 3][c] == piece):
                return True, [(r, c), (r + 1, c), (r + 2, c), (r + 3, c)]

    # Check positively sloped diagonals
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                    board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                return True, [(r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)]

    # Check negatively sloped diagonals
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                    board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                return True, [(r, c), (r - 1, c + 1), (r - 2, c + 2), (r - 3, c + 3)]

    return False, []


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 100
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10
    elif window.count(piece) == 1 and window.count(EMPTY) == 3:
        score += 1

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 120
    elif window.count(opp_piece) == 2 and window.count(EMPTY) == 2:
        score -= 5

    return score



def score_position(board, piece):
    score = 0

    center_array = [int(i) for i in list(board[:, COLS // 2])]
    center_count = center_array.count(piece)
    score += center_count * 8

    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLS - 3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score



def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE)[0] or winning_move(board, AI_PIECE)[0] or len(
        get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizing_player):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE)[0]:
                return (None, 100000000)
            elif winning_move(board, PLAYER_PIECE)[0]:
                return (None, -100000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizing_player:
        value = -np.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:
        value = np.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def get_valid_locations(board):
    """Get all valid locations for a piece"""
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def draw_board(screen, board, winning_positions=None):
    """Draw the game board on the screen"""
    # Fill the entire board area with blue
    pygame.draw.rect(screen, BLUE, (0, SQUARESIZE, WIDTH, HEIGHT - SQUARESIZE))

    # Draw the empty circles for the board
    for c in range(COLS):
        for r in range(ROWS):
            # Calculate the position for each circle
            # r=0 should be at the top of the board visually
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                               RADIUS)

    # Draw the pieces
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == PLAYER_PIECE:
                color = RED
                # Highlight winning pieces
                if winning_positions and (r, c) in winning_positions:
                    color = (255, 150, 150)  # Lighter red for winning pieces

                # Draw at the correct position (r=0 at the top)
                pygame.draw.circle(screen, color,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)

            elif board[r][c] == AI_PIECE:
                color = YELLOW
                # Highlight winning pieces
                if winning_positions and (r, c) in winning_positions:
                    color = (255, 255, 150)  # Lighter yellow for winning pieces

                # Draw at the correct position (r=0 at the top)
                pygame.draw.circle(screen, color,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                   RADIUS)

    pygame.display.update()


def draw_menu(screen):
    """Draw the difficulty selection menu"""
    screen.fill(BLACK)

    # Title
    title = LARGE_FONT.render("CONNECT 4", True, WHITE)
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 50))

    # Instructions
    instructions = FONT.render("Select Difficulty:", True, WHITE)
    screen.blit(instructions, (WIDTH / 2 - instructions.get_width() / 2, 150))

    # Difficulty buttons
    button_width = 200
    button_height = 60
    button_y_positions = [250, 350, 450]
    difficulty_levels = list(DIFFICULTY_LEVELS.keys())

    for i, difficulty in enumerate(difficulty_levels):
        button_x = WIDTH / 2 - button_width / 2
        button_y = button_y_positions[i]

        # Draw button
        pygame.draw.rect(screen, BLUE, (button_x, button_y, button_width, button_height))
        pygame.draw.rect(screen, WHITE, (button_x, button_y, button_width, button_height), 2)

        # Draw text
        text = FONT.render(difficulty, True, WHITE)
        screen.blit(text, (button_x + button_width / 2 - text.get_width() / 2,
                           button_y + button_height / 2 - text.get_height() / 2))

    pygame.display.update()

    # Wait for user selection
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                for i, difficulty in enumerate(difficulty_levels):
                    button_x = WIDTH / 2 - button_width / 2
                    button_y = button_y_positions[i]

                    if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                        return difficulty


def display_winner(screen, winner):
    """Display the winner message"""
    text = "Player Wins!" if winner == PLAYER_PIECE else "AI Wins!"
    label = LARGE_FONT.render(text, True, WHITE)

    # Create a semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))

    # Display the winner text
    screen.blit(label, (WIDTH / 2 - label.get_width() / 2, HEIGHT / 2 - label.get_height() / 2))

    # Display play again message
    play_again = FONT.render("Press SPACE to play again or ESC to quit", True, WHITE)
    screen.blit(play_again, (WIDTH / 2 - play_again.get_width() / 2, HEIGHT / 2 + 50))

    pygame.display.update()


def display_tie(screen):
    """Display tie game message"""
    label = LARGE_FONT.render("Tie Game!", True, WHITE)

    # Create a semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(180)
    overlay.fill(BLACK)
    screen.blit(overlay, (0, 0))

    # Display the tie game text
    screen.blit(label, (WIDTH / 2 - label.get_width() / 2, HEIGHT / 2 - label.get_height() / 2))

    # Display play again message
    play_again = FONT.render("Press SPACE to play again or ESC to quit", True, WHITE)
    screen.blit(play_again, (WIDTH / 2 - play_again.get_width() / 2, HEIGHT / 2 + 50))

    pygame.display.update()


def display_turn_indicator(screen, turn):
    """Display whose turn it is"""
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))

    if turn == 0:  # Player's turn
        text = "Player's Turn (RED)"
        color = RED
    else:  # AI's turn
        text = "AI's Turn (YELLOW)"
        color = YELLOW

    label = FONT.render(text, True, color)
    screen.blit(label, (WIDTH / 2 - label.get_width() / 2, SQUARESIZE / 2 - label.get_height() / 2))

    pygame.display.update()


def main():
    # Set up the game window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")

    # Select difficulty
    difficulty = draw_menu(screen)
    difficulty_depth = DIFFICULTY_LEVELS[difficulty]

    # Game loop
    while True:
        # Initialize game
        board = create_board()
        game_over = False
        turn = 0  # 0 for Player, 1 for AI

        # Draw initial board
        screen.fill(BLACK)
        draw_board(screen, board)
        display_turn_indicator(screen, turn)

        # Main game loop
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Display a piece above the board when moving mouse
                if event.type == pygame.MOUSEMOTION and turn == 0:
                    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                    posx = event.pos[0]
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)
                    display_turn_indicator(screen, turn)
                    pygame.display.update()

                # Player drops a piece
                if event.type == pygame.MOUSEBUTTONDOWN and turn == 0:
                    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))

                    # Get player's move
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if 0 <= col < COLS and is_valid_location(board, col):
                        row = get_next_open_row(board, col)

                        # Update board directly without animation
                        drop_piece(board, row, col, PLAYER_PIECE)
                        draw_board(screen, board)

                        # Check for win
                        win, winning_positions = winning_move(board, PLAYER_PIECE)
                        if win:
                            draw_board(screen, board, winning_positions)
                            display_winner(screen, PLAYER_PIECE)
                            game_over = True

                        # Check for tie
                        if len(get_valid_locations(board)) == 0 and not game_over:
                            display_tie(screen)
                            game_over = True

                        # Switch turns
                        turn = 1
                        display_turn_indicator(screen, turn)

            # AI's turn
            if turn == 1 and not game_over:
                # Add a small delay to make the AI's move visible
                pygame.time.wait(500)

                # Get AI's move using minimax
                col, minimax_score = minimax(board, difficulty_depth, -np.inf, np.inf, True)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)

                    # Update board directly without animation
                    drop_piece(board, row, col, AI_PIECE)
                    draw_board(screen, board)

                    # Check for win
                    win, winning_positions = winning_move(board, AI_PIECE)
                    if win:
                        draw_board(screen, board, winning_positions)
                        display_winner(screen, AI_PIECE)
                        game_over = True

                    # Check for tie
                    if len(get_valid_locations(board)) == 0 and not game_over:
                        display_tie(screen)
                        game_over = True

                    # Switch turns
                    turn = 0
                    display_turn_indicator(screen, turn)

            # If game is over, wait for restart or quit
            if game_over:
                while game_over:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                # Restart the game
                                game_over = False
                                # Return to main to restart
                                return

                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                sys.exit()


if __name__ == "__main__":
    while True:
        main()