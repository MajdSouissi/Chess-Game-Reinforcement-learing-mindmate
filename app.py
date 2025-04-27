from flask import Flask, render_template, request, jsonify
import chess
import chess.engine
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from pettingzoo.classic import chess_v6
import random

app = Flask(__name__)

# Set Stockfish path (Update this if needed)
STOCKFISH_PATH = r"C:\Users\user\PycharmProjects\PythonProject\Projet chess\stockfish\stockfish.exe"

# Famous Chess Openings Database
# Each opening is represented by a list of UCI format moves (e.g., "e2e4")
CHESS_OPENINGS = {
    "Ruy Lopez": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    "Italian Game": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    "Sicilian Defense": ["e2e4", "c7c5"],
    "French Defense": ["e2e4", "e7e6"],
    "Queen's Gambit": ["d2d4", "d7d5", "c2c4"],
    "King's Indian Defense": ["d2d4", "g8f6", "c2c4", "g7g6"],
    "English Opening": ["c2c4"],
    "Caro-Kann Defense": ["e2e4", "c7c6"],
    "Nimzo-Indian Defense": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    "Scandinavian Defense": ["e2e4", "d7d5"],
    "Slav Defense": ["d2d4", "d7d5", "c2c4", "c7c6"],
    "Alekhine's Defense": ["e2e4", "g8f6"],
    "Pirc Defense": ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
    "Modern Defense": ["e2e4", "g7g6"],
    "Dutch Defense": ["d2d4", "f7f5"],
    "Grünfeld Defense": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    "Vienna Game": ["e2e4", "e7e5", "b1c3"],
    "Scotch Game": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],
    "King's Gambit": ["e2e4", "e7e5", "f2f4"],
    "Four Knights Game": ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"],
    "London System": ["d2d4", "d7d5", "c1f4"],
    "Benko Gambit": ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5"],
    "Queen's Indian Defense": ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],
    "Budapest Gambit": ["d2d4", "g8f6", "c2c4", "e7e5"],
    "Bird's Opening": ["f2f4"]
}

# Load Stockfish Engine
try:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    print("Stockfish engine loaded successfully")
except Exception as e:
    print("Error loading Stockfish:", e)
    engine = None  # Set to None so we can check later


# Create the same custom environment class as in train_rl.py
class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()

        # Initialize the PettingZoo chess environment
        self.env = chess_v6.env()
        self.env.reset()

        # Get agent IDs
        self.agents = self.env.possible_agents
        self.current_agent_idx = 0

        # Define action and observation spaces
        # Chess has around 4672 possible moves, but let's simplify for now
        self.action_space = gym.spaces.Discrete(4672)

        # Define a simplified observation space
        # 8x8 board, 12 piece types + empty = 13 possible states per square
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(8 * 8 * 13,),  # Flattened board representation
            dtype=np.float32
        )

    def reset(self, **kwargs):
        self.env.reset()
        self.agents = self.env.possible_agents  # Update agent list
        self.current_agent_idx = 0

        if len(self.agents) > 0:
            obs = self._get_observation()
            return obs, {}
        else:
            # Default observation if no agents
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        # Safety check for agent index
        if self.current_agent_idx >= len(self.agents):
            self.current_agent_idx = 0

        agent = self.agents[self.current_agent_idx]

        # Execute the action
        try:
            self.env.step(action)
        except Exception as e:
            print(f"Error executing action: {e}")
            # Return a default observation and a negative reward
            return (
                np.zeros(self.observation_space.shape, dtype=np.float32),
                -1.0,
                True,  # done
                True,  # truncated
                {}  # info
            )

        # Check if the game is done before updating agent index
        if len(self.env.agents) == 0 or all(self.env.terminations.values()) or all(self.env.truncations.values()):
            # Game is over
            obs = self._get_observation()
            reward = self.env.rewards.get(agent, 0)
            done = True
            return obs, reward, done, done, {}

        # Update agent index
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)

        # Get next agent
        agent = self.agents[self.current_agent_idx]

        # Get observation and reward
        obs = self._get_observation()
        reward = self.env.rewards.get(agent, 0)

        # Check if the game is done
        done = self.env.terminations.get(agent, False) or self.env.truncations.get(agent, False)

        return obs, reward, done, done, {}

    def _get_observation(self):
        """Get a simplified observation that's compatible with our observation space"""
        # For simplicity, return a zero array of the appropriate shape
        # In a real implementation, you'd extract the actual board state
        return np.zeros(self.observation_space.shape, dtype=np.float32)


# Create the environment using the same class as in training
try:
    env = ChessEnv()
    print("Chess environment created successfully")
except Exception as e:
    print(f"Error creating environment: {e}")
    env = None  # Set to None so we can check later

# Load RL Model
MODEL_PATH = "chess_rl_model_policy"  # Use policy path from previous training
try:
    if env is not None:
        # Create a new model with the environment
        model = PPO(MlpPolicy, env)

        # Load just the policy weights
        try:
            # Try to load the policy directly
            model.policy = MlpPolicy.load(MODEL_PATH)
            print(f"Model policy loaded from {MODEL_PATH}")
        except Exception as policy_load_error:
            print(f"Error loading policy directly: {policy_load_error}")

            # Alternative approach - try loading into the existing policy
            try:
                model.policy.load(MODEL_PATH)
                print(f"Model policy loaded via second method from {MODEL_PATH}")
            except Exception as policy_load_error2:
                print(f"Error loading policy via second method: {policy_load_error2}")

                # Check if the params.npz file exists (fallback from train_rl.py)
                import os

                params_path = "chess_rl_model_params.npz"
                if os.path.exists(params_path):
                    try:
                        # Load parameters from numpy file
                        import torch

                        params = np.load(params_path)
                        for name, param in model.policy.named_parameters():
                            if name in params:
                                param.data = torch.tensor(params[name])
                        print(f"Model parameters loaded from {params_path}")
                    except Exception as params_load_error:
                        print(f"Error loading from params.npz: {params_load_error}")
                        model = None
                else:
                    print(f"No params file found at {params_path}")
                    model = None
    else:
        print("Cannot load model: environment not created")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to Stockfish only mode...")
    model = None

# Initialize board and game state
board = chess.Board()
current_opening = None
opening_move_index = 0
game_history = []  # List to track moves played in the current game
player_color = "white"  # Default player color is white


# Chess opening functions
def identify_current_opening():
    """Identify which opening is being played based on the game history"""
    global current_opening, opening_move_index, game_history

    # Reset if this is a new game
    if len(game_history) == 0:
        current_opening = None
        opening_move_index = 0
        return None

    # Check each opening to see if it matches the current move sequence
    potential_openings = []

    for opening_name, moves in CHESS_OPENINGS.items():
        # See how many moves match our current game
        match_length = 0
        for i, move in enumerate(moves):
            if i < len(game_history) and move == game_history[i]:
                match_length += 1
            else:
                break

        if match_length > 0:
            potential_openings.append((opening_name, match_length))

    # Find the opening with the most matching moves
    if potential_openings:
        # Sort by match length (descending)
        potential_openings.sort(key=lambda x: x[1], reverse=True)
        best_match = potential_openings[0]

        current_opening = best_match[0]
        opening_move_index = best_match[1]

        print(f"Current opening: {current_opening}, move index: {opening_move_index}")
        return current_opening

    return None


def get_next_opening_move():
    """Get the next move from the current opening if available"""
    global current_opening, opening_move_index

    if current_opening is None:
        return None

    opening_moves = CHESS_OPENINGS[current_opening]

    # Check if we have more moves in this opening
    if opening_move_index < len(opening_moves):
        next_move_uci = opening_moves[opening_move_index]
        try:
            next_move = chess.Move.from_uci(next_move_uci)
            # Verify it's a legal move in the current position
            if next_move in board.legal_moves:
                opening_move_index += 1
                return next_move
        except ValueError:
            pass

    # If we reach here, either we've completed the opening or the next move isn't legal
    return None


def select_random_opening():
    """Select a random opening that's appropriate for the current board position"""
    if len(game_history) > 0:
        return None  # Only select random openings at the beginning of the game

    # Choose a random opening
    opening_name = random.choice(list(CHESS_OPENINGS.keys()))
    opening_moves = CHESS_OPENINGS[opening_name]

    # Verify the first move is legal
    try:
        first_move = chess.Move.from_uci(opening_moves[0])
        if first_move in board.legal_moves:
            global current_opening, opening_move_index
            current_opening = opening_name
            opening_move_index = 1  # Set to 1 because we're about to play the first move
            print(f"Selected opening: {opening_name}")
            return first_move
    except ValueError:
        pass

    return None


# Function to get a move from the RL model or chess opening theory
def get_ai_move(board):
    """Get a move for the AI, prioritizing opening theory, then RL model, then Stockfish"""
    global game_history, current_opening

    # First, try to follow a chess opening if we're in the early game (first 10 moves)
    if board.fullmove_number <= 10:
        # Identify the current opening if necessary
        if current_opening is None:
            identify_current_opening()

        # If we're not following an opening yet, maybe choose one randomly
        if current_opening is None and board.fullmove_number <= 1 and len(game_history) == 0:
            opening_move = select_random_opening()
            if opening_move:
                print(f"Playing from opening theory: {current_opening}")
                return opening_move

        # Try to get the next move from the current opening
        next_opening_move = get_next_opening_move()
        if next_opening_move:
            print(f"Playing from opening theory: {current_opening}, move: {next_opening_move.uci()}")
            return next_opening_move

    # If no opening move is available or we're past the opening, try the RL model
    if model is not None:
        try:
            # Reset the environment
            obs, _ = env.reset()

            # Add some randomization to the move selection (same as your original code)
            if random.random() < 0.3:  # 30% chance to explore
                # Get a completely random move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return random.choice(legal_moves)
                return None
            else:
                # Get the model's prediction
                action, _ = model.predict(obs, deterministic=False)

                # Convert action to chess move with some randomness
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move_index = action % len(legal_moves)
                    slightly_shuffled = legal_moves.copy()
                    for i in range(min(3, len(slightly_shuffled))):
                        j = random.randint(0, len(slightly_shuffled) - 1)
                        slightly_shuffled[i], slightly_shuffled[j] = slightly_shuffled[j], slightly_shuffled[i]
                    return slightly_shuffled[move_index % len(slightly_shuffled)]
                return None
        except Exception as e:
            print(f"Error getting move from RL model: {e}")

    # If all else fails, use Stockfish
    if engine is not None:
        result = engine.play(board, chess.engine.Limit(time=1))
        return result.move

    # Last resort: pick a random legal move
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return random.choice(legal_moves)
    return None


# Homepage route
@app.route("/")
def index():
    return render_template("index.html")


# Handle setting player color
@app.route("/set_color", methods=["POST"])
def set_color():
    global player_color, board, game_history, current_opening, opening_move_index

    try:
        data = request.json
        new_color = data.get("color", "white").lower()

        if new_color not in ["white", "black"]:
            return jsonify({"error": "Invalid color selection"}), 400

        # Set the player color
        player_color = new_color

        # Reset the game
        board = chess.Board()
        current_opening = None
        opening_move_index = 0
        game_history = []

        response_data = {
            "message": f"Player color set to {player_color}",
            "board": board.fen(),
            "player_color": player_color
        }

        # If player is black, AI (white) goes first
        if player_color == "black":
            ai_move = get_ai_move(board)
            if ai_move and ai_move in board.legal_moves:
                game_history.append(ai_move.uci())
                board.push(ai_move)

                # Update opening info after AI move
                opening_info = identify_current_opening()
                opening_message = f"Current opening: {opening_info}" if opening_info else "No recognized opening"

                engine_type = "Opening Theory" if current_opening and opening_move_index > 0 else "RL" if model is not None else "Stockfish"

                response_data.update({
                    "bot_move": ai_move.uci(),
                    "board": board.fen(),
                    "engine_type": engine_type,
                    "opening_info": opening_message
                })

        return jsonify(response_data)

    except Exception as e:
        import traceback
        print(f"Error in set_color endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# NEW ENDPOINT: Sync position after undo
@app.route("/sync_position", methods=["POST"])
def sync_position():
    global board, current_opening, opening_move_index, game_history
    try:
        data = request.json
        fen = data.get("fen")
        if not fen:
            return jsonify({"error": "No FEN position provided"}), 400

        try:
            # Log the incoming FEN for debugging
            print(f"Received FEN for sync: {fen}")

            # Update the server's board to match client
            board.set_fen(fen)
            print(f"Synchronized board position: {fen}")

            # When undoing moves, we need to reset our opening tracking
            # Only reset if we can't derive the state from the move stack
            if len(board.move_stack) == 0:
                current_opening = None
                opening_move_index = 0
                game_history = []
            else:
                # Rebuild game_history from the board's move stack if needed
                if len(game_history) != len(board.move_stack):
                    game_history = []
                    for move in board.move_stack:
                        game_history.append(move.uci())
                    print(f"Rebuilt game history from move stack: {game_history}")

            # Identify opening after position change
            opening_info = identify_current_opening()
            opening_message = f"Current opening: {opening_info}" if opening_info else "No recognized opening yet."

            # Return complete state information to the client
            return jsonify({
                "message": "Position synchronized successfully",
                "board": board.fen(),
                "opening_info": opening_message,
                "player_color": player_color,
                "move_count": len(board.move_stack)
            })
        except ValueError as e:
            print(f"FEN error: {e}")
            return jsonify({"error": f"Invalid FEN notation: {str(e)}"}), 400
    except Exception as e:
        import traceback
        print(f"Error in sync_position endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Handle user move and AI response
@app.route("/move", methods=["POST"])
def make_move():
    global board, game_history, current_opening, opening_move_index, player_color
    try:
        data = request.json
        print(f"Received move data: {data}")

        fen = data.get("move")
        uci_move = data.get("uci_move")
        is_after_undo = data.get("is_after_undo", False)

        if not fen and not uci_move:
            return jsonify({"error": "No move data provided"}), 400

        if uci_move:
            try:
                move = chess.Move.from_uci(uci_move)
                if move in board.legal_moves:
                    # Track the human move
                    game_history.append(uci_move)
                    board.push(move)
                    print(f"Human move applied: {move}")
                else:
                    return jsonify({"error": "Illegal move"}), 400
            except ValueError as e:
                print(f"UCI move error: {e}")
                return jsonify({"error": "Invalid UCI move"}), 400
        else:
            try:
                # When setting FEN, we might lose track of move history
                old_fen = board.fen()
                board.set_fen(fen)
                if old_fen != fen:
                    # If the position changed significantly, reset our opening tracking
                    current_opening = None
                    opening_move_index = 0
                    game_history = []
                print(f"FEN position set: {fen}")
            except ValueError as e:
                print(f"FEN error: {e}")
                return jsonify({"error": "Invalid FEN notation"}), 400

        # If this is just synchronizing after an undo, don't make an AI move
        if is_after_undo:
            # Reset our opening tracking as undo disrupts the sequence
            current_opening = None
            opening_move_index = 0
            game_history = []
            return jsonify({
                "board": board.fen(),
                "game_over": board.is_game_over(),
                "message": "Board synchronized after undo",
                "player_color": player_color
            })

        # Identify current opening after human move
        opening_info = identify_current_opening()
        opening_message = f"Current opening: {opening_info}" if opening_info else "No recognized opening"

        if not board.is_game_over():
            # Get AI move using our enhanced function
            ai_move = get_ai_move(board)

            if ai_move and ai_move in board.legal_moves:
                # Track the AI move
                game_history.append(ai_move.uci())
                board.push(ai_move)
                print(f"AI move applied: {ai_move}, new position: {board.fen()}")

                # Update opening info after AI move
                opening_info = identify_current_opening()
                opening_message = f"Current opening: {opening_info}" if opening_info else "No recognized opening"

                engine_type = "Opening Theory" if current_opening and opening_move_index > 0 else "RL" if model is not None else "Stockfish"

                return jsonify({
                    "bot_move": ai_move.uci(),
                    "board": board.fen(),
                    "game_over": board.is_game_over(),
                    "engine_type": engine_type,
                    "opening_info": opening_message,
                    "player_color": player_color
                })
            else:
                return jsonify({"error": "Failed to generate a valid AI move"}), 500
        else:
            return jsonify({
                "error": "Game over",
                "result": board.result(),
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "opening_info": opening_message,
                "player_color": player_color
            }), 200
    except Exception as e:
        # Catch any unexpected errors
        import traceback
        print(f"Error in move endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# Restart the game
@app.route("/restart", methods=["POST"])
def restart():
    global board, current_opening, opening_move_index, game_history, player_color
    board = chess.Board()
    current_opening = None
    opening_move_index = 0
    game_history = []

    response_data = {
        "message": "Game restarted",
        "board": board.fen(),
        "player_color": player_color
    }

    # If player is black, AI (white) goes first
    if player_color == "black":
        ai_move = get_ai_move(board)
        if ai_move and ai_move in board.legal_moves:
            game_history.append(ai_move.uci())
            board.push(ai_move)

            # Update opening info after AI move
            opening_info = identify_current_opening()
            opening_message = f"Current opening: {opening_info}" if opening_info else "No recognized opening"

            engine_type = "Opening Theory" if current_opening and opening_move_index > 0 else "RL" if model is not None else "Stockfish"

            response_data.update({
                "bot_move": ai_move.uci(),
                "board": board.fen(),
                "engine_type": engine_type,
                "opening_info": opening_message
            })

    return jsonify(response_data)


# Get information about the current opening
@app.route("/opening_info", methods=["GET"])
def get_opening_info():
    global current_opening, opening_move_index

    if current_opening:
        return jsonify({
            "opening_name": current_opening,
            "moves": CHESS_OPENINGS[current_opening],
            "current_move_index": opening_move_index,
            "description": get_opening_description(current_opening)
        })
    else:
        return jsonify({"message": "No recognized opening in play"})


def get_opening_description(opening_name):
    """Return a brief description of the given opening"""
    descriptions = {
        "Ruy Lopez": "One of the oldest and most classic openings. White develops naturally while pinning Black's knight.",
        "Italian Game": "An old opening aimed at quick development and control of the center.",
        "Sicilian Defense": "The most popular response to e4, giving Black good counterattacking chances.",
        "French Defense": "A solid defense where Black counters in the center after establishing a pawn chain.",
        "Queen's Gambit": "White offers a pawn to gain control of the center. It's not truly a gambit as Black can't easily keep the pawn.",
        "King's Indian Defense": "A hypermodern defense where Black allows White to establish a center and then counterattacks it.",
        "English Opening": "A flexible flank opening that can transpose into many different setups.",
        "Caro-Kann Defense": "A solid defense for Black that avoids creating early weaknesses.",
        "Nimzo-Indian Defense": "A solid defense where Black pins White's knight to hinder development.",
        "Scandinavian Defense": "Black immediately challenges White's center pawn, leading to an early queen exchange.",
        "Slav Defense": "A solid response to the Queen's Gambit where Black supports the d5 pawn with c6.",
        "Alekhine's Defense": "Black tempts White's pawns forward to later counterattack them.",
        "Pirc Defense": "A hypermodern defense where Black develops their kingside fianchetto.",
        "Modern Defense": "Similar to the Pirc but more flexible in move order.",
        "Dutch Defense": "Black immediately counters for the center with the f-pawn.",
        "Grünfeld Defense": "Black allows White to establish a strong pawn center, then attacks it with pieces.",
        "Vienna Game": "White develops the knight to c3 before playing d4, allowing for more tactical possibilities.",
        "Scotch Game": "An open, tactical alternative to the Ruy Lopez and Italian Game.",
        "King's Gambit": "An aggressive opening where White sacrifices a pawn for rapid development and attack.",
        "Four Knights Game": "A symmetric, solid opening with mutual development.",
        "London System": "A solid opening system where White develops the bishop to f4 early on.",
        "Benko Gambit": "Black sacrifices a pawn to open lines on the queenside.",
        "Queen's Indian Defense": "A solid hypermodern approach where Black fianchettoes the queen's bishop.",
        "Budapest Gambit": "Black sacrifices a pawn to disrupt White's pawn center.",
        "Bird's Opening": "An unusual flank opening with White playing f4 on the first move."
    }

    return descriptions.get(opening_name, "A chess opening with no additional information available.")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
