import tkinter as tk
import random

sizeSquare = 40    # size of each little square
gap = 2         # gap between stickers
border = 4      # border around each face
FONT_NAME = "Calibri"
FONT_SIZE = 12

FACE_COLORS = {
    "U": "#FFFFFF",  # white
    "R": "#CC0000",  # red
    "F": "#1E8F1E",  # green
    "D": "#F2E300",  # yellow
    "L": "#FF7F00",  # orange
    "B": "#204CFF",  # blue
}

# Layout
NET_POSITIONS = {
    "U": (3, 2),
    "L": (1.5, 3.5),
    "F": (3, 3.5),
    "R": (4.5, 3.5),
    "B": (6, 3.5),
    "D": (3, 5),
}

# Initialize cube state
CUBE = {
    "U": [  # Up face
        [8, 1, 3],
        [4, 6, 7],
        [2, 9, 5]
    ],
    "L": [  # Left face
        [7, 1, 8],
        [2, 4, 6],
        [9, 3, 5]
    ],
    "F": [  # Front face
        [9, 5, 2],
        [3, 8, 1],
        [6, 7, 4]
    ],
    "R": [  # Right face
        [4, 6, 3],
        [7, 5, 9],
        [1, 2, 8]
    ],
    "B": [  # Back face
        [9, 5, 2],
        [3, 8, 1],
        [6, 7, 4]
    ],
    "D": [  # Down face
        [1, 2, 8],
        [5, 3, 9],
        [7, 4, 6]
    ]
}

def rotate_face_clockwise(face):
    """Rotate clockwise."""
    return [list(row) for row in zip(*face[::-1])]

def rotate_face_counter(face):
    """Rotate counterclockwise."""
    return [list(row) for row in zip(*face)][::-1]

def rotate(cube, face_key, direction="Clockwise"):
    """
    Rotate a face of the cube and its adjacent strips.
    """
    # Rotate face itself
    if direction == "Clockwise":
        cube[face_key] = rotate_face_clockwise(cube[face_key])
    else:
        cube[face_key] = rotate_face_counter(cube[face_key])

    # Handle adjacent strips
    if face_key == "F":
        if direction == "Clockwise":
            # Save top row of U so it doesn't get overwritten
            temp = cube["U"][2][:]
            # Move L -> U, D -> L, R -> D, saved U -> R
            for i in range(3):
                cube["U"][2][i] = cube["L"][2 - i][2]   # left col -> top row
                cube["L"][2 - i][2] = cube["D"][0][2 - i]  # bottom row -> left col
                cube["D"][0][2-i] = cube["R"][i][0]      # right col -> bottom row
                cube["R"][i][0] = temp[i]              # saved U -> right col
        else:  # Counterclockwise
            temp = cube["U"][2][:]
            for i in range(3):
                cube["U"][2][i] = cube["R"][i][0]
                cube["R"][i][0] = cube["D"][0][2 - i]
                cube["D"][0][2-i] = cube["L"][2-i][2]
                cube["L"][2-i][2] = temp[i]
    elif face_key == "B":
        if direction == "Clockwise":
            temp = cube["U"][0][:]
            for i in range(3):
                cube["U"][0][i] = cube["R"][i][2]
                cube["R"][i][2] = cube["D"][2][2 - i]
                cube["D"][2][i] = cube["L"][i][0]
                cube["L"][i][0] = temp[2-i]
        else:
            temp = cube["U"][0][:]
            for i in range(3):
                cube["U"][0][i] = cube["L"][2-i][0]
                cube["L"][i][0] = cube["D"][2][i]
                cube["D"][2][i] = cube["R"][2 - i][2]
                cube["R"][2-i][2] = temp[2-i]

    elif face_key == "U":
        if direction == "Clockwise":
            temp = cube["B"][0][:]
            cube["B"][0] = cube["L"][0][:]
            cube["L"][0] = cube["F"][0][:]
            cube["F"][0] = cube["R"][0][:]
            cube["R"][0] = temp
        else:
            temp = cube["B"][0][:]
            cube["B"][0] = cube["R"][0][:]
            cube["R"][0] = cube["F"][0][:]
            cube["F"][0] = cube["L"][0][:]
            cube["L"][0] = temp

    elif face_key == "D":
        if direction == "Clockwise":
            temp = cube["F"][2][:]
            cube["F"][2] = cube["L"][2][:]
            cube["L"][2] = cube["B"][2][:]
            cube["B"][2] = cube["R"][2][:]
            cube["R"][2] = temp
        else:
            temp = cube["F"][2][:]
            cube["F"][2] = cube["R"][2][:]
            cube["R"][2] = cube["B"][2][:]
            cube["B"][2] = cube["L"][2][:]
            cube["L"][2] = temp

    elif face_key == "L":
        if direction == "Clockwise":
            temp = [cube["U"][i][0] for i in range(3)]
            for i in range(3):
                cube["U"][i][0] = cube["B"][2 - i][2]
                cube["B"][2 - i][2] = cube["D"][i][0]
                cube["D"][i][0] = cube["F"][i][0]
                cube["F"][i][0] = temp[i]
        else:
            temp = [cube["U"][i][0] for i in range(3)]
            for i in range(3):
                cube["U"][i][0] = cube["F"][i][0]
                cube["F"][i][0] = cube["D"][i][0]
                cube["D"][i][0] = cube["B"][2 - i][2]
                cube["B"][2 - i][2] = temp[i]

    elif face_key == "R":
        if direction == "Clockwise":
            temp = [cube["U"][i][2] for i in range(3)]
            for i in range(3):
                cube["U"][i][2] = cube["F"][i][2]
                cube["F"][i][2] = cube["D"][i][2]
                cube["D"][i][2] = cube["B"][2 - i][0]
                cube["B"][2 - i][0] = temp[i]
        else:
            temp = [cube["U"][i][2] for i in range(3)]
            for i in range(3):
                cube["U"][i][2] = cube["B"][2 - i][0]
                cube["B"][2 - i][0] = cube["D"][i][2]
                cube["D"][i][2] = cube["F"][i][2]
                cube["F"][i][2] = temp[i]

def randomize_cube(cube, num_moves):
    faces = list(cube.keys())
    directions = ["Clockwise", "Counterclockwise"]

    #Get random face and direction, then rotate
    for i in range(num_moves):
        f = random.choice(faces)
        d = random.choice(directions)
        rotate(cube, f, d)
    return cube

def face_top_left(face_slot_col, face_slot_row):
    x = face_slot_col * (sizeSquare + gap) * 3 + 10
    y = face_slot_row * (sizeSquare + gap) * 3 + 10
    return x, y

def draw_cube(canvas, cube_state):
    canvas.delete("all")

    # Title
    canvas.create_text(
        (300, 30),
        text="Sudoku Cube Randomizer",
        font=(FONT_NAME, 20, "bold"),
        fill="#111"
    )

    for face_key in cube_state:
        slot_col, slot_row = NET_POSITIONS[face_key]
        face_x, face_y = face_top_left(slot_col, slot_row)

        face_size = 3 * sizeSquare + 2 * gap + 2 * border
        canvas.create_rectangle(
            face_x, face_y,
            face_x + face_size, face_y + face_size,
            outline="#444444", width=1, fill="#222222"
        )

        # Label above face
        canvas.create_text(
            face_x + face_size/2, face_y - 10,
            text=f"{face_key} face",
            font=(FONT_NAME, FONT_SIZE),
            fill="#111"
        )

        color = FACE_COLORS[face_key]
        labels = cube_state[face_key]

        for r in range(3):
            for c in range(3):
                sx = face_x + border + gap + c * (sizeSquare + gap)
                sy = face_y + border + gap + r * (sizeSquare + gap)
                canvas.create_rectangle(
                    sx, sy, sx + sizeSquare, sy + sizeSquare,
                    outline="#111111", width=1, fill=color
                )
                # Draw numbers
                txt = labels[r][c]
                canvas.create_text(
                    sx + sizeSquare/2, sy + sizeSquare/2,
                    text=txt,
                    font=(FONT_NAME, FONT_SIZE, "bold"),
                    fill="#000000" if face_key in ("U", "D") else "#FFFFFF"
                )

def on_submit(moves_entry, canvas):
    try:
        moves = int(moves_entry.get())
    except ValueError:
        moves = 0
    #initial heuristic
    print("Heuristic at start:", heuristic(CUBE))
    randomize_cube(CUBE, moves)
    draw_cube(canvas, CUBE)
    #after scrambling
    print("Heuristic after scramble:", heuristic(CUBE))

def heuristic(cube):
    misplaced = 0
    target = set(range(1, 10)) 

    for face in cube.values():
        face_numbers = {num for row in face for num in row}
        # Count how many numbers are missing from this face
        missing = target - face_numbers
        misplaced += len(missing)

    return misplaced

def main():
    root = tk.Tk()
    root.title("Sudoku Cube (Unfolded Net)")

    max_x, max_y = 1000, 700
    canvas = tk.Canvas(root, width=max_x, height=max_y, bg="#EEEEEE", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    moves_label = tk.Label(root, text='Input Moves', font=('calibre',10,'bold'))
    moves_label.place(x=0,y=100)
    moves_entry=tk.Entry(root, font=('calibre',10,'normal'))
    moves_entry.place(x=0,y=150)

    #when submit button pressed, goes to randomize the cube
    submit_button = tk.Button(root, text="Randomize",
                              command=lambda: on_submit(moves_entry, canvas))
    submit_button.place(x=0,y=200)

    #to refresh the cube drawing
    draw_cube(canvas, CUBE)

    root.mainloop()

if __name__ == "__main__":
    main()
