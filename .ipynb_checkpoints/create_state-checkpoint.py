from pyboy import PyBoy
from pyboy.utils import WindowEvent

pyboy = PyBoy("roms/pokemon_blue.gb")

# Let emulator boot
for _ in range(300):
    pyboy.tick()

# Press START

for _ in range(400):
    pyboy.tick()


pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
for _ in range(30):
    pyboy.tick()
pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

for _ in range(300):
    pyboy.tick()

pyboy.send_input(WindowEvent.PRESS_BUTTON_START)
for _ in range(60):
    pyboy.tick()
pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

for _ in range(300):
    pyboy.tick()

pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
for _ in range(30):
    pyboy.tick()
pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)

pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
for _ in range(30):
    pyboy.tick()
pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

for _ in range(600):
    pyboy.tick()


# Skip remaining dialogue and naming
for _ in range(3000):
    pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    pyboy.tick()
    pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    pyboy.tick()

# Give the overworld time to load
for _ in range(400):
    pyboy.tick()

def press(pyboy, press_event, release_event, frames=20):
    pyboy.send_input(press_event)
    for _ in range(frames):
        pyboy.tick()
    pyboy.send_input(release_event)
    pyboy.tick()

# Movement sequence to exit house

# 1 right
press(pyboy, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)

# 5 up
for _ in range(5):
    press(pyboy, WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP)

# 3 right
for _ in range(3):
    press(pyboy, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)

# 6 down
for _ in range(6):
    press(pyboy, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

# 4 left
for _ in range(4):
    press(pyboy, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT)

# 1 down
press(pyboy, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)
press(pyboy, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

# Wait a moment outside
for _ in range(200):
    pyboy.tick()

def press(pyboy, press, release, frames=20):
    pyboy.send_input(press)
    for _ in range(frames):
        pyboy.tick()
    pyboy.send_input(release)
    pyboy.tick()


# move away from the house door
press(pyboy, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
press(pyboy, WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT)
press(pyboy, WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN)

# wait a moment
for _ in range(200):
    pyboy.tick()

# save state
with open("start.state", "wb") as f:
    pyboy.save_state(f)

# Save savestate
with open("start.state", "wb") as f:
    pyboy.save_state(f)

print("Savestate created outside the house!")