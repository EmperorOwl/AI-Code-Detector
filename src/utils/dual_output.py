import sys
import io


class DualOutput:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)  # Print to screen
        self.buffer.write(message)  # Save to buffer

    def flush(self):
        self.terminal.flush()
