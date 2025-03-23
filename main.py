import time

from model import Model

if __name__ == '__main__':
    model = Model()

    start_time = time.time()
    print("Training model...")
    model.train()
    end_time = time.time()

    print(f"Runtime: {end_time - start_time} seconds")
