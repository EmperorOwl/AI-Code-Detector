import time

from model import Model

if __name__ == '__main__':
    start_time = time.time()
    model = Model()

    print("Training model...")
    model.train()
    print("Model trained")

    model.save()
    print("Model saved")

    end_time = time.time()
    print(f"Runtime: {end_time - start_time} seconds")
