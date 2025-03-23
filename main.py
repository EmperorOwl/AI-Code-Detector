import time

from app import app, model

if __name__ == '__main__':

    isTraining = input("Do you want to train the model? (y/n): ")

    if isTraining == 'y':
        start_time = time.time()

        print("Training model...")
        model.train()
        print("Model trained")
        model.save()
        print("Model saved")

        end_time = time.time()
        print(f"Runtime: {end_time - start_time} seconds")

    else:
        app.run(debug=True)
