import time

from model import Model

if __name__ == '__main__':
    model = Model()

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
        while True:
            print("Enter a code snippet: ")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                lines.append(line)


            code_snippet = '\n'.join(lines)
            label = model.classify_code(code_snippet)
            res = 'AI' if label == 0 else 'Human'
            print("Result:", res, "written")

    print("Done")
