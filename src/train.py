import sys
import time

from src.models.codebert_model import CodeBertModel
from src.models.ast_model import ASTModel
from src.utils.dual_output import DualOutput


def train_codebert():
    dual_output = DualOutput()
    sys.stdout = dual_output

    model = CodeBertModel()
    start_time = time.time()

    print("\nPreparing datasets...")
    model.prepare()

    print("\nTraining model...")
    model.train()

    print("\nEvaluating model...")
    model.evaluate()

    end_time = time.time()
    print(f"\nRuntime: {end_time - start_time} seconds")

    sys.stdout = sys.__stdout__
    filename = input("Save output to (filename): ").strip()
    if filename != "":
        with open(f'outputs/codebert_model/{filename}.log', 'w') as file:
            file.write(dual_output.buffer.getvalue())
        print(f"Model saved to outputs/{filename}.log")
    else:
        print("Output not saved")
    print()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
    else:
        print("Model not saved")


def train_ast():
    dual_output = DualOutput()
    sys.stdout = dual_output

    model = ASTModel()
    start_time = time.time()

    print("\nPreparing datasets...")
    model.prepare()

    print("\nTraining model...")
    model.train()

    print("\nEvaluating model...")
    model.evaluate()

    end_time = time.time()
    print(f"\nRuntime: {end_time - start_time} seconds")

    sys.stdout = sys.__stdout__
    filename = input("Save output to (filename): ").strip()
    if filename != "":
        with open(f'outputs/ast_model/{filename}.log', 'w') as file:
            file.write(dual_output.buffer.getvalue())
        print(f"Output saved to outputs/{filename}.log")
    else:
        print("Output not saved")
    print()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
    else:
        print("Model not saved")
