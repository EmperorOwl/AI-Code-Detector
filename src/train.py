import sys
import time

from src.models.codebert_model import CodeBertModel
from src.models.ast_model import AstModel
from src.utils.dual_output import DualOutput


def train_model(model_type: str,
                train_samples: list,
                test_samples: list,
                dual_output: DualOutput,
                args) -> None:

    if model_type == 'codebert':
        model = CodeBertModel(
            train_samples=train_samples,
            test_samples=test_samples
        )
    elif model_type == 'ast':
        model = AstModel(
            train_samples=train_samples,
            test_samples=test_samples,
            max_iterations=args.max_iterations,
            use_enhanced_features=args.use_enhanced_features
        )

    start_time = time.time()
    model.train()
    model.evaluate()

    end_time = time.time()
    seconds = end_time - start_time
    print(f"\nRuntime: {seconds:.2f} seconds ({seconds / 60:.2f} minutes)")

    sys.stdout = sys.__stdout__
    filename = input("Save output to (filename): ").strip()
    if filename != "":
        save_path = f'outputs/{model_type}_model/{filename}.log'
        with open(save_path, 'w') as file:
            file.write(dual_output.buffer.getvalue())
        print(f"Model saved to {save_path}")
    else:
        print("Output not saved")
    print()

    if input("Save model (y/n): ").strip().lower() == "y":
        model.save()
    else:
        print("Model not saved")
