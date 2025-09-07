from src.pre_processing.ast import get_ast_representation

# Test Python code
python_code = """
def factorial(n):
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial of 5 is: {result}")
"""

# Test Java code
java_code = """
public class HelloWorld {
    public static void main(String[] args) {
        int x = 42;
        String message = "Hello, World!";
        System.out.println(message + " " + x);
    }
}
"""


def test_enhanced_ast():
    print("=== Testing Enhanced AST Serialization ===\n")

    print("--- Python Code ---")
    print("Original code:")
    print(python_code)

    print("\nBasic AST:")
    basic_ast = get_ast_representation(
        python_code, 'python', include_tokens=False)
    print(basic_ast)

    print("\nEnhanced AST (with tokens):")
    enhanced_ast = get_ast_representation(
        python_code, 'python', include_tokens=True)
    print(enhanced_ast)

    print("\n" + "="*60 + "\n")

    print("--- Java Code ---")
    print("Original code:")
    print(java_code)

    print("\nBasic AST:")
    basic_ast_java = get_ast_representation(
        java_code, 'java', include_tokens=False)
    print(basic_ast_java)

    print("\nEnhanced AST (with tokens):")
    enhanced_ast_java = get_ast_representation(
        java_code, 'java', include_tokens=True)
    print(enhanced_ast_java)


if __name__ == "__main__":
    test_enhanced_ast()
