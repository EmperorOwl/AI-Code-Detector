import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ast_node import AstNode, AstParser

class TestAstParser(unittest.TestCase):
    """Test cases for the AstParser class"""

    def test_parse_python_ast_simple_code(self):
        """Test parsing simple Python code"""
        code = "x = 5"
        result = AstParser.parse_python_ast(code)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)
        if result is not None:
            self.assertEqual(result.node_type, "module")
            print('\n', result.to_string())

    def test_parse_python_ast_function_definition(self):
        """Test parsing Python function definition"""
        code = """
def hello():
    return "world"
"""
        result = AstParser.parse_python_ast(code)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)
        if result is not None:
            self.assertEqual(result.node_type, "module")
            # The AST should contain function definition nodes
            ast_string = result.to_string()
            self.assertIn("function_definition", ast_string)
            print('\n', result.to_string())

    def test_parse_python_ast_invalid_code(self):
        """Test parsing invalid Python code"""
        code = "def invalid syntax here"
        result = AstParser.parse_python_ast(code)
        
        # Even invalid code should return an AST node with error nodes
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)

    def test_parse_java_ast_simple_code(self):
        """Test parsing simple Java code"""
        code = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello World");
    }
}
"""
        result = AstParser.parse_java_ast(code)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)
        if result is not None:
            self.assertEqual(result.node_type, "program")
            print('\n', result.to_string())

    def test_parse_java_ast_class_declaration(self):
        """Test parsing Java class declaration"""
        code = "public class Test {}"
        result = AstParser.parse_java_ast(code)
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)
        if result is not None:
            self.assertEqual(result.node_type, "program")
            # The AST should contain class declaration nodes
            ast_string = result.to_string()
            self.assertIn("class_declaration", ast_string)

    def test_parse_java_ast_invalid_code(self):
        """Test parsing invalid Java code"""
        code = "public class invalid syntax here"
        result = AstParser.parse_java_ast(code)
        
        # Even invalid code should return an AST node with error nodes
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AstNode)
        if result is not None:
            print('\n', result.to_string())

    def test_get_ast_representation_python(self):
        """Test get_ast_representation for Python code"""
        code = "x = 5"
        result = AstParser.get_ast_representation(code, "python")
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        if result is not None:
            self.assertTrue(result.startswith("(module"))

    def test_get_ast_representation_java(self):
        """Test get_ast_representation for Java code"""
        code = "public class Test {}"
        result = AstParser.get_ast_representation(code, "java")
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        if result is not None:
            self.assertTrue(result.startswith("(program"))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty code
        result_python = AstParser.parse_python_ast("")
        self.assertIsNotNone(result_python)
        
        result_java = AstParser.parse_java_ast("")
        self.assertIsNotNone(result_java)
        
        # Whitespace only
        result_python_ws = AstParser.parse_python_ast("   \n\t  ")
        self.assertIsNotNone(result_python_ws)
        
        result_java_ws = AstParser.parse_java_ast("   \n\t  ")
        self.assertIsNotNone(result_java_ws)

    def test_complex_python_code(self):
        """Test parsing complex Python code structures"""
        code = """
class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, *args):
        result = 1
        for arg in args:
            result *= arg
        return result

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
"""
        result = AstParser.parse_python_ast(code)
        self.assertIsNotNone(result)
        
        if result is not None:
            ast_string = result.to_string()
            self.assertIn("class_definition", ast_string)
            self.assertIn("function_definition", ast_string)
            self.assertIn("if_statement", ast_string)

    def test_complex_java_code(self):
        """Test parsing complex Java code structures"""
        code = """
public class Calculator {
    private int value;
    
    public Calculator() {
        this.value = 0;
    }
    
    public int add(int x, int y) {
        return x + y;
    }
    
    public static void main(String[] args) {
        Calculator calc = new Calculator();
        System.out.println(calc.add(5, 3));
    }
}
"""
        result = AstParser.parse_java_ast(code)
        self.assertIsNotNone(result)
        
        if result is not None:
            ast_string = result.to_string()
            self.assertIn("class_declaration", ast_string)
            self.assertIn("method_declaration", ast_string)
            self.assertIn("constructor_declaration", ast_string)


if __name__ == '__main__':
    unittest.main()
