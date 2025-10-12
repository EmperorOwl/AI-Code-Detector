"""
Simplified AST processing module for Java and Python code.
Based on the approach from https://arxiv.org/pdf/2203.03850.pdf
"""

import re
from io import StringIO
import tokenize
from logging import Logger

from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava
import tree_sitter_python as tspython


class AstProcessor:

    def __init__(self, logger: Logger):
        """
        Initialize the AST processor.

        Args:
            logger (Logger): Logger instance for logging operations
        """
        self.logger = logger
        self.parsers: dict[str, Parser] = {}
        self._setup_parsers()

    def _setup_parsers(self) -> None:
        """
        Setup tree-sitter parsers for supported languages.
        """
        try:
            self.logger.info("Setting up tree-sitter parsers...")

            # Setup Java parser
            java_parser = Parser(Language(tsjava.language()))
            self.parsers['java'] = java_parser

            # Setup Python parser
            python_parser = Parser(Language(tspython.language()))
            self.parsers['python'] = python_parser

            self.logger.info(
                "✓ Tree-sitter parsers configured for Java and Python\n"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup tree-sitter parsers: {e}")
            raise RuntimeError(f"Tree-sitter setup failed: {e}")

    def remove_comments_and_docstrings(self, source: str, lang: str) -> str:
        """
        Remove comments and docstrings from source code.

        Args:
            source (str): Source code
            lang (str): Programming language ('python' or 'java')

        Returns:
            str: Source code with comments and docstrings removed
        """
        if lang == 'python':
            return self._remove_python_comments(source)
        elif lang == 'java':
            return self._remove_java_comments(source)
        else:
            self.logger.warning(f"Unsupported language: {lang}")
            return source

    def _remove_python_comments(self, source: str) -> str:
        """
        Remove comments and docstrings from Python source code.

        Args:
            source (str): Python source code

        Returns:
            str: Cleaned Python source code
        """
        try:
            io_obj = StringIO(source)
            out = ""
            prev_toktype = tokenize.INDENT
            last_lineno = -1
            last_col = 0

            for tok in tokenize.generate_tokens(io_obj.readline):
                token_type = tok[0]
                token_string = tok[1]
                start_line, start_col = tok[2]
                end_line, end_col = tok[3]

                if start_line > last_lineno:
                    last_col = 0
                if start_col > last_col:
                    out += (" " * (start_col - last_col))

                # Remove comments
                if token_type == tokenize.COMMENT:
                    pass
                # Remove docstrings
                elif token_type == tokenize.STRING:
                    if prev_toktype != tokenize.INDENT:
                        if prev_toktype != tokenize.NEWLINE:
                            if start_col > 0:
                                out += token_string
                else:
                    out += token_string

                prev_toktype = token_type
                last_col = end_col
                last_lineno = end_line

            # Remove empty lines
            temp = []
            for line in out.split('\n'):
                if line.strip():
                    temp.append(line)
            return '\n'.join(temp)

        except Exception as e:
            self.logger.warning(f"Failed to remove Python comments: {e}")
            return source

    def _remove_java_comments(self, source: str) -> str:
        """
        Remove comments from Java source code.

        Args:
            source (str): Java source code

        Returns:
            str: Cleaned Java source code
        """
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # Replace comment with space
            else:
                return s

        # Pattern to match comments while preserving strings
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )

        cleaned = re.sub(pattern, replacer, source)

        # Remove empty lines
        temp = []
        for line in cleaned.split('\n'):
            if line.strip():
                temp.append(line)
        return '\n'.join(temp)

    def generate_ast_sequence(self,
                              code: str,
                              lang: str) -> str:
        """
        Generate AST sequence string for the given code.

        Args:
            code (str): Source code
            lang (str): Programming language ('python' or 'java')

        Returns:
            str: Flattened AST sequence as a single string
        """
        if lang not in self.parsers:
            self.logger.warning(f"Unsupported language: {lang}")
            raise ValueError(f"Unsupported language: {lang}")

        try:
            # Remove comments first
            cleaned_code = self.remove_comments_and_docstrings(code, lang)

            # Parse source code
            parser = self.parsers[lang]
            tree = parser.parse(bytes(cleaned_code, 'utf8'))

            # Get AST sequence
            root_node = tree.root_node

            # Generate AST traversal sequence directly from nodes
            ast_tokens = self._travel_ast(
                root_node, cleaned_code.encode('utf8'))
            # Join all tokens into a single string with spaces
            return ' '.join(ast_tokens)

        except Exception as e:
            self.logger.error(f"AST generation failed: {e}")
            raise RuntimeError(f"AST generation failed: {e}")

    def _travel_ast(self, root_node: Node, source_code: bytes) -> list[str]:
        """
        Traverse AST and generate token sequence.

        Args:
            root_node: Tree-sitter AST node
            source_code: Source code as bytes

        Returns:
            list[str]: AST traversal token sequence
        """
        if (len(root_node.children) == 0 or
            root_node.type == 'string' or
            root_node.type == 'comment' or
                'comment' in root_node.type):

            start_byte = root_node.start_byte
            end_byte = root_node.end_byte
            code_text = source_code[start_byte:end_byte].decode('utf8')
            return [code_text]  # Return raw code token as string
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += self._travel_ast(child, source_code)

            # Add AST structure tokens (reduce nodes with single child)
            if len(root_node.children) != 1:
                node_type = root_node.type.replace("#", "")
                return (["AST#" + node_type + "#Left"] +
                        code_tokens +
                        ["AST#" + node_type + "#Right"])
            else:
                return code_tokens


# Example usage and testing
if __name__ == "__main__":
    from src.utils.logger import get_logger

    # Initialize logger
    logger = get_logger("ast_processor", "outputs_test/ast_processor.log")

    # Create processor
    processor = AstProcessor(logger)

    # Test with sample code
    python_code = '''
def hello_world():
    """This is a docstring"""
    # This is a comment
    print("Hello, World!")
    return True
'''

    java_code = '''
public class HelloWorld {
    // This is a comment
    public static void main(String[] args) {
        /* Multi-line comment 
        This is a multi-line comment
        This is a multi-line comment
        */
        System.out.println("Hello, World!");
    }
}
'''

    # Test comment removal
    print("Original Python code:")
    print(python_code)
    print("\nCleaned Python code:")
    cleaned_python = processor.remove_comments_and_docstrings(
        python_code, 'python')
    print(cleaned_python)

    print("\nPython AST sequence:")
    try:
        ast_sequence = processor.generate_ast_sequence(python_code, 'python')
        print(ast_sequence)
    except Exception as e:
        print(f"AST generation failed: {e}")

    print("\nOriginal Java code:")
    print(java_code)
    print("\nCleaned Java code:")
    cleaned_java = processor.remove_comments_and_docstrings(java_code, 'java')
    print(cleaned_java)

    print("\nJava AST sequence:")
    try:
        ast_sequence = processor.generate_ast_sequence(java_code, 'java')
        print(ast_sequence)
    except Exception as e:
        print(f"AST generation failed: {e}")
