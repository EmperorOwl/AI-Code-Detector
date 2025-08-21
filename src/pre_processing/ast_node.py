from typing import Optional

from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava
import tree_sitter_python as tspython


class AstNode:
    """Represents a node in the Abstract Syntax Tree"""

    def __init__(self,
                 node_type: str,
                 children: list['AstNode'] | None = None):
        self.node_type = node_type
        self.children = children or []

    def to_string(self) -> str:
        if not self.children:
            return self.node_type

        children_str = " ".join([child.to_string() for child in self.children])
        return f"({self.node_type} {children_str})"


class AstParser:
    """Parser for converting code to AST representations using tree-sitter"""
    JAVA_LANGUAGE = Language(tsjava.language())
    PYTHON_LANGUAGE = Language(tspython.language())

    @staticmethod
    def parse_with_treesitter(code: str, language: Language) -> Optional[AstNode]:
        """Parse code using tree-sitter and convert to AstNode format."""
        try:
            parser = Parser(language)
            tree = parser.parse(bytes(code, 'utf8'))
            # if tree.root_node.has_error:
            #     return None
            return AstParser._convert_treesitter_node(tree.root_node)
        except Exception as e:
            return None

    @staticmethod
    def _convert_treesitter_node(node: Node) -> AstNode:
        """Convert tree-sitter node to our AstNode representation."""
        node_type = node.type
        children = []

        for child in node.children:
            children.append(AstParser._convert_treesitter_node(child))

        return AstNode(node_type, children)

    @staticmethod
    def parse_python_ast(code: str) -> Optional[AstNode]:
        """Parse Python code into AST representation using tree-sitter."""
        return AstParser.parse_with_treesitter(code, AstParser.PYTHON_LANGUAGE)

    @staticmethod
    def parse_java_ast(code: str) -> Optional[AstNode]:
        """Parse Java code into AST representation using tree-sitter."""
        return AstParser.parse_with_treesitter(code, AstParser.JAVA_LANGUAGE)

    @staticmethod
    def get_ast_representation(code: str, language: str) -> Optional[str]:
        """Get AST string representation for a code snippet."""
        if language == 'python':
            ast_node = AstParser.parse_python_ast(code)
        elif language == 'java':
            ast_node = AstParser.parse_java_ast(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

        if ast_node is None:
            return None

        return ast_node.to_string()
