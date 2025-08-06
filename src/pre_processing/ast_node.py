

import ast
from typing import Optional


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
    """Parser for converting code to AST representations"""

    @staticmethod
    def parse_python_ast(code: str) -> Optional[AstNode]:
        """Parse Python code into AST representation."""
        try:
            parsed = ast.parse(code)
            return AstParser._convert_ast_node(parsed)
        except (SyntaxError, ValueError) as e:
            # Return None for code with syntax errors
            return None

    @staticmethod
    def parse_java_ast(code: str) -> Optional[AstNode]:
        """Parse Java code into simplified AST representation.

        Note: This is a simplified parser since full Java parsing requires
        additional dependencies. In a production system, you would use
        a proper Java parser like javalang or tree-sitter.
        """
        # Simplified Java parsing - extract basic structural elements
        lines = code.split('\n')
        tokens = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue

            # Extract basic Java constructs
            if 'class ' in line:
                tokens.append('ClassDeclaration')
            elif 'interface ' in line:
                tokens.append('InterfaceDeclaration')
            elif 'public ' in line or 'private ' in line or 'protected ' in line:
                tokens.append('MethodDeclaration')
            elif 'if(' in line or 'if (' in line:
                tokens.append('IfStatement')
            elif 'for(' in line or 'for (' in line:
                tokens.append('ForStatement')
            elif 'while(' in line or 'while (' in line:
                tokens.append('WhileStatement')
            elif 'try' in line:
                tokens.append('TryStatement')
            elif 'catch' in line:
                tokens.append('CatchClause')
            elif '=' in line and ';' in line:
                tokens.append('AssignmentExpression')

        if not tokens:
            return None

        # Create a simplified AST structure
        root = AstNode('CompilationUnit')
        for token in tokens:
            root.children.append(AstNode(token))

        return root

    @staticmethod
    def _convert_ast_node(node) -> AstNode:
        """Convert Python ast node to our AstNode representation."""
        node_type = type(node).__name__
        children = []

        for child in ast.iter_child_nodes(node):
            children.append(AstParser._convert_ast_node(child))

        return AstNode(node_type, children)

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
