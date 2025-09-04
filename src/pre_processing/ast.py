import numpy as np
import tree_sitter_java as tsjava
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node


JAVA_LANGUAGE = Language(tsjava.language())
PYTHON_LANGUAGE = Language(tspython.language())


def parse_with_treesitter(code: str, language: Language) -> Node:
    parser = Parser(language)
    tree = parser.parse(bytes(code, "utf8"))
    return tree.root_node


def get_node_types(node: Node) -> list[str]:
    types = [node.type]
    for child in node.children:
        types.extend(get_node_types(child))
    return types


def serialise_ast(node: Node) -> str:
    types = get_node_types(node)
    return " ".join(types)


def get_ast_representation(code: str, language: str) -> str:
    if language == 'python':
        ast_node = parse_with_treesitter(code, PYTHON_LANGUAGE)
    elif language == 'java':
        ast_node = parse_with_treesitter(code, JAVA_LANGUAGE)
    else:
        raise ValueError(f"Unsupported language: {language}")

    return serialise_ast(ast_node)


def extract_structural_features(ast_repr: str) -> np.ndarray:
    features = []

    # 1. AST Depth estimation (count nested parentheses)
    max_depth = 0
    current_depth = 0
    for char in ast_repr:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    features.append(max_depth)

    # 2. Node type counts - common programming constructs
    # Python-specific nodes
    python_nodes = [
        'function_definition', 'class_definition', 'if_statement',
        'for_statement', 'while_statement', 'assignment', 'call',
        'import_statement', 'import_from_statement', 'return_statement',
        'expression_statement', 'with_statement', 'try_statement',
        'lambda', 'list_comprehension', 'dictionary_comprehension'
    ]

    # Java-specific nodes
    java_nodes = [
        'method_declaration', 'class_declaration', 'variable_declarator',
        'if_statement', 'for_statement', 'while_statement',
        'assignment_expression',
        'method_invocation', 'import_declaration', 'return_statement',
        'expression_statement', 'try_statement', 'catch_clause',
        'synchronized_statement', 'switch_statement'
    ]

    # Count occurrences of common node types
    all_nodes = list(set(python_nodes + java_nodes))
    for node_type in all_nodes:
        count = ast_repr.count(node_type)
        features.append(count)

    # 3. AST size and complexity metrics
    features.append(len(ast_repr))  # Total AST string length
    # Number of nodes (opening parentheses)
    features.append(ast_repr.count('('))
    features.append(ast_repr.count(' '))  # Rough token count (spaces)

    # 4. Identifier and literal patterns
    # Variable/function names
    features.append(ast_repr.count('identifier'))
    features.append(ast_repr.count('string'))  # String literals
    features.append(ast_repr.count('integer'))  # Integer literals
    features.append(ast_repr.count('float'))  # Float literals

    # 5. Control flow complexity indicators
    features.append(ast_repr.count('if_statement') +
                    ast_repr.count('elif_clause'))
    features.append(ast_repr.count('for_statement') +
                    ast_repr.count('while_statement'))
    features.append(ast_repr.count('try_statement') +
                    ast_repr.count('except_clause'))

    # 6. Code style indicators
    features.append(ast_repr.count('comment'))  # Comments
    features.append(ast_repr.count('pass_statement'))  # Empty statements

    return np.array(features, dtype=np.float32)
