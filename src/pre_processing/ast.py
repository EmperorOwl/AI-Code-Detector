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
    if not node.children:
        return node.type

    result = f"({node.type}"
    for child in node.children:
        result += " " + serialise_ast(child)
    result += ")"
    return result


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
    # Control flow structures
    control_flow_nodes = [
        'if_statement', 'for_statement', 'while_statement', 'switch_statement',
        'try_statement', 'catch_clause', 'except_clause', 'finally_clause'
    ]

    # Function/method related
    function_nodes = [
        'function_definition', 'method_declaration', 'method_invocation',
        'call', 'lambda', 'constructor_declaration'
    ]

    # Class and object related
    class_nodes = [
        'class_definition', 'class_declaration', 'interface_declaration',
        'field_declaration', 'variable_declarator'
    ]

    # Assignment and expressions
    assignment_nodes = [
        'assignment', 'assignment_expression', 'expression_statement',
        'binary_expression', 'unary_expression'
    ]

    # Imports and declarations
    import_nodes = [
        'import_statement', 'import_from_statement', 'import_declaration'
    ]

    # Data structures
    data_structure_nodes = [
        'list', 'dictionary', 'array_creation_expression', 'array_access',
        'list_comprehension', 'dictionary_comprehension', 'set_comprehension'
    ]

    # Count grouped node types
    node_groups = [control_flow_nodes, function_nodes, class_nodes,
                   assignment_nodes, import_nodes, data_structure_nodes]

    for node_group in node_groups:
        count = sum(ast_repr.count(node_type) for node_type in node_group)
        features.append(count)

    # 3. Complexity metrics
    features.append(len(ast_repr))  # Total AST string length
    features.append(ast_repr.count('('))  # Number of nodes
    features.append(ast_repr.count(' '))  # Token count approximation

    # Calculate branching factor (average children per node)
    open_parens = ast_repr.count('(')
    if open_parens > 0:
        # Avg branching factor
        features.append(ast_repr.count(' ') / open_parens)
    else:
        features.append(0.0)

    # 4. Identifier and literal patterns
    features.append(ast_repr.count('identifier'))
    features.append(ast_repr.count('string'))
    features.append(ast_repr.count('integer'))
    features.append(ast_repr.count('float'))
    features.append(ast_repr.count('true') +
                    ast_repr.count('false'))  # Boolean literals

    # 5. Advanced control flow complexity
    # Nested control structures
    nested_if_count = 0
    nested_loop_count = 0
    depth = 0
    in_if = False
    in_loop = False

    tokens = ast_repr.split()
    for token in tokens:
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
        elif token in ['if_statement', 'elif_clause']:
            if depth > 1 and in_if:
                nested_if_count += 1
            in_if = True
        elif token in ['for_statement', 'while_statement']:
            if depth > 1 and in_loop:
                nested_loop_count += 1
            in_loop = True

    features.append(nested_if_count)
    features.append(nested_loop_count)

    # 6. Code style and patterns
    features.append(ast_repr.count('comment'))
    features.append(ast_repr.count('pass_statement'))
    features.append(ast_repr.count('return_statement'))

    # Language-specific patterns that might indicate AI generation
    # Long parameter lists (potential AI verbosity)
    features.append(ast_repr.count('parameter_list'))

    # Repetitive patterns (AI tends to be more repetitive)
    tokens = ast_repr.split()
    if len(tokens) > 0:
        unique_tokens = len(set(tokens))
        repetition_ratio = len(tokens) / \
            unique_tokens if unique_tokens > 0 else 0
        features.append(repetition_ratio)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
