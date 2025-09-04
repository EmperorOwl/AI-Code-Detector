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
