

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
