import ast
from pathlib import Path


def test_corrected_training_and_validation_use_matched_controls():
    path = Path(__file__).resolve().parents[1] / "scripts" / "e233_corrected_models.py"
    tree = ast.parse(path.read_text())
    shared = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_shared_step")
    attributes = {
        (node.value.attr, node.attr)
        for node in ast.walk(shared)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name) and node.value.value.id == "batch"
    }
    assert ("controls", "squeeze") in attributes
    assert ("gene_expression", "squeeze") in attributes
