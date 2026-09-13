"""The stubs are hand-written, so they can drift from the extension.

These tests read the .pyi files that actually ship inside the installed
package and check them against the module at runtime, in both directions: a
stub that promises something the extension does not have, and an extension
function that no stub declares, both fail here.
"""

import ast
import inspect
import os

import pytest

import hypors

PACKAGE_DIR = os.path.dirname(hypors.__file__)

SUBMODULES = [
    "anova",
    "chi_square",
    "common",
    "mann_whitney",
    "proportion",
    "t",
    "z",
]


def load_stub(name):
    """Parse the shipped stub for ``name`` ('' for the package itself)."""
    filename = "{}.pyi".format(name) if name else "__init__.pyi"
    path = os.path.join(PACKAGE_DIR, filename)
    assert os.path.exists(path), "{} is not shipped in the package".format(filename)
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def stub_arg_names(node):
    args = node.args
    names = [a.arg for a in list(args.posonlyargs) + list(args.args)]
    return [n for n in names if n != "self"]


def runtime_arg_names(obj):
    return [
        p.name
        for p in inspect.signature(obj).parameters.values()
        if p.name != "self"
    ]


def public(names):
    return sorted(n for n in names if not n.startswith("_"))


def test_py_typed_ships():
    """Without the marker, type checkers ignore the stubs entirely."""
    assert os.path.exists(os.path.join(PACKAGE_DIR, "py.typed"))


def test_package_all_matches_stub():
    tree = load_stub("")
    declared = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "__all__":
            declared = [el.value for el in node.value.elts]
    assert declared is not None, "__init__.pyi declares no __all__"
    assert sorted(declared) == sorted(hypors.__all__)


@pytest.mark.parametrize("name", SUBMODULES)
def test_stub_functions_exist_and_match(name):
    module = getattr(hypors, name)
    for node in load_stub(name).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        assert hasattr(module, node.name), "{}.{} is stubbed but missing".format(
            name, node.name
        )
        func = getattr(module, node.name)
        assert stub_arg_names(node) == runtime_arg_names(func), (
            "{}.{} arguments differ".format(name, node.name)
        )


@pytest.mark.parametrize("name", SUBMODULES)
def test_every_runtime_function_is_stubbed(name):
    module = getattr(hypors, name)
    stubbed = {
        node.name
        for node in load_stub(name).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    at_runtime = {
        attr
        for attr in public(dir(module))
        if callable(getattr(module, attr)) or inspect.isclass(getattr(module, attr))
    }
    assert at_runtime <= stubbed, "{} has undeclared members: {}".format(
        name, sorted(at_runtime - stubbed)
    )


@pytest.mark.parametrize("name", SUBMODULES)
def test_stub_classes_match(name):
    module = getattr(hypors, name)
    for node in load_stub(name).body:
        if not isinstance(node, ast.ClassDef):
            continue
        assert hasattr(module, node.name)
        cls = getattr(module, node.name)

        declared = set()
        for member in node.body:
            if isinstance(member, ast.AnnAssign):
                declared.add(member.target.id)
                assert hasattr(cls, member.target.id)
            elif isinstance(member, ast.FunctionDef):
                if member.name == "__init__":
                    assert stub_arg_names(member) == runtime_arg_names(cls), (
                        "{}.__init__ arguments differ".format(node.name)
                    )
                    continue
                declared.add(member.name)
                assert hasattr(cls, member.name), "{}.{} is stubbed but missing".format(
                    node.name, member.name
                )

        assert set(public(vars(cls))) <= declared, (
            "{} has undeclared members: {}".format(
                node.name, sorted(set(public(vars(cls))) - declared)
            )
        )
