r"""Keep AutoAPI cross-references within their top-level package."""

from sphinx.domains.python import PythonDomain


class IsolatedPythonDomain(PythonDomain):
    r"""Resolve references from ``imtskit`` and ``imtskit_models`` locally."""

    isolated_packages = frozenset({"imtskit", "imtskit_models"})

    def resolve_xref(
        self,
        env,
        fromdocname,
        builder,
        type,
        target,
        node,
        contnode,
    ):
        if not node.get("py:module") and not node.get("py:class"):
            for package in self.isolated_packages:
                if fromdocname.startswith(f"autoapi/{package}/"):
                    node["py:module"] = package
                    break
        return super().resolve_xref(
            env, fromdocname, builder, type, target, node, contnode
        )

    def find_obj(
        self,
        env,
        modname,
        classname,
        name,
        type,
        searchmode=0,
    ):
        matches = super().find_obj(env, modname, classname, name, type, searchmode)
        context = modname or classname
        package = context.split(".", maxsplit=1)[0] if context else None
        if package not in self.isolated_packages:
            return matches
        return [
            match
            for match in matches
            if match[0] == package or match[0].startswith(f"{package}.")
        ]


def setup(app):
    r"""Register the package-isolating Python domain."""
    app.add_domain(IsolatedPythonDomain, override=True)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
