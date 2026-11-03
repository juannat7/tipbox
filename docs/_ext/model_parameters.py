"""Parameter tables derived from the live typed model configurations."""

from dataclasses import fields, is_dataclass

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from tipbox.registry import get_system_info


def parameter_rows(config, prefix=""):
    for item in fields(config):
        value = getattr(config, item.name)
        name = prefix + item.name
        if is_dataclass(value):
            yield from parameter_rows(value, name + ".")
        else:
            default = (
                "automatic"
                if value is None
                else format(value, ".8g")
                if isinstance(value, float)
                else str(value)
            )
            yield name, default, item.metadata["units"], item.metadata["description"]


class ModelParameters(SphinxDirective):
    required_arguments = 1

    def run(self):
        model = get_system_info(self.arguments[0]).create_model()
        table = nodes.table(classes=["parameter-table"])
        group = nodes.tgroup(cols=4)
        table += group
        for width in (20, 13, 18, 49):
            group += nodes.colspec(colwidth=width)
        head, body = nodes.thead(), nodes.tbody()
        group += head
        group += body
        header = nodes.row()
        for label in ("Parameter", "Default", "Units", "Meaning"):
            header += nodes.entry("", nodes.paragraph(text=label))
        head += header
        for values in parameter_rows(model.config):
            row = nodes.row()
            for value in values:
                row += nodes.entry("", nodes.paragraph(text=value))
            body += row
        result = [table]
        if model.critical_parameter is not None:
            value = float(model.critical_parameter)
            result.insert(
                0,
                nodes.paragraph(
                    text=f"Default critical parameter: {value:.8g}. Automatic markers follow the physical coefficients where supported; AMOC uses an explicitly calibrated marker."
                ),
            )
        return result


def setup(app):
    app.add_directive("model-parameters", ModelParameters)
    return {"version": "1", "parallel_read_safe": True}
