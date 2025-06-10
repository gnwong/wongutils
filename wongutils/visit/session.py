__copyright__ = """Copyright (C) 2025 George N. Wong"""
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from xml.etree import ElementTree as ET


class VisitSession:
    """
    Possible development ideas:
      - purge ViewerEngineManager/RunningEngines/MachineProfile ?
      - purge other entries that are defaulted?
    """

    # FILE I/O
    def __init__(self, filepath=None):
        self.root = None
        self.tree = None
        if filepath:
            self.load(filepath)

    def load(self, filepath):
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        if 'name' not in self.root.attrib or self.root.attrib['name'] != 'VisIt':
            raise ValueError(f"Invalid VisIt session file: {filepath}")

    def to_dict(self):
        def parse_element(elem):
            parsed = {
                "tag": elem.tag,
                "attrib": dict(elem.attrib),
                "text": elem.text.strip() if elem.text else None,
                "children": [parse_element(e) for e in elem]
            }
            return parsed

        return parse_element(self.root)

    def save(self, filepath):
        if self.tree is not None:
            self.tree.write(filepath, encoding="utf-8", xml_declaration=True)
        else:
            raise RuntimeError("No session data loaded to save.")

    # UTILITY METHODS
    def get_path(self, *names):
        """
        Navigates nested <Object name="..."> elements by name.
        Usage: get_path("VIEWER", "ViewerSubject", "SourceMap")
        Returns: Element or None
        """
        elem = self.root
        for name in names:
            if elem is None:
                return None
            elem = elem.find(f"./Object[@name='{name}']")
        return elem

    def get_child(self, parent, name, tag=None):
        """
        Returns the first child element with the given name.
        """
        if tag is not None:
            return parent.find(f"./{tag}[@name='{name}']")
        for child in parent:
            if child.attrib.get('name') == name:
                return child
        return None

    def get_child_text(self, parent, name, tag=None):
        """
        Returns the text of a child field element with the given name.
        """
        child = self.get_child(parent, name, tag=tag)
        if child is not None and child.text is not None:
            return child.text.strip()
        return None

    def add_child(self, parent, tag, name, text, type='string'):
        """
        Add child field to parent element.
        """
        new_field = ET.Element(tag, name=name, type=type)
        new_field.text = text
        parent.append(new_field)

    # SOURCE METHODS
    def get_source_map(self):
        """
        Returns a dictionary mapping source names to their file paths and plugins.
        """
        source_map = dict()
        sources = self.get_path('VIEWER', 'ViewerSubject', 'SourceMap')
        for source in sources:
            source_name = source.attrib.get('name')
            source_file = source.text
            if source_name and source_file:
                plugin = self.get_source_plugin(source_file)[0]
                source_map[source_name] = dict(
                    file=source_file,
                    plugin=plugin,
                    node=source
                )
        return source_map

    def get_source_plugin(self, source_name):
        """
        Returns the source plugin for a given source name.
        """
        source_plugins = self.get_path('VIEWER', 'ViewerSubject', 'SourcePlugins')
        if source_plugins is not None:
            for plugin in source_plugins:
                if plugin.attrib.get('name') == source_name:
                    return plugin.text.strip(), plugin
        return None, None

    def update_source(self, source_name, filepath):
        """
        Updates the source map with a new file path for the given source name.
        """
        source_map = self.get_source_map()
        if source_name not in source_map:
            raise ValueError(f"Source '{source_name}' not found in session.")
        # update both source file path and try to set plugin based on extension
        source_map[source_name]['node'].text = filepath
        plugin = self.get_source_plugin(filepath)[1]
        if plugin is None:
            if filepath.strip().lower().endswith('vtk'):
                self.add_child(self.get_path('VIEWER', 'ViewerSubject', 'SourcePlugins'),
                               'Field',
                               filepath,
                               'VTK_1.0')
            else:
                raise ValueError(f"No plugin found for source file '{filepath}'.")

    # PLOT METHODS
    def get_plots(self):
        """
        Returns a list of plot objects from the session.
        """
        plots_info = dict()
        plots_args = ['VIEWER', 'ViewerSubject', 'ViewerWindowManager', 'Windows',
                      'ViewerWindow', 'ViewerPlotList']
        plots = self.get_path(*plots_args)
        for plot in plots:
            plot_name = plot.attrib.get('name')
            if plot_name and plot_name.startswith('plot'):
                plot_info = self.get_path(*plots_args, plot_name)
                plot_dict = dict(
                    plugin=self.get_child_text(plot_info, 'pluginID'),
                    source=self.get_child_text(plot_info, 'sourceID'),
                    variable=self.get_child_text(plot_info, 'variableName'),
                    node=plot_info
                )
                plots_info[plot_name] = plot_dict
        return plots_info
