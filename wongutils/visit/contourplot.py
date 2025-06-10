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


class ContourPlot:
    def __init__(self, viewer_plot_element):
        self.element = viewer_plot_element
        self.attrs = self._get_contour_attrs()

    def _get_contour_attrs(self):
        return self.element.find("./Object[@name='ContourAttributes']")

    def get_field(self, name):
        field = self.attrs.find(f"./Field[@name='{name}']")
        return field.text.strip() if field is not None and field.text else None

    def set_field(self, name, value, field_type="string"):
        field = self.attrs.find(f"./Field[@name='{name}']")
        if field is None:
            field = ET.SubElement(self.attrs, "Field", name=name, type=field_type)
        field.text = str(value)
