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

import numpy as np
from xml.etree import ElementTree as ET


class IntegralCurveOperator:
    def __init__(self, operator_element):
        self.element = operator_element
        self.attrs = self._get_ic_attrs()

    def _get_ic_attrs(self):
        return self.element.find(
            "./Object[@name='ViewerOperator']/Object[@name='IntegralCurveAttributes']"
        )

    def get_field(self, name):
        field = self.attrs.find(f"./Field[@name='{name}']")
        return field.text.strip() if field is not None and field.text else None

    def set_field(self, name, value, field_type="string"):
        field = self.attrs.find(f"./Field[@name='{name}']")
        if field is None:
            field = ET.SubElement(self.attrs, "Field", name=name, type=field_type)
        field.text = str(value)

    def get_point_list(self):
        raw = self.get_field("pointList")
        points = np.array(list(map(float, raw.split())) if raw else [])
        points = points.reshape(-1, 3) if len(points) % 3 == 0 else points
        return points

    def set_point_list(self, points):
        if not hasattr(points, "shape") or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected a numpy array of shape (N, 3)")
        flat = points.flatten()
        text_value = " ".join(map(str, flat))
        self.set_field("pointList", text_value, field_type="doubleVector")
