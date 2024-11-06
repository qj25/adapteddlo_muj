"""Utils for manipulating xml files
"""
import os
import xml.etree.ElementTree as ET
import io
import numpy as np


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]

    Args:
        string (str): String to convert to an array

    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array([float(x) for x in string.split(" ")])


class XMLWrapper:
    def __init__(self, xml_file) -> None:
        self.folder = os.path.dirname(xml_file)
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()

        # resolve file path
        self.resolve_file_path()

    def set_attribute_value(self, element_name, attrib_name, value):
        element = self.get_element(element_name)
        element.set(attrib_name, value)

    def set_EIA_value(self, element_name, iter_name, attrib_name, value):
        # element --> iter --> attribute
        # e.g. worldbody --> joint --> ref
        element = self.get_element(element_name)
        for i, iter in enumerate(element.iter(iter_name)):
            iter.set(attrib_name,value[i])

    # TODO auto_create: auto create the element if fail to find
    def get_element(self, element_name, auto_create=False):
        search_string = f".//{element_name}"
        element = self.root.find(search_string)
        # if element is None and auto_create:
        #     element = ET.Element(element_name)
        #     self.root.append(element)
        return element

    # find an element with a specific attribute value
    def get_element_by_attrib(self, element_name, attrib_name, attrib_value):
        search_string = f".//{element_name}[@{attrib_name}='{attrib_value}']"
        return self.root.findall(search_string)[0]

    def get_attribute_value(self, element_name, attrib_name):
        element = self.get_element(element_name)
        return element.get(attrib_name)

    def get_xml_string(self):
        with io.StringIO() as string:
            string.write(ET.tostring(self.root, encoding="unicode"))
            return string.getvalue()

    # merge with other XMLWrapper object at a particular element
    def merge(
        self,
        xml,
        element_name,
        attrib_name=None,
        attrib_value=None,
        action="extend",
    ):
        merge_point = self.get_element(element_name)
        if attrib_name is not None and attrib_value is not None:
            merge_point = self.get_element_by_attrib(
                element_name, attrib_name, attrib_value
            )
        other_element = xml.get_element(element_name)
        if other_element is not None:
            if action == "extend":
                merge_point.extend(other_element)
            elif action == "append":
                merge_point.append(other_element)

    def merge_multiple(self, xml, element_name_list):
        for e in element_name_list:
            self.merge(xml, e)

    def resolve_file_path(self):
        for node in self.root.findall(".//*[@file]"):
            file = node.get("file")
            abs_path = os.path.abspath(self.folder)
            abs_path = os.path.join(abs_path, file)
            node.set("file", abs_path)
