#!/usr/bin/env python
# coding=utf8

import xml.etree.ElementTree as ET

if __name__ == "__main__":
    tree = ET.parse("./simple_xml.xml")
    root = tree.getroot()

    print("[[[the root element]]]")
    print(root.tag, "--", root.attrib)
    print("")

    print("[[[iterate over root's children]]]")
    for child in root:
        print(child.tag, "--", child.attrib)
    print("")

    print("[[[iterate over all 'neighbor' elments]]]")
    for neighbor in root.iter("neighbor"):
        print(neighbor.attrib)
    print("")

    print("[[[iterate over all 'year' elements]]]")
    for year in root.iter("year"):
        print(year.text)
    print("")

    print("[[[call find() for non-exist element returns None]]]")
    print(root.find("xxx"))
    print("")

    print("[[[call findall() for non-exist elements returns an empty list]]]")
    print(root.findall("xxx"))
    print("")
