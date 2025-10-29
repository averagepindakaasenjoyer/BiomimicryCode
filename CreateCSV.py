import os
import csv
import xml.etree.ElementTree as ET
import dotenv

dotenv.load_dotenv()

images_dir = os.getenv("IMAGES_DIR")
annotations_dir = os.getenv("ANNOTATIONS_DIR")
output_csv = os.getenv("OUTPUT_CSV")

with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["image_path", "annotation_path", "num_boxes", "boxes"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate through all XML annotation files
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Image filename
        filename = root.find("filename").text
        image_path = os.path.join(images_dir, filename)

        # Collect bounding boxes and classes
        objects = root.findall("object")
        num_boxes = len(objects)

        boxes_info = []
        for obj in objects:
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = bbox.find("xmin").text
            ymin = bbox.find("ymin").text
            xmax = bbox.find("xmax").text
            ymax = bbox.find("ymax").text
            boxes_info.append(f"{cls}:({xmin},{ymin},{xmax},{ymax})")

        boxes_str = "[" + ", ".join(boxes_info) + "]"

        writer.writerow({
            "image_path": image_path,
            "annotation_path": xml_path,
            "num_boxes": num_boxes,
            "boxes": boxes_str
        })

print(f"✅ CSV created successfully: {output_csv}")
