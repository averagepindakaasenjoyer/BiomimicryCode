import os
import xml.etree.ElementTree as ET
import dotenv


dotenv.load_dotenv()

images_dir = os.getenv("IMAGES_DIR")
annotations_dir = os.getenv("ANNOTATIONS_DIR")
labels_dir = os.getenv("LABELS_DIR")

print(f"Images directory: {images_dir}")
print(f"Annotations directory: {annotations_dir}")
print(f"Labels directory: {labels_dir}")


classes = ["flower", "flower-bud"]

def voc_to_yolo(size, box):
    # size: (width, height), box: (xmin, ymin, xmax, ymax)
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    box_w = xmax - xmin
    box_h = ymax - ymin
    # normalize
    return x_center / w, y_center / h, box_w / w, box_h / h

for xml_name in os.listdir(annotations_dir):
    if not xml_name.endswith(".xml"):
        continue
    xml_path = os.path.join(annotations_dir, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    label_name = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_name)

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in classes:
            # optionally: append new class or skip
            print(f"Warning: class '{cls_name}' not in classes list; skipping")
            continue
        cls_id = classes.index(cls_name)
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        x_c, y_c, bw, bh = voc_to_yolo((img_w, img_h), (xmin, ymin, xmax, ymax))
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
