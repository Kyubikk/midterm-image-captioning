import json
from pathlib import Path

data_dir = Path("uitviic_dataset")
ann = json.load(open(data_dir/"uitviic_captions_train2017.json"))
print("keys:", ann.keys())
print("#images:", len(ann["images"]), "#ann:", len(ann["annotations"]))

id2fname = {im["id"]: im["file_name"] for im in ann["images"]}
# in 3 cặp ảnh–caption bất kỳ
for a in ann["annotations"][:3]:
    print({
        "image_path": str(data_dir/"coco_uitviic_train"/id2fname[a["image_id"]]),
        "caption": a["caption"]
    })
