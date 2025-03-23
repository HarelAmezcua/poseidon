import json
import glob
import os

from PIL import Image

##################################################

##################################################

def loadjson(path, objectsofinterest, img):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """
    with open(path) as data_file:
        data = json.load(data_file)

    translations = []
    rotations = []

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]

        if objectsofinterest is not None and objectsofinterest.lower() not in info['class'].lower():
            continue        

        # Parse translations
        location = info.get('location', [0, 0, 0])
        translations.append([location[0], location[1], location[2]])

        # Parse quaternion
        rot = info.get("quaternion_xyzw", [0, 0, 0, 1])
        rotations.append(rot)

    return {
        "rotations": rotations,
        "translations": translations,
    }

def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs.
    """
    imgs = []

    def add_json_files(path):
        for ext in ['png', 'jpg']:
            for imgpath in glob.glob(os.path.join(path, f"*.{ext}")):
                jsonpath = imgpath.replace(f".{ext}", ".json")
                if os.path.exists(imgpath) and os.path.exists(jsonpath):
                    relative_path = os.path.relpath(imgpath, root)
                    imgs.append((imgpath, relative_path, jsonpath))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path)
                   if os.path.isdir(os.path.join(path, o))]
        if folders:
            for path_entry in folders:
                explore(path_entry)
        else:
            add_json_files(path)

    explore(root)

    return imgs


def select_image_data(imgs, index):
    """
    Select an element from the list of images and return a dictionary with
    "relative_path": (translations, rotations, imgpath).
    """
    if index < 0 or index >= len(imgs):
        raise IndexError("Index out of range.")

    imgpath, relative_path, jsonpath = imgs[index]
    data = loadjson(jsonpath, 'Ketchup', imgpath)

    return {
        "index": index,
        "translation": data["translations"],
        "rotation": data["rotations"],
        "imgpath": imgpath        
    }
