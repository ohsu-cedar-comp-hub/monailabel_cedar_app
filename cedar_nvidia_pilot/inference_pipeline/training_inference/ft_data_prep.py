import argparse
import os
import json
import skimage
import numpy as np
import random
import cv2

def main(args):
    image_dir = args.image_dir
    anno_dir = args.anno_dir

    image_names = [f[:f.find(".")] for f in os.listdir(image_dir)]
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    anno_paths = []
    for n in image_names:
        matches = [f for f in os.listdir(anno_dir) if n in f]
        if not matches:
            print(f"Warning: No annotation found for {n}")
            continue
        file = matches[0]
        anno_paths.append(os.path.join(anno_dir, file))

    # save image tiles and create json
    tile_size = args.tile_size
    data_json = {"training": []}
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(image_paths)):
        image_name = image_names[i]
        image_path = image_paths[i]
        anno_path = anno_paths[i]

        print(f"Processing {image_name} ({i+1}/{len(image_paths)})")

        # load image and annotations
        image = skimage.io.imread(image_path)
        with open(anno_path) as json_file:
            json_data = json.load(json_file)

        # create mask where each pixel corresponds to annotation class, 0 is unlabeled
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for anno in json_data:
            label = int(anno['properties']['metadata']['class_id'])

            if len(anno['geometry']['coordinates']) > 1: 
                # only use largest polygon if multiple are listed
                sizes = [len(p[0]) for p in anno['geometry']['coordinates']]
                max_ind = sizes.index(max(sizes))
                poly = np.asarray(anno['geometry']['coordinates'][max_ind][0])
            else:
                poly = np.asarray(anno['geometry']['coordinates'][0])
                
            # skip invalid polygons
            if poly.shape[-1] != 2: continue
            if len(poly) < 3: continue

            cv2.fillPoly(mask, pts=[np.array(poly, dtype=np.int32)], color=(label))
        
        # define tile coordinates
        x_points = np.arange(0, image.shape[0] + tile_size, tile_size, dtype=int).tolist()
        y_points = np.arange(0, image.shape[1] + tile_size, tile_size, dtype=int).tolist()

        image_tiles = []
        mask_tiles = []

        for x in x_points[:-1]: 
            for y in y_points[:-1]:
                # only take tiles of tile_size x tile_size dimension
                if (x+tile_size > image.shape[0]) or (y+tile_size > image.shape[1]): continue
                    
                # get image and mask crop for tile
                i_tile = image[x:x+tile_size, y:y+tile_size]
                m_tile = mask[x:x+tile_size, y:y+tile_size]

                image_tiles.append(i_tile)
                mask_tiles.append(m_tile)

        # create random assigment to training set (1) or validation set (0), 80:20 split
        random.seed(42)
        nval = int(.2 * len(image_tiles)) 
        random_folds = [0]*nval + [1]*(len(image_tiles) - nval)
        random.shuffle(random_folds)

        for j in range(len(image_tiles)):
            # save each tile with unique id
            image_tile_name = image_name + "_" + str(j).zfill(4) + "_img.tif"
            mask_tile_name = image_name + "_" + str(j).zfill(4) + "_mask.tif"
            
            skimage.io.imsave(os.path.join(save_dir, image_tile_name), image_tiles[j], check_contrast=False)
            skimage.io.imsave(os.path.join(save_dir, mask_tile_name), mask_tiles[j], check_contrast=False)

            # random 20% of tiles are assigned to validation set
            fold = random_folds[j]
                
            data_json["training"].append({"image": image_tile_name, 
                                        "label": mask_tile_name,
                                        "fold": fold})

    with open(save_dir + "/data_list.json", "w") as f:
        json.dump(data_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")

    parser.add_argument('--image_dir', type=str, required=True,\
                        help='Directory containing the images')
    parser.add_argument('--anno_dir', type=str, required=True,\
                        help='Directory containing the annotations')
    parser.add_argument('--tile_size', type=int, default=1024,\
                        help='Tile dimension')
    parser.add_argument('--save_dir', type=str, required=True,\
                        help='Path to save the generated tiles and JSON')

    args = parser.parse_args()
    main(args)