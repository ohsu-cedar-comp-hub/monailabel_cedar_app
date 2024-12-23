import json
import warnings
warnings.filterwarnings('ignore')

from lib.configs.segmentation_tissue import SegmentationTissue
from lib.infers import SegmentationTissueInferTask

def run_test():
    infer_config = SegmentationTissue()
    conf = {
        'models': 'segmentation_tissue',
    }
    infer_config.init(name='segmentation_tissue', model_dir='model', conf=conf, planner=None)
    infer_task: SegmentationTissueInferTask = infer_config.infer()

    request = {
        # Local test at MacBookPro
        'src_image_dir': '/Users/wug/temp/images/',
        'src_image_file': '2020_08_13__8670_48411.tif',
        # The actual annoation file will be saved into ../annotations.
        'annotation_dir': '/Users/wug/temp/images/',
        # Test background and foreground segementation only
        # 'out_channels': '2',

        # Test at Windows
        # 'src_image_dir': 'test_data/images/',
        # 'src_image_file': '2020_08_13__8670_48411.tif',
        # # 'src_image_file': '004_3_B_(3)_5.tif',
        # 'src_image_file': '004_3_B_(3)_5.ome.tif',
        # # The actual annoation file will be saved into ../annotations.
        # 'annotation_dir': 'test_data/images/'
        # 'src_image_dir': '/Users/wug/temp/images/',
        # 'src_image_file': '2020_08_13__8670_48411.tif',
        # # The actual annoation file will be saved into ../annotations.
        # 'annotation_dir': '/Users/wug/temp/images/'
    }

    json_path, annoation_json = infer_task(request=request, callbacks=None)

    print(annoation_json)
    print(json_path)

    del annoation_json['result_file'] # So that we can dump json
    with open(json_path, 'w') as geojson_file:
        json.dump(annoation_json, geojson_file)

if __name__ == '__main__':
    run_test()