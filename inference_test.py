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
        'src_image_dir': '/Users/wug/temp/images/',
        'src_image_file': '2020_08_13__8670_48411.tif',
        # The actual annoation file will be saved into ../annotations.
        'annotation_dir': '/Users/wug/temp/images/'
    }

    json_path, annoation_json = infer_task(request=request, callbacks=None)

    print(json_path)
    print(annoation_json)


if __name__ == '__main__':
    run_test()