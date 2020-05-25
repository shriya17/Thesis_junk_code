from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
import os

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from detectron2.data.datasets import register_coco_instances

path2json_train  = "/globalwork/mishra/person_annotations_train2017.json"
path2imgdir_train = "/globalwork/mishra/person_train2017"
path2json_test  = "/globalwork/mishra/person_annotations_val2017.json"
path2imgdir_test = "/globalwork/mishra/person_val2017"

register_coco_instances("train_data", {}, path2json_train, path2imgdir_train)
register_coco_instances("test_data", {}, path2json_test, path2imgdir_test)

print("Loading Configurations...")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ('train_data',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TRAIN = 0.01

print("Training starts now ...")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
print("End of Training...")

print("Evaluation starts now ...")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01   # set the testing threshold for this model
cfg.DATASETS.TEST = ('test_data', )
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("test_data", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test_data")
inference_on_dataset(trainer.model, val_loader, evaluator)
print("End of Evaluation")

print("Training / Evaluation over!")
