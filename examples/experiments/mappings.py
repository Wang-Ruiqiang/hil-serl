# from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
# from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
# from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
# from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.tennis_ball_pick.config import TrainConfig as TennisBallPickTrainConfig
from experiments.tennis_ball_pick.config_place import TrainConfig as TennisBallPlaceTrainConfig
from experiments.twist_bottle_cap.config import TrainConfig as TwistBottleCapTrainConfig
from experiments.tube_insertion.config import TrainConfig as TubeInsertionTrainConfig

# CONFIG_MAPPING = {
#                 "ram_insertion": RAMInsertionTrainConfig,
#                 "usb_pickup_insertion": USBPickupInsertionTrainConfig,
#                 "object_handover": ObjectHandoverTrainConfig,
#                 "egg_flip": EggFlipTrainConfig,
                
#                }

NEW_MAPPING = {
    "tennis_ball_pick": TennisBallPickTrainConfig,
    "tennis_ball_place": TennisBallPlaceTrainConfig,
    "twist_bottle_cap": TwistBottleCapTrainConfig,
    "tube_insertion": TubeInsertionTrainConfig,
}