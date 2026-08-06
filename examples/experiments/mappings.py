from experiments.tennis_ball_pick.config import TrainConfig as TennisBallPickTrainConfig
from experiments.tennis_ball_pick.config_place import TrainConfig as TennisBallPlaceTrainConfig
from experiments.twist_bottle_cap.config import TrainConfig as TwistBottleCapTrainConfig
from experiments.twist_bottle_cap.config_lid_grip import TrainConfig as LidGripTrainConfig
from experiments.tube_insertion.config import TrainConfig as TubeInsertionTrainConfig
from experiments.flip_object.config import TrainConfig as FlipObjectTrainConfig

NEW_MAPPING = {
    "tennis_ball_pick": TennisBallPickTrainConfig,
    "tennis_ball_place": TennisBallPlaceTrainConfig,
    "twist_bottle_cap": TwistBottleCapTrainConfig,
    "lid_grip": LidGripTrainConfig,
    "tube_insertion": TubeInsertionTrainConfig,
    "flip_object": FlipObjectTrainConfig,
}
