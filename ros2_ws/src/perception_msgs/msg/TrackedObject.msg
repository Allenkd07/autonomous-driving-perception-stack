# One tracked object with its estimated distance (see perception.core.types.ObjectState)
int32 track_id
int32 class_id
string class_name
float32 score
float32[4] box_xyxy     # pixels
float32 distance_m      # NaN if unknown
float32 ttc_s           # time to collision; NaN if not approaching
bool in_ego_lane
