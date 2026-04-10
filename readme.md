# Mujoco leg simulator

## Model 

The model is created with 2 leg and a box. Right leg is copied from left leg. Pay attention of the pos of the body.

To put the same rotational pos in different motors, as well as to set the same to real object, the axis of right leg is changed into -1.

The mass of legs and objects is near to real object.  

All sensor data is about imu and joint position. Among them, the framequat data, by the way, the imu data, is quat propertion, you should change it into euler degree.

The equality is created by two site model. Because of recriction of equality is not enough, you could set `solref` to strenghen connection.

### Data detail

motor positive direction: countercloc kwise. (inverse: clockwise)

motor zero position: level of the legs

### Param

Notice these parameters: mass, damping, ctrlrange. 