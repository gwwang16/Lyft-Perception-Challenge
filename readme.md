## Lyft Challenge Test

The challenge data is being produced by the [CARLA Simulator](http://carla.org/) an open source autonomous vehicle platform for the testing and derivative of autonomous algorithms.  Gather more data using this simulator



#### Training data

```
https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/train.tar.gz
```



#### Run python client


- Auto mode
```
$ python client_example.py -a -i
```
- Manual mode
```
$ python manual_control -a
```



#### Run Simulator with fixed time-step

```
CarlaUE4.exe -windowed -ResX=800 -ResY=600 -benchmark -fps=5 -carla-server
```



##### Reference:

CARLA Simulator https://github.com/carla-simulator/carla

CARLA Simulator Document http://carla.readthedocs.io/en/latest/

