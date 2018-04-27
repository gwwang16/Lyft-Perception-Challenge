## Lyft Challenge Test

The challenge data is being produced by the [CARLA Simulator](http://carla.org/) an open source autonomous vehicle platform for the testing and derivative of autonomous algorithms.  Gather more data using this simulator



#### Training data

```
https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/train.tar.gz
```



#### Run python client

```
python client_example.py --autopilot --images-to-disk
```



#### Run with fixed time-step

```
CarlaUE4.exe -benchmark -fps=5
```


