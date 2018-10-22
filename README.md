# FGST

### Introduction

Currently, fully stationless bike sharing systems, such as Mobike and Ofo are becoming increasingly popular in both China and some other big cities in the world. Different from traditional bike sharing systems that have to build a set of bike stations at different locations of a city and each station is associated with a fixed number of bike docks, there are no stations in stationless bike sharing systems. Thus users can flexibly check-out/return the bikes at arbitrary locations. Such a brand new bike-sharing mode better meets people's short travel demand but also poses new challenges for performing effective system management due to the extremely unbalanced bike usage demand in different areas and time intervals. Therefore, it is crucial to accurately predict the future bike traffic for helping the service provider rebalance the bikes timely. In this paper, we propose a Fine-Grained Spatial-Temporal based regression model named FGST to predict the future bike traffic in a stationless bike sharing system. We motivate the method via discovering the spatial-temporal correlation and the localized conservative rules of the bike check-out and check-in patterns. Our model also makes use of external factors like POI features to improve the prediction. A number of experiments on a large Mobike trip dataset demonstrate that our approach outperforms baseline methods by a significant margin.

### Overview

Here we proviede the implementation of a Fine-Grained Spatial-Temporal based regression model in Python, along with  the derivation of the STMM optimization problem. We also give the public accessable dataset by posing the download site [here](https://www.dropbox.com/s/vfty8kqb4s1o0rr/MOBIKE_CUP_2017.zip?dl=0)

### Dataset

Mobike dataset can be download [here](	https://www.dropbox.com/s/vfty8kqb4s1o0rr/MOBIKE_CUP_2017.zip?dl=0)


