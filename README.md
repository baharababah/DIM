# Distributed-Intelligence-Model-for-IoT-Applications

<p style='text-align: justify;'> 

This work proposes a model that provides an IoT gateway with  the  intelligence needed  to  extract  the knowledge from sensors’ data in order to make the decision locally without needing to send all raw data to the cloud over the Internet.  When the gateway is unable to process a task locally, the data and task are offloaded to the cloud. 
</p>


## Model Demonstration
The model was demonstrated by building smart home distributed intelligence application. 
The Model architecture is presented below:

## The features of the smart home distributed intelligence application:
* Controll home appliences using machine learning.
* Local data processing of the most real time data for fast decision and action making.
* Offload coming tasks when the gateway is overload. 
* Avoid sending most of the data to the cloud for security and privacy purposes.
* Enable IoT network to be more resourceful in terms of energy and communication bandwidth.


<p align="center">
  <img src="/Project2.jpg" width="200" height="250" class="center" >
</p>

# Connect the devices using Node-Red
<p align="center">
  <img src="/project.jpg" width="250" height="250"  >
</p>

* IoT gateway: Raspberry Pi,  zigbee usb.
* End devices: smart plug, smart bulb, motion sensor, temperature sensor, and two mobile phones.

## Collect the data from sensors:
<p align="center">
  <img src="/CollectDataset.png" width="600" height="300" >
</p>

* ETL (Extract, Transform, Load). 


## Cloud platforms:
* IBM cloud.

## Build Neural Networks algorithm that able to controll the home appliences : 
* Programming: 
 * Python ( NumPy, Pandas, Keras, Matplotlib)
 * Neural Networks (MLPNN, LSTM, and GRUs)
 * Node-Red
 * JSON
 * Java Script
 * train the choosen Neural Networks algorithm on the collected data. 
 * save the model in .h5 file in order to use it for controlling home appliences.

# Build flow-based tool using Node-Red that connects end devices, gateway, and cloud.
* Include the trained Neural Networks algorithm (.h5) in the Node-Red tool.
* When any event happen at the house, the system will response to the event based on Neural Networks trained model that is implemented on Node-Red.
  

# The application now knows when to turn the light on/off:
* Suppose the collected dataset illusrats that you usually go to the bed at 9:00PM on weekdays and 1:00AM on weekends.
  * The appliaction will turn the light off 9:00PM on weekdays. 
  * The appliaction will turn the light off 1:00AM on weekends. 


## For more details, open [DIM](https://mspace.lib.umanitoba.ca/bitstream/handle/1993/35511/Rababah_Baha.pdf?sequence=1)

