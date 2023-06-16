## Fast and Robust Mid-Air Gesture Typing for AR Headsets using 3D Trajectory Decoding
The code in this repository is an implementation of the neural motion model in the paper.
    
We present a fast mid-air gesture keyboard for head-mounted optical see-through augmented reality (OST AR) that supports users in articulating word patterns by merely moving their own physical index finger in relation to a virtual keyboard plane without a need to indirectly control a visual 2D cursor on a keyboard plane. To realize this, we introduce a novel decoding method that directly translates users' three-dimensional fingertip gestural trajectories into their intended text.
We evaluate the efficacy of the system in three studies that investigate various design aspects, such as immediate efficacy, accelerated learning, and whether it is possible to maintain performance without providing visual feedback. We find that the new 3D trajectory decoding design results in significant improvements in entry rates while maintaining low error rates. In addition, we demonstrate that users can maintain their performance even without fingertip and gesture trace visualization.
The implementations in this repository can enable readers better replicate our experiments and use the models as a rapid synthetic tools. 

### File Description 
- **main.py** : The main function for training and evaluating the model.
- **model.py** : The model of the 3D trajectory decoding method.
- **data_process.py** : The data processing method for the model.
- **train.py** : The training method for the model.
- **eval.py** : The evaluation function for the model.
- **utils.py** : The utility functions.
- **metrics.py** : The evaluation metrics for the model.
- **getData.py** : The data generation function for the model.
- **decoders.py** : The decoding methods for the model.
- **kb-layout.txt** : The layout of the keyboard.
- **GestureDecoderServer** : The folder for the server of the 3D trajectory decoder.
- - **GestureDecoder.py** : The gesture decoder.
- - **GestureDecoderClient.py** : The test client of the 3D trajectory decoder.
- - **GestureDecoderServer.py** : The server of the 3D trajectory decoder.
- - **GestureDecoderUtils.py** : The utility functions for the 3D trajectory decoder.


## Getting started 

### 1. Setup Environment

```
  $ pip install -r requrements.txt
```  
  
### 2. Data Preparation

```
  $ python data_process.py
```


### 3. Model Training & Evaluation
```
  $ python main.py
```

### 4. Model Host
Download the n-gram language model from [here](https://www.keithv.com/software/mobiletext/) or it can be any other n-gram language models in arpa format, and put it in the main folder.
Change the n-grame language model path in GestureDecoder.py and the trained 3D trajectory decoder model path in GestureDecoderServer.py and run the server.

```
$ python GestureDecoderServer/GestureDecoderServer.py
```
Run the test client to test the 3D trajectory decoder.

```
$ python GestureDecoderServer/GestureDecoderClient.py
```
