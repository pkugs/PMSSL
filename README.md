# PMSSL

## 1 Environment configuration
1. computer with GPU
2. python	
	* numpy	
	* scipy	
	* tensorflow		
## 2	Test data
  * The test data is given in data/input, 
  * The direct signal and first-order reflections are pre-estimated, as well as the room geometry and sound source localization.`
  * In each .mat file, **data.ref_sig** refers to 7-channel direct and reflected signals; **data.dir_sig1** refers to the sound source signal. **data.room_shape** refers to room geometry. **data.source_pos** refers to source position, and **data.mic_po**s refers to sound source position
## 3 Run` 
  * The PMSSL model is conducted using the pre-trained U-net in [model/enhance-Unet-22-6-4/](https://disk.pku.edu.cn:443/link/5B58574058E6EF47A023BA3EF5018A36
)，Download and place it in the root directory. 
  * The clean speech dataset is better be utilized in the running procee, Download it from https://disk.pku.edu.cn:443/link/50828A4252C055C1AC8D03CC910FED55 and 
release it in the root directory.
  * run c_train_with_phy_model.py 
