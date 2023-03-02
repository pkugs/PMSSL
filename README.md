# PMSSL

## 1 Environment configuration
1. computer with GPU

2. python	
	* numpy	
	* scipy	
	* tensorflow	
	* matplotlib	
## 2	Test data
  * The test data is given in data/input, 
  * The direct signal and first-order reflections are pre-extracted, as well as the room geometry and sound source localization.`
## 3 Run` 
  * The PMSSL model is conducted using the pre-trained U-net in model/enhance-Unet-22-6-4/...
  * run c_train_with_phy_model.py 
