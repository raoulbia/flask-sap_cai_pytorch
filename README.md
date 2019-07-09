## Flask Application with SAP Conversational AI and PyTorch

This repository is based on the [this PyTorch Chatbot Tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).
Originally I had set out to figure out how to isolate the *load model* functionality such that I could 
implement an SAP CAI chatbot using the pre-trained model. In the end I expanded this task to a general
refactoring of the original code into modules. WHhle I managed to do the former, the latter is, at the 
time of writing, not completed. Putting this on hold pending better mastery of the PyTorch framework.

Some Notes:

* the model I used for running the chatbot was produced in Google Colabs with GPU.
* the chatbot works but somewhat unsatisfactory (from a conversational point of view).
* based on the above, the Flask component of this project is on hold for now. 
* the building blocks for developing a working Flask application are in place though. Specifically, 
the script `utils_load_model.py` should be called by `app.py`. 
 

 