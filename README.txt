Workflow:

1. Run app.py
2. In a new terminal run python create_dummy_files.py
3. In the same terminal run create_queues.py
4. In the same terminal run push.py (populates the create queues with the dummy csv) I usually wait until all 20 transactions are pushed (idk if thats necessary)
5. Now in an bash terminal run the following:
    mpiexec -n 6 python ml_prediction_service.py
    this takes a lot of time usually 3-5 minutes
    (if you go to queue_data.json you can see that stuff is happening during this runtime)
6. Now run pull.py in a different terminal (fetch all the stuff that ml_prediction_service did i guess idk)