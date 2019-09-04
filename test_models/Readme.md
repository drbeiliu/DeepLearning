Test all model at once.

1. put Model.pkl files in the model folder
2. In test_all.py, find python dictionary model_dict and add your model following this format:
    
    "model_name":[".pkl name(not adding date)", data_in_normalization, data_out_normalization, train_in_size, mix_in_image(AVG or MAX)]

    e.x.:

    "HE_4_AVG_HER":["HE_X2_4fAVG_HER",5542.0,8029.0,4,'AVG']

3. in test_all.py, all model.pkl has a date added to the file name, be sure to add/change correct date in test_all.py by control+F or command+F variable state_dict_path in function modelPredictHER and LE_HEtest. The current models are dated _0825. This issue should be fix later.
4. Run main.py following the examples in test.sh.
    --enlarge and --notenlarge is used only in LE_HE models. HER assumes all input images have size 256 by 256.

    --afterLEPred and --notafterLEPred is used only in HER models. AfterLEPred uses the results in LE_HE predictions as input, whereas notafterLEPred uses original HE images as input.

