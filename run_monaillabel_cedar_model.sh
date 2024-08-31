#!/bin/bash

# In case an null exception is thrown with next_sample menu at QuPath, try re-download the data file. Not sure why!
# curl "https://demo.kitware.com/histomicstk/api/v1/item/5d5c07509114c049342b66f8/download" > "downloaded-datasets/JP2K-33003-1.svs"
monailabel start_server --app . --studies downloaded-datasets/ --conf models segmentation_tissue


