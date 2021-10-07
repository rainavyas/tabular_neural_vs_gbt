The aim is to fold in more data into the canonical train data split to assess the following

1. Add all climates into train, so there is only a time shift to eval_out data
2. Add all time into train, so there is only a climate shift to eval_out data
3. Add all time and climate into train, so there is no shift to eval_out data

Note that the dev data (for tuning) is also expanded proportionally to the train data

To view the original canonical split, look at the Shifts Challenge.