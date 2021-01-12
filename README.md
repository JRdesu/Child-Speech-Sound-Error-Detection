# Child Speech Sound Disorder Detection
 Training data  	  Dev data            Test data
--------------      ------------       ---------------------------
	150					30	                  36+20
(train_man)			  (dev_man)			test_atypical+test_man 


Prepare Data
> text feature
mannual syllable level  --> text (pickle format)
> acoustic feature:
frame-level obtained from kaldi

train_man.text dev_man.text test_man.text  test_atypical_normal.text test_atypical.text
