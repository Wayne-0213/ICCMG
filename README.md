# ICCMG

## Environment
- Python 3.9

## Dataset
- The dataset we used are `train_x.json` and `test_x.json`  
- The commit categories are `"perfective"` (x=0), `"adaptive"` (x=1), and `"corrective"` (x=2)  
- `train_id_type3_x.txt` and `test_id_type3_x.txt` are their corresponding identifiers

## Experimental
- `ICCMG_RQy (y=1,2,3,4,5)` is the source code and all experimental results of RQy  
- `Java_data_x_x_500 (x=1,2,3)` is a similar example of the same category retrieved  
- `Java_data_x_500 (x=1,2,3)` is a similar example not retrieved by category  
- The results of the experiment are in `ICCMG_RQy(y=1,2,3,4,5)/Result/Java`

## Get Started
- Set the value of `api_key` in `utils.py`  
- Run the corresponding `.py` file
