#############################
####### Slot Filling ########
#############################

python main.py --tgt_dm PlayMusic --n_samples 0 --epoch 5
python main.py --tgt_dm GetWeather --n_samples 0 --epoch 5
python main.py --tgt_dm SearchScreeningEvent --n_samples 0 --epoch 5
python main.py --tgt_dm RateBook --n_samples 0 --epoch 5
python main.py --tgt_dm SearchCreativeWork --n_samples 0 --epoch 5
python main.py --tgt_dm BookRestaurant --n_samples 0 --epoch 5
python main.py --tgt_dm AddToPlaylist --n_samples 0 --epoch 5

python main.py --tgt_dm PlayMusic --n_samples 20 --epoch 5
python main.py --tgt_dm GetWeather --n_samples 20 --epoch 5
python main.py --tgt_dm SearchScreeningEvent --n_samples 20 --epoch 5
python main.py --tgt_dm RateBook --n_samples 20 --epoch 5
python main.py --tgt_dm SearchCreativeWork --n_samples 20 --epoch 5
python main.py --tgt_dm BookRestaurant --n_samples 20 --epoch 5
python main.py --tgt_dm AddToPlaylist --n_samples 20 --epoch 5

python main.py --tgt_dm PlayMusic --n_samples 50 --epoch 5
python main.py --tgt_dm GetWeather --n_samples 50 --epoch 5
python main.py --tgt_dm SearchScreeningEvent --n_samples 50 --epoch 5
python main.py --tgt_dm RateBook --n_samples 50 --epoch 5
python main.py --tgt_dm SearchCreativeWork --n_samples 50 --epoch 5
python main.py --tgt_dm BookRestaurant --n_samples 50 --epoch 5
python main.py --tgt_dm AddToPlaylist --n_samples 50 --epoch 5