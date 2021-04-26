# coach + slot type regularization

# PLAYMUSIC
python test.py --tgt_dm PlayMusic --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm PlayMusic --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm PlayMusic --epoch 5 --n_samples 50 --test_mode testset

# ADDTOPLAYLIST
python test.py --tgt_dm AddToPlaylist --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm AddToPlaylist --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm AddToPlaylist --epoch 5 --n_samples 50 --test_mode testset

# BOOKRESTAURANT
python test.py --tgt_dm BookRestaurant --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm BookRestaurant --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm BookRestaurant --epoch 5 --n_samples 50 --test_mode testset

# GETWEATHER
python test.py --tgt_dm GetWeather --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm GetWeather --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm GetWeather --epoch 5 --n_samples 50 --test_mode testset

# RATEBOOK
python test.py --tgt_dm RateBook --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm RateBook --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm RateBook --epoch 5 --n_samples 50 --test_mode testset

# SEARCHCREATEWORK
python test.py --tgt_dm SearchCreativeWork --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm SearchCreativeWork --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm SearchCreativeWork --epoch 5 --n_samples 50 --test_mode testset

# FINDSCREENINGEVENT
python test.py --tgt_dm SearchScreeningEvent --epoch 5 --n_samples 0 --test_mode testset
python test.py --tgt_dm SearchScreeningEvent --epoch 5 --n_samples 20 --test_mode testset
python test.py --tgt_dm SearchScreeningEvent --epoch 5 --n_samples 50 --test_mode testset
