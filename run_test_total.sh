# coach + slot type regularization

# PLAYMUSIC
python test.py --tgt_dm PlayMusic --n_samples 0 --test_mode testset
python test.py --tgt_dm PlayMusic --n_samples 50 --test_mode testset

# ADDTOPLAYLIST
python test.py --tgt_dm AddToPlaylist --n_samples 0 --test_mode testset
python test.py --tgt_dm AddToPlaylist --n_samples 50 --test_mode testset

# BOOKRESTAURANT
python test.py --tgt_dm BookRestaurant --n_samples 0 --test_mode testset
python test.py --tgt_dm BookRestaurant --n_samples 50 --test_mode testset

# GETWEATHER
python test.py --tgt_dm GetWeather --n_samples 0 --test_mode testset
python test.py --tgt_dm GetWeather --n_samples 50 --test_mode testset

# RATEBOOK
python test.py --tgt_dm RateBook --n_samples 0 --test_mode testset
python test.py --tgt_dm RateBook --n_samples 50 --test_mode testset

# SEARCHCREATEWORK
python test.py --tgt_dm SearchCreativeWork --n_samples 0 --test_mode testset
python test.py --tgt_dm SearchCreativeWork --n_samples 50 --test_mode testset

# FINDSCREENINGEVENT
python test.py --tgt_dm SearchScreeningEvent --n_samples 0 --test_mode testset
python test.py --tgt_dm SearchScreeningEvent --n_samples 50 --test_mode testset
